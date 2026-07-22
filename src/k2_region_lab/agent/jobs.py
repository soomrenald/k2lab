from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import math
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from PIL import Image

from k2_region_lab.agent.domain import (
    FileKind,
    GenerationJob,
    JobEvent,
    JobEventPage,
    JobKind,
    JobState,
    JobSubmitRequest,
)
from k2_region_lab.agent.storage import WorkspaceLayout
from k2_region_lab.agent.transfers import TransferError, TransferManager
from k2_region_lab.output import validate_filename_prefix
from k2_region_lab.project import PROJECT_SCHEMA, PROJECT_VERSION, ProjectState, project_state
from k2_region_lab.worker.protocol import CommandKind


class JobError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class WorkerExecutor(Protocol):
    async def run(
        self,
        commands: list[dict[str, Any]],
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> int: ...

    async def cancel(self) -> None: ...


class SubprocessWorkerExecutor:
    def __init__(self, worker_python: Path, working_directory: Path) -> None:
        self._worker_python = worker_python
        self._working_directory = working_directory
        self._process: asyncio.subprocess.Process | None = None

    async def run(
        self,
        commands: list[dict[str, Any]],
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> int:
        if not self._worker_python.is_file() or not os.access(self._worker_python, os.X_OK):
            raise JobError(
                "worker_unavailable",
                "The configured GPU worker runtime is unavailable.",
                503,
            )
        environment = self._worker_environment()
        self._process = await asyncio.create_subprocess_exec(
            str(self._worker_python),
            "-m",
            "k2_region_lab.worker.entrypoint",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._working_directory,
            env=environment,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        for command in commands:
            encoded = json.dumps(command, separators=(",", ":"), allow_nan=False)
            self._process.stdin.write((encoded + "\n").encode("utf-8"))
        await self._process.stdin.drain()
        self._process.stdin.close()
        stderr_task = asyncio.create_task(self._discard_stderr(self._process.stderr))
        try:
            async for encoded in self._process.stdout:
                if len(encoded) > 1024 * 1024:
                    raise JobError(
                        "worker_protocol_invalid",
                        "The remote worker emitted an oversized event.",
                        502,
                    )
                try:
                    event = json.loads(encoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise JobError(
                        "worker_protocol_invalid",
                        "The remote worker emitted an invalid event.",
                        502,
                    ) from error
                if not isinstance(event, dict):
                    raise JobError(
                        "worker_protocol_invalid",
                        "The remote worker emitted an invalid event.",
                        502,
                    )
                await on_event(event)
            return await self._process.wait()
        except BaseException:
            await self.cancel()
            raise
        finally:
            await stderr_task

    async def cancel(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    async def _discard_stderr(stream: asyncio.StreamReader) -> None:
        while await stream.read(64 * 1024):
            pass

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        allowed_names = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "PATH",
            "PYTHONPATH",
            "TMPDIR",
            "VIRTUAL_ENV",
        }
        allowed_prefixes = ("CUDA_", "HIP_", "NVIDIA_", "ROCM_")
        return {
            key: value
            for key, value in os.environ.items()
            if key in allowed_names or key.startswith(allowed_prefixes)
        }


class JobManager:
    def __init__(
        self,
        layout: WorkspaceLayout,
        transfers: TransferManager,
        *,
        worker_python: Path,
        comfyui_root: Path,
        executor_factory: Callable[[], WorkerExecutor] | None = None,
        readiness_callback: Callable[[bool], None] | None = None,
    ) -> None:
        self._layout = layout
        self._transfers = transfers
        self._worker_python = worker_python
        self._comfyui_root = comfyui_root
        self._executor_factory = executor_factory or (
            lambda: SubprocessWorkerExecutor(worker_python, layout.root)
        )
        self._readiness_callback = readiness_callback or (lambda _ready: None)
        self._state_directory = layout.state_directory / "jobs"
        self._event_directory = self._state_directory / "events"
        self._state_directory.mkdir(parents=True, exist_ok=True)
        self._event_directory.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._worker_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._executors: dict[str, WorkerExecutor] = {}
        self._cancelled: set[str] = set()
        self._recover_interrupted()

    async def submit(self, request: JobSubmitRequest) -> GenerationJob:
        state, project = self._validate_request(request)
        async with self._lock:
            existing = self._find_command(request.command_id)
            if existing is not None:
                return existing
            now = datetime.now(UTC)
            job = GenerationJob(
                id=uuid4().hex,
                command_id=request.command_id,
                kind=request.kind,
                project_id=request.project_id,
                state=JobState.QUEUED,
                created_at=now,
                updated_at=now,
            )
            self._write_job(job)
            self._write_project(request.project_id, project)
            task = asyncio.create_task(self._run_job(job.id, request, state, project))
            self._tasks[job.id] = task
            task.add_done_callback(lambda _task: self._tasks.pop(job.id, None))
            return job

    async def get(self, job_id: str) -> GenerationJob:
        async with self._lock:
            return self._read_job(job_id)

    async def events(
        self, job_id: str, *, cursor: str | None = None, limit: int = 200
    ) -> JobEventPage:
        if limit < 1 or limit > 500:
            raise JobError("invalid_page_size", "Event limit must be 1 to 500.")
        offset = self._decode_cursor(cursor)
        async with self._lock:
            self._read_job(job_id)
            items = self._read_events(job_id)
        page = items[offset : offset + limit]
        return JobEventPage(items=page, next_cursor=self._encode_cursor(offset + len(page)))

    async def cancel(self, job_id: str) -> GenerationJob:
        async with self._lock:
            job = self._read_job(job_id)
            if job.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return job
            self._cancelled.add(job_id)
            executor = self._executors.get(job_id)
            job = self._update_job(job, state=JobState.CANCELLED)
        if executor is not None:
            await executor.cancel()
        await self._append_event(
            job_id,
            state=JobState.CANCELLED.value,
            message="Remote job cancelled; the isolated worker was stopped.",
        )
        return await self.get(job_id)

    async def close(self) -> None:
        for job_id in list(self._tasks):
            self._cancelled.add(job_id)
        for executor in list(self._executors.values()):
            await executor.cancel()
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def release_worker_memory(self) -> list[str]:
        job_ids = list(self._tasks)
        for job_id in job_ids:
            try:
                await self.cancel(job_id)
            except JobError:
                continue
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._readiness_callback(False)
        return job_ids

    async def _run_job(
        self,
        job_id: str,
        request: JobSubmitRequest,
        project_state_value: ProjectState,
        project: dict[str, Any],
    ) -> None:
        try:
            async with self._worker_lock:
                if job_id in self._cancelled:
                    return
                await self._set_job_state(job_id, JobState.STARTING)
                await self._append_event(
                    job_id,
                    state=JobState.STARTING.value,
                    message="Starting an isolated GPU worker.",
                )
                payload = await self._job_payload(job_id, request, project_state_value, project)
                executor = self._executor_factory()
                async with self._lock:
                    self._executors[job_id] = executor
                worker_errors: list[str] = []
                output_ids: list[str] = []

                async def on_event(raw: dict[str, Any]) -> None:
                    output_id = await self._handle_worker_event(job_id, raw)
                    if output_id is not None and output_id not in output_ids:
                        output_ids.append(output_id)
                    if str(raw.get("state")) == "error":
                        worker_errors.append(
                            str(raw.get("payload", {}).get("exception_type", "worker_error"))
                        )

                exit_code = await executor.run(self._commands(job_id, request, payload), on_event)
                async with self._lock:
                    self._executors.pop(job_id, None)
                if job_id in self._cancelled:
                    return
                if exit_code != 0 or worker_errors:
                    raise JobError(
                        self._worker_error_code(worker_errors),
                        "The GPU worker could not complete this job.",
                        500,
                    )
                if not output_ids:
                    raise JobError(
                        "worker_output_missing",
                        "The GPU worker completed without a valid output file.",
                        500,
                    )
                await self._set_job_state(
                    job_id,
                    JobState.COMPLETED,
                    output_file_ids=output_ids,
                )
                self._readiness_callback(True)
                await self._append_event(
                    job_id,
                    state=JobState.COMPLETED.value,
                    message="Remote job complete.",
                    payload={"output_file_ids": output_ids},
                )
        except JobError as error:
            if job_id not in self._cancelled:
                self._readiness_callback(False)
                await self._fail_job(job_id, error.code, error.message)
        except TransferError as error:
            if job_id not in self._cancelled:
                self._readiness_callback(False)
                await self._fail_job(job_id, error.code, error.message)
        except Exception:
            if job_id not in self._cancelled:
                self._readiness_callback(False)
                await self._fail_job(job_id, "worker_failed", "The remote GPU worker failed.")
        finally:
            async with self._lock:
                self._executors.pop(job_id, None)

    async def _handle_worker_event(self, job_id: str, raw: dict[str, Any]) -> str | None:
        raw_state = str(raw.get("state", "unknown"))
        message = str(raw.get("message", "Worker event"))[:512]
        raw_payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        output_id: str | None = None
        image_path = raw_payload.get("image_path")
        if isinstance(image_path, str):
            record = await self._transfers.index_existing_file(FileKind.OUTPUTS, Path(image_path))
            output_id = record.id
            raw_payload = {**raw_payload, "output_file_id": output_id}
            raw_payload.pop("image_path", None)
        if raw_state == "error":
            message = "The remote worker reported an error."
            payload = {
                "exception_type": str(raw_payload.get("exception_type", "worker_error"))[:128]
            }
        else:
            payload = self._sanitize_payload(raw_payload)
        step = int(payload.get("step", 0) or 0)
        total = int(payload.get("total_steps", 0) or 0)
        if raw_state == "running":
            progress: dict[str, int] = {}
            if "step" in payload:
                progress["progress_current"] = max(0, step)
            if "total_steps" in payload:
                progress["progress_total"] = max(0, total)
            await self._set_job_state(job_id, JobState.RUNNING, **progress)
            self._readiness_callback(True)
        await self._append_event(job_id, state=raw_state, message=message, payload=payload)
        return output_id

    async def _job_payload(
        self,
        job_id: str,
        request: JobSubmitRequest,
        state: ProjectState,
        project: dict[str, Any],
    ) -> dict[str, Any]:
        loras = await self._resolve_loras(request, state)
        base = await self._base_worker_payload(request)
        try:
            filename_prefix = validate_filename_prefix(request.filename_prefix)
        except ValueError as error:
            raise JobError("filename_prefix_invalid", str(error)) from error
        regions = [self._region_payload(region) for region in state.regions]
        emphases = [
            {
                "scope_id": item.scope_id,
                "phrase": item.phrase,
                "strength": item.strength,
                "occurrence": item.occurrence,
            }
            for item in state.prompt_emphases
        ]
        base.update(
            {
                "output_directory": str(self._layout.destination(FileKind.OUTPUTS.value)),
                "project_json": project,
                "filename_prefix": filename_prefix,
            }
        )
        if request.kind == JobKind.GENERATE:
            base.update(
                {
                    "prompt": state.global_prompt,
                    "width": state.canvas_width,
                    "height": state.canvas_height,
                    "steps": state.steps,
                    "sampler": state.sampler,
                    "scheduler": state.scheduler,
                    "seed": state.seed,
                    "regions": regions,
                    "prompt_emphases": emphases,
                    "regional_prompting": state.regional_prompting,
                    "regional_prompt_strength": state.regional_prompt_strength,
                    "regional_outside_penalty": state.regional_outside_penalty,
                    "regional_feather_pixels": state.regional_feather_pixels,
                    "regional_subject_competition": state.regional_subject_competition,
                    "regional_subject_fill": state.regional_subject_fill,
                    "regional_late_step_scale": (
                        state.regional_late_step_scale if state.regional_relaxation else 1.0
                    ),
                    "regional_lora_delta_adaptation": state.regional_lora_delta_adaptation,
                    "regional_lora_delta_adaptation_gain": state.regional_lora_delta_adaptation_gain,
                    "projector_enabled": state.projector_enabled,
                    "projector_preset": state.projector_preset,
                    "projector_values": list(state.projector_values),
                    "projector_multiplier": state.projector_multiplier,
                    "projector_identity_protection": state.projector_identity_protection,
                    "post_upscale": state.post_upscale,
                    "upscale_scale": state.upscale_scale,
                    "upscale_method": state.upscale_method,
                    "upscale_model_path": await self._optional_file_path(
                        request.upscale_model_file_id, FileKind.UPSCALE_MODELS
                    ),
                    "loras": loras,
                }
            )
            return base

        input_path = await self._input_path(request.input_file_id)
        edit = state.image_edit
        if request.kind == JobKind.EDIT_IMAGE:
            base.update(
                {
                    "image_path": str(input_path),
                    "prompt": edit.global_prompt,
                    "regions": [self._region_payload(region) for region in edit.regions],
                    "reference_prompt": edit.reference_global_prompt,
                    "reference_regions": [
                        self._region_payload(region) for region in edit.reference_regions
                    ],
                    "prompt_emphases": [
                        {
                            "scope_id": item.scope_id,
                            "phrase": item.phrase,
                            "strength": item.strength,
                            "occurrence": item.occurrence,
                        }
                        for item in edit.reference_prompt_emphases
                    ],
                    "loras": self._edit_loras(loras, state),
                    "seed": edit.seed,
                    "steps": edit.steps,
                    "sampler": edit.sampler,
                    "scheduler": edit.scheduler,
                    "denoise": edit.denoise,
                    "latent_feather_pixels": edit.latent_feather_pixels,
                    "composite_feather_pixels": edit.composite_feather_pixels,
                    "edit_entire_image": edit.edit_entire_image,
                    "preserve_identity": edit.preserve_identity,
                    "reference_description_retention": edit.reference_description_retention,
                    "regional_prompt_strength": edit.regional_prompt_strength,
                    "regional_outside_penalty": edit.regional_outside_penalty,
                    "regional_feather_pixels": edit.regional_feather_pixels,
                    "regional_subject_competition": edit.regional_subject_competition,
                    "regional_subject_fill": edit.regional_subject_fill,
                    "regional_late_step_scale": edit.regional_late_step_scale,
                    "regional_lora_delta_adaptation": edit.regional_lora_delta_adaptation,
                    "regional_lora_delta_adaptation_gain": edit.regional_lora_delta_adaptation_gain,
                    "projector_enabled": edit.reference_projector_enabled,
                    "projector_preset": edit.reference_projector_preset,
                    "projector_values": list(edit.reference_projector_values),
                    "projector_multiplier": edit.reference_projector_multiplier,
                    "projector_identity_protection": edit.reference_projector_identity_protection,
                }
            )
            return base

        width, height = await asyncio.to_thread(self._image_size, input_path)
        scale_x = width / max(1, state.canvas_width)
        scale_y = height / max(1, state.canvas_height)
        base.update(
            {
                "image_path": str(input_path),
                "regions": [
                    self._region_payload(region, scale_x=scale_x, scale_y=scale_y)
                    for region in state.regions
                ],
                "loras": loras,
                "seed": state.face_detail_seed,
                "steps": state.face_detail_steps,
                "denoise": state.face_detail_denoise,
                "crop_size": state.face_detail_crop_size,
                "padding": state.face_detail_padding,
                "feather": state.face_detail_feather,
                "blend": state.face_detail_blend,
                "lora_scale": state.face_detail_lora_scale,
                "detector_threshold": state.face_detail_detector_threshold,
                "detector_provider": state.face_detail_detector_provider,
                "selected_face_indices": request.selected_face_indices,
                "manual_face_paths": request.manual_face_paths,
            }
        )
        return base

    async def _base_worker_payload(self, request: JobSubmitRequest) -> dict[str, Any]:
        face_detector_path = await self._optional_file_path(
            request.face_detector_file_id, FileKind.FACE_DETECTION
        )
        if face_detector_path is None:
            face_files = sorted(
                path
                for path in self._layout.destination(FileKind.FACE_DETECTION.value).iterdir()
                if path.is_file() and not path.is_symlink()
            )
            face_detector_path = str(face_files[0]) if face_files else None
        return {
            "comfyui_root": str(self._comfyui_root),
            "diffusion_models": str(self._layout.destination(FileKind.DIFFUSION_MODELS.value)),
            "text_encoders": str(self._layout.destination(FileKind.TEXT_ENCODERS.value)),
            "vae": str(self._layout.destination(FileKind.VAE.value)),
            "loras": str(self._layout.destination(FileKind.LORAS.value)),
            "upscale_models": str(self._layout.destination(FileKind.UPSCALE_MODELS.value)),
            "diffusion_model_file": await self._optional_file_path(
                request.diffusion_model_file_id, FileKind.DIFFUSION_MODELS
            ),
            "text_encoder_file": await self._optional_file_path(
                request.text_encoder_file_id, FileKind.TEXT_ENCODERS
            ),
            "vae_file": await self._optional_file_path(request.vae_file_id, FileKind.VAE),
            "face_detector_path": face_detector_path,
            "manifest_directory": str(self._layout.state_directory / "manifests"),
            "memory_policy": "safe_16gb",
            "reserve_vram_gb": 4.0,
            "minimum_system_ram_gb": 14.0,
            "cpu_vae": False,
            "oom_recovery": True,
        }

    async def _resolve_loras(
        self, request: JobSubmitRequest, state: ProjectState
    ) -> list[dict[str, Any]]:
        if len(request.lora_file_ids) != len(state.loras):
            raise JobError(
                "lora_binding_mismatch",
                "Every project LoRA must have one opaque cloud file binding.",
                409,
            )
        payload = []
        for index, (file_id, lora) in enumerate(zip(request.lora_file_ids, state.loras)):
            record, path = await self._transfers.resolve_file(file_id, required_kind=FileKind.LORAS)
            payload.append(
                {
                    "id": record.id,
                    "name": record.display_name,
                    "path": str(path),
                    "strength": lora.strength,
                    "global": lora.global_scope,
                    "region_ids": list(lora.region_ids),
                    "routing_mode": lora.routing_mode,
                    "trigger_phrase": lora.trigger_phrase,
                    "project_index": index,
                }
            )
        return payload

    @staticmethod
    def _edit_loras(resolved: list[dict[str, Any]], state: ProjectState) -> list[dict[str, Any]]:
        payload = []
        for item, lora in zip(resolved, state.loras):
            if lora.reference_enabled:
                payload.append(
                    {
                        **item,
                        "id": f"reference:{item['id']}",
                        "global": lora.reference_global_scope,
                        "region_ids": list(lora.reference_region_ids),
                        "routing_mode": lora.reference_routing_mode,
                        "trigger_phrase": lora.reference_trigger_phrase,
                    }
                )
            if lora.edit_enabled:
                payload.append(
                    {
                        **item,
                        "id": f"edit:{item['id']}",
                        "global": lora.edit_global_scope,
                        "region_ids": list(lora.edit_region_ids),
                        "routing_mode": lora.edit_routing_mode,
                        "trigger_phrase": lora.edit_trigger_phrase,
                    }
                )
        return payload

    async def _input_path(self, file_id: str | None) -> Path:
        if not file_id:
            raise JobError(
                "input_file_required", "Image editing and face refinement require an input file."
            )
        record, path = await self._transfers.resolve_file(file_id)
        if record.kind not in {FileKind.INPUTS, FileKind.OUTPUTS}:
            raise JobError("input_file_invalid", "The selected file is not an input image.")
        return path

    async def _optional_file_path(self, file_id: str | None, kind: FileKind) -> str | None:
        if not file_id:
            return None
        _record, path = await self._transfers.resolve_file(file_id, required_kind=kind)
        return str(path)

    def _commands(
        self, job_id: str, request: JobSubmitRequest, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        kind = {
            JobKind.GENERATE: CommandKind.GENERATE_BASELINE,
            JobKind.EDIT_IMAGE: CommandKind.EDIT_IMAGE,
            JobKind.REFINE_FACES: CommandKind.REFINE_FACES,
        }[request.kind]
        return [
            {"command_id": f"{job_id}:probe", "kind": CommandKind.PROBE.value, "payload": payload},
            {
                "command_id": f"{job_id}:load",
                "kind": CommandKind.LOAD_MODEL.value,
                "payload": payload,
            },
            {"command_id": request.command_id, "kind": kind.value, "payload": payload},
        ]

    def _validate_request(self, request: JobSubmitRequest) -> tuple[ProjectState, dict[str, Any]]:
        try:
            encoded = json.dumps(request.project, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise JobError("project_invalid", "The project document is invalid.") from error
        if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
            raise JobError("project_too_large", "The project document exceeds 2 MiB.", 413)
        try:
            state = project_state(request.project)
        except (KeyError, TypeError, ValueError) as error:
            raise JobError("project_invalid", "The project document is invalid.") from error
        if (
            request.project.get("schema") != PROJECT_SCHEMA
            or request.project.get("version") != PROJECT_VERSION
        ):
            raise JobError(
                "project_version_mismatch",
                f"Remote jobs require {PROJECT_SCHEMA} version {PROJECT_VERSION}.",
                409,
            )
        if len(request.lora_file_ids) != len(state.loras):
            raise JobError(
                "lora_binding_mismatch",
                "Every project LoRA must have one opaque cloud file binding.",
                409,
            )
        if request.kind in {JobKind.EDIT_IMAGE, JobKind.REFINE_FACES} and not request.input_file_id:
            raise JobError(
                "input_file_required",
                "Image editing and face refinement require an input file.",
            )
        if not 256 <= state.canvas_width <= 4096 or not 256 <= state.canvas_height <= 4096:
            raise JobError("canvas_size_invalid", "Canvas dimensions must be 256 to 4096 pixels.")
        if state.canvas_width % 32 or state.canvas_height % 32:
            raise JobError("canvas_alignment_invalid", "Canvas dimensions must be divisible by 32.")
        points = sum(len(path) for path in request.manual_face_paths)
        if points > 4096 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for path in request.manual_face_paths
            for point in path
        ):
            raise JobError("manual_face_path_invalid", "Manual face paths are invalid.")
        project = copy.deepcopy(request.project)
        for index, lora in enumerate(project.get("loras", [])):
            lora["path"] = (
                f"opaque:{request.lora_file_ids[index]}"
                if index < len(request.lora_file_ids)
                else "opaque:unbound"
            )
        if isinstance(project.get("image_edit"), dict):
            project["image_edit"]["source_image"] = (
                f"opaque:{request.input_file_id}" if request.input_file_id else None
            )
            project["image_edit"]["associated_project"] = None
        project["background_image"] = None
        return state, project

    def _write_project(self, project_id: str, project: dict[str, Any]) -> None:
        path = self._layout.resolve_child(FileKind.PROJECTS.value, f"{project_id}.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(project, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _region_payload(region, *, scale_x: float = 1.0, scale_y: float = 1.0) -> dict[str, Any]:
        return {
            "id": region.region_id,
            "name": region.name,
            "box": {
                "x0": region.box.x0 * scale_x,
                "y0": region.box.y0 * scale_y,
                "x1": region.box.x1 * scale_x,
                "y1": region.box.y1 * scale_y,
            },
            "prompt": region.prompt,
            "face_identity_prompt": region.face_identity_prompt,
            "enabled": region.enabled,
            "priority": region.priority,
            "spatial_role": region.spatial_role,
        }

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                image.verify()
                return image.size
        except (OSError, ValueError) as error:
            raise JobError("input_image_invalid", "The input image is invalid.") from error

    async def _fail_job(self, job_id: str, code: str, message: str) -> None:
        await self._set_job_state(job_id, JobState.FAILED, error_code=code, error_message=message)
        await self._append_event(
            job_id,
            state=JobState.FAILED.value,
            message=message,
            payload={"error_code": code},
        )

    async def _set_job_state(self, job_id: str, state: JobState, **updates: Any) -> GenerationJob:
        async with self._lock:
            return self._update_job(self._read_job(job_id), state=state, **updates)

    def _update_job(self, job: GenerationJob, *, state: JobState, **updates: Any) -> GenerationJob:
        updated = job.model_copy(
            update={"state": state, "updated_at": datetime.now(UTC), **updates}
        )
        self._write_job(updated)
        return updated

    async def _append_event(
        self,
        job_id: str,
        *,
        state: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> JobEvent:
        async with self._lock:
            events = self._read_events(job_id)
            event = JobEvent(
                sequence=len(events),
                state=state[:64],
                message=message[:512],
                payload=payload or {},
                created_at=datetime.now(UTC),
            )
            with self._event_path(job_id).open("a", encoding="utf-8") as output:
                output.write(event.model_dump_json() + "\n")
            return event

    def _read_job(self, job_id: str) -> GenerationJob:
        if not job_id.isalnum() or len(job_id) > 64:
            raise JobError("job_not_found", "The remote job does not exist.", 404)
        try:
            return GenerationJob.model_validate_json(
                (self._state_directory / f"{job_id}.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ValueError) as error:
            raise JobError("job_not_found", "The remote job does not exist.", 404) from error

    def _write_job(self, job: GenerationJob) -> None:
        path = self._state_directory / f"{job.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(job.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def _find_command(self, command_id: str) -> GenerationJob | None:
        for path in self._state_directory.glob("*.json"):
            try:
                job = GenerationJob.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.command_id == command_id:
                return job
        return None

    def _read_events(self, job_id: str) -> list[JobEvent]:
        try:
            return [
                JobEvent.model_validate_json(line)
                for line in self._event_path(job_id).read_text(encoding="utf-8").splitlines()
                if line
            ]
        except FileNotFoundError:
            return []

    def _event_path(self, job_id: str) -> Path:
        return self._event_directory / f"{job_id}.jsonl"

    def _recover_interrupted(self) -> None:
        active = {JobState.QUEUED, JobState.STARTING, JobState.RUNNING}
        for path in self._state_directory.glob("*.json"):
            try:
                job = GenerationJob.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.state in active:
                self._write_job(
                    job.model_copy(
                        update={
                            "state": JobState.FAILED,
                            "error_code": "agent_restarted",
                            "error_message": "The workspace agent restarted during the job.",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )

    @staticmethod
    def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        sensitive = ("path", "prompt", "project", "token", "secret", "credential")

        def clean(value: Any, depth: int = 0) -> Any:
            if depth > 5:
                return None
            if isinstance(value, dict):
                return {
                    str(key)[:128]: clean(item, depth + 1)
                    for key, item in list(value.items())[:256]
                    if not any(part in str(key).casefold() for part in sensitive)
                }
            if isinstance(value, list):
                return [clean(item, depth + 1) for item in value[:256]]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value[:512] if isinstance(value, str) else value
            return str(value)[:512]

        return clean(payload)

    @staticmethod
    def _worker_error_code(errors: list[str]) -> str:
        combined = " ".join(errors).casefold()
        if "outofmemory" in combined or "out_of_memory" in combined:
            return "worker_oom"
        return "worker_failed"

    @staticmethod
    def _encode_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(str(offset).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            offset = int(base64.urlsafe_b64decode(padded).decode())
        except (binascii.Error, UnicodeDecodeError, ValueError) as error:
            raise JobError("invalid_cursor", "The job event cursor is invalid.") from error
        if offset < 0:
            raise JobError("invalid_cursor", "The job event cursor is invalid.")
        return offset
