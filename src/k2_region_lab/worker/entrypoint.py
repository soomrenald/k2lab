from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from k2_region_lab.config import ModelDirectories
from k2_region_lab.debug import configure_debug_logging
from k2core.backends import ComfyUIBackend, NativeK2Backend
from k2core.inference import (
    BackendName,
    FaceRefinementRequest,
    GenerationRequest,
    ImageEditRequest,
    K2InferenceError,
    PipelineConfig,
    ProgressEvent,
    configured_backend_name,
    convert_error,
)
from k2core.model import discover_model_artifacts
from k2core.worker.protocol import CommandKind, WorkerState
from k2core.worker.runtime import (
    ComfyBaselineRuntime,
    diagnose_accelerator,
    probe_runtime,
    validate_model_artifacts,
)


def emit(
    state: WorkerState,
    message: str,
    *,
    command_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    print(
        json.dumps(
            {
                "command_id": command_id,
                "state": state.value,
                "message": message,
                "payload": payload or {},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def model_directories(payload: dict[str, Any]) -> ModelDirectories:
    return ModelDirectories(
        diffusion_models=Path(payload["diffusion_models"]),
        text_encoders=Path(payload["text_encoders"]),
        vae=Path(payload["vae"]),
        loras=Path(payload.get("loras", "~/ComfyUI/models/loras")).expanduser(),
        upscale_models=Path(
            payload.get("upscale_models", "~/ComfyUI/models/upscale_models")
        ).expanduser(),
        diffusion_model_file=(
            Path(payload["diffusion_model_file"])
            if payload.get("diffusion_model_file") else None
        ),
        text_encoder_file=(
            Path(payload["text_encoder_file"])
            if payload.get("text_encoder_file") else None
        ),
        vae_file=Path(payload["vae_file"]) if payload.get("vae_file") else None,
    )


def forward_progress(callback, event: ProgressEvent) -> None:
    callback(
        int(event.step or 0),
        int(event.total_steps or 0),
        dict(event.detail),
    )


def main() -> int:
    configure_debug_logging("worker")
    logger = logging.getLogger("k2_region_lab.worker.entrypoint")
    logger.debug("worker starting with executable=%s argv=%r", sys.executable, sys.argv)
    try:
        selected_backend = configured_backend_name(logger=logger)
    except K2InferenceError as error:
        logger.error("backend selection failed: %s", error)
        emit(
            WorkerState.ERROR,
            str(error),
            payload={
                "exception_type": type(error).__name__,
                "error": error.to_payload(),
            },
        )
        return 1
    runtime: ComfyBaselineRuntime | None = None
    backend: ComfyUIBackend | NativeK2Backend | None = None
    artifacts = None
    emit(WorkerState.UNLOADED, "GPU worker started")
    for encoded in sys.stdin:
        command_id: str | None = None
        kind: CommandKind | None = None
        try:
            command = json.loads(encoded)
            command_id = command.get("command_id")
            kind = CommandKind(command["kind"])
            payload = command.get("payload", {})
            logger.debug("received worker command id=%s kind=%s", command_id, kind.value)
            comfyui_root = Path(payload.get("comfyui_root", "~/ComfyUI")).expanduser()
            if kind == CommandKind.PROBE:
                emit(WorkerState.PROBING, "Probing worker runtime", command_id=command_id)
                result = probe_runtime(comfyui_root)
                emit(
                    WorkerState.UNLOADED,
                    "Worker runtime probe complete",
                    command_id=command_id,
                    payload=result,
                )
            elif kind == CommandKind.DIAGNOSE_ACCELERATOR:
                emit(WorkerState.PROBING, "Running accelerator diagnostics", command_id=command_id)
                result = diagnose_accelerator(comfyui_root)
                logger.debug("accelerator diagnostics: %r", result)
                emit(
                    WorkerState.READY if result.get("accelerator_available") else WorkerState.ERROR,
                    "Accelerator diagnostics complete",
                    command_id=command_id,
                    payload=result,
                )
            elif kind in (CommandKind.DISCOVER_MODELS, CommandKind.VALIDATE_MODELS):
                emit(WorkerState.VALIDATING, "Validating model artifacts", command_id=command_id)
                directories = model_directories(payload)
                artifacts, manifests = validate_model_artifacts(
                    directories, Path(payload["manifest_directory"])
                )
                compatible = artifacts.complete and all(item["compatible"] for item in manifests)
                emit(
                    WorkerState.READY if compatible else WorkerState.ERROR,
                    "Model artifacts validated" if compatible else "Model validation failed",
                    command_id=command_id,
                    payload={"complete": artifacts.complete, "manifests": manifests},
                )
            elif kind == CommandKind.LOAD_MODEL:
                if runtime is not None and runtime.loaded:
                    emit(
                        WorkerState.READY,
                        "Krea 2 baseline already loaded",
                        command_id=command_id,
                        payload={"reused": True},
                    )
                    continue
                if artifacts is None:
                    directories = model_directories(payload)
                    artifacts = discover_model_artifacts(directories)
                emit(
                    WorkerState.LOADING,
                    "Loading Krea 2 baseline components",
                    command_id=command_id,
                )
                runtime = runtime or ComfyBaselineRuntime(
                    comfyui_root,
                    face_detector_path=(
                        Path(payload["face_detector_path"])
                        if payload.get("face_detector_path") else None
                    ),
                )
                backend = (
                    ComfyUIBackend(runtime)
                    if selected_backend is BackendName.COMFYUI
                    else NativeK2Backend()
                )
                loaded_pipeline = backend.load(
                    PipelineConfig(
                        artifacts=artifacts,
                        memory_policy=str(
                            payload.get("memory_policy", "safe_16gb")
                        ),
                        reserve_vram_gb=float(
                            payload.get("reserve_vram_gb", 4.0)
                        ),
                        minimum_system_ram_gb=float(
                            payload.get("minimum_system_ram_gb", 14.0)
                        ),
                        cpu_vae=bool(payload.get("cpu_vae", False)),
                        oom_recovery=bool(payload.get("oom_recovery", True)),
                    )
                )
                loaded = dict(loaded_pipeline.metadata)
                emit(
                    WorkerState.READY,
                    "Krea 2 baseline components loaded",
                    command_id=command_id,
                    payload=loaded,
                )
            elif kind == CommandKind.VALIDATE_LORAS:
                if runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before validating LoRAs")
                emit(
                    WorkerState.VALIDATING,
                    "Validating LoRA compatibility",
                    command_id=command_id,
                )
                reports = runtime.diagnose_loras(list(payload.get("loras", [])))
                compatible = bool(reports) and all(report["compatible"] for report in reports)
                emit(
                    WorkerState.READY if compatible else WorkerState.ERROR,
                    "LoRA diagnostics complete",
                    command_id=command_id,
                    payload={"compatible": compatible, "loras": reports},
                )
            elif kind == CommandKind.GENERATE_BASELINE:
                if backend is None or runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before generating")
                generation_started_at = time.monotonic()
                emit(
                    WorkerState.RUNNING,
                    "Generation started",
                    command_id=command_id,
                )

                def progress(step: int, total: int, memory: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        f"Denoising step {step}/{total}",
                        command_id=command_id,
                        payload={
                            "step": step,
                            "total_steps": total,
                            "memory": memory,
                        },
                    )

                def runtime_event(message: str, event_payload: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                request = GenerationRequest.from_payload(
                    payload, correlation_id=str(command_id or "")
                )
                generated = backend.generate(
                    request,
                    progress=lambda event: forward_progress(progress, event),
                    diagnostic=runtime_event,
                ).to_payload()
                duration_seconds = time.monotonic() - generation_started_at
                emit(
                    WorkerState.RUNNING,
                    f"Generation run finished in {duration_seconds:.2f} seconds",
                    command_id=command_id,
                    payload={"duration_seconds": duration_seconds},
                )
                emit(
                    WorkerState.READY,
                    "Generation complete",
                    command_id=command_id,
                    payload=generated,
                )
                emit(
                    WorkerState.COMPLETE,
                    "Generation worker releasing GPU and system RAM",
                    command_id=command_id,
                )
                return 0
            elif kind == CommandKind.EDIT_IMAGE:
                if backend is None or runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before image editing")
                emit(
                    WorkerState.RUNNING,
                    "Image editing started",
                    command_id=command_id,
                )

                def edit_progress(step: int, total: int, memory: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        f"Image-edit denoising step {step}/{total}",
                        command_id=command_id,
                        payload={"step": step, "total_steps": total, "memory": memory},
                    )

                def edit_event(message: str, event_payload: dict[str, Any]) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                request = ImageEditRequest.from_payload(
                    payload, correlation_id=str(command_id or "")
                )
                edited = backend.generate(
                    request,
                    progress=lambda event: forward_progress(edit_progress, event),
                    diagnostic=edit_event,
                ).to_payload()
                emit(
                    WorkerState.READY,
                    "Image editing complete",
                    command_id=command_id,
                    payload=edited,
                )
                emit(
                    WorkerState.COMPLETE,
                    "Image-edit worker releasing GPU and system RAM",
                    command_id=command_id,
                )
                return 0
            elif kind == CommandKind.REFINE_FACES:
                if backend is None or runtime is None or not runtime.loaded:
                    raise RuntimeError("load the Krea 2 baseline before refining faces")
                emit(
                    WorkerState.RUNNING,
                    "Face refinement started",
                    command_id=command_id,
                )

                def refinement_event(
                    message: str, event_payload: dict[str, Any]
                ) -> None:
                    emit(
                        WorkerState.RUNNING,
                        message,
                        command_id=command_id,
                        payload=event_payload,
                    )

                request = FaceRefinementRequest.from_payload(
                    payload, correlation_id=str(command_id or "")
                )
                refined = backend.generate(
                    request,
                    diagnostic=refinement_event,
                ).to_payload()
                emit(
                    WorkerState.READY,
                    "Face refinement complete",
                    command_id=command_id,
                    payload=refined,
                )
                emit(
                    WorkerState.COMPLETE,
                    "Face refinement worker releasing GPU and system RAM",
                    command_id=command_id,
                )
                return 0
            elif kind == CommandKind.SHUTDOWN:
                emit(WorkerState.COMPLETE, "GPU worker stopped", command_id=command_id)
                return 0
            else:
                raise ValueError(f"unsupported worker command: {kind.value}")
        except Exception as error:
            logger.exception("worker command failed")
            traceback.print_exc(file=sys.stderr)
            structured = (
                error
                if isinstance(error, K2InferenceError)
                else convert_error(
                    error,
                    backend_name=selected_backend.value,
                    phase=kind.value if kind is not None else "worker_command",
                    correlation_id=str(command_id or ""),
                    gpu_work_started=kind
                    in {
                        CommandKind.LOAD_MODEL,
                        CommandKind.GENERATE_BASELINE,
                        CommandKind.EDIT_IMAGE,
                        CommandKind.REFINE_FACES,
                    },
                )
            )
            emit(
                WorkerState.ERROR,
                str(error),
                command_id=command_id,
                payload={
                    "exception_type": type(error).__name__,
                    "error": structured.to_payload(),
                },
            )
            if kind in {
                CommandKind.LOAD_MODEL,
                CommandKind.GENERATE_BASELINE,
                CommandKind.EDIT_IMAGE,
                CommandKind.REFINE_FACES,
            }:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
