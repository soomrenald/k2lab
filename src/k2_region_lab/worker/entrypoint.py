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
from k2_region_lab.model import discover_model_artifacts
from k2_region_lab.projector import DEFAULT_PROJECTOR_PRESET
from k2_region_lab.regional_prompting import (
    prompt_emphases_from_payload,
    region_definitions_from_payload,
)
from k2_region_lab.worker.protocol import CommandKind, WorkerState
from k2_region_lab.worker.runtime import (
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


def main() -> int:
    configure_debug_logging("worker")
    logger = logging.getLogger("k2_region_lab.worker.entrypoint")
    logger.debug("worker starting with executable=%s argv=%r", sys.executable, sys.argv)
    runtime: ComfyBaselineRuntime | None = None
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
                loaded = runtime.load(
                    artifacts,
                    memory_policy_key=str(payload.get("memory_policy", "safe_16gb")),
                    reserve_vram_gb=float(payload.get("reserve_vram_gb", 4.0)),
                    minimum_system_ram_gb=float(
                        payload.get("minimum_system_ram_gb", 14.0)
                    ),
                    cpu_vae=bool(payload.get("cpu_vae", False)),
                    oom_recovery=bool(payload.get("oom_recovery", True)),
                )
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
                if runtime is None or not runtime.loaded:
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

                generated = runtime.generate(
                    prompt=str(payload.get("prompt", "")),
                    width=int(payload.get("width", 1024)),
                    height=int(payload.get("height", 1024)),
                    steps=int(payload.get("steps", 8)),
                    sampler=str(payload.get("sampler", "euler")),
                    scheduler=str(payload.get("scheduler", "simple")),
                    seed=int(payload.get("seed", 0)),
                    output_directory=Path(payload["output_directory"]),
                    filename_prefix=str(payload.get("filename_prefix", "baseline")),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    emphases=prompt_emphases_from_payload(
                        payload.get("prompt_emphases", [])
                    ),
                    regional_prompting=bool(payload.get("regional_prompting", True)),
                    regional_prompt_strength=float(
                        payload.get("regional_prompt_strength", 1.0)
                    ),
                    regional_outside_penalty=float(
                        payload.get("regional_outside_penalty", 1.0)
                    ),
                    regional_feather_pixels=float(
                        payload.get("regional_feather_pixels", 128.0)
                    ),
                    regional_subject_competition=bool(
                        payload.get("regional_subject_competition", True)
                    ),
                    regional_subject_fill=bool(
                        payload.get("regional_subject_fill", True)
                    ),
                    regional_late_step_scale=float(
                        payload.get("regional_late_step_scale", 0.35)
                    ),
                    regional_lora_delta_adaptation=bool(
                        payload.get("regional_lora_delta_adaptation", False)
                    ),
                    regional_lora_delta_adaptation_gain=float(
                        payload.get("regional_lora_delta_adaptation_gain", 0.35)
                    ),
                    projector_enabled=bool(payload.get("projector_enabled", False)),
                    projector_preset=str(
                        payload.get("projector_preset", "filter_bypass2")
                    ),
                    projector_values=tuple(payload.get("projector_values", ())),
                    projector_multiplier=float(payload.get("projector_multiplier", 1.0)),
                    projector_identity_protection=float(
                        payload.get("projector_identity_protection", 1.0)
                    ),
                    post_upscale=bool(payload.get("post_upscale", False)),
                    upscale_scale=int(payload.get("upscale_scale", 2)),
                    upscale_method=str(payload.get("upscale_method", "lanczos")),
                    upscale_model_path=(
                        Path(payload["upscale_model_path"])
                        if payload.get("upscale_model_path")
                        else None
                    ),
                    loras=list(payload.get("loras", [])),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    progress=progress,
                    event=runtime_event,
                )
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
                if runtime is None or not runtime.loaded:
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

                edited = runtime.edit_image(
                    image_path=Path(payload["image_path"]),
                    output_directory=(
                        Path(payload["output_directory"])
                        if payload.get("output_directory")
                        else None
                    ),
                    prompt=str(payload.get("prompt", "")),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    reference_prompt=str(payload.get("reference_prompt", "")),
                    reference_regions=region_definitions_from_payload(
                        payload.get("reference_regions", [])
                    ),
                    prompt_emphases=prompt_emphases_from_payload(
                        payload.get("prompt_emphases", [])
                    ),
                    loras=list(payload.get("loras", [])),
                    seed=int(payload.get("seed", 0)),
                    steps=int(payload.get("steps", 8)),
                    sampler=str(payload.get("sampler", "euler")),
                    scheduler=str(payload.get("scheduler", "simple")),
                    denoise=float(payload.get("denoise", 0.15)),
                    latent_feather_pixels=int(
                        payload.get("latent_feather_pixels", 64)
                    ),
                    composite_feather_pixels=int(
                        payload.get("composite_feather_pixels", 48)
                    ),
                    edit_entire_image=bool(payload.get("edit_entire_image", False)),
                    preserve_identity=bool(payload.get("preserve_identity", True)),
                    reference_description_retention=float(
                        payload.get("reference_description_retention", 1.0)
                    ),
                    regional_prompt_strength=float(
                        payload.get("regional_prompt_strength", 1.0)
                    ),
                    regional_outside_penalty=float(
                        payload.get("regional_outside_penalty", 1.0)
                    ),
                    regional_feather_pixels=float(
                        payload.get("regional_feather_pixels", 128.0)
                    ),
                    regional_subject_competition=bool(
                        payload.get("regional_subject_competition", True)
                    ),
                    regional_subject_fill=bool(
                        payload.get("regional_subject_fill", True)
                    ),
                    regional_late_step_scale=float(
                        payload.get("regional_late_step_scale", 0.35)
                    ),
                    regional_lora_delta_adaptation=bool(
                        payload.get("regional_lora_delta_adaptation", False)
                    ),
                    regional_lora_delta_adaptation_gain=float(
                        payload.get("regional_lora_delta_adaptation_gain", 0.35)
                    ),
                    projector_enabled=bool(payload.get("projector_enabled", False)),
                    projector_preset=str(
                        payload.get("projector_preset", DEFAULT_PROJECTOR_PRESET)
                    ),
                    projector_values=(
                        tuple(float(value) for value in payload["projector_values"])
                        if payload.get("projector_values") is not None
                        else None
                    ),
                    projector_multiplier=float(
                        payload.get("projector_multiplier", 1.0)
                    ),
                    projector_identity_protection=float(
                        payload.get("projector_identity_protection", 1.0)
                    ),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    progress=edit_progress,
                    event=edit_event,
                )
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
                if runtime is None or not runtime.loaded:
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

                refined = runtime.refine_faces(
                    image_path=Path(payload["image_path"]),
                    output_directory=(
                        Path(payload["output_directory"])
                        if payload.get("output_directory")
                        else None
                    ),
                    regions=region_definitions_from_payload(payload.get("regions", [])),
                    loras=list(payload.get("loras", [])),
                    seed=int(payload.get("seed", 0)),
                    steps=int(payload.get("steps", 8)),
                    denoise=float(payload.get("denoise", 0.15)),
                    crop_size=int(payload.get("crop_size", 512)),
                    padding=float(payload.get("padding", 2.0)),
                    feather=float(payload.get("feather", 0.12)),
                    blend=float(payload.get("blend", 0.5)),
                    lora_scale=float(payload.get("lora_scale", 0.5)),
                    detector_threshold=float(
                        payload.get("detector_threshold", 0.15)
                    ),
                    detector_provider=str(payload.get("detector_provider", "auto")),
                    selected_face_indices=(
                        tuple(int(index) for index in payload["selected_face_indices"])
                        if payload.get("selected_face_indices") is not None
                        else None
                    ),
                    manual_face_paths=tuple(
                        tuple((float(point[0]), float(point[1])) for point in path)
                        for path in payload.get("manual_face_paths", ())
                    ),
                    project_json=(
                        dict(payload["project_json"])
                        if isinstance(payload.get("project_json"), dict)
                        else None
                    ),
                    event=refinement_event,
                )
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
            emit(
                WorkerState.ERROR,
                str(error),
                command_id=command_id,
                payload={"exception_type": type(error).__name__},
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
