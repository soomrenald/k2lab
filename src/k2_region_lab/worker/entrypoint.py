from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from k2_region_lab.config import ModelDirectories
from k2_region_lab.model import discover_model_artifacts
from k2_region_lab.worker.protocol import CommandKind, WorkerState
from k2_region_lab.worker.runtime import (
    ComfyBaselineRuntime,
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
    )


def main() -> int:
    runtime: ComfyBaselineRuntime | None = None
    artifacts = None
    emit(WorkerState.UNLOADED, "GPU worker started")
    for encoded in sys.stdin:
        command_id: str | None = None
        try:
            command = json.loads(encoded)
            command_id = command.get("command_id")
            kind = CommandKind(command["kind"])
            payload = command.get("payload", {})
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
                if artifacts is None:
                    directories = model_directories(payload)
                    artifacts = discover_model_artifacts(directories)
                emit(WorkerState.LOADING, "Loading Krea 2 baseline components", command_id=command_id)
                runtime = runtime or ComfyBaselineRuntime(comfyui_root)
                loaded = runtime.load(artifacts)
                emit(
                    WorkerState.READY,
                    "Krea 2 baseline components loaded",
                    command_id=command_id,
                    payload=loaded,
                )
            elif kind == CommandKind.SHUTDOWN:
                emit(WorkerState.COMPLETE, "GPU worker stopped", command_id=command_id)
                return 0
            else:
                raise ValueError(f"unsupported worker command: {kind.value}")
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(
                WorkerState.ERROR,
                str(error),
                command_id=command_id,
                payload={"exception_type": type(error).__name__},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
