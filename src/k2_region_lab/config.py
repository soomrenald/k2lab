from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _configured_path(environment_name: str, default: str) -> Path:
    return Path(os.environ.get(environment_name, default)).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ModelDirectories:
    """Directories searched for local model components.

    Explicit environment variables make this portable while the defaults match
    the user's existing ComfyUI installation.
    """

    diffusion_models: Path
    text_encoders: Path
    vae: Path

    @classmethod
    def from_environment(cls) -> "ModelDirectories":
        return cls(
            diffusion_models=_configured_path(
                "K2_TURBO_DIR", "~/ComfyUI/models/diffusion_models"
            ),
            text_encoders=_configured_path(
                "K2_TEXT_ENCODER_DIR", "~/ComfyUI/models/text_encoders"
            ),
            vae=_configured_path("K2_VAE_DIR", "~/ComfyUI/models/vae"),
        )


@dataclass(frozen=True, slots=True)
class AppSettings:
    model_directories: ModelDirectories
    data_directory: Path
    worker_python: Path = field(
        default_factory=lambda: Path("~/ComfyUI/venv_rocm/bin/python").expanduser()
    )
    comfyui_root: Path = field(default_factory=lambda: Path("~/ComfyUI").expanduser())
    auto_start_worker: bool = True
    reserve_vram_gb: float = 2.0
    default_width: int = 1024
    default_height: int = 1024

    @classmethod
    def from_environment(cls) -> "AppSettings":
        return cls(
            model_directories=ModelDirectories.from_environment(),
            data_directory=_configured_path("K2LAB_DATA_DIR", "~/.local/share/k2-region-lab"),
            worker_python=_configured_path(
                "K2LAB_WORKER_PYTHON", "~/ComfyUI/venv_rocm/bin/python"
            ),
            comfyui_root=_configured_path("K2LAB_COMFYUI_ROOT", "~/ComfyUI"),
            auto_start_worker=os.environ.get("K2LAB_AUTO_START_WORKER", "1")
            not in {"0", "false", "False"},
            reserve_vram_gb=float(os.environ.get("K2LAB_RESERVE_VRAM_GB", "2.0")),
        )
