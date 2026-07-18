from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from k2_region_lab.memory import memory_policy
from k2_region_lab.output import default_output_directory, validate_filename_prefix


def _configured_path(environment_name: str, default: str) -> Path:
    return Path(os.environ.get(environment_name, default)).expanduser().resolve()


def _configured_executable(environment_name: str, default: str) -> Path:
    # Resolving a venv's Python symlink changes how Python discovers pyvenv.cfg
    # and can silently select the system environment instead.
    return Path(os.environ.get(environment_name, default)).expanduser().absolute()


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
        default_factory=lambda: Path("~/ComfyUI/venv_rocm7/bin/python").expanduser()
    )
    comfyui_root: Path = field(default_factory=lambda: Path("~/ComfyUI").expanduser())
    auto_start_worker: bool = True
    memory_policy: str = "safe_16gb"
    reserve_vram_gb: float = 4.0
    minimum_system_ram_gb: float = 14.0
    cpu_vae: bool = False
    oom_recovery: bool = True
    output_directory: Path | None = None
    filename_prefix: str = "baseline"
    default_width: int = 1024
    default_height: int = 1024

    @classmethod
    def from_environment(cls) -> "AppSettings":
        policy = memory_policy(os.environ.get("K2LAB_MEMORY_POLICY", "safe_16gb"))
        data_directory = _configured_path(
            "K2LAB_DATA_DIR", "~/.local/share/k2-region-lab"
        )
        configured_output = os.environ.get("K2LAB_OUTPUT_DIRECTORY")
        return cls(
            model_directories=ModelDirectories.from_environment(),
            data_directory=data_directory,
            worker_python=_configured_executable(
                "K2LAB_WORKER_PYTHON", "~/ComfyUI/venv_rocm7/bin/python"
            ),
            comfyui_root=_configured_path("K2LAB_COMFYUI_ROOT", "~/ComfyUI"),
            auto_start_worker=os.environ.get("K2LAB_AUTO_START_WORKER", "1")
            not in {"0", "false", "False"},
            memory_policy=policy.key,
            reserve_vram_gb=float(
                os.environ.get("K2LAB_RESERVE_VRAM_GB", policy.reserve_vram_gb)
            ),
            minimum_system_ram_gb=float(
                os.environ.get("K2LAB_MINIMUM_SYSTEM_RAM_GB", policy.minimum_system_ram_gb)
            ),
            cpu_vae=os.environ.get("K2LAB_CPU_VAE", "1" if policy.cpu_vae else "0")
            in {"1", "true", "True"},
            oom_recovery=os.environ.get(
                "K2LAB_OOM_RECOVERY", "1" if policy.oom_recovery else "0"
            )
            not in {"0", "false", "False"},
            output_directory=(
                Path(configured_output).expanduser().resolve()
                if configured_output
                else default_output_directory(data_directory)
            ),
            filename_prefix=validate_filename_prefix(
                os.environ.get("K2LAB_FILENAME_PREFIX", "baseline")
            ),
        )
