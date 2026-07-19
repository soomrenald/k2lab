from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from k2_region_lab.memory import memory_policy
from k2_region_lab.output import default_output_directory, validate_filename_prefix
from k2_region_lab.sampling import validate_sampler, validate_scheduler


CONFIG_ENVIRONMENT_NAME = "K2LAB_CONFIG_FILE"
PROJECT_CONFIG_NAME = "k2_region_lab.toml"
USER_CONFIG_PATH = Path("~/.config/k2-region-lab/config.toml").expanduser()


def _absolute_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _absolute_executable(value: str | Path, *, base: Path | None = None) -> Path:
    # Resolving a venv's Python symlink changes how Python discovers pyvenv.cfg
    # and can silently select the system environment instead.
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.absolute()


def discover_worker_python(comfyui_root: Path) -> Path:
    """Choose a GPU-enabled ComfyUI interpreter without assuming CUDA or ROCm."""

    root = comfyui_root.expanduser().resolve()
    preferred = (
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
        root / "venv_rocm7" / "bin" / "python",
        root / "venv_rocm" / "bin" / "python",
        root / "python_embeded" / "python.exe",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate.absolute()
    try:
        discovered = sorted(
            candidate
            for environment in root.iterdir()
            if environment.is_dir()
            for candidate in (environment / "bin" / "python",)
            if candidate.is_file()
        )
    except OSError:
        discovered = []
    if discovered:
        return discovered[0].absolute()
    # Keep a useful, platform-neutral path in diagnostics when no environment
    # exists yet. The GUI also lets the user select an interpreter explicitly.
    return (root / ".venv" / "bin" / "python").absolute()


def _optional_path(value: object, *, base: Path | None = None) -> Path | None:
    text = str(value or "").strip()
    return _absolute_path(text, base=base) if text else None


def _config_file() -> Path | None:
    supplied = os.environ.get(CONFIG_ENVIRONMENT_NAME, "").strip()
    if supplied:
        path = _absolute_path(supplied)
        if not path.is_file():
            raise FileNotFoundError(f"configured K2 Region Lab file does not exist: {path}")
        return path
    project = Path.cwd() / PROJECT_CONFIG_NAME
    if project.is_file():
        return project.resolve()
    if USER_CONFIG_PATH.is_file():
        return USER_CONFIG_PATH.resolve()
    return None


def load_config_file() -> tuple[dict[str, Any], Path | None]:
    path = _config_file()
    if path is None:
        return {}, None
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"K2 Region Lab config must be a TOML table: {path}")
    return document, path


def _table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section [{name}] must be a TOML table")
    return value


def _env_or(table: dict[str, Any], key: str, environment_name: str, default: object):
    return os.environ.get(environment_name, table.get(key, default))


def _boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True, slots=True)
class ModelDirectories:
    """Directories and optional exact files used for local model discovery."""

    diffusion_models: Path
    text_encoders: Path
    vae: Path
    loras: Path = field(
        default_factory=lambda: Path("~/ComfyUI/models/loras").expanduser()
    )
    upscale_models: Path = field(
        default_factory=lambda: Path("~/ComfyUI/models/upscale_models").expanduser()
    )
    diffusion_model_file: Path | None = None
    text_encoder_file: Path | None = None
    vae_file: Path | None = None

    @classmethod
    def from_config(
        cls,
        paths: dict[str, Any],
        models: dict[str, Any],
        *,
        base: Path | None,
    ) -> "ModelDirectories":
        return cls(
            diffusion_models=_absolute_path(
                _env_or(
                    paths,
                    "diffusion_models",
                    "K2_TURBO_DIR",
                    "~/ComfyUI/models/diffusion_models",
                ),
                base=base,
            ),
            text_encoders=_absolute_path(
                _env_or(
                    paths,
                    "text_encoders",
                    "K2_TEXT_ENCODER_DIR",
                    "~/ComfyUI/models/text_encoders",
                ),
                base=base,
            ),
            vae=_absolute_path(
                _env_or(paths, "vae", "K2_VAE_DIR", "~/ComfyUI/models/vae"),
                base=base,
            ),
            loras=_absolute_path(
                _env_or(paths, "loras", "K2_LORA_DIR", "~/ComfyUI/models/loras"),
                base=base,
            ),
            upscale_models=_absolute_path(
                _env_or(
                    paths,
                    "upscale_models",
                    "K2_UPSCALE_MODEL_DIR",
                    "~/ComfyUI/models/upscale_models",
                ),
                base=base,
            ),
            diffusion_model_file=_optional_path(
                _env_or(models, "diffusion_model", "K2_TURBO_MODEL", ""),
                base=base,
            ),
            text_encoder_file=_optional_path(
                _env_or(models, "text_encoder", "K2_TEXT_ENCODER_MODEL", ""),
                base=base,
            ),
            vae_file=_optional_path(
                _env_or(models, "vae", "K2_VAE_MODEL", ""), base=base
            ),
        )

    @classmethod
    def from_environment(cls) -> "ModelDirectories":
        document, path = load_config_file()
        return cls.from_config(
            _table(document, "paths"),
            _table(document, "models"),
            base=path.parent if path else None,
        )


@dataclass(frozen=True, slots=True)
class AppSettings:
    model_directories: ModelDirectories
    data_directory: Path
    worker_python: Path = field(
        default_factory=lambda: discover_worker_python(Path("~/ComfyUI").expanduser())
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
    face_detector_path: Path | None = None
    default_upscale_model: Path | None = None
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 8
    default_sampler: str = "euler"
    default_scheduler: str = "simple"
    default_seed: int = 0
    default_seed_mode: str = "fixed"
    config_file: Path | None = None

    def __post_init__(self) -> None:
        if not 256 <= self.default_width <= 4096:
            raise ValueError("default generation width must be between 256 and 4096")
        if not 256 <= self.default_height <= 4096:
            raise ValueError("default generation height must be between 256 and 4096")
        if not 1 <= self.default_steps <= 100:
            raise ValueError("default generation steps must be between 1 and 100")
        validate_sampler(self.default_sampler)
        validate_scheduler(self.default_scheduler)
        if not 0 <= self.default_seed <= 2_147_483_647:
            raise ValueError("default seed must be between 0 and 2147483647")
        if self.default_seed_mode not in {"fixed", "random", "increment"}:
            raise ValueError(
                "default seed_mode must be fixed, random, or increment"
            )

    @classmethod
    def from_environment(cls) -> "AppSettings":
        document, config_path = load_config_file()
        base = config_path.parent if config_path else None
        paths = _table(document, "paths")
        models = _table(document, "models")
        runtime = _table(document, "runtime")
        generation = _table(document, "generation")
        comfyui_root = _absolute_path(
            _env_or(
                paths,
                "comfyui_root",
                "K2LAB_COMFYUI_ROOT",
                "~/ComfyUI",
            ),
            base=base,
        )
        configured_worker = _env_or(
            paths,
            "worker_python",
            "K2LAB_WORKER_PYTHON",
            "auto",
        )
        policy_from_environment = os.environ.get("K2LAB_MEMORY_POLICY")
        policy = memory_policy(
            str(policy_from_environment or runtime.get("memory_policy", "safe_16gb"))
        )
        policy_controls = {} if policy_from_environment else runtime
        data_directory = _absolute_path(
            _env_or(
                paths,
                "data_directory",
                "K2LAB_DATA_DIR",
                "~/.local/share/k2-region-lab",
            ),
            base=base,
        )
        configured_output = _env_or(
            paths, "output_directory", "K2LAB_OUTPUT_DIRECTORY", ""
        )
        output_directory = (
            _optional_path(configured_output, base=base)
            or default_output_directory(data_directory)
        )
        model_directories = ModelDirectories.from_config(
            paths, models, base=base
        )
        return cls(
            model_directories=model_directories,
            data_directory=data_directory,
            worker_python=(
                discover_worker_python(comfyui_root)
                if str(configured_worker).strip().casefold() in {"", "auto"}
                else _absolute_executable(configured_worker, base=base)
            ),
            comfyui_root=comfyui_root,
            auto_start_worker=_boolean(
                _env_or(
                    policy_controls,
                    "auto_start_worker",
                    "K2LAB_AUTO_START_WORKER",
                    True,
                )
            ),
            memory_policy=policy.key,
            reserve_vram_gb=float(
                _env_or(
                    policy_controls,
                    "reserve_vram_gb",
                    "K2LAB_RESERVE_VRAM_GB",
                    policy.reserve_vram_gb,
                )
            ),
            minimum_system_ram_gb=float(
                _env_or(
                    policy_controls,
                    "minimum_system_ram_gb",
                    "K2LAB_MINIMUM_SYSTEM_RAM_GB",
                    policy.minimum_system_ram_gb,
                )
            ),
            cpu_vae=_boolean(
                _env_or(policy_controls, "cpu_vae", "K2LAB_CPU_VAE", policy.cpu_vae)
            ),
            oom_recovery=_boolean(
                _env_or(
                    policy_controls,
                    "oom_recovery",
                    "K2LAB_OOM_RECOVERY",
                    policy.oom_recovery,
                )
            ),
            output_directory=output_directory,
            filename_prefix=validate_filename_prefix(
                str(
                    _env_or(
                        generation,
                        "filename_prefix",
                        "K2LAB_FILENAME_PREFIX",
                        "baseline",
                    )
                )
            ),
            face_detector_path=_optional_path(
                _env_or(models, "face_detector", "K2_FACE_DETECTOR_MODEL", ""),
                base=base,
            ),
            default_upscale_model=_optional_path(
                _env_or(models, "upscale_model", "K2_UPSCALE_MODEL", ""),
                base=base,
            ),
            default_width=int(generation.get("width", 1024)),
            default_height=int(generation.get("height", 1024)),
            default_steps=int(generation.get("steps", 8)),
            default_sampler=str(generation.get("sampler", "euler")),
            default_scheduler=str(generation.get("scheduler", "simple")),
            default_seed=int(generation.get("seed", 0)),
            default_seed_mode=str(generation.get("seed_mode", "fixed")),
            config_file=config_path,
        )
