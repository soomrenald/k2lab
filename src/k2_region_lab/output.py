from __future__ import annotations

from pathlib import Path


def project_workspace_directory() -> Path:
    """Return the checkout that owns the desktop's prompt and output folders."""

    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / "pyproject.toml").is_file():
        return source_checkout
    configured = Path("~/k2lab").expanduser()
    if configured.is_dir():
        return configured.resolve()
    return Path.cwd().resolve()


def default_prompt_directory() -> Path:
    return project_workspace_directory() / "prompts"


def default_output_directory(_data_directory: Path | None = None) -> Path:
    return project_workspace_directory() / "outputs"


def validate_filename_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value:
        raise ValueError("output filename prefix must not be empty")
    if len(value) > 128:
        raise ValueError("output filename prefix must be at most 128 characters")
    if value in {".", ".."} or any(character in value for character in ("/", "\\", "\0")):
        raise ValueError("output filename prefix must be a filename, not a path")
    return value
