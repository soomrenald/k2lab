from __future__ import annotations

from pathlib import Path


def default_output_directory(data_directory: Path) -> Path:
    return data_directory / "baseline_outputs"


def validate_filename_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value:
        raise ValueError("output filename prefix must not be empty")
    if len(value) > 128:
        raise ValueError("output filename prefix must be at most 128 characters")
    if value in {".", ".."} or any(character in value for character in ("/", "\\", "\0")):
        raise ValueError("output filename prefix must be a filename, not a path")
    return value
