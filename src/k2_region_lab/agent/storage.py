from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


LAYOUT_VERSION: Final = 1
MODEL_KINDS: Final = {
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "upscale_models",
    "face_detection",
}
FILE_KINDS: Final = MODEL_KINDS | {"projects", "inputs", "outputs"}


@dataclass(frozen=True)
class WorkspaceLayout:
    root: Path

    @property
    def state_directory(self) -> Path:
        return self.root / "state"

    @property
    def marker_path(self) -> Path:
        return self.state_directory / "layout.json"

    def initialize(self) -> None:
        directories = [
            self.root / "projects",
            self.root / "inputs",
            self.root / "outputs",
            self.root / "downloads" / "incomplete",
            self.root / "cache" / "huggingface",
            self.root / "state" / "migrations",
            self.root / "state" / "inventory",
            self.root / "state" / "jobs",
            self.root / "state" / "manifests",
            self.root / "state" / "uploads",
        ]
        directories.extend(self.root / "models" / kind for kind in sorted(MODEL_KINDS))
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        self._validate_root()
        self._write_or_validate_marker()

    def destination(self, kind: str) -> Path:
        if kind not in FILE_KINDS:
            raise ValueError(f"unsupported workspace file kind: {kind}")
        if kind in MODEL_KINDS:
            return self.root / "models" / kind
        return self.root / kind

    def resolve_child(self, kind: str, filename: str) -> Path:
        if not filename or filename in {".", ".."}:
            raise ValueError("filename is required")
        candidate_name = Path(filename)
        if candidate_name.is_absolute() or len(candidate_name.parts) != 1:
            raise ValueError("nested and absolute paths are not allowed")
        destination = self.destination(kind).resolve(strict=True)
        candidate = destination / candidate_name.name
        if candidate.is_symlink():
            raise ValueError("symbolic links are not allowed")
        resolved_parent = candidate.parent.resolve(strict=True)
        if resolved_parent != destination:
            raise ValueError("path escapes the workspace destination")
        return candidate

    def resolve_relative(self, kind: str, relative_path: str, *, create: bool = False) -> Path:
        candidate_name = Path(relative_path)
        if (
            not relative_path
            or candidate_name.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in candidate_name.parts)
        ):
            raise ValueError("relative workspace path is unsafe")
        destination = self.destination(kind).resolve(strict=True)
        parent = destination
        if create:
            for part in candidate_name.parts[:-1]:
                parent /= part
                if parent.is_symlink():
                    raise ValueError("symbolic links are not allowed")
                parent.mkdir(exist_ok=True)
        candidate = destination.joinpath(*candidate_name.parts)
        if candidate.is_symlink():
            raise ValueError("symbolic links are not allowed")
        resolved_parent = candidate.parent.resolve(strict=True)
        if resolved_parent != destination and destination not in resolved_parent.parents:
            raise ValueError("path escapes the workspace destination")
        return candidate

    def is_writable(self) -> bool:
        probe = self.state_directory / f".write-probe-{os.getpid()}"
        try:
            probe.write_bytes(b"ok")
            return True
        except OSError:
            return False
        finally:
            try:
                probe.unlink()
            except FileNotFoundError:
                pass

    def model_inventory_ready(self) -> bool:
        required = ("diffusion_models", "text_encoders", "vae")
        return all(any(self.destination(kind).iterdir()) for kind in required)

    def _validate_root(self) -> None:
        if self.root.is_symlink():
            raise RuntimeError("workspace root must not be a symbolic link")
        resolved = self.root.resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError("workspace root must be a directory")

    def _write_or_validate_marker(self) -> None:
        if self.marker_path.exists():
            try:
                marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise RuntimeError("workspace layout marker is unreadable") from error
            version = marker.get("layout_version")
            if version != LAYOUT_VERSION:
                raise RuntimeError(
                    f"unsupported workspace layout version {version!r}; "
                    f"expected {LAYOUT_VERSION}"
                )
            return
        temporary = self.marker_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"layout_version": LAYOUT_VERSION}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.marker_path)
