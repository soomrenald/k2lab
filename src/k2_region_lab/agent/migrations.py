from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from k2_region_lab.agent.domain import (
    ManifestEntry,
    MigrationChunkReceipt,
    WorkspaceManifest,
)
from k2_region_lab.agent.storage import LAYOUT_VERSION, WorkspaceLayout
from k2_region_lab.agent.transfers import TransferError


class WorkspaceMigrationManager:
    """Seals a workspace, produces deterministic manifests, and stages verified files."""

    _ALLOWED_DIRECTORIES = (
        "inputs",
        "models",
        "outputs",
        "projects",
        "state/inventory",
        "state/jobs",
        "state/transfers",
    )
    _ALLOWED_FILES = frozenset({"state/layout.json"})
    _CHUNK_LIMIT = 64 * 1024 * 1024

    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout
        self._state_directory = layout.state_directory / "migrations"
        self._manifest_directory = layout.state_directory / "manifests"
        self._staging_directory = self._state_directory / "staging"
        self._seal_path = self._state_directory / "sealed.json"
        self._generation_path = self._manifest_directory / "generation"
        self._lock = asyncio.Lock()
        self._state_directory.mkdir(parents=True, exist_ok=True)
        self._manifest_directory.mkdir(parents=True, exist_ok=True)
        self._staging_directory.mkdir(parents=True, exist_ok=True)

    @property
    def sealed(self) -> bool:
        return self._seal_path.is_file()

    async def seal(self) -> None:
        async with self._lock:
            temporary = self._seal_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"sealed_at": datetime.now(UTC).isoformat()}) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._seal_path)

    async def unseal(self) -> None:
        async with self._lock:
            self._seal_path.unlink(missing_ok=True)

    async def create_manifest(self) -> WorkspaceManifest:
        if not self.sealed:
            raise TransferError(
                "workspace_not_sealed",
                "Seal the workspace before creating a migration manifest.",
                409,
            )
        async with self._lock:
            generation = self._next_generation()
            manifest = await asyncio.to_thread(self._scan, generation)
            path = self._manifest_path(generation)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(manifest.model_dump_json(), encoding="utf-8")
            temporary.replace(path)
            self._generation_path.write_text(f"{generation}\n", encoding="ascii")
            return manifest

    async def get_manifest(self, generation: int) -> WorkspaceManifest:
        async with self._lock:
            try:
                return WorkspaceManifest.model_validate_json(
                    self._manifest_path(generation).read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise TransferError(
                    "manifest_not_found", "The workspace manifest does not exist.", 404
                ) from error

    async def read_file(
        self, generation: int, relative_path: str, start: int, end: int | None
    ) -> tuple[ManifestEntry, bytes, int, int]:
        manifest = await self.get_manifest(generation)
        normalized = self._safe_relative(relative_path)
        entry = next((item for item in manifest.files if item.path == normalized), None)
        if entry is None:
            raise TransferError(
                "manifest_file_not_found", "The file is not present in this manifest.", 404
            )
        if start < 0 or start >= entry.size_bytes and entry.size_bytes != 0:
            raise TransferError("invalid_range", "The migration byte range is invalid.", 416)
        last = entry.size_bytes - 1 if end is None else min(end, entry.size_bytes - 1)
        if entry.size_bytes == 0:
            return entry, b"", 0, -1
        if last < start:
            raise TransferError("invalid_range", "The migration byte range is invalid.", 416)
        path = self._resolve_source(normalized)

        def read() -> bytes:
            with path.open("rb") as source:
                source.seek(start)
                return source.read(last - start + 1)

        return entry, await asyncio.to_thread(read), start, last

    async def write_chunk(
        self,
        *,
        migration_id: str,
        relative_path: str,
        offset: int,
        total_size: int,
        file_sha256: str,
        chunk_sha256: str,
        content: bytes,
    ) -> MigrationChunkReceipt:
        if not self.sealed:
            raise TransferError(
                "workspace_not_sealed",
                "Seal the target workspace before importing migration files.",
                409,
            )
        if not migration_id.isalnum() or len(migration_id) > 64:
            raise TransferError("migration_id_invalid", "The migration ID is invalid.")
        normalized = self._safe_relative(relative_path)
        if not self._allowed(normalized):
            raise TransferError(
                "migration_path_not_allowed", "This workspace path cannot be migrated."
            )
        if len(content) > self._CHUNK_LIMIT:
            raise TransferError("chunk_too_large", "Migration chunks may not exceed 64 MiB.", 413)
        if offset < 0 or total_size < 0 or offset + len(content) > total_size:
            raise TransferError("chunk_range_invalid", "The migration chunk range is invalid.")
        if hashlib.sha256(content).hexdigest() != chunk_sha256.lower():
            raise TransferError("chunk_hash_mismatch", "The migration chunk checksum failed.", 409)
        if len(file_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in file_sha256
        ):
            raise TransferError("file_hash_invalid", "The migration file checksum is invalid.")

        async with self._lock:
            destination = self._resolve_destination(normalized)
            if destination.is_file():
                destination_size = destination.stat().st_size
                if (
                    destination_size == total_size
                    and self._sha256(destination) == file_sha256.lower()
                ):
                    return MigrationChunkReceipt(
                        path=normalized,
                        next_offset=total_size,
                        completed=True,
                    )
            staged = self._staging_path(migration_id, normalized)
            staged.parent.mkdir(parents=True, exist_ok=True)
            current_size = staged.stat().st_size if staged.exists() else 0
            if current_size > offset:
                with staged.open("rb") as existing:
                    existing.seek(offset)
                    already_written = existing.read(len(content))
                if already_written != content:
                    raise TransferError(
                        "migration_offset_conflict",
                        "The staged migration file conflicts with this retry.",
                        409,
                    )
            elif current_size == offset:
                with staged.open("ab") as staged_file:
                    staged_file.write(content)
                    staged_file.flush()
            else:
                raise TransferError(
                    "migration_offset_gap",
                    "Resume the migration from the last accepted byte.",
                    409,
                )
            next_offset = max(current_size, offset + len(content))
            completed = next_offset == total_size
            if completed:
                digest = await asyncio.to_thread(self._sha256, staged)
                if digest != file_sha256.lower():
                    raise TransferError(
                        "file_hash_mismatch", "The migrated file checksum failed.", 409
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(destination)
            return MigrationChunkReceipt(
                path=normalized, next_offset=next_offset, completed=completed
            )

    def _scan(self, generation: int) -> WorkspaceManifest:
        entries: list[ManifestEntry] = []
        for directory in self._ALLOWED_DIRECTORIES:
            root = self._layout.root / directory
            if not root.exists():
                continue
            if root.is_symlink():
                raise TransferError(
                    "manifest_symlink_unsafe", "Workspace manifests cannot include symlinks.", 409
                )
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise TransferError(
                        "manifest_symlink_unsafe",
                        "Workspace manifests cannot include symlinks.",
                        409,
                    )
                if path.is_file():
                    relative = path.relative_to(self._layout.root).as_posix()
                    entries.append(
                        ManifestEntry(
                            path=relative,
                            size_bytes=path.stat().st_size,
                            sha256=self._sha256(path),
                        )
                    )
        for relative in sorted(self._ALLOWED_FILES):
            path = self._layout.root / relative
            if path.is_file() and not path.is_symlink():
                entries.append(
                    ManifestEntry(
                        path=relative,
                        size_bytes=path.stat().st_size,
                        sha256=self._sha256(path),
                    )
                )
        entries.sort(key=lambda item: item.path)
        root_digest = hashlib.sha256()
        for entry in entries:
            root_digest.update(entry.path.encode("utf-8"))
            root_digest.update(b"\0")
            root_digest.update(str(entry.size_bytes).encode("ascii"))
            root_digest.update(b"\0")
            root_digest.update(entry.sha256.encode("ascii"))
            root_digest.update(b"\n")
        return WorkspaceManifest(
            generation=generation,
            layout_version=LAYOUT_VERSION,
            files=entries,
            file_count=len(entries),
            total_bytes=sum(item.size_bytes for item in entries),
            root_sha256=root_digest.hexdigest(),
            created_at=datetime.now(UTC),
        )

    def _next_generation(self) -> int:
        try:
            current = int(self._generation_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            current = 0
        return current + 1

    def _manifest_path(self, generation: int) -> Path:
        if generation < 1:
            raise TransferError("manifest_not_found", "The workspace manifest does not exist.", 404)
        return self._manifest_directory / f"{generation:020d}.json"

    def _resolve_source(self, relative_path: str) -> Path:
        path = self._layout.root.joinpath(*PurePosixPath(relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise TransferError("manifest_file_not_found", "The manifest file is unavailable.", 404)
        resolved = path.resolve(strict=True)
        root = self._layout.root.resolve(strict=True)
        if root not in resolved.parents:
            raise TransferError("migration_path_unsafe", "The migration path is unsafe.")
        return resolved

    def _resolve_destination(self, relative_path: str) -> Path:
        root = self._layout.root.resolve(strict=True)
        candidate = root.joinpath(*PurePosixPath(relative_path).parts)
        parent = candidate.parent
        cursor = root
        for part in parent.relative_to(root).parts:
            cursor /= part
            if cursor.is_symlink():
                raise TransferError("migration_path_unsafe", "The migration path is unsafe.")
            cursor.mkdir(exist_ok=True)
        if candidate.is_symlink():
            raise TransferError("migration_path_unsafe", "The migration path is unsafe.")
        return candidate

    def _staging_path(self, migration_id: str, relative_path: str) -> Path:
        root = self._staging_directory / migration_id
        path = root.joinpath(*PurePosixPath(relative_path).parts).with_suffix(
            PurePosixPath(relative_path).suffix + ".part"
        )
        resolved_parent = path.parent.resolve(strict=False)
        staging_root = self._staging_directory.resolve(strict=True)
        if resolved_parent != staging_root and staging_root not in resolved_parent.parents:
            raise TransferError("migration_path_unsafe", "The migration path is unsafe.")
        return path

    @classmethod
    def _safe_relative(cls, relative_path: str) -> str:
        candidate = PurePosixPath(relative_path)
        if (
            not relative_path
            or candidate.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise TransferError("migration_path_unsafe", "The migration path is unsafe.")
        return candidate.as_posix()

    @classmethod
    def _allowed(cls, relative_path: str) -> bool:
        if relative_path in cls._ALLOWED_FILES:
            return True
        return any(
            relative_path.startswith(f"{directory}/") for directory in cls._ALLOWED_DIRECTORIES
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
