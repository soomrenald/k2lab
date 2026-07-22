from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import shutil
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from k2_region_lab.agent.domain import (
    ChunkReceipt,
    FileKind,
    FilePage,
    FileRecord,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
)
from k2_region_lab.agent.storage import WorkspaceLayout


class TransferError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class TransferManager:
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout
        self._lock = asyncio.Lock()
        self._index_path = layout.state_directory / "inventory" / "files.json"
        self._upload_directory = layout.state_directory / "uploads"
        self._incomplete_directory = layout.root / "downloads" / "incomplete"

    async def inventory(
        self, kind: FileKind, *, cursor: str | None = None, limit: int = 100
    ) -> FilePage:
        if limit < 1 or limit > 250:
            raise TransferError("invalid_page_size", "Inventory limit must be 1 to 250.")
        offset = self._decode_cursor(cursor)
        async with self._lock:
            records = await asyncio.to_thread(self._scan_kind, kind)
        page = records[offset : offset + limit]
        next_offset = offset + len(page)
        return FilePage(
            items=page,
            next_cursor=(
                self._encode_cursor(next_offset) if next_offset < len(records) else None
            ),
        )

    async def create_upload(self, request: UploadCreateRequest) -> UploadSession:
        filename = self._safe_filename(request.filename)
        destination = self._layout.resolve_child(request.destination_kind.value, filename)
        if destination.exists():
            raise TransferError(
                "destination_exists",
                "A file with this name already exists in the destination.",
                409,
            )
        free_bytes = shutil.disk_usage(self._layout.root).free
        if request.size_bytes * 2 > free_bytes:
            raise TransferError(
                "storage_full",
                "The workspace does not have enough free space for this upload.",
                409,
            )
        now = datetime.now(UTC)
        session = UploadSession(
            id=uuid4().hex,
            filename=filename,
            display_name=request.filename,
            destination_kind=request.destination_kind,
            size_bytes=request.size_bytes,
            sha256=request.sha256.lower(),
            chunk_size_bytes=request.chunk_size_bytes,
            chunk_count=math.ceil(request.size_bytes / request.chunk_size_bytes),
            state="uploading",
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            self._chunk_directory(session.id).mkdir(parents=False, exist_ok=False)
            self._write_session(session)
        return session

    async def get_upload(self, upload_id: str) -> UploadSession:
        async with self._lock:
            return self._read_session(upload_id)

    async def write_chunk(
        self, upload_id: str, index: int, content: bytes, supplied_sha256: str
    ) -> ChunkReceipt:
        if len(content) > 64 * 1024 * 1024:
            raise TransferError("chunk_too_large", "Upload chunks may not exceed 64 MiB.", 413)
        digest = hashlib.sha256(content).hexdigest()
        if digest != supplied_sha256.lower():
            raise TransferError("chunk_hash_mismatch", "The upload chunk checksum failed.", 409)
        async with self._lock:
            session = self._read_session(upload_id)
            if session.state != "uploading":
                raise TransferError("upload_not_active", "This upload is not active.", 409)
            if index < 0 or index >= session.chunk_count:
                raise TransferError("chunk_index_invalid", "The chunk index is out of range.")
            expected = self._expected_chunk_size(session, index)
            if len(content) != expected:
                raise TransferError(
                    "chunk_size_mismatch",
                    f"Chunk {index} must contain exactly {expected} bytes.",
                    409,
                )
            destination = self._chunk_directory(upload_id) / f"{index:08d}.part"
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
            completed = sorted(set(session.completed_chunks) | {index})
            session = session.model_copy(
                update={"completed_chunks": completed, "updated_at": datetime.now(UTC)}
            )
            self._write_session(session)
        return ChunkReceipt(
            upload_id=upload_id,
            index=index,
            size_bytes=len(content),
            sha256=digest,
        )

    async def complete_upload(self, upload_id: str) -> UploadCompleteResponse:
        async with self._lock:
            session = self._read_session(upload_id)
            expected_indices = list(range(session.chunk_count))
            if session.completed_chunks != expected_indices:
                raise TransferError(
                    "upload_incomplete",
                    "Every upload chunk must be present before completion.",
                    409,
                )
            existing = self._find_by_sha256(session.sha256)
            if existing is not None:
                self._remove_upload(session.id)
                return UploadCompleteResponse(file=existing, duplicate=True)

            destination = self._layout.resolve_child(
                session.destination_kind.value, session.filename
            )
            if destination.exists():
                raise TransferError(
                    "destination_exists",
                    "A file with this name already exists in the destination.",
                    409,
                )
            temporary = self._chunk_directory(upload_id) / "assembled.tmp"
            digest = hashlib.sha256()
            written = 0
            with temporary.open("wb") as output:
                for index in expected_indices:
                    chunk = self._chunk_directory(upload_id) / f"{index:08d}.part"
                    with chunk.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                            digest.update(block)
                            written += len(block)
                output.flush()
                os.fsync(output.fileno())
            if written != session.size_bytes or digest.hexdigest() != session.sha256:
                temporary.unlink(missing_ok=True)
                raise TransferError(
                    "upload_hash_mismatch",
                    "The completed upload failed size or SHA-256 verification.",
                    409,
                )
            temporary.replace(destination)
            record = self._record_file(session.destination_kind, destination, session.sha256)
            self._upsert_record(record)
            self._remove_upload(session.id)
            return UploadCompleteResponse(file=record)

    async def cancel_upload(self, upload_id: str) -> None:
        async with self._lock:
            self._read_session(upload_id)
            self._remove_upload(upload_id)

    async def resolve_file(
        self, file_id: str, *, required_kind: FileKind | None = None
    ) -> tuple[FileRecord, Path]:
        if not file_id.isalnum() or len(file_id) > 64:
            raise TransferError("file_not_found", "The file does not exist.", 404)
        async with self._lock:
            if required_kind is not None:
                self._scan_kind(required_kind)
            else:
                for kind in FileKind:
                    self._scan_kind(kind)
            for value in self._read_index().values():
                record_data = value.get("record") if isinstance(value, dict) else None
                if not record_data or record_data.get("id") != file_id:
                    continue
                record = FileRecord.model_validate(record_data)
                if required_kind is not None and record.kind != required_kind:
                    break
                path = self._layout.resolve_relative(record.kind.value, record.display_name)
                if path.is_file() and not path.is_symlink():
                    return record, path
        raise TransferError("file_not_found", "The file does not exist.", 404)

    async def index_existing_file(self, kind: FileKind, path: Path) -> FileRecord:
        destination = self._layout.destination(kind.value).resolve(strict=True)
        if path.is_symlink():
            raise TransferError(
                "worker_output_invalid", "The worker output is not a regular file.", 409
            )
        try:
            resolved = path.resolve(strict=True)
            relative_path = resolved.relative_to(destination).as_posix()
        except (FileNotFoundError, ValueError) as error:
            raise TransferError(
                "worker_output_invalid",
                "The worker output is outside the workspace output directory.",
                409,
            ) from error
        if not resolved.is_file():
            raise TransferError(
                "worker_output_invalid", "The worker output is not a regular file.", 409
            )
        async with self._lock:
            records = self._scan_kind(kind)
            for record in records:
                if record.display_name == relative_path:
                    return record
        raise TransferError(
            "worker_output_invalid", "The worker output could not be indexed.", 409
        )

    async def save_project(self, filename: str, project: dict[str, Any]) -> FileRecord:
        safe_name = self._safe_filename(filename)
        if not safe_name.casefold().endswith(".json"):
            raise TransferError("project_name_invalid", "Project filenames must end in .json.")
        encoded = (json.dumps(project, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise TransferError("project_too_large", "The project document exceeds 2 MiB.", 413)
        destination = self._layout.resolve_child(FileKind.PROJECTS.value, safe_name)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        async with self._lock:
            temporary.write_bytes(encoded)
            temporary.replace(destination)
            records = self._scan_kind(FileKind.PROJECTS)
        return next(record for record in records if record.display_name == safe_name)

    async def install_download(
        self,
        staged_path: Path,
        kind: FileKind,
        relative_path: str,
        sha256: str,
    ) -> tuple[FileRecord, bool]:
        installed = await self.install_download_batch(
            [(staged_path, kind, relative_path, sha256)]
        )
        return installed[0]

    async def install_download_batch(
        self,
        downloads: list[tuple[Path, FileKind, str, str]],
    ) -> list[tuple[FileRecord, bool]]:
        if not downloads:
            return []
        for staged_path, _kind, _relative_path, _sha256 in downloads:
            if not staged_path.is_file() or staged_path.is_symlink():
                raise TransferError(
                    "download_missing", "A downloaded file is unavailable.", 409
                )
        async with self._lock:
            planned: list[tuple[Path, Path, FileKind, str, str, FileRecord | None]] = []
            destinations: set[Path] = set()
            for staged_path, kind, relative_path, sha256 in downloads:
                existing = self._find_by_sha256(
                    sha256, kind=kind, display_name=relative_path
                )
                try:
                    destination = self._layout.resolve_relative(
                        kind.value, relative_path, create=True
                    )
                except ValueError as error:
                    raise TransferError(
                        "unsafe_filename", "A downloaded filename is unsafe."
                    ) from error
                if destination in destinations:
                    raise TransferError(
                        "destination_conflict",
                        "The download contains duplicate destination paths.",
                        409,
                    )
                destinations.add(destination)
                if existing is None and destination.exists():
                    raise TransferError(
                        "destination_exists",
                        "A file with this name already exists in the destination.",
                        409,
                    )
                planned.append(
                    (staged_path, destination, kind, relative_path, sha256, existing)
                )
            results: list[tuple[FileRecord, bool]] = []
            for staged_path, destination, kind, _relative_path, sha256, existing in planned:
                if existing is not None:
                    staged_path.unlink(missing_ok=True)
                    results.append((existing, True))
                    continue
                staged_path.replace(destination)
                record = self._record_file(kind, destination, sha256)
                self._upsert_record(record)
                results.append((record, False))
            return results

    def _scan_kind(self, kind: FileKind) -> list[FileRecord]:
        index = self._read_index()
        records: list[FileRecord] = []
        destination = self._layout.destination(kind.value)
        for path in sorted(destination.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative_path = path.relative_to(destination).as_posix()
            key = f"{kind.value}/{relative_path}"
            stat = path.stat()
            cached = index.get(key)
            if (
                cached
                and cached.get("size_bytes") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
            ):
                record = FileRecord.model_validate(cached["record"])
            else:
                record = self._record_file(kind, path)
                index[key] = {
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "record": record.model_dump(mode="json"),
                }
            records.append(record)
        self._write_index(index)
        return records

    def _record_file(
        self, kind: FileKind, path: Path, known_sha256: str | None = None
    ) -> FileRecord:
        stat = path.stat()
        return FileRecord(
            id=uuid4().hex,
            kind=kind,
            display_name=path.relative_to(self._layout.destination(kind.value)).as_posix(),
            size_bytes=stat.st_size,
            sha256=known_sha256 or self._hash_file(path),
            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
        )

    def _upsert_record(self, record: FileRecord) -> None:
        index = self._read_index()
        path = self._layout.resolve_relative(record.kind.value, record.display_name)
        stat = path.stat()
        index[f"{record.kind.value}/{record.display_name}"] = {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "record": record.model_dump(mode="json"),
        }
        self._write_index(index)

    def _find_by_sha256(
        self,
        sha256: str,
        *,
        kind: FileKind | None = None,
        display_name: str | None = None,
    ) -> FileRecord | None:
        required_kind = kind
        for scan_kind in FileKind:
            self._scan_kind(scan_kind)
        for value in self._read_index().values():
            record_data = value.get("record") if isinstance(value, dict) else None
            if record_data and record_data.get("sha256") == sha256:
                record = FileRecord.model_validate(record_data)
                if required_kind is not None and record.kind != required_kind:
                    continue
                if display_name is not None and record.display_name != display_name:
                    continue
                path = self._layout.resolve_relative(record.kind.value, record.display_name)
                if path.is_file() and not path.is_symlink():
                    return record
        return None

    def _read_session(self, upload_id: str) -> UploadSession:
        if not upload_id.isalnum() or len(upload_id) > 64:
            raise TransferError("upload_not_found", "The upload session does not exist.", 404)
        path = self._upload_directory / f"{upload_id}.json"
        try:
            return UploadSession.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise TransferError(
                "upload_not_found", "The upload session does not exist.", 404
            ) from error

    def _write_session(self, session: UploadSession) -> None:
        path = self._upload_directory / f"{session.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(session.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def _remove_upload(self, upload_id: str) -> None:
        (self._upload_directory / f"{upload_id}.json").unlink(missing_ok=True)
        shutil.rmtree(self._chunk_directory(upload_id), ignore_errors=True)

    def _chunk_directory(self, upload_id: str) -> Path:
        return self._incomplete_directory / upload_id

    def _read_index(self) -> dict[str, Any]:
        try:
            value = json.loads(self._index_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_index(self, index: dict[str, Any]) -> None:
        temporary = self._index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        temporary.replace(self._index_path)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_filename(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip().strip(".")
        if not normalized or Path(normalized).name != normalized or "\\" in normalized:
            raise TransferError("unsafe_filename", "The upload filename is unsafe.")
        if any(ord(character) < 32 for character in normalized):
            raise TransferError("unsafe_filename", "The upload filename is unsafe.")
        return normalized[:191]

    @staticmethod
    def _expected_chunk_size(session: UploadSession, index: int) -> int:
        if index < session.chunk_count - 1:
            return session.chunk_size_bytes
        return session.size_bytes - session.chunk_size_bytes * (session.chunk_count - 1)

    @staticmethod
    def _encode_cursor(offset: int) -> str:
        return base64.urlsafe_b64encode(str(offset).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str | None) -> int:
        if not cursor:
            return 0
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            offset = int(base64.urlsafe_b64decode(padded).decode())
        except (binascii.Error, ValueError, UnicodeDecodeError) as error:
            raise TransferError("invalid_cursor", "The inventory cursor is invalid.") from error
        if offset < 0:
            raise TransferError("invalid_cursor", "The inventory cursor is invalid.")
        return offset
