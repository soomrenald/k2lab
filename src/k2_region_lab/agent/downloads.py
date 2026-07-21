from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from uuid import uuid4

import httpx

from k2_region_lab.agent.domain import (
    CivitaiDownloadRequest,
    CivitaiFilePreview,
    CivitaiPreview,
    FileKind,
    HuggingFaceDownloadRequest,
    HuggingFaceFilePreview,
    HuggingFacePreview,
    RemoteProvider,
    RemoteTransfer,
    TransferState,
)
from k2_region_lab.agent.storage import WorkspaceLayout
from k2_region_lab.agent.transfers import TransferError, TransferManager
from k2_region_lab.model import read_safetensors_header


CIVITAI_SOURCE_HOSTS = frozenset({"civitai.com", "www.civitai.com"})
CIVITAI_DOWNLOAD_HOSTS = frozenset(
    {"civitai.com", "www.civitai.com", "files.civitai.com"}
)
SECRET_QUERY_KEYS = frozenset({"token", "api_key", "apikey", "authorization", "auth"})
UNSAFE_MODEL_EXTENSIONS = frozenset({".bin", ".ckpt", ".pt", ".pth", ".pkl", ".pickle"})
SAFE_MODEL_EXTENSIONS = frozenset({".safetensors", ".onnx"})
SAFE_REPOSITORY_EXTENSIONS = frozenset(
    {".json", ".md", ".model", ".tiktoken", ".txt", ".yaml", ".yml"}
)
HUGGINGFACE_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass(frozen=True)
class CivitaiSource:
    model_id: str | None
    version_id: str | None


@dataclass(frozen=True)
class HuggingFaceSource:
    repo_id: str
    repo_type: str
    revision: str
    filename: str | None
    mirror_repository: bool


def parse_civitai_url(value: str) -> CivitaiSource:
    parsed = _validated_https_url(value, CIVITAI_SOURCE_HOSTS)
    query = parse_qs(parsed.query)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    model_id: str | None = None
    version_id: str | None = None
    if len(parts) >= 4 and parts[:3] == ["api", "download", "models"]:
        version_id = parts[3]
    elif len(parts) >= 2 and parts[0] == "models":
        model_id = parts[1].split("-", 1)[0]
        version_values = query.get("modelVersionId", [])
        version_id = version_values[0] if version_values else None
    if model_id is not None and not model_id.isdigit():
        model_id = None
    if version_id is not None and not version_id.isdigit():
        version_id = None
    if model_id is None and version_id is None:
        raise TransferError(
            "download_url_invalid",
            "Enter a Civitai model or model-version download URL.",
        )
    return CivitaiSource(model_id=model_id, version_id=version_id)


def parse_huggingface_url(value: str) -> HuggingFaceSource:
    parsed = _validated_https_url(value, frozenset({"huggingface.co"}))
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    repo_type = "model"
    if parts and parts[0] in {"datasets", "spaces"}:
        repo_type = "dataset" if parts[0] == "datasets" else "space"
        parts = parts[1:]
    if len(parts) < 2 or any(part in {".", ".."} for part in parts):
        raise TransferError(
            "download_url_invalid", "Enter a canonical Hugging Face repository or file URL."
        )
    repo_id = "/".join(parts[:2])
    if not all(HUGGINGFACE_REPO_SEGMENT.fullmatch(part) for part in parts[:2]):
        raise TransferError("download_url_invalid", "The Hugging Face repository ID is invalid.")
    remainder = parts[2:]
    if not remainder:
        return HuggingFaceSource(repo_id, repo_type, "main", None, True)
    if remainder[0] not in {"blob", "resolve", "tree"} or len(remainder) < 2:
        raise TransferError(
            "download_url_invalid", "Enter a canonical Hugging Face repository or file URL."
        )
    revision = remainder[1]
    if not _safe_relative_path(revision):
        raise TransferError("download_url_invalid", "The Hugging Face revision is unsafe.")
    if remainder[0] == "tree":
        return HuggingFaceSource(repo_id, repo_type, revision, None, True)
    filename = "/".join(remainder[2:])
    if not filename or not _safe_relative_path(filename):
        raise TransferError("download_url_invalid", "The Hugging Face file path is unsafe.")
    return HuggingFaceSource(repo_id, repo_type, revision, filename, False)


class RemoteDownloadManager:
    def __init__(
        self,
        layout: WorkspaceLayout,
        transfers: TransferManager,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        hf_file_download: Callable[..., Any] | None = None,
        hf_snapshot_download: Callable[..., Any] | None = None,
        hf_repo_info: Callable[..., Any] | None = None,
    ) -> None:
        self._layout = layout
        self._transfers = transfers
        self._transport = transport
        self._hf_file_download = hf_file_download
        self._hf_snapshot_download = hf_snapshot_download
        self._hf_repo_info = hf_repo_info
        self._state_directory = layout.state_directory / "transfers"
        self._incomplete_directory = layout.root / "downloads" / "incomplete"
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cancel: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._state_directory.mkdir(parents=True, exist_ok=True)
        self._recover_interrupted()

    async def preview_civitai(self, source_url: str, token: str | None) -> CivitaiPreview:
        source = parse_civitai_url(source_url)
        metadata = await self._civitai_metadata(source, token)
        return self._civitai_preview(metadata)

    async def preview_huggingface(
        self, source_url: str, token: str | None, allow_patterns: list[str]
    ) -> HuggingFacePreview:
        source = parse_huggingface_url(source_url)
        self._validate_patterns(allow_patterns)
        try:
            items = await asyncio.to_thread(
                self._huggingface_metadata,
                source,
                token,
                allow_patterns,
            )
        except TransferError:
            raise
        except Exception as error:
            raise self._huggingface_error(error) from error
        files = [
            HuggingFaceFilePreview(
                filename=filename,
                size_bytes=size,
            )
            for filename, size in items
        ]
        return HuggingFacePreview(
            repo_id=source.repo_id,
            repo_type=source.repo_type,
            revision=source.revision,
            filename=source.filename,
            mirror_repository=source.mirror_repository,
            files=files,
            required_bytes=sum(file.size_bytes for file in files if not file.cached),
        )

    async def start_civitai(
        self, request: CivitaiDownloadRequest, token: str | None
    ) -> RemoteTransfer:
        parse_civitai_url(request.source_url)
        transfer = await self._prepare_transfer(
            RemoteProvider.CIVITAI,
            request.source_url,
            request.destination_kind,
            request.resume_transfer_id,
        )
        self._launch(
            transfer.id,
            self._run_civitai(transfer.id, request, token),
        )
        return transfer

    async def start_huggingface(
        self, request: HuggingFaceDownloadRequest, token: str | None
    ) -> RemoteTransfer:
        parse_huggingface_url(request.source_url)
        self._validate_patterns(request.allow_patterns)
        transfer = await self._prepare_transfer(
            RemoteProvider.HUGGINGFACE,
            request.source_url,
            request.destination_kind,
            request.resume_transfer_id,
        )
        self._launch(
            transfer.id,
            self._run_huggingface(transfer.id, request, token),
        )
        return transfer

    async def get(self, transfer_id: str) -> RemoteTransfer:
        async with self._lock:
            return self._read(transfer_id)

    async def cancel(self, transfer_id: str) -> RemoteTransfer:
        async with self._lock:
            transfer = self._read(transfer_id)
            event = self._cancel.setdefault(transfer_id, asyncio.Event())
            event.set()
            if transfer.state not in {TransferState.COMPLETED, TransferState.FAILED}:
                transfer = self._update(transfer, state=TransferState.CANCELLED)
            return transfer

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for event in self._cancel.values():
            event.set()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _prepare_transfer(
        self,
        provider: RemoteProvider,
        source_url: str,
        destination_kind: FileKind,
        resume_transfer_id: str | None,
    ) -> RemoteTransfer:
        async with self._lock:
            if resume_transfer_id:
                transfer = self._read(resume_transfer_id)
                if (
                    transfer.provider != provider
                    or transfer.source_url != source_url
                    or transfer.destination_kind != destination_kind
                    or transfer.state
                    not in {TransferState.PAUSED, TransferState.CANCELLED, TransferState.FAILED}
                ):
                    raise TransferError(
                        "transfer_not_resumable", "This transfer cannot be resumed.", 409
                    )
                transfer = self._update(
                    transfer,
                    state=TransferState.PENDING,
                    error_code=None,
                    error_message=None,
                )
            else:
                now = datetime.now(UTC)
                transfer = RemoteTransfer(
                    id=uuid4().hex,
                    provider=provider,
                    source_url=source_url,
                    destination_kind=destination_kind,
                    state=TransferState.PENDING,
                    created_at=now,
                    updated_at=now,
                )
                self._write(transfer)
            self._cancel[transfer.id] = asyncio.Event()
            return transfer

    def _launch(self, transfer_id: str, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks[transfer_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(transfer_id, None))

    async def _run_civitai(
        self, transfer_id: str, request: CivitaiDownloadRequest, token: str | None
    ) -> None:
        try:
            await self._set_state(transfer_id, TransferState.RESOLVING)
            metadata = await self._civitai_metadata(parse_civitai_url(request.source_url), token)
            preview = self._civitai_preview(metadata)
            selected = next((file for file in preview.files if file.id == request.file_id), None)
            if selected is None:
                raise TransferError("download_file_missing", "The selected Civitai file is missing.", 404)
            previous = await self.get(transfer_id)
            if previous.filename is not None and previous.filename != selected.filename:
                raise TransferError(
                    "transfer_not_resumable",
                    "The selected file does not match the interrupted transfer.",
                    409,
                )
            self._validate_scan_results(selected)
            self._validate_model_format(selected.filename, request.allow_unsafe_format)
            self._ensure_space(selected.size_bytes)
            await self._set_state(
                transfer_id,
                TransferState.DOWNLOADING,
                filename=selected.filename,
                bytes_total=selected.size_bytes,
            )
            staged = self._incomplete_directory / f"{transfer_id}.part"
            await self._download_civitai_file(transfer_id, selected, staged, token)
            await self._set_state(transfer_id, TransferState.VERIFYING)
            digest = await asyncio.to_thread(self._verify_download, staged, selected)
            record, _duplicate = await self._transfers.install_download(
                staged,
                request.destination_kind,
                selected.filename,
                digest,
            )
            await self._set_state(
                transfer_id,
                TransferState.COMPLETED,
                bytes_complete=record.size_bytes,
                sha256=digest,
                files=[record],
            )
        except asyncio.CancelledError:
            await self._set_state(transfer_id, TransferState.PAUSED)
            raise
        except TransferError as error:
            state = (
                TransferState.CANCELLED
                if self._cancelled(transfer_id)
                else TransferState.FAILED
            )
            await self._set_state(
                transfer_id, state, error_code=error.code, error_message=error.message
            )
        except Exception:
            await self._set_state(
                transfer_id,
                TransferState.FAILED,
                error_code="download_failed",
                error_message="The remote download failed.",
            )
        finally:
            token = None

    async def _run_huggingface(
        self, transfer_id: str, request: HuggingFaceDownloadRequest, token: str | None
    ) -> None:
        staging_directory = self._incomplete_directory / transfer_id
        try:
            source = parse_huggingface_url(request.source_url)
            await self._set_state(transfer_id, TransferState.RESOLVING)
            preview = await self.preview_huggingface(
                request.source_url, token, request.allow_patterns
            )
            self._ensure_space(sum(file.size_bytes for file in preview.files))
            await self._set_state(
                transfer_id,
                TransferState.DOWNLOADING,
                filename=source.filename,
                bytes_total=sum(file.size_bytes for file in preview.files),
            )
            if self._cancelled(transfer_id):
                raise TransferError("transfer_cancelled", "The transfer was cancelled.", 409)
            downloaded = await asyncio.to_thread(
                self._call_huggingface,
                source,
                token,
                request.allow_patterns,
            )
            if self._cancelled(transfer_id):
                raise TransferError("transfer_cancelled", "The transfer was cancelled.", 409)
            source_root = Path(str(downloaded))
            source_files = (
                [source_root]
                if source.filename is not None
                else sorted(path for path in source_root.rglob("*") if path.is_file())
            )
            if not source_files:
                raise TransferError("download_file_missing", "The repository contains no files.", 404)
            staging_directory.mkdir(exist_ok=True)
            await self._set_state(transfer_id, TransferState.VERIFYING)
            bytes_complete = 0
            staged_downloads: list[tuple[Path, FileKind, str, str]] = []
            for source_file in source_files:
                relative = (
                    source.filename
                    if source.filename is not None
                    else source_file.relative_to(source_root).as_posix()
                )
                assert relative is not None
                if relative.startswith(".cache/"):
                    continue
                if self._cancelled(transfer_id):
                    raise TransferError(
                        "transfer_cancelled", "The transfer was cancelled.", 409
                    )
                self._validate_model_format(
                    relative, request.allow_unsafe_format, allow_repository_file=True
                )
                staged = staging_directory / uuid4().hex
                await asyncio.to_thread(shutil.copyfile, source_file, staged)
                digest = await asyncio.to_thread(self._hash_file, staged)
                if Path(relative).suffix.casefold() == ".safetensors":
                    try:
                        await asyncio.to_thread(read_safetensors_header, staged)
                    except (OSError, ValueError) as error:
                        raise TransferError(
                            "safetensors_header_invalid",
                            "A downloaded safetensors header is invalid.",
                            409,
                        ) from error
                staged_downloads.append(
                    (staged, request.destination_kind, relative, digest)
                )
                bytes_complete += staged.stat().st_size
                await self._set_state(
                    transfer_id, TransferState.VERIFYING, bytes_complete=bytes_complete
                )
            installed_results = await self._transfers.install_download_batch(
                staged_downloads
            )
            if not installed_results:
                raise TransferError(
                    "download_file_missing",
                    "The repository contains no supported files.",
                    404,
                )
            installed = [record for record, _duplicate in installed_results]
            await self._set_state(
                transfer_id,
                TransferState.COMPLETED,
                bytes_complete=bytes_complete,
                files=installed,
            )
        except TransferError as error:
            state = (
                TransferState.CANCELLED
                if self._cancelled(transfer_id)
                else TransferState.FAILED
            )
            await self._set_state(
                transfer_id, state, error_code=error.code, error_message=error.message
            )
        except Exception as error:
            mapped = self._huggingface_error(error)
            await self._set_state(
                transfer_id,
                TransferState.FAILED,
                error_code=mapped.code,
                error_message=mapped.message,
            )
        finally:
            shutil.rmtree(staging_directory, ignore_errors=True)
            token = None

    async def _civitai_metadata(
        self, source: CivitaiSource, token: str | None
    ) -> dict[str, Any]:
        headers = self._authorization(token)
        if source.version_id:
            metadata = await self._get_json(
                f"https://civitai.com/api/v1/model-versions/{source.version_id}", headers
            )
        else:
            metadata = await self._get_json(
                f"https://civitai.com/api/v1/models/{source.model_id}", headers
            )
            versions = metadata.get("modelVersions")
            if not isinstance(versions, list) or not versions:
                raise TransferError("download_file_missing", "The Civitai model has no versions.", 404)
            selected = versions[0]
            if not isinstance(selected, dict) or "id" not in selected:
                raise TransferError("provider_response_invalid", "Civitai returned invalid metadata.", 502)
            metadata = await self._get_json(
                f"https://civitai.com/api/v1/model-versions/{selected['id']}", headers
            )
        return metadata

    async def _get_json(self, url: str, headers: dict[str, str]) -> dict[str, Any]:
        async with httpx.AsyncClient(transport=self._transport, timeout=30) as client:
            response = await self._request_with_redirects(client, "GET", url, headers=headers)
        if response.status_code in {401, 403}:
            raise TransferError(
                "download_unauthorized",
                "Civitai denied access. Check the download-only token and model access.",
                403,
            )
        if response.status_code == 404:
            raise TransferError("download_file_missing", "The Civitai model was not found.", 404)
        if response.status_code >= 400:
            raise TransferError("provider_unavailable", "Civitai metadata is unavailable.", 502)
        try:
            body = response.json()
        except ValueError as error:
            raise TransferError("provider_response_invalid", "Civitai returned invalid metadata.", 502) from error
        if not isinstance(body, dict):
            raise TransferError("provider_response_invalid", "Civitai returned invalid metadata.", 502)
        return body

    async def _download_civitai_file(
        self,
        transfer_id: str,
        selected: CivitaiFilePreview,
        staged: Path,
        token: str | None,
    ) -> None:
        offset = staged.stat().st_size if staged.exists() else 0
        headers = self._authorization(token)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        async with httpx.AsyncClient(transport=self._transport, timeout=None) as client:
            response = await self._request_with_redirects(
                client, "GET", selected.download_url, headers=headers, stream=True
            )
            if response.status_code in {401, 403}:
                await response.aclose()
                raise TransferError(
                    "download_unauthorized",
                    "Civitai denied the download. Check the token and model access.",
                    403,
                )
            if response.status_code == 416 and offset:
                await response.aclose()
                if selected.size_bytes is not None and offset == selected.size_bytes:
                    return
                staged.unlink(missing_ok=True)
                return await self._download_civitai_file(
                    transfer_id, selected, staged, token
                )
            if response.status_code not in {200, 206}:
                await response.aclose()
                raise TransferError("download_failed", "Civitai could not download the file.", 502)
            append = response.status_code == 206 and offset > 0
            if not append:
                offset = 0
            content_type = response.headers.get("content-type", "").casefold()
            if "text/html" in content_type:
                await response.aclose()
                raise TransferError("download_payload_invalid", "Civitai returned HTML instead of a model file.", 502)
            with staged.open("ab" if append else "wb") as output:
                async for block in response.aiter_bytes(1024 * 1024):
                    if self._cancelled(transfer_id):
                        await response.aclose()
                        raise TransferError("transfer_cancelled", "The transfer was cancelled.", 409)
                    if offset == 0 and block.lstrip().lower().startswith((b"<html", b"<!doctype html")):
                        await response.aclose()
                        raise TransferError("download_payload_invalid", "Civitai returned HTML instead of a model file.", 502)
                    output.write(block)
                    offset += len(block)
                    if selected.size_bytes is not None and offset > selected.size_bytes:
                        raise TransferError(
                            "download_size_mismatch",
                            "The provider sent more data than declared.",
                            409,
                        )
                    await self._set_state(
                        transfer_id, TransferState.DOWNLOADING, bytes_complete=offset
                    )
                output.flush()
                os.fsync(output.fileno())
            await response.aclose()

    async def _request_with_redirects(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        stream: bool = False,
    ) -> httpx.Response:
        current = url
        for _attempt in range(6):
            _validated_https_url(current, CIVITAI_DOWNLOAD_HOSTS)
            request = client.build_request(method, current, headers=headers)
            response = await client.send(request, stream=stream)
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = response.headers.get("location")
            await response.aclose()
            if not location:
                raise TransferError("redirect_unsafe", "The provider redirect is invalid.", 502)
            current = urljoin(current, location)
        raise TransferError("redirect_unsafe", "The provider sent too many redirects.", 502)

    def _civitai_preview(self, metadata: dict[str, Any]) -> CivitaiPreview:
        raw_files = metadata.get("files")
        if not isinstance(raw_files, list):
            raise TransferError("provider_response_invalid", "Civitai returned invalid file metadata.", 502)
        files: list[CivitaiFilePreview] = []
        for raw in raw_files:
            if not isinstance(raw, dict):
                continue
            filename = str(raw.get("name") or "")
            download_url = str(raw.get("downloadUrl") or "")
            if (
                not filename
                or Path(filename).name != filename
                or not download_url
                or raw.get("id") is None
            ):
                continue
            _validated_https_url(download_url, CIVITAI_DOWNLOAD_HOSTS)
            hashes = raw.get("hashes") if isinstance(raw.get("hashes"), dict) else {}
            metadata_value = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            suffix = Path(filename).suffix.casefold()
            size_kb = raw.get("sizeKB")
            files.append(
                CivitaiFilePreview(
                    id=str(raw.get("id")),
                    filename=filename,
                    size_bytes=round(float(size_kb) * 1024) if size_kb is not None else None,
                    format=str(metadata_value.get("format")) if metadata_value.get("format") else None,
                    sha256=str(hashes.get("SHA256")).lower() if hashes.get("SHA256") else None,
                    pickle_scan=str(raw.get("pickleScanResult")) if raw.get("pickleScanResult") else None,
                    virus_scan=str(raw.get("virusScanResult")) if raw.get("virusScanResult") else None,
                    download_url=download_url,
                    preferred=bool(raw.get("primary")) or suffix == ".safetensors",
                    requires_unsafe_confirmation=suffix in UNSAFE_MODEL_EXTENSIONS,
                )
            )
        if not files:
            raise TransferError("download_file_missing", "The Civitai version has no downloadable files.", 404)
        model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
        return CivitaiPreview(
            model_id=str(model.get("id") or metadata.get("modelId") or "unknown"),
            version_id=str(metadata.get("id") or "unknown"),
            model_name=str(model.get("name") or "Civitai model"),
            version_name=str(metadata.get("name") or "Version"),
            model_type=str(model.get("type")) if model.get("type") else None,
            base_model=str(metadata.get("baseModel")) if metadata.get("baseModel") else None,
            training_words=[str(word) for word in metadata.get("trainedWords", []) if isinstance(word, str)],
            files=files,
        )

    def _call_huggingface(
        self,
        source: HuggingFaceSource,
        token: str | None,
        allow_patterns: list[str],
    ) -> Any:
        file_download = self._hf_file_download
        snapshot_download = self._hf_snapshot_download
        if file_download is None or snapshot_download is None:
            from huggingface_hub import hf_hub_download, snapshot_download as snapshot

            file_download = file_download or hf_hub_download
            snapshot_download = snapshot_download or snapshot
        common = {
            "repo_id": source.repo_id,
            "repo_type": None if source.repo_type == "model" else source.repo_type,
            "revision": source.revision,
            "cache_dir": self._layout.root / "cache" / "huggingface",
            "token": token,
        }
        if source.filename is not None:
            return file_download(filename=source.filename, **common)
        return snapshot_download(allow_patterns=allow_patterns or None, **common)

    def _huggingface_metadata(
        self,
        source: HuggingFaceSource,
        token: str | None,
        allow_patterns: list[str],
    ) -> list[tuple[str, int]]:
        repo_info = self._hf_repo_info
        if repo_info is None:
            from huggingface_hub import HfApi

            repo_info = HfApi(token=token).repo_info
        info = repo_info(
            repo_id=source.repo_id,
            repo_type=None if source.repo_type == "model" else source.repo_type,
            revision=source.revision,
            files_metadata=True,
            token=token,
        )
        items: list[tuple[str, int]] = []
        for sibling in getattr(info, "siblings", []) or []:
            filename = str(getattr(sibling, "rfilename", ""))
            if not filename:
                continue
            if source.filename is not None and filename != source.filename:
                continue
            if allow_patterns and not any(
                fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns
            ):
                continue
            items.append((filename, int(getattr(sibling, "size", 0) or 0)))
        if source.filename is not None and not items:
            raise TransferError(
                "download_file_missing", "The Hugging Face file was not found.", 404
            )
        if source.filename is None and not items:
            raise TransferError(
                "download_file_missing",
                "No Hugging Face repository files match the selected filters.",
                404,
            )
        return items

    @staticmethod
    def _huggingface_error(error: Exception) -> TransferError:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {401, 403}:
            return TransferError(
                "download_gated",
                "Hugging Face denied access. Use a read token and accept the repository access terms.",
                403,
            )
        if status_code == 404:
            return TransferError("download_file_missing", "The Hugging Face file was not found.", 404)
        return TransferError("download_failed", "The Hugging Face download failed.", 502)

    @staticmethod
    def _validate_patterns(patterns: list[str]) -> None:
        if any(
            not pattern
            or len(pattern) > 191
            or pattern.startswith(("/", "\\"))
            or ".." in PurePosixPath(pattern).parts
            for pattern in patterns
        ):
            raise TransferError("download_pattern_invalid", "A repository file pattern is unsafe.")

    def _ensure_space(self, size_bytes: int | None) -> None:
        if size_bytes is None:
            return
        if size_bytes > 1_099_511_627_776:
            raise TransferError(
                "download_too_large", "A download may not exceed 1 TiB.", 413
            )
        if size_bytes * 3 > shutil.disk_usage(self._layout.root).free:
            raise TransferError(
                "storage_full",
                "The workspace does not have enough free space for this download.",
                409,
            )

    @staticmethod
    def _validate_model_format(
        filename: str, allow_unsafe: bool, *, allow_repository_file: bool = False
    ) -> None:
        suffix = Path(filename).suffix.casefold()
        if suffix in UNSAFE_MODEL_EXTENSIONS and not allow_unsafe:
            raise TransferError(
                "unsafe_format_requires_confirmation",
                "Pickle-based model files require explicit confirmation.",
                409,
            )
        supported = SAFE_MODEL_EXTENSIONS | UNSAFE_MODEL_EXTENSIONS
        if allow_repository_file:
            supported |= SAFE_REPOSITORY_EXTENSIONS
        if suffix not in supported:
            raise TransferError(
                "download_format_unsupported",
                "Only supported model artifact formats can be installed.",
                409,
            )

    @staticmethod
    def _validate_scan_results(selected: CivitaiFilePreview) -> None:
        accepted = {"success", "pass", "passed"}
        for result in (selected.virus_scan, selected.pickle_scan):
            if result is not None and result.casefold() not in accepted:
                raise TransferError(
                    "download_scan_failed",
                    "Civitai has not reported a successful safety scan for this file.",
                    409,
                )

    @staticmethod
    def _verify_download(path: Path, selected: CivitaiFilePreview) -> str:
        if selected.size_bytes is not None and path.stat().st_size != selected.size_bytes:
            raise TransferError("download_size_mismatch", "The downloaded file size did not match.", 409)
        digest = RemoteDownloadManager._hash_file(path)
        if selected.sha256 is not None and digest.casefold() != selected.sha256.casefold():
            raise TransferError("download_hash_mismatch", "The downloaded file checksum failed.", 409)
        if path.suffix.casefold() == ".safetensors":
            try:
                read_safetensors_header(path)
            except (OSError, ValueError) as error:
                raise TransferError(
                    "safetensors_header_invalid",
                    "The downloaded safetensors header is invalid.",
                    409,
                ) from error
        return digest

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    async def _set_state(
        self, transfer_id: str, state: TransferState, **updates: Any
    ) -> RemoteTransfer:
        async with self._lock:
            return self._update(self._read(transfer_id), state=state, **updates)

    def _cancelled(self, transfer_id: str) -> bool:
        event = self._cancel.get(transfer_id)
        return bool(event and event.is_set())

    def _update(
        self, transfer: RemoteTransfer, *, state: TransferState, **updates: Any
    ) -> RemoteTransfer:
        updated = transfer.model_copy(
            update={"state": state, "updated_at": datetime.now(UTC), **updates}
        )
        self._write(updated)
        return updated

    def _read(self, transfer_id: str) -> RemoteTransfer:
        if not transfer_id.isalnum() or len(transfer_id) > 64:
            raise TransferError("transfer_not_found", "The transfer does not exist.", 404)
        try:
            return RemoteTransfer.model_validate_json(
                (self._state_directory / f"{transfer_id}.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ValueError) as error:
            raise TransferError("transfer_not_found", "The transfer does not exist.", 404) from error

    def _write(self, transfer: RemoteTransfer) -> None:
        path = self._state_directory / f"{transfer.id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(transfer.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def _recover_interrupted(self) -> None:
        active = {
            TransferState.PENDING,
            TransferState.RESOLVING,
            TransferState.DOWNLOADING,
            TransferState.VERIFYING,
        }
        for path in self._state_directory.glob("*.json"):
            try:
                transfer = RemoteTransfer.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if transfer.state in active:
                self._write(
                    transfer.model_copy(
                        update={
                            "state": TransferState.PAUSED,
                            "error_code": "agent_restarted",
                            "error_message": "Resume the transfer to provide a fresh provider token.",
                            "updated_at": datetime.now(UTC),
                        }
                    )
                )

    @staticmethod
    def _authorization(token: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"} if token else {}


def _validated_https_url(value: str, hosts: frozenset[str]):
    parsed = urlsplit(value)
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
    ):
        raise TransferError("download_url_unsafe", "The remote download URL is not allowed.")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if any(key.casefold() in SECRET_QUERY_KEYS for key in query):
        raise TransferError(
            "authenticated_url_rejected",
            "Provider tokens must be supplied separately, not embedded in a URL.",
        )
    return parsed


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        "\\" not in value
        and not path.is_absolute()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
    )
