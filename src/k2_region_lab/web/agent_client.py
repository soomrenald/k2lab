from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from k2_region_lab.agent.domain import (
    AgentCapabilities,
    AgentHealth,
    ChunkReceipt,
    CivitaiDownloadRequest,
    CivitaiPreview,
    CivitaiPreviewRequest,
    FaceDetectionRequest,
    FaceDetectionResult,
    FileKind,
    FilePage,
    FileRecord,
    GenerationJob,
    HuggingFaceDownloadRequest,
    HuggingFacePreview,
    HuggingFacePreviewRequest,
    JobEventPage,
    JobSubmitRequest,
    MigrationChunkReceipt,
    ProjectSaveRequest,
    RemoteTransfer,
    StorageStatus,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
    WorkerReleaseResult,
    WorkspaceManifest,
)
from k2_region_lab.web.domain import WorkspaceError, WorkspaceOutput


class WorkspaceAgentApi(Protocol):
    async def health(self) -> AgentHealth: ...

    async def capabilities(self) -> AgentCapabilities: ...

    async def storage(self) -> StorageStatus: ...

    async def seal_for_migration(self) -> WorkspaceManifest: ...

    async def unseal_after_migration(self) -> None: ...

    async def create_migration_manifest(self) -> WorkspaceManifest: ...

    async def migration_file(
        self, generation: int, relative_path: str, *, start: int, end: int
    ) -> bytes: ...

    async def import_migration_chunk(
        self,
        migration_id: str,
        relative_path: str,
        *,
        offset: int,
        total_size: int,
        file_sha256: str,
        content: bytes,
    ) -> MigrationChunkReceipt: ...

    async def inventory(self, kind: FileKind, *, cursor: str | None = None) -> FilePage: ...

    async def save_project(self, filename: str, request: ProjectSaveRequest) -> FileRecord: ...

    async def create_upload(self, request: UploadCreateRequest) -> UploadSession: ...

    async def upload_status(self, upload_id: str) -> UploadSession: ...

    async def write_chunk(
        self, upload_id: str, index: int, content: bytes, sha256: str
    ) -> ChunkReceipt: ...

    async def complete_upload(self, upload_id: str) -> UploadCompleteResponse: ...

    async def cancel_upload(self, upload_id: str) -> None: ...

    async def preview_civitai(
        self, request: CivitaiPreviewRequest, token: str | None
    ) -> CivitaiPreview: ...

    async def start_civitai(
        self, request: CivitaiDownloadRequest, token: str | None
    ) -> RemoteTransfer: ...

    async def preview_huggingface(
        self, request: HuggingFacePreviewRequest, token: str | None
    ) -> HuggingFacePreview: ...

    async def start_huggingface(
        self, request: HuggingFaceDownloadRequest, token: str | None
    ) -> RemoteTransfer: ...

    async def transfer_status(self, transfer_id: str) -> RemoteTransfer: ...

    async def cancel_transfer(self, transfer_id: str) -> RemoteTransfer: ...

    async def submit_job(self, request: JobSubmitRequest) -> GenerationJob: ...

    async def job_status(self, job_id: str) -> GenerationJob: ...

    async def job_events(self, job_id: str, *, cursor: str | None = None) -> JobEventPage: ...

    async def cancel_job(self, job_id: str) -> GenerationJob: ...

    async def detect_faces(self, request: FaceDetectionRequest) -> FaceDetectionResult: ...

    async def release_worker_memory(self) -> WorkerReleaseResult: ...

    async def output(self, file_id: str, *, range_header: str | None = None) -> WorkspaceOutput: ...

    async def file_content(
        self, file_id: str, *, range_header: str | None = None
    ) -> WorkspaceOutput: ...


class WorkspaceAgentClient:
    """Authenticated client for one Pod agent through RunPod's HTTPS proxy."""

    _POD_ID = re.compile(r"^[a-zA-Z0-9-]{1,191}$")

    def __init__(
        self,
        pod_id: str,
        session_token: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not self._POD_ID.fullmatch(pod_id):
            raise ValueError("invalid RunPod Pod ID")
        self._base_url = f"https://{pod_id}-8080.proxy.runpod.net"
        self._session_token = session_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def health(self) -> AgentHealth:
        return AgentHealth.model_validate(await self._request("/v1/health"))

    async def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities.model_validate(await self._request("/v1/capabilities"))

    async def storage(self) -> StorageStatus:
        return StorageStatus.model_validate(await self._request("/v1/storage"))

    async def seal_for_migration(self) -> WorkspaceManifest:
        return WorkspaceManifest.model_validate(
            await self._request("/v1/migrations/seal", method="POST")
        )

    async def unseal_after_migration(self) -> None:
        await self._request("/v1/migrations/seal", method="DELETE")

    async def create_migration_manifest(self) -> WorkspaceManifest:
        return WorkspaceManifest.model_validate(
            await self._request("/v1/migrations/manifests", method="POST")
        )

    async def migration_file(
        self, generation: int, relative_path: str, *, start: int, end: int
    ) -> bytes:
        path = quote(relative_path, safe="/")
        headers = {
            "Authorization": f"Bearer {self._session_token}",
            "Range": f"bytes={start}-{end}",
        }
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=max(self._timeout_seconds, 60.0),
                transport=self._transport,
            ) as client:
                response = await client.get(
                    f"/v1/migrations/files/{path}",
                    params={"generation": generation},
                    headers=headers,
                )
        except httpx.HTTPError as error:
            raise WorkspaceError(
                "agent_unavailable",
                "The source workspace file could not be retrieved.",
                status_code=502,
            ) from error
        if response.status_code not in {200, 206}:
            raise WorkspaceError(
                "migration_read_failed",
                "The source workspace file could not be retrieved.",
                status_code=response.status_code,
            )
        return response.content

    async def import_migration_chunk(
        self,
        migration_id: str,
        relative_path: str,
        *,
        offset: int,
        total_size: int,
        file_sha256: str,
        content: bytes,
    ) -> MigrationChunkReceipt:
        return MigrationChunkReceipt.model_validate(
            await self._request(
                f"/v1/migrations/files/{quote(relative_path, safe='/')}",
                method="PUT",
                content=content,
                extra_headers={
                    "X-Migration-ID": migration_id,
                    "X-File-Offset": str(offset),
                    "X-File-Size": str(total_size),
                    "X-File-SHA256": file_sha256,
                    "X-Chunk-SHA256": hashlib.sha256(content).hexdigest(),
                },
            )
        )

    async def inventory(self, kind: FileKind, *, cursor: str | None = None) -> FilePage:
        params = {"kind": kind.value}
        if cursor:
            params["cursor"] = cursor
        return FilePage.model_validate(await self._request("/v1/files", params=params))

    async def save_project(self, filename: str, request: ProjectSaveRequest) -> FileRecord:
        return FileRecord.model_validate(
            await self._request(
                f"/v1/projects/{quote(filename, safe='')}",
                method="PUT",
                json=request.model_dump(mode="json"),
            )
        )

    async def create_upload(self, request: UploadCreateRequest) -> UploadSession:
        return UploadSession.model_validate(
            await self._request("/v1/uploads", method="POST", json=request.model_dump(mode="json"))
        )

    async def upload_status(self, upload_id: str) -> UploadSession:
        return UploadSession.model_validate(await self._request(f"/v1/uploads/{upload_id}"))

    async def write_chunk(
        self, upload_id: str, index: int, content: bytes, sha256: str
    ) -> ChunkReceipt:
        return ChunkReceipt.model_validate(
            await self._request(
                f"/v1/uploads/{upload_id}/chunks/{index}",
                method="PUT",
                content=content,
                extra_headers={"X-Chunk-SHA256": sha256},
            )
        )

    async def complete_upload(self, upload_id: str) -> UploadCompleteResponse:
        return UploadCompleteResponse.model_validate(
            await self._request(f"/v1/uploads/{upload_id}/complete", method="POST")
        )

    async def cancel_upload(self, upload_id: str) -> None:
        await self._request(f"/v1/uploads/{upload_id}", method="DELETE")

    async def preview_civitai(
        self, request: CivitaiPreviewRequest, token: str | None
    ) -> CivitaiPreview:
        return CivitaiPreview.model_validate(
            await self._request(
                "/v1/downloads/civitai/preview",
                method="POST",
                json=request.model_dump(mode="json"),
                extra_headers=self._provider_header(token),
            )
        )

    async def start_civitai(
        self, request: CivitaiDownloadRequest, token: str | None
    ) -> RemoteTransfer:
        return RemoteTransfer.model_validate(
            await self._request(
                "/v1/downloads/civitai",
                method="POST",
                json=request.model_dump(mode="json"),
                extra_headers=self._provider_header(token),
            )
        )

    async def preview_huggingface(
        self, request: HuggingFacePreviewRequest, token: str | None
    ) -> HuggingFacePreview:
        return HuggingFacePreview.model_validate(
            await self._request(
                "/v1/downloads/huggingface/preview",
                method="POST",
                json=request.model_dump(mode="json"),
                extra_headers=self._provider_header(token),
            )
        )

    async def start_huggingface(
        self, request: HuggingFaceDownloadRequest, token: str | None
    ) -> RemoteTransfer:
        return RemoteTransfer.model_validate(
            await self._request(
                "/v1/downloads/huggingface",
                method="POST",
                json=request.model_dump(mode="json"),
                extra_headers=self._provider_header(token),
            )
        )

    async def transfer_status(self, transfer_id: str) -> RemoteTransfer:
        return RemoteTransfer.model_validate(await self._request(f"/v1/transfers/{transfer_id}"))

    async def cancel_transfer(self, transfer_id: str) -> RemoteTransfer:
        return RemoteTransfer.model_validate(
            await self._request(f"/v1/transfers/{transfer_id}/cancel", method="POST")
        )

    async def submit_job(self, request: JobSubmitRequest) -> GenerationJob:
        return GenerationJob.model_validate(
            await self._request(
                "/v1/jobs",
                method="POST",
                json=request.model_dump(mode="json"),
            )
        )

    async def job_status(self, job_id: str) -> GenerationJob:
        return GenerationJob.model_validate(await self._request(f"/v1/jobs/{job_id}"))

    async def job_events(self, job_id: str, *, cursor: str | None = None) -> JobEventPage:
        params = {"cursor": cursor} if cursor else None
        return JobEventPage.model_validate(
            await self._request(f"/v1/jobs/{job_id}/events", params=params)
        )

    async def cancel_job(self, job_id: str) -> GenerationJob:
        return GenerationJob.model_validate(
            await self._request(f"/v1/jobs/{job_id}/cancel", method="POST")
        )

    async def detect_faces(self, request: FaceDetectionRequest) -> FaceDetectionResult:
        return FaceDetectionResult.model_validate(
            await self._request(
                "/v1/faces/detect",
                method="POST",
                json=request.model_dump(mode="json"),
            )
        )

    async def release_worker_memory(self) -> WorkerReleaseResult:
        return WorkerReleaseResult.model_validate(
            await self._request("/v1/worker/release", method="POST")
        )

    async def output(self, file_id: str, *, range_header: str | None = None) -> WorkspaceOutput:
        return await self._file_response(
            f"/v1/outputs/{file_id}", range_header=range_header
        )

    async def file_content(
        self, file_id: str, *, range_header: str | None = None
    ) -> WorkspaceOutput:
        return await self._file_response(
            f"/v1/files/{file_id}/content", range_header=range_header
        )

    async def _file_response(
        self, path: str, *, range_header: str | None = None
    ) -> WorkspaceOutput:
        headers = {"Authorization": f"Bearer {self._session_token}"}
        if range_header:
            headers["Range"] = range_header
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(path, headers=headers)
        except httpx.HTTPError as error:
            raise WorkspaceError(
                "agent_unavailable",
                "The workspace output could not be retrieved.",
                status_code=502,
            ) from error
        if response.status_code not in {200, 206}:
            raise WorkspaceError(
                "output_unavailable",
                "The workspace output could not be retrieved.",
                status_code=response.status_code,
            )
        forwarded = {
            key: value
            for key, value in response.headers.items()
            if key.casefold()
            in {
                "accept-ranges",
                "content-disposition",
                "content-length",
                "content-range",
                "content-type",
            }
        }
        return WorkspaceOutput(
            content=response.content,
            status_code=response.status_code,
            headers=forwarded,
        )

    @staticmethod
    def _provider_header(token: str | None) -> dict[str, str] | None:
        return {"X-Provider-Token": token} if token else None

    async def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.request(
                    method,
                    path,
                    headers={
                        "Authorization": f"Bearer {self._session_token}",
                        **(extra_headers or {}),
                    },
                    **kwargs,
                )
        except httpx.TimeoutException as error:
            raise WorkspaceError(
                "agent_timeout",
                "The workspace agent is not responding yet.",
                status_code=504,
            ) from error
        except httpx.HTTPError as error:
            raise WorkspaceError(
                "agent_unavailable",
                "The workspace agent is currently unreachable.",
                status_code=502,
            ) from error
        if response.status_code == 401:
            raise WorkspaceError(
                "agent_authentication_failed",
                "The workspace agent rejected its session credential.",
                status_code=502,
            )
        if response.status_code == 204:
            return {}
        if response.status_code < 200 or response.status_code >= 300:
            try:
                error_body = response.json()
            except ValueError:
                error_body = {}
            if isinstance(error_body, dict):
                code = error_body.get("code")
                message = error_body.get("message")
                if isinstance(code, str) and isinstance(message, str):
                    raise WorkspaceError(code, message, status_code=response.status_code)
            raise WorkspaceError(
                "agent_unavailable",
                "The workspace agent could not complete its health request.",
                status_code=502,
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise WorkspaceError(
                "agent_response_invalid",
                "The workspace agent returned invalid JSON.",
                status_code=502,
            ) from error
        if not isinstance(payload, dict):
            raise WorkspaceError(
                "agent_response_invalid",
                "The workspace agent returned an unexpected response.",
                status_code=502,
            )
        return payload
