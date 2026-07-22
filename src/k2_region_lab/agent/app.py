from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import shutil
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

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
    ReadinessStages,
    RemoteTransfer,
    StorageStatus,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
    WorkerReleaseResult,
    WorkspaceManifest,
)
from k2_region_lab.agent.downloads import RemoteDownloadManager
from k2_region_lab.agent.faces import FaceDetectionRunner, FaceDetectionService
from k2_region_lab.agent.jobs import JobError, JobManager
from k2_region_lab.agent.migrations import WorkspaceMigrationManager
from k2_region_lab.agent.storage import LAYOUT_VERSION, WorkspaceLayout
from k2_region_lab.agent.transfers import TransferError, TransferManager
from k2_region_lab.http_security import SlidingWindowRateLimiter
from k2_region_lab.project import PROJECT_SCHEMA, PROJECT_VERSION, project_state


class AgentSettings:
    def __init__(
        self,
        *,
        session_token: str,
        workspace_id: str,
        image_version: str,
        workspace_root: Path,
        worker_python: Path,
        comfyui_root: Path = Path("/opt/ComfyUI"),
        cuda_version: str | None = None,
        pytorch_version: str | None = None,
        read_requests_per_minute: int = 600,
        write_requests_per_minute: int = 120,
        provisioning_requests_per_minute: int = 30,
        upload_chunk_requests_per_minute: int = 600,
        max_request_bytes: int = 65 * 1024 * 1024,
    ) -> None:
        if len(session_token) < 32:
            raise ValueError("agent session token must contain at least 32 characters")
        if not workspace_id:
            raise ValueError("workspace ID is required")
        if not image_version:
            raise ValueError("image version is required")
        self.session_token = session_token
        self.workspace_id = workspace_id
        self.image_version = image_version
        self.workspace_root = workspace_root
        self.worker_python = worker_python
        self.comfyui_root = comfyui_root
        self.cuda_version = cuda_version
        self.pytorch_version = pytorch_version
        self.read_requests_per_minute = read_requests_per_minute
        self.write_requests_per_minute = write_requests_per_minute
        self.provisioning_requests_per_minute = provisioning_requests_per_minute
        self.upload_chunk_requests_per_minute = upload_chunk_requests_per_minute
        self.max_request_bytes = max_request_bytes

    @classmethod
    def from_environment(cls) -> AgentSettings:
        token = os.environ.get("K2LAB_AGENT_SESSION_TOKEN", "")
        workspace_id = os.environ.get("K2LAB_WORKSPACE_ID", "")
        image_version = os.environ.get("K2LAB_IMAGE_VERSION", "")
        return cls(
            session_token=token,
            workspace_id=workspace_id,
            image_version=image_version,
            workspace_root=Path(os.environ.get("K2LAB_WORKSPACE_ROOT", "/workspace/k2lab")),
            worker_python=Path(
                os.environ.get("K2LAB_WORKER_PYTHON", "/opt/comfyui-venv/bin/python")
            ),
            comfyui_root=Path(os.environ.get("K2LAB_COMFYUI_ROOT", "/opt/ComfyUI")),
            cuda_version=os.environ.get("K2LAB_CUDA_VERSION"),
            pytorch_version=os.environ.get("K2LAB_PYTORCH_VERSION"),
            read_requests_per_minute=int(os.environ.get("K2LAB_AGENT_READ_RATE_LIMIT", "600")),
            write_requests_per_minute=int(os.environ.get("K2LAB_AGENT_WRITE_RATE_LIMIT", "120")),
            provisioning_requests_per_minute=int(
                os.environ.get("K2LAB_AGENT_JOB_RATE_LIMIT", "30")
            ),
            upload_chunk_requests_per_minute=int(
                os.environ.get("K2LAB_AGENT_CHUNK_RATE_LIMIT", "600")
            ),
        )


def _agent_rate_limit(request: Request, settings: AgentSettings) -> tuple[str, int]:
    path = request.url.path
    if "/uploads/" in path and "/chunks/" in path:
        return "upload-chunk", settings.upload_chunk_requests_per_minute
    if request.method == "POST" and (path == "/v1/jobs" or path.startswith("/v1/downloads/")):
        return "job-start", settings.provisioning_requests_per_minute
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return "read", settings.read_requests_per_minute
    return "write", settings.write_requests_per_minute


def create_agent_app(
    settings: AgentSettings | None = None,
    *,
    download_transport: Any | None = None,
    hf_file_download: Any | None = None,
    hf_snapshot_download: Any | None = None,
    hf_repo_info: Any | None = None,
    job_executor_factory: Any | None = None,
    face_detection_runner: FaceDetectionRunner | None = None,
) -> FastAPI:
    configured = settings or AgentSettings.from_environment()
    layout = WorkspaceLayout(configured.workspace_root)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        layout.initialize()
        try:
            yield
        finally:
            await download_manager.close()
            await job_manager.close()

    application = FastAPI(
        title="K2 Region Lab Workspace Agent",
        version=configured.image_version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.settings = configured
    application.state.layout = layout
    application.state.worker_ready = False
    transfer_manager = TransferManager(layout)
    application.state.transfer_manager = transfer_manager
    download_manager = RemoteDownloadManager(
        layout,
        transfer_manager,
        transport=download_transport,
        hf_file_download=hf_file_download,
        hf_snapshot_download=hf_snapshot_download,
        hf_repo_info=hf_repo_info,
    )
    application.state.download_manager = download_manager
    job_manager = JobManager(
        layout,
        transfer_manager,
        worker_python=configured.worker_python,
        comfyui_root=configured.comfyui_root,
        executor_factory=job_executor_factory,
        readiness_callback=lambda ready: setattr(application.state, "worker_ready", ready),
    )
    application.state.job_manager = job_manager
    face_detection_service = FaceDetectionService(
        layout,
        transfer_manager,
        worker_python=configured.worker_python,
        comfyui_root=configured.comfyui_root,
        runner=face_detection_runner,
    )
    application.state.face_detection_service = face_detection_service
    migration_manager = WorkspaceMigrationManager(layout)
    application.state.migration_manager = migration_manager
    rate_limiter = SlidingWindowRateLimiter()

    @application.middleware("http")
    async def secure_agent_requests(request: Request, call_next):
        try:
            content_length = int(request.headers.get("Content-Length", "0"))
        except ValueError:
            response = JSONResponse(
                status_code=400,
                content={"code": "request_size_invalid", "message": "Invalid request size."},
            )
        else:
            if content_length > configured.max_request_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "code": "request_too_large",
                        "message": "The request body is too large.",
                    },
                )
            else:
                authorization = request.headers.get("Authorization", "")
                identity = hashlib.sha256(authorization.encode()).hexdigest()[:16]
                rate_class, limit = _agent_rate_limit(request, configured)
                allowed, retry_after = await rate_limiter.allow(
                    f"{identity}:{rate_class}", limit=limit
                )
                if not allowed:
                    response = JSONResponse(
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                        content={
                            "code": "rate_limit_exceeded",
                            "message": "Too many agent requests; retry shortly.",
                        },
                    )
                elif (
                    migration_manager.sealed
                    and request.method not in {"GET", "HEAD", "OPTIONS"}
                    and not request.url.path.startswith("/v1/migrations/")
                ):
                    response = JSONResponse(
                        status_code=423,
                        content={
                            "code": "workspace_sealed_for_migration",
                            "message": "Workspace writes are sealed during migration.",
                        },
                    )
                else:
                    response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.exception_handler(TransferError)
    async def transfer_error_handler(_request: Request, error: TransferError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.message},
        )

    @application.exception_handler(JobError)
    async def job_error_handler(_request: Request, error: JobError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"code": error.code, "message": error.message},
        )

    async def require_agent_token(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(
            supplied.encode("utf-8"), configured.session_token.encode("utf-8")
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Agent authentication failed.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authentication = [Depends(require_agent_token)]

    @application.get("/v1/health", response_model=AgentHealth, dependencies=authentication)
    async def health() -> AgentHealth:
        writable = layout.is_writable()
        readiness = ReadinessStages(
            container=True,
            agent=True,
            storage=writable,
            models=layout.model_inventory_ready(),
            worker=bool(application.state.worker_ready),
        )
        core_ready = readiness.container and readiness.agent and readiness.storage
        return AgentHealth(
            status="ready" if core_ready else "starting",
            workspace_id=configured.workspace_id,
            image_version=configured.image_version,
            readiness=readiness,
            observed_at=datetime.now(UTC),
        )

    @application.get(
        "/v1/capabilities",
        response_model=AgentCapabilities,
        dependencies=authentication,
    )
    async def capabilities() -> AgentCapabilities:
        return AgentCapabilities(
            workspace_layout_version=LAYOUT_VERSION,
            image_version=configured.image_version,
            cuda_version=configured.cuda_version,
            pytorch_version=configured.pytorch_version,
        )

    @application.get("/v1/storage", response_model=StorageStatus, dependencies=authentication)
    async def storage() -> StorageStatus:
        usage = shutil.disk_usage(layout.root)
        return StorageStatus(
            root="/workspace/k2lab",
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            writable=layout.is_writable(),
            layout_version=LAYOUT_VERSION,
        )

    @application.post(
        "/v1/migrations/seal",
        response_model=WorkspaceManifest,
        dependencies=authentication,
    )
    async def seal_for_migration() -> WorkspaceManifest:
        await migration_manager.seal()
        await job_manager.close()
        await download_manager.close()
        return await migration_manager.create_manifest()

    @application.delete(
        "/v1/migrations/seal",
        dependencies=authentication,
        status_code=204,
    )
    async def unseal_after_migration() -> None:
        await migration_manager.unseal()

    @application.post(
        "/v1/migrations/manifests",
        response_model=WorkspaceManifest,
        dependencies=authentication,
    )
    async def create_migration_manifest() -> WorkspaceManifest:
        return await migration_manager.create_manifest()

    @application.get(
        "/v1/migrations/manifests/{generation}",
        response_model=WorkspaceManifest,
        dependencies=authentication,
    )
    async def migration_manifest(generation: int) -> WorkspaceManifest:
        return await migration_manager.get_manifest(generation)

    @application.get("/v1/migrations/files/{relative_path:path}", dependencies=authentication)
    async def migration_file(
        relative_path: str,
        generation: int = Query(ge=1),
        range_header: str | None = Header(default=None, alias="Range"),
    ) -> StreamingResponse:
        start = 0
        end: int | None = None
        if range_header:
            if not range_header.startswith("bytes=") or "," in range_header:
                raise TransferError("invalid_range", "Only one byte range is supported.", 416)
            start_text, separator, end_text = range_header[6:].partition("-")
            if not separator or not start_text:
                raise TransferError("invalid_range", "The migration byte range is invalid.", 416)
            try:
                start = int(start_text)
                end = int(end_text) if end_text else None
            except ValueError as error:
                raise TransferError(
                    "invalid_range", "The migration byte range is invalid.", 416
                ) from error
        entry, content, first, last = await migration_manager.read_file(
            generation, relative_path, start, end
        )
        status_code = 206 if range_header else 200
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(content)),
            "X-File-SHA256": entry.sha256,
        }
        if range_header:
            headers["Content-Range"] = f"bytes {first}-{last}/{entry.size_bytes}"
        return StreamingResponse(
            iter([content]),
            status_code=status_code,
            media_type="application/octet-stream",
            headers=headers,
        )

    @application.put(
        "/v1/migrations/files/{relative_path:path}",
        response_model=MigrationChunkReceipt,
        dependencies=authentication,
    )
    async def import_migration_file(
        relative_path: str,
        request: Request,
        migration_id: str = Header(alias="X-Migration-ID"),
        offset: int = Header(alias="X-File-Offset", ge=0),
        total_size: int = Header(alias="X-File-Size", ge=0),
        file_sha256: str = Header(alias="X-File-SHA256"),
        chunk_sha256: str = Header(alias="X-Chunk-SHA256"),
    ) -> MigrationChunkReceipt:
        return await migration_manager.write_chunk(
            migration_id=migration_id,
            relative_path=relative_path,
            offset=offset,
            total_size=total_size,
            file_sha256=file_sha256,
            chunk_sha256=chunk_sha256,
            content=await request.body(),
        )

    @application.get("/v1/files", response_model=FilePage, dependencies=authentication)
    async def files(
        kind: FileKind,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=250),
    ) -> FilePage:
        return await transfer_manager.inventory(kind, cursor=cursor, limit=limit)

    @application.put(
        "/v1/projects/{filename}", response_model=FileRecord, dependencies=authentication
    )
    async def save_project(filename: str, request: ProjectSaveRequest) -> FileRecord:
        try:
            project_state(request.project)
        except (KeyError, TypeError, ValueError) as error:
            raise JobError("project_invalid", "The project document is invalid.") from error
        if (
            request.project.get("schema") != PROJECT_SCHEMA
            or request.project.get("version") != PROJECT_VERSION
        ):
            raise JobError("project_version_mismatch", "The project version is unsupported.", 409)
        return await transfer_manager.save_project(filename, request.project)

    @application.post(
        "/v1/uploads",
        response_model=UploadSession,
        dependencies=authentication,
        status_code=201,
    )
    async def create_upload(request: UploadCreateRequest) -> UploadSession:
        return await transfer_manager.create_upload(request)

    @application.get(
        "/v1/uploads/{upload_id}",
        response_model=UploadSession,
        dependencies=authentication,
    )
    async def upload_status(upload_id: str) -> UploadSession:
        return await transfer_manager.get_upload(upload_id)

    @application.put(
        "/v1/uploads/{upload_id}/chunks/{index}",
        response_model=ChunkReceipt,
        dependencies=authentication,
    )
    async def upload_chunk(
        upload_id: str,
        index: int,
        request: Request,
        x_chunk_sha256: str = Header(alias="X-Chunk-SHA256"),
    ) -> ChunkReceipt:
        return await transfer_manager.write_chunk(
            upload_id, index, await request.body(), x_chunk_sha256
        )

    @application.post(
        "/v1/uploads/{upload_id}/complete",
        response_model=UploadCompleteResponse,
        dependencies=authentication,
    )
    async def complete_upload(upload_id: str) -> UploadCompleteResponse:
        return await transfer_manager.complete_upload(upload_id)

    @application.delete(
        "/v1/uploads/{upload_id}",
        dependencies=authentication,
        status_code=204,
    )
    async def cancel_upload(upload_id: str) -> None:
        await transfer_manager.cancel_upload(upload_id)

    @application.post(
        "/v1/downloads/civitai/preview",
        response_model=CivitaiPreview,
        dependencies=authentication,
    )
    async def preview_civitai(
        request: CivitaiPreviewRequest,
        provider_token: str | None = Header(default=None, alias="X-Provider-Token"),
    ) -> CivitaiPreview:
        return await download_manager.preview_civitai(request.source_url, provider_token)

    @application.post(
        "/v1/downloads/civitai",
        response_model=RemoteTransfer,
        dependencies=authentication,
        status_code=202,
    )
    async def start_civitai_download(
        request: CivitaiDownloadRequest,
        provider_token: str | None = Header(default=None, alias="X-Provider-Token"),
    ) -> RemoteTransfer:
        return await download_manager.start_civitai(request, provider_token)

    @application.post(
        "/v1/downloads/huggingface/preview",
        response_model=HuggingFacePreview,
        dependencies=authentication,
    )
    async def preview_huggingface(
        request: HuggingFacePreviewRequest,
        provider_token: str | None = Header(default=None, alias="X-Provider-Token"),
    ) -> HuggingFacePreview:
        return await download_manager.preview_huggingface(
            request.source_url, provider_token, request.allow_patterns
        )

    @application.post(
        "/v1/downloads/huggingface",
        response_model=RemoteTransfer,
        dependencies=authentication,
        status_code=202,
    )
    async def start_huggingface_download(
        request: HuggingFaceDownloadRequest,
        provider_token: str | None = Header(default=None, alias="X-Provider-Token"),
    ) -> RemoteTransfer:
        return await download_manager.start_huggingface(request, provider_token)

    @application.get(
        "/v1/transfers/{transfer_id}",
        response_model=RemoteTransfer,
        dependencies=authentication,
    )
    async def transfer_status(transfer_id: str) -> RemoteTransfer:
        return await download_manager.get(transfer_id)

    @application.post(
        "/v1/transfers/{transfer_id}/cancel",
        response_model=RemoteTransfer,
        dependencies=authentication,
    )
    async def cancel_transfer(transfer_id: str) -> RemoteTransfer:
        return await download_manager.cancel(transfer_id)

    @application.post(
        "/v1/jobs",
        response_model=GenerationJob,
        dependencies=authentication,
        status_code=202,
    )
    async def submit_job(request: JobSubmitRequest) -> GenerationJob:
        return await job_manager.submit(request)

    @application.post(
        "/v1/faces/detect",
        response_model=FaceDetectionResult,
        dependencies=authentication,
    )
    async def detect_faces(request: FaceDetectionRequest) -> FaceDetectionResult:
        return await face_detection_service.detect(request)

    @application.post(
        "/v1/worker/release",
        response_model=WorkerReleaseResult,
        dependencies=authentication,
    )
    async def release_worker_memory() -> WorkerReleaseResult:
        return WorkerReleaseResult(
            cancelled_job_ids=await job_manager.release_worker_memory()
        )

    @application.get(
        "/v1/jobs/{job_id}",
        response_model=GenerationJob,
        dependencies=authentication,
    )
    async def job_status(job_id: str) -> GenerationJob:
        return await job_manager.get(job_id)

    @application.get(
        "/v1/jobs/{job_id}/events",
        response_model=JobEventPage,
        dependencies=authentication,
    )
    async def job_events(
        job_id: str,
        cursor: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> JobEventPage:
        return await job_manager.events(job_id, cursor=cursor, limit=limit)

    @application.post(
        "/v1/jobs/{job_id}/cancel",
        response_model=GenerationJob,
        dependencies=authentication,
    )
    async def cancel_job(job_id: str) -> GenerationJob:
        return await job_manager.cancel(job_id)

    @application.get("/v1/outputs/{file_id}", dependencies=authentication)
    async def output_file(
        file_id: str, range_header: str | None = Header(default=None, alias="Range")
    ):
        return await _workspace_file_response(
            transfer_manager, file_id, range_header, required_kind=FileKind.OUTPUTS
        )

    @application.get("/v1/files/{file_id}/content", dependencies=authentication)
    async def file_content(
        file_id: str, range_header: str | None = Header(default=None, alias="Range")
    ):
        return await _workspace_file_response(transfer_manager, file_id, range_header)

    return application


async def _workspace_file_response(
    transfer_manager: TransferManager,
    file_id: str,
    range_header: str | None,
    *,
    required_kind: FileKind | None = None,
):
    record, path = await transfer_manager.resolve_file(file_id, required_kind=required_kind)
    if record.kind not in {FileKind.INPUTS, FileKind.OUTPUTS, FileKind.PROJECTS}:
        raise JobError(
            "file_content_unavailable",
            "Only input, output, and project content is readable.",
            404,
        )
    if not range_header:
        media_type = {
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.casefold(), "application/octet-stream")
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": "inline; filename*=UTF-8''" + quote(record.display_name),
            },
        )
    start, end = _parse_byte_range(range_header, record.size_bytes)

    with path.open("rb") as source:
        source.seek(start)
        content = source.read(end - start + 1)

    return Response(
        content=content,
        status_code=206,
        media_type="application/octet-stream",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{record.size_bytes}",
            "Content-Length": str(end - start + 1),
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(record.display_name),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k2lab-agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_agent_app(), host=args.host, port=args.port)
    return 0


def _parse_byte_range(value: str, size: int) -> tuple[int, int]:
    if not value.startswith("bytes=") or "," in value:
        raise TransferError("invalid_range", "Only one byte range is supported.", 416)
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise TransferError("invalid_range", "The byte range is invalid.", 416)
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            suffix = int(end_text)
            start = max(0, size - suffix)
            end = size - 1
    except ValueError as error:
        raise TransferError("invalid_range", "The byte range is invalid.", 416) from error
    if start < 0 or end < start or start >= size:
        raise TransferError("invalid_range", "The byte range is outside the file.", 416)
    return start, min(end, size - 1)


if __name__ == "__main__":
    raise SystemExit(main())
