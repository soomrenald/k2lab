from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from k2_region_lab.agent.domain import (
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
    ProjectSaveRequest,
    RemoteProvider,
    RemoteTransfer,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
    WorkerReleaseResult,
)
from k2_region_lab.project import project_state
from k2_region_lab.regional_lora import character_identity_triggers
from k2_region_lab.regional_prompting import compile_regional_prompt_plan

from k2_region_lab.web.development_backend import DevelopmentWorkspaceBackend
from k2_region_lab.web.domain import (
    CapabilityManifest,
    CostSnapshot,
    CredentialStatus,
    DatacenterOption,
    GpuOption,
    NetworkVolumeOption,
    WorkspaceBackend,
    WorkspaceCreateRequest,
    WorkspaceError,
    WorkspacePlan,
    WorkspacePlanRequest,
    WorkspaceRecord,
    WorkspaceMode,
    WorkspaceMigrationConfirmRequest,
    WorkspaceMigrationCreateRequest,
    WorkspaceMigrationRecord,
    WorkspaceTerminateRequest,
)
from k2_region_lab.web.security import (
    BrowserSession,
    BrowserSessionManager,
    ControlPlaneSecurityMiddleware,
    ControlPlaneSecuritySettings,
    SessionSecurityError,
)


def backend_from_environment() -> WorkspaceBackend:
    backend_name = os.environ.get("K2LAB_WEB_BACKEND", "development").strip().lower()
    if backend_name == "development":
        return DevelopmentWorkspaceBackend()
    if backend_name != "runpod":
        raise RuntimeError("K2LAB_WEB_BACKEND must be 'development' or 'runpod'")

    from k2_region_lab.web.credential_vault import DatabaseCredentialVault
    from k2_region_lab.web.runpod_backend import RunPodPersistentPodBackend
    from k2_region_lab.web.state_store import SqlRunPodStateStore

    encryption_key = os.environ.get("K2LAB_CREDENTIAL_FERNET_KEY")
    database_url = os.environ.get("K2LAB_DATABASE_URL")
    if not encryption_key:
        raise RuntimeError("K2LAB_CREDENTIAL_FERNET_KEY is required for the RunPod backend")
    if not image_digest:
    if not database_url:
        raise RuntimeError("K2LAB_DATABASE_URL is required for the RunPod backend")
    state_store = SqlRunPodStateStore(database_url)
    return RunPodPersistentPodBackend(
        credential_vault=DatabaseCredentialVault(state_store, encryption_key),
        state_store=state_store,
        image_digest=image_digest,
    )


class RunPodCredentialRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=512)


class ProviderTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class ErrorBody(BaseModel):
    code: str
    message: str


class UnifiedPromptPreviewRequest(BaseModel):
    project: dict[str, Any]


class UnifiedPromptPreviewRegion(BaseModel):
    id: str
    name: str
    spatial_role: str
    clause: str


class UnifiedPromptPreview(BaseModel):
    prompt: str
    regions: list[UnifiedPromptPreviewRegion]


def create_app(
    backend: WorkspaceBackend | None = None,
    *,
    security: ControlPlaneSecuritySettings | None = None,
    static_directory: Path | None = None,
) -> FastAPI:
    workspace_backend = backend or DevelopmentWorkspaceBackend()
    security_settings = security or ControlPlaneSecuritySettings()
    session_manager = BrowserSessionManager(security_settings)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        reaper_task: asyncio.Task[None] | None = None
        state_store = getattr(workspace_backend, "state_store", None)
        if state_store is not None:
            from k2_region_lab.web.lease_reaper import WorkspaceLeaseReaper

            await state_store.initialize()
            await workspace_backend.reconcile_workspaces()
            interval = float(os.environ.get("K2LAB_REAPER_INTERVAL_SECONDS", "30"))
            reaper = WorkspaceLeaseReaper(workspace_backend, interval_seconds=interval)
            reaper_task = asyncio.create_task(reaper.run_forever())
        try:
            yield
        finally:
            if reaper_task is not None:
                await WorkspaceLeaseReaper.cancel(reaper_task)
            if state_store is not None:
                await state_store.close()

    application = FastAPI(
        title="K2 Region Lab Control Plane",
        version="0.1.0",
        description=(
            "Provider-neutral workspace lifecycle API. The default development backend "
            "does not create or bill cloud resources."
        ),
        lifespan=lifespan,
    )
    application.state.workspace_backend = workspace_backend
    application.state.security_settings = security_settings
    application.state.session_manager = session_manager
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(security_settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Range", "X-Chunk-SHA256", "X-CSRF-Token"],
    )
    application.add_middleware(
        ControlPlaneSecurityMiddleware,
        settings=security_settings,
        sessions=session_manager,
    )

    @application.exception_handler(WorkspaceError)
    async def workspace_error_handler(_request: Request, error: WorkspaceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorBody(code=error.code, message=error.message).model_dump(),
        )

    @application.exception_handler(SessionSecurityError)
    async def session_security_error_handler(
        _request: Request, error: SessionSecurityError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorBody(code=error.code, message=error.message).model_dump(),
        )

    @application.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(workspace_backend).__name__}

    @application.post("/api/v1/auth/session", response_model=BrowserSession)
    async def open_browser_session(request: Request, response: Response) -> BrowserSession:
        return session_manager.open(request, response)

    @application.delete("/api/v1/auth/session", status_code=204)
    async def close_browser_session(request: Request, response: Response) -> None:
        session_manager.close(request, response)

    @application.get("/api/v1/capabilities", response_model=CapabilityManifest)
    async def capabilities() -> CapabilityManifest:
        return CapabilityManifest(
            development_backend=isinstance(workspace_backend, DevelopmentWorkspaceBackend),
            workspace_modes=[
                WorkspaceMode.PERSISTENT_POD,
                WorkspaceMode.PORTABLE_WORKSPACE,
            ],
        )

    @application.post(
        "/api/v1/projects/unified-prompt-preview",
        response_model=UnifiedPromptPreview,
    )
    async def preview_unified_prompt(
        request: UnifiedPromptPreviewRequest,
    ) -> UnifiedPromptPreview:
        try:
            state = project_state(request.project)
            plan = compile_regional_prompt_plan(
                state.canvas_width,
                state.canvas_height,
                state.global_prompt,
                state.regions,
                strength=state.regional_prompt_strength,
                outside_penalty=state.regional_outside_penalty,
                falloff_pixels=state.regional_feather_pixels,
                subject_competition=state.regional_subject_competition,
                subject_fill=state.regional_subject_fill,
                late_step_scale=(
                    state.regional_late_step_scale if state.regional_relaxation else 1.0
                ),
                emphases=state.prompt_emphases,
                character_identity_triggers=character_identity_triggers(
                    list(request.project.get("loras", []))
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WorkspaceError(
                "invalid_project",
                f"The project cannot be compiled: {error}",
                status_code=422,
            ) from error
        return UnifiedPromptPreview(
            prompt=plan.prompt,
            regions=[
                UnifiedPromptPreviewRegion(
                    id=region.region_id,
                    name=region.name,
                    spatial_role=region.spatial_role,
                    clause=region.clause,
                )
                for region in plan.regions
            ],
        )

    @application.get("/api/v1/credentials/runpod", response_model=CredentialStatus)
    async def credential_status() -> CredentialStatus:
        return await workspace_backend.credential_status()

    @application.post("/api/v1/credentials/runpod", response_model=CredentialStatus)
    async def connect_runpod(request: RunPodCredentialRequest) -> CredentialStatus:
        return await workspace_backend.validate_credentials(request.api_key)

    @application.delete("/api/v1/credentials/runpod", response_model=CredentialStatus)
    async def disconnect_runpod() -> CredentialStatus:
        return await workspace_backend.clear_credentials()

    @application.get("/api/v1/gpus", response_model=list[GpuOption])
    async def list_gpus() -> list[GpuOption]:
        return await workspace_backend.list_gpu_options()

    @application.get("/api/v1/datacenters", response_model=list[DatacenterOption])
    async def list_datacenters() -> list[DatacenterOption]:
        return await workspace_backend.list_datacenters()

    @application.get("/api/v1/network-volumes", response_model=list[NetworkVolumeOption])
    async def list_network_volumes() -> list[NetworkVolumeOption]:
        return await workspace_backend.list_network_volumes()

    @application.post("/api/v1/workspace-plans", response_model=WorkspacePlan)
    async def plan_workspace(request: WorkspacePlanRequest) -> WorkspacePlan:
        return await workspace_backend.plan_workspace(request)

    @application.post("/api/v1/workspaces", response_model=WorkspaceRecord)
    async def create_workspace(request: WorkspaceCreateRequest) -> WorkspaceRecord:
        return await workspace_backend.create_workspace(request)

    @application.get("/api/v1/workspaces", response_model=list[WorkspaceRecord])
    async def list_workspaces() -> list[WorkspaceRecord]:
        return await workspace_backend.list_workspaces()

    @application.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceRecord)
    async def workspace_status(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.get_workspace_status(workspace_id)

    @application.post("/api/v1/workspaces/{workspace_id}/start", response_model=WorkspaceRecord)
    async def start_workspace(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.start_workspace(workspace_id)

    @application.post("/api/v1/workspaces/{workspace_id}/stop", response_model=WorkspaceRecord)
    async def stop_workspace(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.stop_workspace(workspace_id)

    @application.post("/api/v1/workspaces/{workspace_id}/terminate", response_model=WorkspaceRecord)
    async def terminate_workspace(
        workspace_id: str, request: WorkspaceTerminateRequest
    ) -> WorkspaceRecord:
        return await workspace_backend.terminate_workspace(workspace_id, request.confirmation)

    @application.post("/api/v1/workspaces/{workspace_id}/lease", response_model=WorkspaceRecord)
    async def extend_lease(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.extend_lease(workspace_id)

    @application.get("/api/v1/workspaces/{workspace_id}/cost", response_model=CostSnapshot)
    async def cost_snapshot(workspace_id: str) -> CostSnapshot:
        return await workspace_backend.get_cost_snapshot(workspace_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/migrations",
        response_model=WorkspaceMigrationRecord,
        status_code=202,
    )
    async def create_workspace_migration(
        workspace_id: str, request: WorkspaceMigrationCreateRequest
    ) -> WorkspaceMigrationRecord:
        return await workspace_backend.create_workspace_migration(workspace_id, request)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/migrations",
        response_model=list[WorkspaceMigrationRecord],
    )
    async def list_workspace_migrations(
        workspace_id: str,
    ) -> list[WorkspaceMigrationRecord]:
        return await workspace_backend.list_workspace_migrations(workspace_id)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/migrations/{migration_id}",
        response_model=WorkspaceMigrationRecord,
    )
    async def get_workspace_migration(
        workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        return await workspace_backend.get_workspace_migration(workspace_id, migration_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/migrations/{migration_id}/resume",
        response_model=WorkspaceMigrationRecord,
    )
    async def resume_workspace_migration(
        workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        return await workspace_backend.resume_workspace_migration(workspace_id, migration_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/migrations/{migration_id}/confirm",
        response_model=WorkspaceMigrationRecord,
    )
    async def confirm_workspace_migration(
        workspace_id: str,
        migration_id: str,
        request: WorkspaceMigrationConfirmRequest,
    ) -> WorkspaceMigrationRecord:
        return await workspace_backend.confirm_workspace_migration(
            workspace_id, migration_id, request.confirmation
        )

    @application.get("/api/v1/workspaces/{workspace_id}/files", response_model=FilePage)
    async def file_inventory(
        workspace_id: str,
        kind: FileKind,
        cursor: str | None = None,
    ) -> FilePage:
        return await workspace_backend.get_file_inventory(workspace_id, kind, cursor)

    @application.put(
        "/api/v1/workspaces/{workspace_id}/projects/{filename}", response_model=FileRecord
    )
    async def save_project(
        workspace_id: str, filename: str, request: ProjectSaveRequest
    ) -> FileRecord:
        return await workspace_backend.save_project(workspace_id, filename, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/uploads",
        response_model=UploadSession,
        status_code=201,
    )
    async def create_upload(workspace_id: str, request: UploadCreateRequest) -> UploadSession:
        return await workspace_backend.create_upload(workspace_id, request)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/uploads/{upload_id}",
        response_model=UploadSession,
    )
    async def upload_status(workspace_id: str, upload_id: str) -> UploadSession:
        return await workspace_backend.get_upload(workspace_id, upload_id)

    @application.put(
        "/api/v1/workspaces/{workspace_id}/uploads/{upload_id}/chunks/{index}",
        response_model=ChunkReceipt,
    )
    async def upload_chunk(
        workspace_id: str,
        upload_id: str,
        index: int,
        request: Request,
        x_chunk_sha256: str = Header(alias="X-Chunk-SHA256"),
    ) -> ChunkReceipt:
        return await workspace_backend.write_upload_chunk(
            workspace_id,
            upload_id,
            index,
            await request.body(),
            x_chunk_sha256,
        )

    @application.post(
        "/api/v1/workspaces/{workspace_id}/uploads/{upload_id}/complete",
        response_model=UploadCompleteResponse,
    )
    async def complete_upload(workspace_id: str, upload_id: str) -> UploadCompleteResponse:
        return await workspace_backend.complete_upload(workspace_id, upload_id)

    @application.delete("/api/v1/workspaces/{workspace_id}/uploads/{upload_id}", status_code=204)
    async def cancel_upload(workspace_id: str, upload_id: str) -> None:
        await workspace_backend.cancel_upload(workspace_id, upload_id)

    @application.get("/api/v1/credentials/downloads/{provider}", response_model=CredentialStatus)
    async def download_credential_status(provider: RemoteProvider) -> CredentialStatus:
        return await workspace_backend.download_credential_status(provider)

    @application.post("/api/v1/credentials/downloads/{provider}", response_model=CredentialStatus)
    async def store_download_credential(
        provider: RemoteProvider, request: ProviderTokenRequest
    ) -> CredentialStatus:
        return await workspace_backend.store_download_credential(provider, request.token)

    @application.delete("/api/v1/credentials/downloads/{provider}", response_model=CredentialStatus)
    async def clear_download_credential(provider: RemoteProvider) -> CredentialStatus:
        return await workspace_backend.clear_download_credential(provider)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/downloads/civitai/preview",
        response_model=CivitaiPreview,
    )
    async def preview_civitai_download(
        workspace_id: str, request: CivitaiPreviewRequest
    ) -> CivitaiPreview:
        return await workspace_backend.preview_civitai_download(workspace_id, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/downloads/civitai",
        response_model=RemoteTransfer,
        status_code=202,
    )
    async def start_civitai_download(
        workspace_id: str, request: CivitaiDownloadRequest
    ) -> RemoteTransfer:
        return await workspace_backend.start_civitai_download(workspace_id, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/downloads/huggingface/preview",
        response_model=HuggingFacePreview,
    )
    async def preview_huggingface_download(
        workspace_id: str, request: HuggingFacePreviewRequest
    ) -> HuggingFacePreview:
        return await workspace_backend.preview_huggingface_download(workspace_id, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/downloads/huggingface",
        response_model=RemoteTransfer,
        status_code=202,
    )
    async def start_huggingface_download(
        workspace_id: str, request: HuggingFaceDownloadRequest
    ) -> RemoteTransfer:
        return await workspace_backend.start_huggingface_download(workspace_id, request)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/transfers/{transfer_id}",
        response_model=RemoteTransfer,
    )
    async def transfer_status(workspace_id: str, transfer_id: str) -> RemoteTransfer:
        return await workspace_backend.get_transfer(workspace_id, transfer_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/transfers/{transfer_id}/cancel",
        response_model=RemoteTransfer,
    )
    async def cancel_transfer(workspace_id: str, transfer_id: str) -> RemoteTransfer:
        return await workspace_backend.cancel_transfer(workspace_id, transfer_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/jobs",
        response_model=GenerationJob,
        status_code=202,
    )
    async def submit_job(workspace_id: str, request: JobSubmitRequest) -> GenerationJob:
        return await workspace_backend.submit_job(workspace_id, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/faces/detect",
        response_model=FaceDetectionResult,
    )
    async def detect_faces(workspace_id: str, request: FaceDetectionRequest) -> FaceDetectionResult:
        return await workspace_backend.detect_faces(workspace_id, request)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/worker/release",
        response_model=WorkerReleaseResult,
    )
    async def release_worker_memory(workspace_id: str) -> WorkerReleaseResult:
        return await workspace_backend.release_worker_memory(workspace_id)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/jobs/{job_id}",
        response_model=GenerationJob,
    )
    async def job_status(workspace_id: str, job_id: str) -> GenerationJob:
        return await workspace_backend.get_job(workspace_id, job_id)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/jobs/{job_id}/events",
        response_model=JobEventPage,
    )
    async def job_events(workspace_id: str, job_id: str, cursor: str | None = None) -> JobEventPage:
        return await workspace_backend.get_job_events(workspace_id, job_id, cursor)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/jobs/{job_id}/cancel",
        response_model=GenerationJob,
    )
    async def cancel_job(workspace_id: str, job_id: str) -> GenerationJob:
        return await workspace_backend.cancel_job(workspace_id, job_id)

    @application.get("/api/v1/workspaces/{workspace_id}/outputs/{file_id}")
    async def output_file(
        workspace_id: str,
        file_id: str,
        range_header: str | None = Header(default=None, alias="Range"),
    ) -> Response:
        output = await workspace_backend.get_output(workspace_id, file_id, range_header)
        return Response(
            content=output.content,
            status_code=output.status_code,
            headers=output.headers,
        )

    @application.get("/api/v1/workspaces/{workspace_id}/files/{file_id}/content")
    async def file_content(
        workspace_id: str,
        file_id: str,
        range_header: str | None = Header(default=None, alias="Range"),
    ) -> Response:
        output = await workspace_backend.get_file_content(workspace_id, file_id, range_header)
        return Response(
            content=output.content,
            status_code=output.status_code,
            headers=output.headers,
        )

    if static_directory is not None:
        application.mount(
            "/",
            StaticFiles(directory=static_directory, html=True, check_dir=True),
            name="studio",
        )

    return application


_configured_backend = backend_from_environment()
_local_single_user = os.environ.get("K2LAB_LOCAL_SINGLE_USER", "").casefold() in {
    "1",
    "true",
    "yes",
}
_local_port = int(os.environ.get("K2LAB_LOCAL_PORT", "8000"))
_static_directory = (
    Path(__file__).with_name("static")
    if os.environ.get("K2LAB_SERVE_WEB_UI", "").casefold() in {"1", "true", "yes"}
    else None
)
app = create_app(
    _configured_backend,
    security=(
        ControlPlaneSecuritySettings.local_single_user(port=_local_port)
        if _local_single_user
        else ControlPlaneSecuritySettings.from_environment(
            production=not isinstance(_configured_backend, DevelopmentWorkspaceBackend)
        )
    ),
    static_directory=_static_directory,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k2lab-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "k2_region_lab.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
