from __future__ import annotations

import argparse
from collections.abc import Sequence

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from k2_region_lab.web.development_backend import DevelopmentWorkspaceBackend
from k2_region_lab.web.domain import (
    CapabilityManifest,
    CostSnapshot,
    CredentialStatus,
    GpuOption,
    WorkspaceBackend,
    WorkspaceCreateRequest,
    WorkspaceError,
    WorkspacePlan,
    WorkspacePlanRequest,
    WorkspaceRecord,
    WorkspaceTerminateRequest,
)


class RunPodCredentialRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=512)


class ErrorBody(BaseModel):
    code: str
    message: str


def create_app(backend: WorkspaceBackend | None = None) -> FastAPI:
    workspace_backend = backend or DevelopmentWorkspaceBackend()
    application = FastAPI(
        title="K2 Region Lab Control Plane",
        version="0.1.0",
        description=(
            "Provider-neutral workspace lifecycle API. The default development backend "
            "does not create or bill cloud resources."
        ),
    )
    application.state.workspace_backend = workspace_backend
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(WorkspaceError)
    async def workspace_error_handler(
        _request: Request, error: WorkspaceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorBody(code=error.code, message=error.message).model_dump(),
        )

    @application.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": type(workspace_backend).__name__}

    @application.get("/api/v1/capabilities", response_model=CapabilityManifest)
    async def capabilities() -> CapabilityManifest:
        return CapabilityManifest(development_backend=isinstance(
            workspace_backend, DevelopmentWorkspaceBackend
        ))

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

    @application.post(
        "/api/v1/workspaces/{workspace_id}/start", response_model=WorkspaceRecord
    )
    async def start_workspace(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.start_workspace(workspace_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/stop", response_model=WorkspaceRecord
    )
    async def stop_workspace(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.stop_workspace(workspace_id)

    @application.post(
        "/api/v1/workspaces/{workspace_id}/terminate", response_model=WorkspaceRecord
    )
    async def terminate_workspace(
        workspace_id: str, request: WorkspaceTerminateRequest
    ) -> WorkspaceRecord:
        return await workspace_backend.terminate_workspace(
            workspace_id, request.confirmation
        )

    @application.post(
        "/api/v1/workspaces/{workspace_id}/lease", response_model=WorkspaceRecord
    )
    async def extend_lease(workspace_id: str) -> WorkspaceRecord:
        return await workspace_backend.extend_lease(workspace_id)

    @application.get(
        "/api/v1/workspaces/{workspace_id}/cost", response_model=CostSnapshot
    )
    async def cost_snapshot(workspace_id: str) -> CostSnapshot:
        return await workspace_backend.get_cost_snapshot(workspace_id)

    return application


app = create_app()


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
