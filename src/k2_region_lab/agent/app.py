from __future__ import annotations

import argparse
import hmac
import os
import shutil
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status

from k2_region_lab.agent.domain import (
    AgentCapabilities,
    AgentHealth,
    ReadinessStages,
    StorageStatus,
)
from k2_region_lab.agent.storage import LAYOUT_VERSION, WorkspaceLayout


class AgentSettings:
    def __init__(
        self,
        *,
        session_token: str,
        workspace_id: str,
        image_version: str,
        workspace_root: Path,
        worker_python: Path,
        cuda_version: str | None = None,
        pytorch_version: str | None = None,
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
        self.cuda_version = cuda_version
        self.pytorch_version = pytorch_version

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
            cuda_version=os.environ.get("K2LAB_CUDA_VERSION"),
            pytorch_version=os.environ.get("K2LAB_PYTORCH_VERSION"),
        )


def create_agent_app(settings: AgentSettings | None = None) -> FastAPI:
    configured = settings or AgentSettings.from_environment()
    layout = WorkspaceLayout(configured.workspace_root)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        layout.initialize()
        yield

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

    @application.get(
        "/v1/health", response_model=AgentHealth, dependencies=authentication
    )
    async def health() -> AgentHealth:
        writable = layout.is_writable()
        readiness = ReadinessStages(
            container=True,
            agent=True,
            storage=writable,
            models=layout.model_inventory_ready(),
            worker=bool(application.state.worker_ready),
        )
        return AgentHealth(
            status="ready" if all(readiness.model_dump().values()) else "starting",
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

    @application.get(
        "/v1/storage", response_model=StorageStatus, dependencies=authentication
    )
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

    return application


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k2lab-agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_agent_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
