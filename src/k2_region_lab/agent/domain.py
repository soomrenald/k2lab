from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from k2_region_lab.agent import AGENT_API_VERSION, AGENT_VERSION, WORKER_PROTOCOL_VERSION
from k2_region_lab.project import PROJECT_SCHEMA, PROJECT_VERSION


class ReadinessStages(BaseModel):
    container: bool
    agent: bool
    storage: bool
    models: bool
    worker: bool


class AgentHealth(BaseModel):
    status: str
    workspace_id: str
    image_version: str
    readiness: ReadinessStages
    observed_at: datetime


class AgentCapabilities(BaseModel):
    api_version: str = AGENT_API_VERSION
    agent_version: str = AGENT_VERSION
    worker_protocol_version: int = WORKER_PROTOCOL_VERSION
    project_schema: str = PROJECT_SCHEMA
    project_schema_version: int = PROJECT_VERSION
    workspace_layout_version: int
    image_version: str
    cuda_version: str | None = None
    pytorch_version: str | None = None
    supported_job_kinds: list[str] = Field(
        default_factory=lambda: ["generate", "edit_image", "refine_faces"]
    )


class StorageStatus(BaseModel):
    root: str
    total_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)
    writable: bool
    layout_version: int
