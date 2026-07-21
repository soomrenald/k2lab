from __future__ import annotations

from datetime import datetime
from enum import StrEnum

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


class FileKind(StrEnum):
    DIFFUSION_MODELS = "diffusion_models"
    TEXT_ENCODERS = "text_encoders"
    VAE = "vae"
    LORAS = "loras"
    UPSCALE_MODELS = "upscale_models"
    FACE_DETECTION = "face_detection"
    PROJECTS = "projects"
    INPUTS = "inputs"
    OUTPUTS = "outputs"


class FileRecord(BaseModel):
    id: str
    kind: FileKind
    display_name: str
    size_bytes: int = Field(ge=0)
    sha256: str
    modified_at: datetime


class FilePage(BaseModel):
    items: list[FileRecord]
    next_cursor: str | None = None


class UploadCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    destination_kind: FileKind
    size_bytes: int = Field(gt=0, le=1_099_511_627_776)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    chunk_size_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)


class UploadSession(BaseModel):
    id: str
    filename: str
    display_name: str
    destination_kind: FileKind
    size_bytes: int
    sha256: str
    chunk_size_bytes: int
    chunk_count: int
    completed_chunks: list[int] = Field(default_factory=list)
    state: str
    created_at: datetime
    updated_at: datetime


class ChunkReceipt(BaseModel):
    upload_id: str
    index: int
    size_bytes: int
    sha256: str


class UploadCompleteResponse(BaseModel):
    file: FileRecord
    duplicate: bool = False
