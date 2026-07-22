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


class ManifestEntry(BaseModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class WorkspaceManifest(BaseModel):
    generation: int = Field(ge=1)
    layout_version: int = Field(ge=1)
    files: list[ManifestEntry]
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    root_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime


class MigrationChunkReceipt(BaseModel):
    path: str
    next_offset: int = Field(ge=0)
    completed: bool = False


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


class ProjectSaveRequest(BaseModel):
    project: dict


class RemoteProvider(StrEnum):
    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"


class TransferState(StrEnum):
    PENDING = "pending"
    RESOLVING = "resolving"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CivitaiPreviewRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2048)


class CivitaiFilePreview(BaseModel):
    id: str
    filename: str
    size_bytes: int | None = Field(default=None, ge=0)
    format: str | None = None
    sha256: str | None = None
    pickle_scan: str | None = None
    virus_scan: str | None = None
    download_url: str
    preferred: bool = False
    requires_unsafe_confirmation: bool = False


class CivitaiPreview(BaseModel):
    model_id: str
    version_id: str
    model_name: str
    version_name: str
    model_type: str | None = None
    base_model: str | None = None
    training_words: list[str] = Field(default_factory=list)
    files: list[CivitaiFilePreview]


class CivitaiDownloadRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2048)
    file_id: str = Field(min_length=1, max_length=128)
    destination_kind: FileKind
    allow_unsafe_format: bool = False
    resume_transfer_id: str | None = Field(default=None, max_length=64)


class HuggingFacePreviewRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2048)
    allow_patterns: list[str] = Field(default_factory=list, max_length=20)


class HuggingFaceFilePreview(BaseModel):
    filename: str
    size_bytes: int = Field(ge=0)
    cached: bool = False


class HuggingFacePreview(BaseModel):
    repo_id: str
    repo_type: str
    revision: str
    filename: str | None = None
    mirror_repository: bool
    files: list[HuggingFaceFilePreview]
    required_bytes: int = Field(ge=0)


class HuggingFaceDownloadRequest(BaseModel):
    source_url: str = Field(min_length=1, max_length=2048)
    destination_kind: FileKind
    allow_patterns: list[str] = Field(default_factory=list, max_length=20)
    allow_unsafe_format: bool = False
    resume_transfer_id: str | None = Field(default=None, max_length=64)


class RemoteTransfer(BaseModel):
    id: str
    provider: RemoteProvider
    source_url: str
    destination_kind: FileKind
    state: TransferState
    filename: str | None = None
    bytes_total: int | None = Field(default=None, ge=0)
    bytes_complete: int = Field(default=0, ge=0)
    sha256: str | None = None
    files: list[FileRecord] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class JobKind(StrEnum):
    GENERATE = "generate"
    EDIT_IMAGE = "edit_image"
    REFINE_FACES = "refine_faces"


class JobState(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobSubmitRequest(BaseModel):
    command_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    kind: JobKind
    project_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    project: dict
    input_file_id: str | None = Field(default=None, max_length=64)
    diffusion_model_file_id: str | None = Field(default=None, max_length=64)
    text_encoder_file_id: str | None = Field(default=None, max_length=64)
    vae_file_id: str | None = Field(default=None, max_length=64)
    face_detector_file_id: str | None = Field(default=None, max_length=64)
    lora_file_ids: list[str] = Field(default_factory=list, max_length=128)
    upscale_model_file_id: str | None = Field(default=None, max_length=64)
    filename_prefix: str = Field(default="baseline", min_length=1, max_length=128)
    selected_face_indices: list[int] | None = Field(default=None, max_length=128)
    manual_face_paths: list[list[list[float]]] = Field(default_factory=list, max_length=128)


class FaceDetectionRequest(BaseModel):
    input_file_id: str = Field(min_length=1, max_length=64)
    face_detector_file_id: str | None = Field(default=None, max_length=64)
    threshold: float = Field(default=0.15, gt=0.0, lt=1.0)
    provider: str = Field(default="auto", pattern=r"^(auto|cpu|cuda)$")


class DetectedFaceRecord(BaseModel):
    index: int = Field(ge=0)
    box: list[float] = Field(min_length=4, max_length=4)
    score: float = Field(ge=0.0, le=1.0)


class FaceDetectionResult(BaseModel):
    width: int = Field(gt=0, le=4096)
    height: int = Field(gt=0, le=4096)
    execution_provider: str
    faces: list[DetectedFaceRecord]


class WorkerReleaseResult(BaseModel):
    released: bool = True
    cancelled_job_ids: list[str] = Field(default_factory=list)


class GenerationJob(BaseModel):
    id: str
    command_id: str
    kind: JobKind
    project_id: str
    state: JobState
    progress_current: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    output_file_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class JobEvent(BaseModel):
    sequence: int = Field(ge=0)
    state: str
    message: str
    payload: dict = Field(default_factory=dict)
    created_at: datetime


class JobEventPage(BaseModel):
    items: list[JobEvent]
    next_cursor: str
