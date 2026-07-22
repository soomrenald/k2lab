from __future__ import annotations

from abc import abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    WorkspaceManifest,
)


class WorkspaceMode(StrEnum):
    PERSISTENT_POD = "persistent_pod"
    PORTABLE_WORKSPACE = "portable_workspace"


class WorkspaceState(StrEnum):
    PROVISIONING = "provisioning"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"


class MigrationState(StrEnum):
    PREPARING = "preparing"
    COPYING = "copying"
    VERIFYING = "verifying"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"


class CloudType(StrEnum):
    SECURE = "secure"
    COMMUNITY = "community"


class StorageTier(StrEnum):
    POD_VOLUME = "pod_volume"
    NETWORK_VOLUME = "network_volume"


class GpuOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    memory_gb: int = Field(ge=1)
    secure_available: bool
    community_available: bool
    on_demand_price_per_hour: float = Field(ge=0)
    interruptible_price_per_hour: float | None = Field(default=None, ge=0)
    secure_on_demand_price_per_hour: float | None = Field(default=None, ge=0)
    community_on_demand_price_per_hour: float | None = Field(default=None, ge=0)
    secure_interruptible_price_per_hour: float | None = Field(default=None, ge=0)
    community_interruptible_price_per_hour: float | None = Field(default=None, ge=0)
    available: bool = True


class CredentialStatus(BaseModel):
    configured: bool
    key_hint: str | None = None
    validated_at: datetime | None = None
    development_only: bool = False


class GpuAvailability(BaseModel):
    gpu_type_id: str
    display_name: str
    stock_status: str


class DatacenterOption(BaseModel):
    id: str
    name: str
    location: str
    gpu_availability: list[GpuAvailability] = Field(default_factory=list)


class NetworkVolumeOption(BaseModel):
    id: str
    name: str
    size_gb: int = Field(ge=1, le=4_000)
    datacenter_id: str


class WorkspacePlanRequest(BaseModel):
    mode: WorkspaceMode = WorkspaceMode.PERSISTENT_POD
    gpu_priority_ids: list[str] = Field(min_length=1, max_length=12)
    cloud_type: CloudType = CloudType.SECURE
    interruptible: bool = False
    container_disk_gb: int = Field(default=50, ge=30, le=500)
    workspace_disk_gb: int = Field(default=200, ge=50, le=4_000)
    idle_timeout_seconds: int = Field(default=900, ge=300, le=86_400)
    hard_deadline_seconds: int = Field(default=28_800, ge=900, le=604_800)
    network_volume_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$"
    )
    datacenter_priority_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("gpu_priority_ids")
    @classmethod
    def unique_gpu_priorities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("GPU priorities must not contain duplicates")
        return value

    @field_validator("datacenter_priority_ids")
    @classmethod
    def unique_datacenter_priorities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Datacenter priorities must not contain duplicates")
        return value


class WorkspacePlan(BaseModel):
    id: str
    request: WorkspacePlanRequest
    selected_gpu: GpuOption
    estimated_compute_per_hour: float
    estimated_storage_per_month: float
    image_digest: str
    provider_gpu_priority_ids: list[str] = Field(default_factory=list)
    selected_datacenter_id: str | None = None
    selected_network_volume: NetworkVolumeOption | None = None
    create_network_volume: bool = False
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class WorkspaceCreateRequest(BaseModel):
    plan_id: str
    name: str = Field(default="K2 Cloud Workspace", min_length=1, max_length=80)


class WorkspaceTerminateRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=80)


class WorkspaceMigrationCreateRequest(BaseModel):
    network_volume_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,190}$"
    )
    workspace_disk_gb: int | None = Field(default=None, ge=50, le=4_000)
    datacenter_priority_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("datacenter_priority_ids")
    @classmethod
    def unique_datacenter_priorities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Datacenter priorities must not contain duplicates")
        return value


class WorkspaceMigrationConfirmRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=80)


class WorkspaceRecord(BaseModel):
    id: str
    name: str
    mode: WorkspaceMode
    state: WorkspaceState
    gpu: GpuOption
    cloud_type: CloudType
    interruptible: bool
    container_disk_gb: int
    workspace_disk_gb: int
    estimated_compute_per_hour: float
    estimated_storage_per_month: float
    idle_timeout_seconds: int
    hard_deadline_seconds: int
    lease_expires_at: datetime
    hard_expires_at: datetime
    created_at: datetime
    updated_at: datetime
    provider_resource_id: str | None = None
    readiness: dict[str, bool] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    gpu_priority_ids: list[str] = Field(default_factory=list)
    network_volume_id: str | None = None
    datacenter_id: str | None = None
    owns_network_volume: bool = False
    storage_tier: StorageTier = StorageTier.POD_VOLUME
    workspace_layout_version: int = Field(default=1, ge=1)
    retained_original_provider_resource_id: str | None = None


class WorkspaceMigrationRecord(BaseModel):
    id: str
    operation_id: str | None = None
    workspace_id: str
    state: MigrationState
    source_provider_resource_id: str
    target_provider_resource_id: str | None = None
    target_network_volume_id: str | None = None
    target_datacenter_id: str | None = None
    target_gpu: GpuOption | None = None
    target_compute_per_hour: float = Field(default=0, ge=0)
    source_storage_per_month: float = Field(default=0, ge=0)
    target_workspace_disk_gb: int = Field(ge=50, le=4_000)
    owns_target_volume: bool = False
    source_manifest: WorkspaceManifest | None = None
    target_manifest: WorkspaceManifest | None = None
    current_file_index: int = Field(default=0, ge=0)
    current_file_offset: int = Field(default=0, ge=0)
    bytes_copied: int = Field(default=0, ge=0)
    bytes_total: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class CostSnapshot(BaseModel):
    workspace_id: str
    state: WorkspaceState
    compute_per_hour: float
    storage_per_month: float
    accrued_compute_estimate: float
    observed_at: datetime


class WorkspaceOutput(BaseModel):
    content: bytes
    status_code: int
    headers: dict[str, str]


class CapabilityManifest(BaseModel):
    api_version: str = "v1"
    project_schema: str = "k2-region-lab-project"
    project_schema_version: int = 18
    minimum_gpu_memory_gb: int = 24
    workspace_modes: list[WorkspaceMode] = Field(
        default_factory=lambda: [WorkspaceMode.PERSISTENT_POD]
    )
    development_backend: bool = False


class WorkspaceError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class WorkspaceBackend(Protocol):
    @abstractmethod
    async def credential_status(self) -> CredentialStatus: ...

    @abstractmethod
    async def validate_credentials(self, api_key: str) -> CredentialStatus: ...

    @abstractmethod
    async def clear_credentials(self) -> CredentialStatus: ...

    @abstractmethod
    async def list_gpu_options(self) -> list[GpuOption]: ...

    @abstractmethod
    async def list_datacenters(self) -> list[DatacenterOption]: ...

    @abstractmethod
    async def list_network_volumes(self) -> list[NetworkVolumeOption]: ...

    @abstractmethod
    async def plan_workspace(self, request: WorkspacePlanRequest) -> WorkspacePlan: ...

    @abstractmethod
    async def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRecord: ...

    @abstractmethod
    async def list_workspaces(self) -> list[WorkspaceRecord]: ...

    @abstractmethod
    async def get_workspace_status(self, workspace_id: str) -> WorkspaceRecord: ...

    @abstractmethod
    async def start_workspace(self, workspace_id: str) -> WorkspaceRecord: ...

    @abstractmethod
    async def stop_workspace(self, workspace_id: str) -> WorkspaceRecord: ...

    @abstractmethod
    async def terminate_workspace(
        self, workspace_id: str, confirmation: str
    ) -> WorkspaceRecord: ...

    @abstractmethod
    async def extend_lease(self, workspace_id: str) -> WorkspaceRecord: ...

    @abstractmethod
    async def get_cost_snapshot(self, workspace_id: str) -> CostSnapshot: ...

    async def create_workspace_migration(
        self, workspace_id: str, request: WorkspaceMigrationCreateRequest
    ) -> WorkspaceMigrationRecord: ...

    async def list_workspace_migrations(
        self, workspace_id: str
    ) -> list[WorkspaceMigrationRecord]: ...

    async def get_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord: ...

    async def resume_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord: ...

    async def confirm_workspace_migration(
        self, workspace_id: str, migration_id: str, confirmation: str
    ) -> WorkspaceMigrationRecord: ...

    async def get_file_inventory(
        self, workspace_id: str, kind: FileKind, cursor: str | None = None
    ) -> FilePage: ...

    async def save_project(
        self, workspace_id: str, filename: str, request: ProjectSaveRequest
    ) -> FileRecord: ...

    async def create_upload(
        self, workspace_id: str, request: UploadCreateRequest
    ) -> UploadSession: ...

    async def get_upload(self, workspace_id: str, upload_id: str) -> UploadSession: ...

    async def write_upload_chunk(
        self,
        workspace_id: str,
        upload_id: str,
        index: int,
        content: bytes,
        sha256: str,
    ) -> ChunkReceipt: ...

    async def complete_upload(
        self, workspace_id: str, upload_id: str
    ) -> UploadCompleteResponse: ...

    async def cancel_upload(self, workspace_id: str, upload_id: str) -> None: ...

    async def download_credential_status(self, provider: RemoteProvider) -> CredentialStatus: ...

    async def store_download_credential(
        self, provider: RemoteProvider, token: str
    ) -> CredentialStatus: ...

    async def clear_download_credential(self, provider: RemoteProvider) -> CredentialStatus: ...

    async def preview_civitai_download(
        self, workspace_id: str, request: CivitaiPreviewRequest
    ) -> CivitaiPreview: ...

    async def start_civitai_download(
        self, workspace_id: str, request: CivitaiDownloadRequest
    ) -> RemoteTransfer: ...

    async def preview_huggingface_download(
        self, workspace_id: str, request: HuggingFacePreviewRequest
    ) -> HuggingFacePreview: ...

    async def start_huggingface_download(
        self, workspace_id: str, request: HuggingFaceDownloadRequest
    ) -> RemoteTransfer: ...

    async def get_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer: ...

    async def cancel_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer: ...

    async def submit_job(self, workspace_id: str, request: JobSubmitRequest) -> GenerationJob: ...

    async def get_job(self, workspace_id: str, job_id: str) -> GenerationJob: ...

    async def get_job_events(
        self, workspace_id: str, job_id: str, cursor: str | None = None
    ) -> JobEventPage: ...

    async def cancel_job(self, workspace_id: str, job_id: str) -> GenerationJob: ...

    async def detect_faces(
        self, workspace_id: str, request: FaceDetectionRequest
    ) -> FaceDetectionResult: ...

    async def release_worker_memory(self, workspace_id: str) -> WorkerReleaseResult: ...

    async def get_output(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput: ...

    async def get_file_content(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput: ...


def utc_now() -> datetime:
    return datetime.now(UTC)
