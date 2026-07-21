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
    FileKind,
    FilePage,
    HuggingFaceDownloadRequest,
    HuggingFacePreview,
    HuggingFacePreviewRequest,
    RemoteProvider,
    RemoteTransfer,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
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


class CloudType(StrEnum):
    SECURE = "secure"
    COMMUNITY = "community"


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


class WorkspacePlanRequest(BaseModel):
    mode: WorkspaceMode = WorkspaceMode.PERSISTENT_POD
    gpu_priority_ids: list[str] = Field(min_length=1, max_length=12)
    cloud_type: CloudType = CloudType.SECURE
    interruptible: bool = False
    container_disk_gb: int = Field(default=50, ge=30, le=500)
    workspace_disk_gb: int = Field(default=200, ge=50, le=4_000)
    idle_timeout_seconds: int = Field(default=900, ge=300, le=86_400)
    hard_deadline_seconds: int = Field(default=28_800, ge=900, le=604_800)

    @field_validator("gpu_priority_ids")
    @classmethod
    def unique_gpu_priorities(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("GPU priorities must not contain duplicates")
        return value


class WorkspacePlan(BaseModel):
    id: str
    request: WorkspacePlanRequest
    selected_gpu: GpuOption
    estimated_compute_per_hour: float
    estimated_storage_per_month: float
    image_digest: str
    provider_gpu_priority_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class WorkspaceCreateRequest(BaseModel):
    plan_id: str
    name: str = Field(default="K2 Cloud Workspace", min_length=1, max_length=80)


class WorkspaceTerminateRequest(BaseModel):
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


class CostSnapshot(BaseModel):
    workspace_id: str
    state: WorkspaceState
    compute_per_hour: float
    storage_per_month: float
    accrued_compute_estimate: float
    observed_at: datetime


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

    async def get_file_inventory(
        self, workspace_id: str, kind: FileKind, cursor: str | None = None
    ) -> FilePage: ...

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

    async def download_credential_status(
        self, provider: RemoteProvider
    ) -> CredentialStatus: ...

    async def store_download_credential(
        self, provider: RemoteProvider, token: str
    ) -> CredentialStatus: ...

    async def clear_download_credential(
        self, provider: RemoteProvider
    ) -> CredentialStatus: ...

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

    async def get_transfer(
        self, workspace_id: str, transfer_id: str
    ) -> RemoteTransfer: ...

    async def cancel_transfer(
        self, workspace_id: str, transfer_id: str
    ) -> RemoteTransfer: ...


def utc_now() -> datetime:
    return datetime.now(UTC)
