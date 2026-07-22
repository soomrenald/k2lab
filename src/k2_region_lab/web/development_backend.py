from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import NoReturn
from uuid import uuid4

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
from k2_region_lab.web.domain import (
    CloudType,
    CostSnapshot,
    CredentialStatus,
    DatacenterOption,
    GpuOption,
    GpuAvailability,
    MigrationState,
    NetworkVolumeOption,
    StorageTier,
    WorkspaceCreateRequest,
    WorkspaceError,
    WorkspaceMode,
    WorkspaceMigrationCreateRequest,
    WorkspaceMigrationRecord,
    WorkspacePlan,
    WorkspacePlanRequest,
    WorkspaceRecord,
    WorkspaceState,
    WorkspaceOutput,
    utc_now,
)


class DevelopmentWorkspaceBackend:
    """Safe local backend for UI development; it never calls or bills RunPod."""

    IMAGE_DIGEST = "ghcr.io/k2-region-lab/runtime@sha256:development-placeholder"
    STORAGE_PRICE_PER_GB_MONTH = 0.10

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._credential = CredentialStatus(configured=False, development_only=True)
        self._plans: dict[str, WorkspacePlan] = {}
        self._workspaces: dict[str, WorkspaceRecord] = {}
        self._migrations: dict[str, WorkspaceMigrationRecord] = {}
        self._gpus = [
            GpuOption(
                id="NVIDIA RTX A6000",
                display_name="RTX A6000",
                memory_gb=48,
                secure_available=True,
                community_available=True,
                on_demand_price_per_hour=0.49,
                interruptible_price_per_hour=0.24,
            ),
            GpuOption(
                id="NVIDIA RTX 4090",
                display_name="RTX 4090",
                memory_gb=24,
                secure_available=False,
                community_available=True,
                on_demand_price_per_hour=0.44,
                interruptible_price_per_hour=0.21,
            ),
            GpuOption(
                id="NVIDIA A40",
                display_name="A40",
                memory_gb=48,
                secure_available=True,
                community_available=True,
                on_demand_price_per_hour=0.40,
                interruptible_price_per_hour=0.19,
            ),
        ]
        self._datacenters = [
            DatacenterOption(
                id="US-GA-2",
                name="US-GA-2",
                location="United States",
                gpu_availability=[
                    GpuAvailability(
                        gpu_type_id=gpu.id,
                        display_name=gpu.display_name,
                        stock_status="High",
                    )
                    for gpu in self._gpus
                    if gpu.secure_available
                ],
            ),
            DatacenterOption(
                id="EU-RO-1",
                name="EU-RO-1",
                location="Europe",
                gpu_availability=[
                    GpuAvailability(
                        gpu_type_id="NVIDIA A40",
                        display_name="A40",
                        stock_status="Medium",
                    )
                ],
            ),
        ]
        self._network_volumes = [
            NetworkVolumeOption(
                id="dev-volume-existing",
                name="Existing development volume",
                size_gb=200,
                datacenter_id="US-GA-2",
            )
        ]

    async def credential_status(self) -> CredentialStatus:
        return self._credential.model_copy(deep=True)

    async def validate_credentials(self, api_key: str) -> CredentialStatus:
        value = api_key.strip()
        if len(value) < 8:
            raise WorkspaceError(
                "invalid_api_key",
                "The development API key must contain at least eight characters.",
                status_code=401,
            )
        self._credential = CredentialStatus(
            configured=True,
            key_hint=f"••••{value[-4:]}",
            validated_at=utc_now(),
            development_only=True,
        )
        return self._credential.model_copy(deep=True)

    async def clear_credentials(self) -> CredentialStatus:
        self._credential = CredentialStatus(configured=False, development_only=True)
        return self._credential.model_copy(deep=True)

    async def list_gpu_options(self) -> list[GpuOption]:
        self._require_credentials()
        return [gpu.model_copy(deep=True) for gpu in self._gpus]

    async def list_datacenters(self) -> list[DatacenterOption]:
        self._require_credentials()
        return [item.model_copy(deep=True) for item in self._datacenters]

    async def list_network_volumes(self) -> list[NetworkVolumeOption]:
        self._require_credentials()
        return [item.model_copy(deep=True) for item in self._network_volumes]

    async def plan_workspace(self, request: WorkspacePlanRequest) -> WorkspacePlan:
        self._require_credentials()
        if (
            request.mode == WorkspaceMode.PORTABLE_WORKSPACE
            and request.cloud_type != CloudType.SECURE
        ):
            raise WorkspaceError(
                "portable_secure_cloud_required",
                "Portable network volumes are available only in Secure Cloud.",
            )
        selected_volume = next(
            (item for item in self._network_volumes if item.id == request.network_volume_id),
            None,
        )
        if request.network_volume_id and selected_volume is None:
            raise WorkspaceError(
                "network_volume_not_found",
                "The selected development network volume does not exist.",
                status_code=404,
            )
        if (
            request.mode == WorkspaceMode.PORTABLE_WORKSPACE
            and selected_volume
            and selected_volume.size_gb < request.workspace_disk_gb
        ):
            raise WorkspaceError(
                "network_volume_too_small",
                "The selected network volume is smaller than the requested workspace.",
                status_code=409,
            )
        if request.mode == WorkspaceMode.PORTABLE_WORKSPACE and selected_volume:
            request = request.model_copy(update={"workspace_disk_gb": selected_volume.size_gb})
        selected_datacenter_id = (
            selected_volume.datacenter_id
            if selected_volume
            else request.datacenter_priority_ids[0]
            if request.datacenter_priority_ids
            else self._datacenters[0].id
        )
        datacenter = next(
            (item for item in self._datacenters if item.id == selected_datacenter_id),
            None,
        )
        datacenter_gpu_ids = (
            {item.gpu_type_id for item in datacenter.gpu_availability}
            if request.mode == WorkspaceMode.PORTABLE_WORKSPACE and datacenter
            else None
        )
        available = {
            gpu.id: gpu
            for gpu in self._gpus
            if gpu.available and (datacenter_gpu_ids is None or gpu.id in datacenter_gpu_ids)
        }
        selected = None
        for gpu_id in request.gpu_priority_ids:
            candidate = available.get(gpu_id)
            if candidate is None:
                continue
            cloud_available = (
                candidate.secure_available
                if request.cloud_type == CloudType.SECURE
                else candidate.community_available
            )
            if cloud_available:
                selected = candidate
                break
        if selected is None:
            raise WorkspaceError(
                "requested_gpu_unavailable",
                "None of the preferred GPUs is available for the selected cloud type.",
                status_code=409,
            )
        if selected.memory_gb < 24:
            raise WorkspaceError(
                "gpu_memory_unsupported",
                "Phase one requires a GPU with at least 24 GiB of VRAM.",
            )
        if request.interruptible and selected.interruptible_price_per_hour is None:
            raise WorkspaceError(
                "interruptible_unavailable",
                "The selected GPU does not advertise interruptible capacity.",
            )
        compute_price = (
            selected.interruptible_price_per_hour
            if request.interruptible
            else selected.on_demand_price_per_hour
        )
        assert compute_price is not None
        warnings = ["Development preview only: no RunPod resource will be created."]
        if request.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            warnings.append("Stopping terminates ephemeral compute; the network volume remains.")
        else:
            warnings.append(
                "Stopping releases GPU compute but persistent storage continues to incur cost."
            )
        if request.interruptible:
            warnings.append("Interruptible Pods may stop without notice.")
        plan = WorkspacePlan(
            id=uuid4().hex,
            request=request,
            selected_gpu=selected,
            estimated_compute_per_hour=compute_price,
            estimated_storage_per_month=round(
                request.workspace_disk_gb
                * (0.07 if request.mode == WorkspaceMode.PORTABLE_WORKSPACE else 0.10),
                2,
            ),
            image_digest=self.IMAGE_DIGEST,
            provider_gpu_priority_ids=[selected.id],
            selected_datacenter_id=(
                selected_datacenter_id if request.mode == WorkspaceMode.PORTABLE_WORKSPACE else None
            ),
            selected_network_volume=selected_volume,
            create_network_volume=(
                request.mode == WorkspaceMode.PORTABLE_WORKSPACE and selected_volume is None
            ),
            warnings=warnings,
            created_at=utc_now(),
        )
        async with self._lock:
            self._plans[plan.id] = plan
        return plan.model_copy(deep=True)

    async def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRecord:
        self._require_credentials()
        async with self._lock:
            plan = self._plans.pop(request.plan_id, None)
            if plan is None:
                raise WorkspaceError(
                    "workspace_plan_missing",
                    "The workspace plan is missing or has already been used.",
                    status_code=409,
                )
            now = utc_now()
            network_volume = plan.selected_network_volume
            if plan.create_network_volume:
                network_volume = NetworkVolumeOption(
                    id=f"dev-volume-{uuid4().hex[:8]}",
                    name=request.name.strip(),
                    size_gb=plan.request.workspace_disk_gb,
                    datacenter_id=plan.selected_datacenter_id or "US-GA-2",
                )
                self._network_volumes.append(network_volume)
            workspace = WorkspaceRecord(
                id=uuid4().hex,
                name=request.name.strip(),
                mode=plan.request.mode,
                state=WorkspaceState.READY,
                gpu=plan.selected_gpu,
                cloud_type=plan.request.cloud_type,
                interruptible=plan.request.interruptible,
                container_disk_gb=plan.request.container_disk_gb,
                workspace_disk_gb=plan.request.workspace_disk_gb,
                estimated_compute_per_hour=plan.estimated_compute_per_hour,
                estimated_storage_per_month=plan.estimated_storage_per_month,
                idle_timeout_seconds=plan.request.idle_timeout_seconds,
                hard_deadline_seconds=plan.request.hard_deadline_seconds,
                lease_expires_at=now + timedelta(seconds=plan.request.idle_timeout_seconds),
                hard_expires_at=now + timedelta(seconds=plan.request.hard_deadline_seconds),
                created_at=now,
                updated_at=now,
                provider_resource_id=f"dev-pod-{uuid4().hex[:8]}",
                readiness={
                    "container": True,
                    "agent": True,
                    "storage": True,
                    "models": False,
                    "worker": False,
                },
                gpu_priority_ids=plan.provider_gpu_priority_ids,
                network_volume_id=network_volume.id if network_volume else None,
                datacenter_id=plan.selected_datacenter_id,
                owns_network_volume=plan.create_network_volume,
                storage_tier=(
                    StorageTier.NETWORK_VOLUME
                    if plan.request.mode == WorkspaceMode.PORTABLE_WORKSPACE
                    else StorageTier.POD_VOLUME
                ),
            )
            self._workspaces[workspace.id] = workspace
        return workspace.model_copy(deep=True)

    async def list_workspaces(self) -> list[WorkspaceRecord]:
        return [item.model_copy(deep=True) for item in self._workspaces.values()]

    async def get_workspace_status(self, workspace_id: str) -> WorkspaceRecord:
        return self._workspace(workspace_id).model_copy(deep=True)

    async def start_workspace(self, workspace_id: str) -> WorkspaceRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if workspace.state not in {WorkspaceState.STOPPED, WorkspaceState.ERROR}:
                raise WorkspaceError(
                    "invalid_workspace_transition",
                    f"A {workspace.state.value} workspace cannot be started.",
                    status_code=409,
                )
            now = utc_now()
            workspace = workspace.model_copy(
                update={
                    "state": WorkspaceState.READY,
                    "updated_at": now,
                    "lease_expires_at": now + timedelta(seconds=workspace.idle_timeout_seconds),
                    "hard_expires_at": now + timedelta(seconds=workspace.hard_deadline_seconds),
                    "provider_resource_id": f"dev-pod-{uuid4().hex[:8]}",
                }
            )
            self._workspaces[workspace_id] = workspace
        return workspace.model_copy(deep=True)

    async def stop_workspace(self, workspace_id: str) -> WorkspaceRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if workspace.state != WorkspaceState.READY:
                raise WorkspaceError(
                    "invalid_workspace_transition",
                    f"A {workspace.state.value} workspace cannot be stopped.",
                    status_code=409,
                )
            for migration_id, migration in self._migrations.items():
                if migration.workspace_id == workspace_id and migration.state in {
                    MigrationState.PREPARING,
                    MigrationState.COPYING,
                    MigrationState.VERIFYING,
                }:
                    self._migrations[migration_id] = migration.model_copy(
                        update={
                            "state": MigrationState.FAILED,
                            "error_code": "migration_aborted_by_stop",
                            "error_message": (
                                "The migration was stopped; its volume was retained."
                            ),
                            "updated_at": utc_now(),
                        }
                    )
            workspace = workspace.model_copy(
                update={
                    "state": WorkspaceState.STOPPED,
                    "updated_at": utc_now(),
                    "provider_resource_id": (
                        None
                        if workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE
                        else workspace.provider_resource_id
                    ),
                }
            )
            self._workspaces[workspace_id] = workspace
        return workspace.model_copy(deep=True)

    async def terminate_workspace(self, workspace_id: str, confirmation: str) -> WorkspaceRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if workspace.state == WorkspaceState.DELETED:
                return workspace.model_copy(deep=True)
            if workspace.retained_original_provider_resource_id:
                raise WorkspaceError(
                    "migration_confirmation_required",
                    "Confirm the migrated workspace before deleting cloud resources.",
                    status_code=409,
                )
            if any(
                migration.workspace_id == workspace_id
                and migration.state
                in {
                    MigrationState.PREPARING,
                    MigrationState.COPYING,
                    MigrationState.VERIFYING,
                }
                for migration in self._migrations.values()
            ):
                raise WorkspaceError(
                    "migration_in_progress",
                    "Finish or stop the migration before deleting the workspace.",
                    status_code=409,
                )
            if confirmation != workspace.name:
                raise WorkspaceError(
                    "workspace_delete_confirmation_mismatch",
                    "Type the workspace name exactly to confirm permanent deletion.",
                    status_code=409,
                )
            workspace = workspace.model_copy(
                update={
                    "state": WorkspaceState.DELETED,
                    "updated_at": utc_now(),
                    "provider_resource_id": None,
                    "readiness": {},
                }
            )
            self._workspaces[workspace_id] = workspace
        return workspace.model_copy(deep=True)

    async def extend_lease(self, workspace_id: str) -> WorkspaceRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if workspace.state != WorkspaceState.READY:
                raise WorkspaceError(
                    "workspace_not_running",
                    "Only a running workspace has an active compute lease.",
                    status_code=409,
                )
            now = utc_now()
            proposed = now + timedelta(seconds=workspace.idle_timeout_seconds)
            workspace = workspace.model_copy(
                update={
                    "lease_expires_at": min(proposed, workspace.hard_expires_at),
                    "updated_at": now,
                }
            )
            self._workspaces[workspace_id] = workspace
        return workspace.model_copy(deep=True)

    async def get_cost_snapshot(self, workspace_id: str) -> CostSnapshot:
        workspace = self._workspace(workspace_id)
        elapsed = max(0.0, (utc_now() - workspace.created_at).total_seconds())
        accrued = 0.0
        if workspace.state == WorkspaceState.READY:
            accrued = elapsed / 3600 * workspace.estimated_compute_per_hour
        compute_per_hour = (
            workspace.estimated_compute_per_hour if workspace.state == WorkspaceState.READY else 0.0
        )
        storage_per_month = (
            workspace.estimated_storage_per_month
            if workspace.network_volume_id
            else 0.0
            if workspace.state == WorkspaceState.DELETED
            else workspace.estimated_storage_per_month
        )
        migrations = [
            item for item in self._migrations.values() if item.workspace_id == workspace_id
        ]
        migration = migrations[-1] if migrations else None
        if migration:
            target_storage = round(migration.target_workspace_disk_gb * 0.07, 2)
            if migration.state in {
                MigrationState.PREPARING,
                MigrationState.COPYING,
                MigrationState.VERIFYING,
            }:
                compute_per_hour += migration.target_compute_per_hour
                storage_per_month += target_storage
            elif migration.state == MigrationState.AWAITING_CONFIRMATION:
                storage_per_month += migration.source_storage_per_month
            elif migration.state == MigrationState.FAILED:
                storage_per_month += target_storage
        return CostSnapshot(
            workspace_id=workspace.id,
            state=workspace.state,
            compute_per_hour=compute_per_hour,
            storage_per_month=storage_per_month,
            accrued_compute_estimate=accrued,
            observed_at=utc_now(),
        )

    async def create_workspace_migration(
        self, workspace_id: str, request: WorkspaceMigrationCreateRequest
    ) -> WorkspaceMigrationRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if (
                workspace.mode != WorkspaceMode.PERSISTENT_POD
                or workspace.state != WorkspaceState.READY
            ):
                raise WorkspaceError(
                    "migration_source_invalid",
                    "A ready persistent-Pod workspace is required.",
                    status_code=409,
                )
            if any(
                item.workspace_id == workspace_id
                and item.state
                in {MigrationState.PREPARING, MigrationState.COPYING, MigrationState.VERIFYING}
                for item in self._migrations.values()
            ):
                raise WorkspaceError(
                    "migration_in_progress",
                    "Finish the active workspace migration first.",
                    status_code=409,
                )
            volume = next(
                (item for item in self._network_volumes if item.id == request.network_volume_id),
                None,
            )
            target_size = request.workspace_disk_gb or workspace.workspace_disk_gb
            if request.network_volume_id and volume is None:
                raise WorkspaceError(
                    "network_volume_not_found",
                    "The selected development network volume does not exist.",
                    status_code=404,
                )
            if volume and volume.size_gb < target_size:
                raise WorkspaceError(
                    "network_volume_too_small",
                    "The selected network volume is smaller than the migration target.",
                    status_code=409,
                )
            if volume:
                target_size = volume.size_gb
            else:
                datacenter_id = (
                    request.datacenter_priority_ids[0]
                    if request.datacenter_priority_ids
                    else "US-GA-2"
                )
                volume = NetworkVolumeOption(
                    id=f"dev-volume-{uuid4().hex[:8]}",
                    name=f"Migration for {workspace.name}",
                    size_gb=target_size,
                    datacenter_id=datacenter_id,
                )
                self._network_volumes.append(volume)
            now = utc_now()
            empty_manifest = WorkspaceManifest(
                generation=1,
                layout_version=1,
                files=[],
                file_count=0,
                total_bytes=0,
                root_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                created_at=now,
            )
            migration = WorkspaceMigrationRecord(
                id=uuid4().hex,
                workspace_id=workspace.id,
                state=MigrationState.PREPARING,
                source_provider_resource_id=workspace.provider_resource_id or "dev-source",
                target_provider_resource_id=f"dev-pod-{uuid4().hex[:8]}",
                target_network_volume_id=volume.id,
                target_datacenter_id=volume.datacenter_id,
                target_gpu=workspace.gpu,
                target_compute_per_hour=workspace.estimated_compute_per_hour,
                source_storage_per_month=workspace.estimated_storage_per_month,
                target_workspace_disk_gb=volume.size_gb,
                owns_target_volume=request.network_volume_id is None,
                source_manifest=empty_manifest,
                bytes_total=0,
                created_at=now,
                updated_at=now,
            )
            self._migrations[migration.id] = migration
            return migration.model_copy(deep=True)

    async def get_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        migration = self._migration(workspace_id, migration_id)
        return migration.model_copy(deep=True)

    async def list_workspace_migrations(self, workspace_id: str) -> list[WorkspaceMigrationRecord]:
        self._workspace(workspace_id)
        return [
            item.model_copy(deep=True)
            for item in self._migrations.values()
            if item.workspace_id == workspace_id
        ]

    async def resume_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        async with self._lock:
            migration = self._migration(workspace_id, migration_id)
            if migration.state != MigrationState.PREPARING:
                return migration.model_copy(deep=True)
            workspace = self._workspace(workspace_id)
            if migration.source_manifest is None:
                raise WorkspaceError(
                    "migration_state_invalid",
                    "The development migration manifest is missing.",
                    status_code=409,
                )
            target_manifest = migration.source_manifest.model_copy(
                update={"generation": 2, "created_at": utc_now()}
            )
            updated_workspace = workspace.model_copy(
                update={
                    "mode": WorkspaceMode.PORTABLE_WORKSPACE,
                    "provider_resource_id": migration.target_provider_resource_id,
                    "network_volume_id": migration.target_network_volume_id,
                    "datacenter_id": migration.target_datacenter_id,
                    "workspace_disk_gb": migration.target_workspace_disk_gb,
                    "estimated_storage_per_month": round(
                        migration.target_workspace_disk_gb * 0.07, 2
                    ),
                    "owns_network_volume": migration.owns_target_volume,
                    "storage_tier": StorageTier.NETWORK_VOLUME,
                    "workspace_layout_version": target_manifest.layout_version,
                    "retained_original_provider_resource_id": (
                        migration.source_provider_resource_id
                    ),
                    "updated_at": utc_now(),
                }
            )
            self._workspaces[workspace_id] = updated_workspace
            migration = migration.model_copy(
                update={
                    "state": MigrationState.AWAITING_CONFIRMATION,
                    "target_manifest": target_manifest,
                    "updated_at": utc_now(),
                }
            )
            self._migrations[migration.id] = migration
            return migration.model_copy(deep=True)

    async def confirm_workspace_migration(
        self, workspace_id: str, migration_id: str, confirmation: str
    ) -> WorkspaceMigrationRecord:
        async with self._lock:
            migration = self._migration(workspace_id, migration_id)
            workspace = self._workspace(workspace_id)
            if migration.state == MigrationState.COMPLETED:
                return migration.model_copy(deep=True)
            if migration.state != MigrationState.AWAITING_CONFIRMATION:
                raise WorkspaceError(
                    "migration_not_verified",
                    "The migration has not been verified.",
                    status_code=409,
                )
            if confirmation != workspace.name:
                raise WorkspaceError(
                    "migration_confirmation_mismatch",
                    "Type the workspace name exactly to delete the original Pod volume.",
                    status_code=409,
                )
            self._workspaces[workspace_id] = workspace.model_copy(
                update={
                    "retained_original_provider_resource_id": None,
                    "updated_at": utc_now(),
                }
            )
            migration = migration.model_copy(
                update={"state": MigrationState.COMPLETED, "updated_at": utc_now()}
            )
            self._migrations[migration.id] = migration
            return migration.model_copy(deep=True)

    async def get_file_inventory(
        self, workspace_id: str, kind: FileKind, cursor: str | None = None
    ) -> FilePage:
        del kind, cursor
        self._workspace(workspace_id)
        return FilePage(items=[])

    async def save_project(
        self, workspace_id: str, filename: str, request: ProjectSaveRequest
    ) -> FileRecord:
        del filename, request
        self._transfer_unavailable(workspace_id)

    async def create_upload(self, workspace_id: str, request: UploadCreateRequest) -> UploadSession:
        del request
        self._transfer_unavailable(workspace_id)

    async def get_upload(self, workspace_id: str, upload_id: str) -> UploadSession:
        del upload_id
        self._transfer_unavailable(workspace_id)

    async def write_upload_chunk(
        self,
        workspace_id: str,
        upload_id: str,
        index: int,
        content: bytes,
        sha256: str,
    ) -> ChunkReceipt:
        del upload_id, index, content, sha256
        self._transfer_unavailable(workspace_id)

    async def complete_upload(self, workspace_id: str, upload_id: str) -> UploadCompleteResponse:
        del upload_id
        self._transfer_unavailable(workspace_id)

    async def cancel_upload(self, workspace_id: str, upload_id: str) -> None:
        del upload_id
        self._transfer_unavailable(workspace_id)

    def _transfer_unavailable(self, workspace_id: str) -> NoReturn:
        self._workspace(workspace_id)
        raise WorkspaceError(
            "development_feature_unavailable",
            "Uploads require a connected workspace agent.",
            status_code=501,
        )

    async def download_credential_status(self, provider: RemoteProvider) -> CredentialStatus:
        del provider
        return CredentialStatus(configured=False, development_only=True)

    async def store_download_credential(
        self, provider: RemoteProvider, token: str
    ) -> CredentialStatus:
        del provider, token
        raise WorkspaceError(
            "development_feature_unavailable",
            "Provider downloads require a connected workspace agent.",
            status_code=501,
        )

    async def clear_download_credential(self, provider: RemoteProvider) -> CredentialStatus:
        return await self.download_credential_status(provider)

    async def preview_civitai_download(
        self, workspace_id: str, request: CivitaiPreviewRequest
    ) -> CivitaiPreview:
        del request
        self._transfer_unavailable(workspace_id)

    async def start_civitai_download(
        self, workspace_id: str, request: CivitaiDownloadRequest
    ) -> RemoteTransfer:
        del request
        self._transfer_unavailable(workspace_id)

    async def preview_huggingface_download(
        self, workspace_id: str, request: HuggingFacePreviewRequest
    ) -> HuggingFacePreview:
        del request
        self._transfer_unavailable(workspace_id)

    async def start_huggingface_download(
        self, workspace_id: str, request: HuggingFaceDownloadRequest
    ) -> RemoteTransfer:
        del request
        self._transfer_unavailable(workspace_id)

    async def get_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer:
        del transfer_id
        self._transfer_unavailable(workspace_id)

    async def cancel_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer:
        del transfer_id
        self._transfer_unavailable(workspace_id)

    async def submit_job(self, workspace_id: str, request: JobSubmitRequest) -> GenerationJob:
        del request
        self._transfer_unavailable(workspace_id)

    async def get_job(self, workspace_id: str, job_id: str) -> GenerationJob:
        del job_id
        self._transfer_unavailable(workspace_id)

    async def get_job_events(
        self, workspace_id: str, job_id: str, cursor: str | None = None
    ) -> JobEventPage:
        del job_id, cursor
        self._transfer_unavailable(workspace_id)

    async def cancel_job(self, workspace_id: str, job_id: str) -> GenerationJob:
        del job_id
        self._transfer_unavailable(workspace_id)

    async def detect_faces(
        self, workspace_id: str, request: FaceDetectionRequest
    ) -> FaceDetectionResult:
        del request
        self._transfer_unavailable(workspace_id)

    async def release_worker_memory(self, workspace_id: str) -> WorkerReleaseResult:
        self._transfer_unavailable(workspace_id)

    async def get_output(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput:
        del file_id, range_header
        self._transfer_unavailable(workspace_id)

    async def get_file_content(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput:
        del file_id, range_header
        self._transfer_unavailable(workspace_id)

    def _require_credentials(self) -> None:
        if not self._credential.configured:
            raise WorkspaceError(
                "credentials_required",
                "Connect a RunPod account before planning a workspace.",
                status_code=401,
            )

    def _workspace(self, workspace_id: str) -> WorkspaceRecord:
        try:
            return self._workspaces[workspace_id]
        except KeyError as error:
            raise WorkspaceError(
                "workspace_not_found",
                "The requested workspace does not exist.",
                status_code=404,
            ) from error

    def _migration(self, workspace_id: str, migration_id: str) -> WorkspaceMigrationRecord:
        migration = self._migrations.get(migration_id)
        if migration is None or migration.workspace_id != workspace_id:
            raise WorkspaceError(
                "migration_not_found",
                "The requested workspace migration does not exist.",
                status_code=404,
            )
        return migration
