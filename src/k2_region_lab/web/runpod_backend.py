from __future__ import annotations

import math
import secrets
from collections.abc import Callable
from datetime import timedelta
from typing import Any
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
    JobState,
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
from k2_region_lab.web.agent_client import WorkspaceAgentApi, WorkspaceAgentClient
from k2_region_lab.web.credential_vault import CredentialVault
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
from k2_region_lab.web.runpod_api import RunPodApi, RunPodApiClient, RunPodGpuType
from k2_region_lab.web.state_store import RunPodStateStore


class RunPodPersistentPodBackend:
    """Durable RunPod backend for persistent Pods and portable network volumes."""

    PROVIDER_CREDENTIAL_ID = "provider:runpod"
    STORAGE_PRICE_PER_GB_MONTH = 0.10
    NETWORK_VOLUME_PRICE_UNDER_1TB = 0.07
    NETWORK_VOLUME_PRICE_OVER_1TB = 0.05
    MIGRATION_CHUNK_SIZE = 8 * 1024 * 1024
    MIGRATION_COPY_BUDGET = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        credential_vault: CredentialVault,
        state_store: RunPodStateStore,
        image_digest: str,
        image_version: str,
        api_factory: Callable[[str], RunPodApi] = RunPodApiClient,
        agent_factory: Callable[[str, str], WorkspaceAgentApi] = WorkspaceAgentClient,
    ) -> None:
        if "@sha256:" not in image_digest:
            raise ValueError("RunPod runtime image must use an immutable sha256 digest")
        self._vault = credential_vault
        self.state_store = state_store
        self._image_digest = image_digest
        self._image_version = image_version
        self._api_factory = api_factory
        self._agent_factory = agent_factory

    async def credential_status(self) -> CredentialStatus:
        return await self._vault.status(self.PROVIDER_CREDENTIAL_ID)

    async def validate_credentials(self, api_key: str) -> CredentialStatus:
        value = api_key.strip()
        if len(value) < 8:
            raise WorkspaceError(
                "invalid_api_key",
                "Enter a complete RunPod API key.",
                status_code=401,
            )
        api = self._api_factory(value)
        await api.validate_credentials()
        await api.list_gpu_types()
        status = CredentialStatus(
            configured=True,
            key_hint=f"••••{value[-4:]}",
            validated_at=utc_now(),
        )
        await self._vault.store(
            self.PROVIDER_CREDENTIAL_ID,
            value,
            key_hint=status.key_hint,
            validated_at=status.validated_at,
        )
        await self.state_store.append_audit(action="runpod.credentials.validate", result="success")
        return status

    async def clear_credentials(self) -> CredentialStatus:
        active = any(
            workspace.state != WorkspaceState.DELETED
            and (workspace.provider_resource_id or workspace.network_volume_id)
            for workspace in await self.state_store.list_workspaces()
        )
        if active:
            raise WorkspaceError(
                "credentials_in_use",
                "Delete every RunPod workspace before disconnecting the account.",
                status_code=409,
            )
        await self._vault.delete(self.PROVIDER_CREDENTIAL_ID)
        await self.state_store.append_audit(action="runpod.credentials.revoke", result="success")
        return CredentialStatus(configured=False)

    async def list_gpu_options(self) -> list[GpuOption]:
        api = await self._api()
        inventory = await api.list_gpu_types()
        options = [self._gpu_option(item) for item in inventory if item.memory_gb >= 24]
        return sorted(options, key=lambda item: (-item.memory_gb, item.display_name))

    async def list_datacenters(self) -> list[DatacenterOption]:
        items = await (await self._api()).list_datacenters()
        return [
            DatacenterOption(
                id=item.id,
                name=item.name,
                location=item.location,
                gpu_availability=[
                    GpuAvailability(
                        gpu_type_id=gpu.gpu_type_id,
                        display_name=gpu.display_name,
                        stock_status=gpu.stock_status,
                    )
                    for gpu in item.gpu_availability
                ],
            )
            for item in items
        ]

    async def list_network_volumes(self) -> list[NetworkVolumeOption]:
        items = await (await self._api()).list_network_volumes()
        return [
            NetworkVolumeOption(
                id=item.id,
                name=item.name,
                size_gb=item.size_gb,
                datacenter_id=item.datacenter_id,
            )
            for item in items
        ]

    async def plan_workspace(self, request: WorkspacePlanRequest) -> WorkspacePlan:
        if (
            request.mode == WorkspaceMode.PORTABLE_WORKSPACE
            and request.cloud_type != CloudType.SECURE
        ):
            raise WorkspaceError(
                "portable_secure_cloud_required",
                "Portable network volumes are available only in Secure Cloud.",
            )
        options = {item.id: item for item in await self.list_gpu_options()}
        selected_datacenter_id: str | None = None
        selected_network_volume: NetworkVolumeOption | None = None
        create_network_volume = False
        if request.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            datacenters = await self.list_datacenters()
            if request.network_volume_id:
                volumes = {volume.id: volume for volume in await self.list_network_volumes()}
                selected_network_volume = volumes.get(request.network_volume_id)
                if selected_network_volume is None:
                    raise WorkspaceError(
                        "network_volume_not_found",
                        "The selected RunPod network volume no longer exists.",
                        status_code=404,
                    )
                if selected_network_volume.size_gb < request.workspace_disk_gb:
                    raise WorkspaceError(
                        "network_volume_too_small",
                        "The selected network volume is smaller than the requested workspace.",
                        status_code=409,
                    )
                request = request.model_copy(
                    update={"workspace_disk_gb": selected_network_volume.size_gb}
                )
                selected_datacenter_id = selected_network_volume.datacenter_id
            candidates = self._portable_candidates(
                request, options, datacenters, selected_datacenter_id
            )
            if not candidates:
                raise WorkspaceError(
                    "portable_capacity_unavailable",
                    "None of the preferred GPUs is available in a compatible datacenter.",
                    status_code=409,
                )
            selected, selected_datacenter_id = candidates[0]
            eligible = [item for item, _datacenter in candidates]
            create_network_volume = selected_network_volume is None
        else:
            eligible = [
                options[gpu_id]
                for gpu_id in request.gpu_priority_ids
                if gpu_id in options
                and self._cloud_available(options[gpu_id], request.cloud_type)
                and self._price(options[gpu_id], request.cloud_type, request.interruptible)
                is not None
            ]
            selected = eligible[0] if eligible else None
        if selected is None:
            raise WorkspaceError(
                "requested_gpu_unavailable",
                "None of the preferred GPUs is available for the selected cloud type.",
                status_code=409,
            )
        price = self._price(selected, request.cloud_type, request.interruptible)
        if price is None:
            kind = "Interruptible" if request.interruptible else "On-demand"
            raise WorkspaceError(
                "requested_gpu_unavailable",
                f"{kind} pricing is unavailable for the selected GPU and cloud type.",
                status_code=409,
            )
        if request.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            unavailable = [
                gpu_id
                for gpu_id in request.gpu_priority_ids
                if gpu_id not in {item.id for item in eligible}
            ]
            warnings = [
                "Stopping terminates the ephemeral Pod; the network volume remains billable.",
                "Network volumes are datacenter-bound and are not a permanent backup.",
            ]
            if unavailable:
                warnings.append("Unavailable in the selected datacenter: " + ", ".join(unavailable))
        else:
            warnings = [
                "Stopping releases GPU compute but persistent storage continues to incur cost.",
                "Persistent Pod storage is deleted permanently when this workspace is deleted.",
            ]
        if request.interruptible:
            warnings.append("Interruptible Pods may stop without notice.")
        plan = WorkspacePlan(
            id=uuid4().hex,
            request=request,
            selected_gpu=selected,
            estimated_compute_per_hour=price,
            estimated_storage_per_month=self._storage_price(request),
            image_digest=self._image_digest,
            provider_gpu_priority_ids=[item.id for item in eligible],
            selected_datacenter_id=selected_datacenter_id,
            selected_network_volume=selected_network_volume,
            create_network_volume=create_network_volume,
            warnings=warnings,
            created_at=utc_now(),
        )
        await self.state_store.save_plan(plan)
        return plan.model_copy(deep=True)

    async def create_workspace(self, request: WorkspaceCreateRequest) -> WorkspaceRecord:
        plan = await self.state_store.consume_plan(request.plan_id)
        if plan is None:
            raise WorkspaceError(
                "workspace_plan_missing",
                "The workspace plan is missing or has already been used.",
                status_code=409,
            )

        workspace_id = uuid4().hex
        operation_id = await self.state_store.begin_operation(
            operation="runpod.workspace.create",
            workspace_id=workspace_id,
            context={"plan_id": plan.id, "mode": plan.request.mode.value},
        )
        secret_id = f"agent:{workspace_id}"
        agent_secret = secrets.token_urlsafe(32)
        await self._vault.store(secret_id, agent_secret)
        api = await self._api()
        network_volume = plan.selected_network_volume
        try:
            if plan.create_network_volume:
                assert plan.selected_datacenter_id is not None
                network_volume_api = await api.create_network_volume(
                    name=f"k2lab-{workspace_id[:8]}-{request.name.strip()}"[:191],
                    size_gb=plan.request.workspace_disk_gb,
                    datacenter_id=plan.selected_datacenter_id,
                )
                network_volume = NetworkVolumeOption(
                    id=network_volume_api.id,
                    name=network_volume_api.name,
                    size_gb=network_volume_api.size_gb,
                    datacenter_id=network_volume_api.datacenter_id,
                )
                await self.state_store.update_operation(
                    operation_id,
                    state="volume_created",
                    context={"network_volume_id": network_volume.id},
                )
            payload = self._create_payload(
                workspace_id,
                request.name.strip(),
                plan,
                agent_secret,
                network_volume_id=network_volume.id if network_volume else None,
            )
            provider = await api.create_pod(payload)
            provider_id = self._required_string(provider, "id")
        except Exception:
            await self._vault.delete(secret_id)
            await self.state_store.update_operation(operation_id, state="failed")
            await self.state_store.append_audit(
                action="runpod.workspace.create",
                result="failure",
                workspace_id=workspace_id,
                context={
                    "plan_id": plan.id,
                    "retained_network_volume_id": (
                        network_volume.id if plan.create_network_volume and network_volume else None
                    ),
                },
            )
            raise
        await self.state_store.update_operation(
            operation_id,
            state="provider_created",
            context={"provider_resource_id": provider_id},
        )

        now = utc_now()
        provider_status = str(provider.get("desiredStatus", "RUNNING"))
        workspace = WorkspaceRecord(
            id=workspace_id,
            name=request.name.strip(),
            mode=plan.request.mode,
            state=self._state_from_provider(provider_status),
            gpu=plan.selected_gpu,
            cloud_type=plan.request.cloud_type,
            interruptible=plan.request.interruptible,
            container_disk_gb=plan.request.container_disk_gb,
            workspace_disk_gb=plan.request.workspace_disk_gb,
            estimated_compute_per_hour=float(
                provider.get("adjustedCostPerHr")
                or provider.get("costPerHr")
                or plan.estimated_compute_per_hour
            ),
            estimated_storage_per_month=plan.estimated_storage_per_month,
            idle_timeout_seconds=plan.request.idle_timeout_seconds,
            hard_deadline_seconds=plan.request.hard_deadline_seconds,
            lease_expires_at=now + timedelta(seconds=plan.request.idle_timeout_seconds),
            hard_expires_at=now + timedelta(seconds=plan.request.hard_deadline_seconds),
            created_at=now,
            updated_at=now,
            provider_resource_id=provider_id,
            readiness=self._readiness(provider_status),
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
        await self.state_store.save_workspace(workspace, image_digest=self._image_digest)
        await self.state_store.update_operation(operation_id, state="completed")
        await self.state_store.append_audit(
            action="runpod.workspace.create",
            result="success",
            workspace_id=workspace.id,
            context={"pod_id": provider_id, "gpu_type_id": workspace.gpu.id},
        )
        return workspace.model_copy(deep=True)

    async def list_workspaces(self) -> list[WorkspaceRecord]:
        return await self.state_store.list_workspaces()

    async def get_workspace_status(self, workspace_id: str) -> WorkspaceRecord:
        workspace = await self._workspace(workspace_id)
        if workspace.state == WorkspaceState.DELETED or (
            workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE
            and workspace.state == WorkspaceState.STOPPED
        ):
            return workspace.model_copy(deep=True)
        provider_id = self._provider_id(workspace)
        provider = await (await self._api()).get_pod(provider_id)
        updated = await self._workspace_from_provider(workspace, provider)
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        return updated.model_copy(deep=True)

    async def start_workspace(self, workspace_id: str) -> WorkspaceRecord:
        await self._ensure_no_active_migration(workspace_id)
        workspace = await self.state_store.claim_workspace_transition(
            workspace_id,
            allowed_states={WorkspaceState.STOPPED, WorkspaceState.ERROR},
            claimed_state=WorkspaceState.STARTING,
        )
        agent_secret: str | None = None
        try:
            if workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE:
                agent_secret = secrets.token_urlsafe(32)
                await self._vault.store(f"agent:{workspace.id}", agent_secret)
                provider = await (await self._api()).create_pod(
                    self._portable_create_payload(workspace, agent_secret)
                )
            else:
                provider = await (await self._api()).start_pod(self._provider_id(workspace))
        except Exception as error:
            if agent_secret is not None:
                await self._vault.delete(f"agent:{workspace.id}")
            await self._record_provider_failure(workspace, "runpod.workspace.start", error)
            raise
        now = utc_now()
        status = str(provider.get("desiredStatus", "RUNNING"))
        updated = workspace.model_copy(
            update={
                "state": self._state_from_provider(status),
                "updated_at": now,
                "lease_expires_at": now + timedelta(seconds=workspace.idle_timeout_seconds),
                "hard_expires_at": now + timedelta(seconds=workspace.hard_deadline_seconds),
                "readiness": self._readiness(status),
                "error_code": None,
                "error_message": None,
                "provider_resource_id": self._required_string(provider, "id"),
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action="runpod.workspace.start", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def stop_workspace(self, workspace_id: str) -> WorkspaceRecord:
        await self._abort_active_migrations(workspace_id)
        workspace = await self.state_store.claim_workspace_transition(
            workspace_id,
            allowed_states={
                WorkspaceState.PROVISIONING,
                WorkspaceState.STARTING,
                WorkspaceState.READY,
                WorkspaceState.ERROR,
            },
            claimed_state=WorkspaceState.STOPPING,
        )
        try:
            if workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE:
                await (await self._api()).delete_pod(self._provider_id(workspace))
                await self._vault.delete(f"agent:{workspace.id}")
            else:
                await (await self._api()).stop_pod(self._provider_id(workspace))
        except Exception as error:
            await self._record_provider_failure(workspace, "runpod.workspace.stop", error)
            raise
        updated = workspace.model_copy(
            update={
                "state": WorkspaceState.STOPPED,
                "updated_at": utc_now(),
                "readiness": {},
                "error_code": None,
                "error_message": None,
                "provider_resource_id": (
                    None
                    if workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE
                    else workspace.provider_resource_id
                ),
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action="runpod.workspace.stop", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def terminate_workspace(self, workspace_id: str, confirmation: str) -> WorkspaceRecord:
        await self._ensure_no_active_migration(workspace_id)
        workspace = await self._workspace(workspace_id)
        if workspace.retained_original_provider_resource_id:
            raise WorkspaceError(
                "migration_confirmation_required",
                "Confirm the migrated workspace before deleting cloud resources.",
                status_code=409,
            )
        if workspace.state == WorkspaceState.DELETED:
            return workspace.model_copy(deep=True)
        if confirmation != workspace.name:
            raise WorkspaceError(
                "workspace_delete_confirmation_mismatch",
                "Type the workspace name exactly to confirm permanent deletion.",
                status_code=409,
            )
        workspace = await self.state_store.claim_workspace_transition(
            workspace_id,
            allowed_states={
                WorkspaceState.PROVISIONING,
                WorkspaceState.STARTING,
                WorkspaceState.READY,
                WorkspaceState.STOPPING,
                WorkspaceState.STOPPED,
                WorkspaceState.ERROR,
            },
            claimed_state=WorkspaceState.DELETING,
        )
        operation_id = await self.state_store.begin_operation(
            operation="runpod.workspace.delete",
            workspace_id=workspace_id,
            context={"provider_resource_id": workspace.provider_resource_id},
        )
        try:
            if workspace.provider_resource_id:
                await (await self._api()).delete_pod(self._provider_id(workspace))
        except Exception as error:
            await self.state_store.update_operation(operation_id, state="failed")
            await self._record_provider_failure(workspace, "runpod.workspace.delete", error)
            raise
        await self.state_store.update_operation(operation_id, state="provider_deleted")
        updated = workspace.model_copy(
            update={
                "state": WorkspaceState.DELETED,
                "updated_at": utc_now(),
                "provider_resource_id": None,
                "readiness": {},
                "error_code": None,
                "error_message": None,
            }
        )
        await self._vault.delete(f"agent:{workspace_id}")
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.update_operation(operation_id, state="completed")
        await self.state_store.append_audit(
            action="runpod.workspace.delete", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def extend_lease(self, workspace_id: str) -> WorkspaceRecord:
        workspace = await self._workspace(workspace_id)
        if workspace.state not in {WorkspaceState.STARTING, WorkspaceState.READY}:
            raise WorkspaceError(
                "workspace_not_running",
                "Only a running workspace has an active compute lease.",
                status_code=409,
            )
        now = utc_now()
        updated = workspace.model_copy(
            update={
                "lease_expires_at": min(
                    now + timedelta(seconds=workspace.idle_timeout_seconds),
                    workspace.hard_expires_at,
                ),
                "updated_at": now,
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action="workspace.lease.extend", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def get_cost_snapshot(self, workspace_id: str) -> CostSnapshot:
        workspace = await self._workspace(workspace_id)
        running = workspace.state in {
            WorkspaceState.PROVISIONING,
            WorkspaceState.STARTING,
            WorkspaceState.READY,
            WorkspaceState.STOPPING,
        }
        elapsed = max(0.0, (utc_now() - workspace.created_at).total_seconds())
        storage_billable = (
            workspace.network_volume_id is not None or workspace.state != WorkspaceState.DELETED
        )
        compute_per_hour = workspace.estimated_compute_per_hour if running else 0.0
        storage_per_month = workspace.estimated_storage_per_month if storage_billable else 0.0
        accrued_compute = elapsed / 3600 * workspace.estimated_compute_per_hour if running else 0.0
        migrations = await self.state_store.list_migrations(workspace_id)
        migration = migrations[-1] if migrations else None
        if migration is not None:
            target_storage = round(
                migration.target_workspace_disk_gb
                * (
                    self.NETWORK_VOLUME_PRICE_OVER_1TB
                    if migration.target_workspace_disk_gb > 1_000
                    else self.NETWORK_VOLUME_PRICE_UNDER_1TB
                ),
                2,
            )
            if migration.state in {
                MigrationState.PREPARING,
                MigrationState.COPYING,
                MigrationState.VERIFYING,
            }:
                compute_per_hour += migration.target_compute_per_hour
                storage_per_month += target_storage
                migration_elapsed = max(0.0, (utc_now() - migration.created_at).total_seconds())
                accrued_compute += migration_elapsed / 3600 * migration.target_compute_per_hour
            elif migration.state == MigrationState.AWAITING_CONFIRMATION:
                storage_per_month += migration.source_storage_per_month
            elif migration.state == MigrationState.FAILED and migration.target_network_volume_id:
                storage_per_month += target_storage
        return CostSnapshot(
            workspace_id=workspace.id,
            state=workspace.state,
            compute_per_hour=compute_per_hour,
            storage_per_month=storage_per_month,
            accrued_compute_estimate=accrued_compute,
            observed_at=utc_now(),
        )

    async def create_workspace_migration(
        self, workspace_id: str, request: WorkspaceMigrationCreateRequest
    ) -> WorkspaceMigrationRecord:
        workspace = await self._workspace(workspace_id)
        if workspace.mode != WorkspaceMode.PERSISTENT_POD:
            raise WorkspaceError(
                "migration_source_invalid",
                "Only a persistent-Pod workspace can be migrated.",
                status_code=409,
            )
        if workspace.state != WorkspaceState.READY:
            raise WorkspaceError(
                "migration_source_not_ready",
                "Start the persistent workspace and wait for readiness before migrating.",
                status_code=409,
            )
        await self._ensure_no_active_migration(workspace_id)
        source_pod_id = self._provider_id(workspace)
        source_agent = await self._workspace_agent(workspace_id)
        source_manifest = await source_agent.seal_for_migration()
        required_gb = max(50, math.ceil(source_manifest.total_bytes * 1.10 / (1024**3)))
        target_size = max(required_gb, request.workspace_disk_gb or workspace.workspace_disk_gb)
        try:
            plan = await self.plan_workspace(
                WorkspacePlanRequest(
                    mode=WorkspaceMode.PORTABLE_WORKSPACE,
                    gpu_priority_ids=workspace.gpu_priority_ids or [workspace.gpu.id],
                    cloud_type=CloudType.SECURE,
                    interruptible=workspace.interruptible,
                    container_disk_gb=workspace.container_disk_gb,
                    workspace_disk_gb=target_size,
                    idle_timeout_seconds=workspace.idle_timeout_seconds,
                    hard_deadline_seconds=workspace.hard_deadline_seconds,
                    network_volume_id=request.network_volume_id,
                    datacenter_priority_ids=request.datacenter_priority_ids,
                )
            )
            await self.state_store.consume_plan(plan.id)
        except Exception:
            await source_agent.unseal_after_migration()
            raise

        now = utc_now()
        migration = WorkspaceMigrationRecord(
            id=uuid4().hex,
            workspace_id=workspace.id,
            state=MigrationState.PREPARING,
            source_provider_resource_id=source_pod_id,
            target_network_volume_id=(
                plan.selected_network_volume.id if plan.selected_network_volume else None
            ),
            target_datacenter_id=plan.selected_datacenter_id,
            target_gpu=plan.selected_gpu,
            target_compute_per_hour=plan.estimated_compute_per_hour,
            source_storage_per_month=workspace.estimated_storage_per_month,
            target_workspace_disk_gb=plan.request.workspace_disk_gb,
            owns_target_volume=plan.create_network_volume,
            source_manifest=source_manifest,
            bytes_total=source_manifest.total_bytes,
            created_at=now,
            updated_at=now,
        )
        await self.state_store.save_migration(migration)
        operation_id = await self.state_store.begin_operation(
            operation="runpod.workspace.migrate",
            workspace_id=workspace.id,
            context={"migration_id": migration.id, "source_pod_id": source_pod_id},
        )
        migration = migration.model_copy(
            update={"operation_id": operation_id, "updated_at": utc_now()}
        )
        await self.state_store.save_migration(migration)
        target_secret_id = f"migration-agent:{migration.id}"
        target_secret = secrets.token_urlsafe(32)
        await self._vault.store(target_secret_id, target_secret)
        network_volume = plan.selected_network_volume
        target_pod_id: str | None = None
        try:
            api = await self._api()
            if plan.create_network_volume:
                assert plan.selected_datacenter_id is not None
                created_volume = await api.create_network_volume(
                    name=f"k2lab-migration-{migration.id[:8]}"[:191],
                    size_gb=plan.request.workspace_disk_gb,
                    datacenter_id=plan.selected_datacenter_id,
                )
                network_volume = NetworkVolumeOption(
                    id=created_volume.id,
                    name=created_volume.name,
                    size_gb=created_volume.size_gb,
                    datacenter_id=created_volume.datacenter_id,
                )
                migration = migration.model_copy(
                    update={
                        "target_network_volume_id": network_volume.id,
                        "updated_at": utc_now(),
                    }
                )
                await self.state_store.save_migration(migration)
                await self.state_store.update_operation(
                    operation_id,
                    state="volume_created",
                    context={"network_volume_id": network_volume.id},
                )
            if network_volume is None:
                raise WorkspaceError(
                    "migration_target_invalid",
                    "The migration target network volume is missing.",
                    status_code=409,
                )
            provider = await api.create_pod(
                self._create_payload(
                    workspace.id,
                    f"migration-{workspace.name}",
                    plan,
                    target_secret,
                    network_volume_id=network_volume.id,
                )
            )
            target_pod_id = self._required_string(provider, "id")
            await self.state_store.update_operation(
                operation_id,
                state="target_created",
                context={"target_pod_id": target_pod_id},
            )
            migration = migration.model_copy(
                update={
                    "target_provider_resource_id": target_pod_id,
                    "target_network_volume_id": network_volume.id,
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_migration(migration)
            extended = workspace.model_copy(
                update={
                    "lease_expires_at": workspace.hard_expires_at,
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_workspace(extended, image_digest=self._image_digest)
        except Exception as error:
            if target_pod_id:
                try:
                    await (await self._api()).delete_pod(target_pod_id)
                except Exception:
                    pass
            await self._vault.delete(target_secret_id)
            await source_agent.unseal_after_migration()
            migration = migration.model_copy(
                update={
                    "state": MigrationState.FAILED,
                    "error_code": getattr(error, "code", "migration_target_failed"),
                    "error_message": "The migration target could not be provisioned.",
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_migration(migration)
            await self.state_store.update_operation(operation_id, state="failed")
            await self.state_store.append_audit(
                action="runpod.workspace.migrate",
                result="failure",
                workspace_id=workspace.id,
                context={
                    "migration_id": migration.id,
                    "retained_network_volume_id": migration.target_network_volume_id,
                },
            )
            raise
        await self.state_store.append_audit(
            action="runpod.workspace.migrate",
            result="started",
            workspace_id=workspace.id,
            context={"migration_id": migration.id},
        )
        return migration.model_copy(deep=True)

    async def get_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        return (await self._migration(workspace_id, migration_id)).model_copy(deep=True)

    async def list_workspace_migrations(self, workspace_id: str) -> list[WorkspaceMigrationRecord]:
        await self._workspace(workspace_id)
        return [
            item.model_copy(deep=True)
            for item in await self.state_store.list_migrations(workspace_id)
        ]

    async def resume_workspace_migration(
        self, workspace_id: str, migration_id: str
    ) -> WorkspaceMigrationRecord:
        migration = await self._migration(workspace_id, migration_id)
        if migration.state in {
            MigrationState.AWAITING_CONFIRMATION,
            MigrationState.COMPLETED,
            MigrationState.FAILED,
        }:
            return migration.model_copy(deep=True)
        workspace = await self._workspace(workspace_id)
        source_manifest = migration.source_manifest
        if source_manifest is None or not migration.target_provider_resource_id:
            raise WorkspaceError(
                "migration_state_invalid",
                "The durable migration record is incomplete.",
                status_code=409,
            )
        source_secret = await self._vault.retrieve(f"agent:{workspace_id}")
        target_secret = await self._vault.retrieve(f"migration-agent:{migration.id}")
        if not source_secret or not target_secret:
            raise WorkspaceError(
                "migration_credential_missing",
                "A migration agent credential is missing.",
                status_code=500,
            )
        source_agent = self._agent_factory(migration.source_provider_resource_id, source_secret)
        target_agent = self._agent_factory(migration.target_provider_resource_id, target_secret)
        if migration.state == MigrationState.PREPARING:
            health = await target_agent.health()
            if health.workspace_id != workspace_id:
                raise WorkspaceError(
                    "agent_identity_mismatch",
                    "The migration target reported a different workspace identity.",
                    status_code=502,
                )
            await target_agent.seal_for_migration()
            migration = migration.model_copy(
                update={"state": MigrationState.COPYING, "updated_at": utc_now()}
            )
            await self.state_store.save_migration(migration)

        if migration.state == MigrationState.COPYING:
            copied_this_call = 0
            while migration.current_file_index < len(source_manifest.files):
                entry = source_manifest.files[migration.current_file_index]
                offset = migration.current_file_offset
                remaining = entry.size_bytes - offset
                amount = min(self.MIGRATION_CHUNK_SIZE, remaining)
                content = (
                    await source_agent.migration_file(
                        source_manifest.generation,
                        entry.path,
                        start=offset,
                        end=offset + amount - 1,
                    )
                    if amount
                    else b""
                )
                receipt = await target_agent.import_migration_chunk(
                    migration.id,
                    entry.path,
                    offset=offset,
                    total_size=entry.size_bytes,
                    file_sha256=entry.sha256,
                    content=content,
                )
                delta = max(0, receipt.next_offset - offset)
                copied_this_call += delta
                if receipt.completed:
                    file_index = migration.current_file_index + 1
                    file_offset = 0
                else:
                    file_index = migration.current_file_index
                    file_offset = receipt.next_offset
                migration = migration.model_copy(
                    update={
                        "current_file_index": file_index,
                        "current_file_offset": file_offset,
                        "bytes_copied": min(migration.bytes_total, migration.bytes_copied + delta),
                        "updated_at": utc_now(),
                    }
                )
                await self.state_store.save_migration(migration)
                if copied_this_call >= self.MIGRATION_COPY_BUDGET:
                    return migration.model_copy(deep=True)
            migration = migration.model_copy(
                update={"state": MigrationState.VERIFYING, "updated_at": utc_now()}
            )
            await self.state_store.save_migration(migration)

        if migration.state == MigrationState.VERIFYING:
            target_manifest = (
                migration.target_manifest or await target_agent.create_migration_manifest()
            )
            migration = migration.model_copy(
                update={"target_manifest": target_manifest, "updated_at": utc_now()}
            )
            await self.state_store.save_migration(migration)
            if not self._manifests_match(source_manifest, target_manifest):
                await (await self._api()).delete_pod(migration.target_provider_resource_id)
                await self._vault.delete(f"migration-agent:{migration.id}")
                await source_agent.unseal_after_migration()
                failed = migration.model_copy(
                    update={
                        "state": MigrationState.FAILED,
                        "error_code": "migration_manifest_mismatch",
                        "error_message": (
                            "The target manifest does not match the source; "
                            "the original Pod was retained."
                        ),
                        "updated_at": utc_now(),
                    }
                )
                await self.state_store.save_migration(failed)
                if migration.operation_id:
                    await self.state_store.update_operation(migration.operation_id, state="failed")
                await self.state_store.append_audit(
                    action="runpod.workspace.migrate.verify",
                    result="failure",
                    workspace_id=workspace_id,
                    context={"migration_id": migration.id},
                )
                return failed.model_copy(deep=True)

            await (await self._api()).stop_pod(migration.source_provider_resource_id)
            await target_agent.unseal_after_migration()
            target_health = await target_agent.health()
            switched = workspace.model_copy(
                update={
                    "mode": WorkspaceMode.PORTABLE_WORKSPACE,
                    "state": (
                        WorkspaceState.READY
                        if target_health.status == "ready"
                        else WorkspaceState.STARTING
                    ),
                    "gpu": migration.target_gpu or workspace.gpu,
                    "estimated_compute_per_hour": migration.target_compute_per_hour,
                    "cloud_type": CloudType.SECURE,
                    "workspace_disk_gb": migration.target_workspace_disk_gb,
                    "estimated_storage_per_month": round(
                        migration.target_workspace_disk_gb
                        * (
                            self.NETWORK_VOLUME_PRICE_OVER_1TB
                            if migration.target_workspace_disk_gb > 1_000
                            else self.NETWORK_VOLUME_PRICE_UNDER_1TB
                        ),
                        2,
                    ),
                    "provider_resource_id": migration.target_provider_resource_id,
                    "network_volume_id": migration.target_network_volume_id,
                    "datacenter_id": migration.target_datacenter_id,
                    "owns_network_volume": migration.owns_target_volume,
                    "storage_tier": StorageTier.NETWORK_VOLUME,
                    "workspace_layout_version": source_manifest.layout_version,
                    "retained_original_provider_resource_id": (
                        migration.source_provider_resource_id
                    ),
                    "readiness": target_health.readiness.model_dump(),
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_workspace(switched, image_digest=self._image_digest)
            await self._vault.store(f"agent:{workspace_id}", target_secret)
            await self._vault.delete(f"migration-agent:{migration.id}")
            migration = migration.model_copy(
                update={
                    "state": MigrationState.AWAITING_CONFIRMATION,
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_migration(migration)
            if migration.operation_id:
                await self.state_store.update_operation(migration.operation_id, state="verified")
            await self.state_store.append_audit(
                action="runpod.workspace.migrate.verify",
                result="success",
                workspace_id=workspace_id,
                context={
                    "migration_id": migration.id,
                    "source_root_sha256": source_manifest.root_sha256,
                },
            )
        return migration.model_copy(deep=True)

    async def confirm_workspace_migration(
        self, workspace_id: str, migration_id: str, confirmation: str
    ) -> WorkspaceMigrationRecord:
        migration = await self._migration(workspace_id, migration_id)
        if migration.state == MigrationState.COMPLETED:
            return migration.model_copy(deep=True)
        if migration.state != MigrationState.AWAITING_CONFIRMATION:
            raise WorkspaceError(
                "migration_not_verified",
                "The migration must be verified before deleting the original Pod.",
                status_code=409,
            )
        workspace = await self._workspace(workspace_id)
        if confirmation != workspace.name:
            raise WorkspaceError(
                "migration_confirmation_mismatch",
                "Type the workspace name exactly to delete the original Pod volume.",
                status_code=409,
            )
        retained = workspace.retained_original_provider_resource_id
        if not retained or retained != migration.source_provider_resource_id:
            raise WorkspaceError(
                "migration_original_missing",
                "The retained original Pod record is missing.",
                status_code=409,
            )
        await (await self._api()).delete_pod(retained)
        updated = workspace.model_copy(
            update={
                "retained_original_provider_resource_id": None,
                "updated_at": utc_now(),
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        completed = migration.model_copy(
            update={"state": MigrationState.COMPLETED, "updated_at": utc_now()}
        )
        await self.state_store.save_migration(completed)
        if migration.operation_id:
            await self.state_store.update_operation(migration.operation_id, state="completed")
        await self.state_store.append_audit(
            action="runpod.workspace.migrate.confirm",
            result="success",
            workspace_id=workspace_id,
            context={"migration_id": migration.id},
        )
        return completed.model_copy(deep=True)

    async def reconcile_workspaces(self) -> list[WorkspaceRecord]:
        """Refresh every durable provider resource after control-plane startup."""
        if not (await self.credential_status()).configured:
            return []
        reconciled: list[WorkspaceRecord] = []
        api = await self._api()
        await self._reconcile_operation_journal(api)
        for workspace in await self.state_store.list_workspaces():
            if workspace.state == WorkspaceState.DELETED or not workspace.provider_resource_id:
                continue
            try:
                provider = await api.get_pod(workspace.provider_resource_id)
                status = str(provider.get("desiredStatus", ""))
                updated = await self._workspace_from_provider(workspace, provider)
                state = updated.state
                if (
                    state == WorkspaceState.DELETED
                    and workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE
                ):
                    updated = updated.model_copy(
                        update={
                            "state": WorkspaceState.STOPPED,
                            "provider_resource_id": None,
                            "readiness": {},
                        }
                    )
                    await self._vault.delete(f"agent:{workspace.id}")
                elif state == WorkspaceState.DELETED:
                    updated = updated.model_copy(update={"provider_resource_id": None})
                    await self._vault.delete(f"agent:{workspace.id}")
                await self.state_store.save_workspace(updated, image_digest=self._image_digest)
                await self.state_store.append_audit(
                    action="runpod.workspace.reconcile",
                    result="success",
                    workspace_id=workspace.id,
                    context={"provider_state": status},
                )
                reconciled.append(updated)
            except Exception as error:
                await self._record_provider_failure(workspace, "runpod.workspace.reconcile", error)
        return reconciled

    async def _reconcile_operation_journal(self, api: RunPodApi) -> None:
        for operation in await self.state_store.incomplete_operations():
            operation_id = str(operation["id"])
            workspace_id = operation.get("workspace_id")
            context = operation.get("context", {})
            provider_id = context.get("provider_resource_id")
            workspace = (
                await self.state_store.get_workspace(str(workspace_id)) if workspace_id else None
            )
            if operation["operation"] == "runpod.workspace.create":
                if workspace is not None:
                    await self.state_store.update_operation(operation_id, state="completed")
                    continue
                if isinstance(provider_id, str) and provider_id:
                    portable = context.get("mode") == WorkspaceMode.PORTABLE_WORKSPACE.value
                    if portable:
                        await api.delete_pod(provider_id)
                    else:
                        await api.stop_pod(provider_id)
                    await self.state_store.append_audit(
                        action=(
                            "runpod.workspace.orphan_delete"
                            if portable
                            else "runpod.workspace.orphan_stop"
                        ),
                        result="success",
                        workspace_id=str(workspace_id) if workspace_id else None,
                        context={"provider_resource_id": provider_id},
                    )
                if workspace_id:
                    await self._vault.delete(f"agent:{workspace_id}")
                await self.state_store.update_operation(
                    operation_id,
                    state="compensated" if provider_id else "failed",
                )
            elif (
                operation["operation"] == "runpod.workspace.delete"
                and operation["state"] == "provider_deleted"
                and workspace is not None
            ):
                deleted = workspace.model_copy(
                    update={
                        "state": WorkspaceState.DELETED,
                        "provider_resource_id": None,
                        "readiness": {},
                        "updated_at": utc_now(),
                    }
                )
                await self._vault.delete(f"agent:{workspace.id}")
                await self.state_store.save_workspace(deleted, image_digest=self._image_digest)
                await self.state_store.update_operation(operation_id, state="completed")

    async def get_file_inventory(
        self, workspace_id: str, kind: FileKind, cursor: str | None = None
    ) -> FilePage:
        return await (await self._workspace_agent(workspace_id)).inventory(kind, cursor=cursor)

    async def save_project(
        self, workspace_id: str, filename: str, request: ProjectSaveRequest
    ) -> FileRecord:
        return await (await self._workspace_agent(workspace_id)).save_project(filename, request)

    async def create_upload(self, workspace_id: str, request: UploadCreateRequest) -> UploadSession:
        return await (await self._workspace_agent(workspace_id)).create_upload(request)

    async def get_upload(self, workspace_id: str, upload_id: str) -> UploadSession:
        return await (await self._workspace_agent(workspace_id)).upload_status(upload_id)

    async def write_upload_chunk(
        self,
        workspace_id: str,
        upload_id: str,
        index: int,
        content: bytes,
        sha256: str,
    ) -> ChunkReceipt:
        return await (await self._workspace_agent(workspace_id)).write_chunk(
            upload_id, index, content, sha256
        )

    async def complete_upload(self, workspace_id: str, upload_id: str) -> UploadCompleteResponse:
        return await (await self._workspace_agent(workspace_id)).complete_upload(upload_id)

    async def cancel_upload(self, workspace_id: str, upload_id: str) -> None:
        await (await self._workspace_agent(workspace_id)).cancel_upload(upload_id)

    async def download_credential_status(self, provider: RemoteProvider) -> CredentialStatus:
        return await self._vault.status(self._download_credential_id(provider))

    async def store_download_credential(
        self, provider: RemoteProvider, token: str
    ) -> CredentialStatus:
        value = token.strip()
        if len(value) < 8:
            raise WorkspaceError(
                "invalid_provider_token",
                "Enter a complete read-only provider token.",
                status_code=400,
            )
        await self._vault.store(
            self._download_credential_id(provider),
            value,
            key_hint=f"••••{value[-4:]}",
        )
        await self.state_store.append_audit(
            action="download_credential.store",
            result="success",
            context={"provider": provider.value},
        )
        return await self.download_credential_status(provider)

    async def clear_download_credential(self, provider: RemoteProvider) -> CredentialStatus:
        await self._vault.delete(self._download_credential_id(provider))
        await self.state_store.append_audit(
            action="download_credential.delete",
            result="success",
            context={"provider": provider.value},
        )
        return await self.download_credential_status(provider)

    async def preview_civitai_download(
        self, workspace_id: str, request: CivitaiPreviewRequest
    ) -> CivitaiPreview:
        token = await self._download_token(RemoteProvider.CIVITAI)
        return await (await self._workspace_agent(workspace_id)).preview_civitai(request, token)

    async def start_civitai_download(
        self, workspace_id: str, request: CivitaiDownloadRequest
    ) -> RemoteTransfer:
        token = await self._download_token(RemoteProvider.CIVITAI)
        transfer = await (await self._workspace_agent(workspace_id)).start_civitai(request, token)
        await self.state_store.save_transfer(workspace_id, transfer)
        await self._touch_workspace_lease(workspace_id)
        await self.state_store.append_audit(
            action="download.civitai.start",
            result="success",
            workspace_id=workspace_id,
            context={"transfer_id": transfer.id, "destination": request.destination_kind.value},
        )
        return transfer

    async def preview_huggingface_download(
        self, workspace_id: str, request: HuggingFacePreviewRequest
    ) -> HuggingFacePreview:
        token = await self._download_token(RemoteProvider.HUGGINGFACE)
        return await (await self._workspace_agent(workspace_id)).preview_huggingface(request, token)

    async def start_huggingface_download(
        self, workspace_id: str, request: HuggingFaceDownloadRequest
    ) -> RemoteTransfer:
        token = await self._download_token(RemoteProvider.HUGGINGFACE)
        transfer = await (await self._workspace_agent(workspace_id)).start_huggingface(
            request, token
        )
        await self.state_store.save_transfer(workspace_id, transfer)
        await self._touch_workspace_lease(workspace_id)
        await self.state_store.append_audit(
            action="download.huggingface.start",
            result="success",
            workspace_id=workspace_id,
            context={"transfer_id": transfer.id, "destination": request.destination_kind.value},
        )
        return transfer

    async def get_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer:
        transfer = await (await self._workspace_agent(workspace_id)).transfer_status(transfer_id)
        await self.state_store.save_transfer(workspace_id, transfer)
        if transfer.state.value in {"pending", "resolving", "downloading", "verifying"}:
            await self._touch_workspace_lease(workspace_id)
        return transfer

    async def cancel_transfer(self, workspace_id: str, transfer_id: str) -> RemoteTransfer:
        transfer = await (await self._workspace_agent(workspace_id)).cancel_transfer(transfer_id)
        await self.state_store.save_transfer(workspace_id, transfer)
        await self.state_store.append_audit(
            action="download.cancel",
            result="success",
            workspace_id=workspace_id,
            context={"transfer_id": transfer.id},
        )
        return transfer

    async def _download_token(self, provider: RemoteProvider) -> str | None:
        return await self._vault.retrieve(self._download_credential_id(provider))

    @staticmethod
    def _download_credential_id(provider: RemoteProvider) -> str:
        return f"provider:{provider.value}"

    async def _touch_workspace_lease(self, workspace_id: str) -> None:
        workspace = await self._workspace(workspace_id)
        if workspace.state not in {WorkspaceState.STARTING, WorkspaceState.READY}:
            return
        now = utc_now()
        updated = workspace.model_copy(
            update={
                "lease_expires_at": min(
                    now + timedelta(seconds=workspace.idle_timeout_seconds),
                    workspace.hard_expires_at,
                ),
                "updated_at": now,
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)

    async def submit_job(self, workspace_id: str, request: JobSubmitRequest) -> GenerationJob:
        job = await (await self._workspace_agent(workspace_id)).submit_job(request)
        await self.state_store.save_generation_job(workspace_id, job)
        await self._touch_workspace_lease(workspace_id)
        await self.state_store.append_audit(
            action="generation_job.submit",
            result="success",
            workspace_id=workspace_id,
            context={"job_id": job.id, "kind": job.kind.value},
        )
        return job

    async def get_job(self, workspace_id: str, job_id: str) -> GenerationJob:
        job = await (await self._workspace_agent(workspace_id)).job_status(job_id)
        await self.state_store.save_generation_job(workspace_id, job)
        if job.state in {JobState.QUEUED, JobState.STARTING, JobState.RUNNING}:
            await self._touch_workspace_lease(workspace_id)
        return job

    async def get_job_events(
        self, workspace_id: str, job_id: str, cursor: str | None = None
    ) -> JobEventPage:
        events = await (await self._workspace_agent(workspace_id)).job_events(job_id, cursor=cursor)
        await self.state_store.save_job_events(job_id, events.items)
        return events

    async def cancel_job(self, workspace_id: str, job_id: str) -> GenerationJob:
        job = await (await self._workspace_agent(workspace_id)).cancel_job(job_id)
        await self.state_store.save_generation_job(workspace_id, job)
        await self.state_store.append_audit(
            action="generation_job.cancel",
            result="success",
            workspace_id=workspace_id,
            context={"job_id": job.id},
        )
        return job

    async def detect_faces(
        self, workspace_id: str, request: FaceDetectionRequest
    ) -> FaceDetectionResult:
        result = await (await self._workspace_agent(workspace_id)).detect_faces(request)
        await self._touch_workspace_lease(workspace_id)
        return result

    async def release_worker_memory(self, workspace_id: str) -> WorkerReleaseResult:
        return await (await self._workspace_agent(workspace_id)).release_worker_memory()

    async def get_output(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput:
        return await (await self._workspace_agent(workspace_id)).output(
            file_id, range_header=range_header
        )

    async def get_file_content(
        self, workspace_id: str, file_id: str, range_header: str | None = None
    ) -> WorkspaceOutput:
        return await (await self._workspace_agent(workspace_id)).file_content(
            file_id, range_header=range_header
        )

    async def _workspace_agent(self, workspace_id: str) -> WorkspaceAgentApi:
        workspace = await self._workspace(workspace_id)
        if workspace.state not in {WorkspaceState.STARTING, WorkspaceState.READY}:
            raise WorkspaceError(
                "workspace_not_running",
                "Start the workspace before accessing its files.",
                status_code=409,
            )
        secret = await self._vault.retrieve(f"agent:{workspace.id}")
        if not secret:
            raise WorkspaceError(
                "agent_credential_missing",
                "The workspace agent credential is missing.",
                status_code=500,
            )
        return self._agent_factory(self._provider_id(workspace), secret)

    async def _workspace_from_provider(
        self, workspace: WorkspaceRecord, provider: dict[str, Any]
    ) -> WorkspaceRecord:
        status = str(provider.get("desiredStatus", ""))
        state = self._state_from_provider(status)
        readiness = self._readiness(status)
        provider_resource_id = workspace.provider_resource_id
        error_code = None
        error_message = None
        if state == WorkspaceState.DELETED and workspace.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            state = WorkspaceState.STOPPED
            readiness = {}
            provider_resource_id = None
            await self._vault.delete(f"agent:{workspace.id}")
        if status == "RUNNING":
            secret = await self._vault.retrieve(f"agent:{workspace.id}")
            if not secret:
                state = WorkspaceState.ERROR
                error_code = "agent_credential_missing"
                error_message = "The workspace agent credential is missing."
            else:
                try:
                    health = await self._agent_factory(
                        self._provider_id(workspace), secret
                    ).health()
                    if health.workspace_id != workspace.id:
                        raise WorkspaceError(
                            "agent_identity_mismatch",
                            "The Pod agent reported a different workspace identity.",
                            status_code=502,
                        )
                    if health.image_version != self._image_version:
                        raise WorkspaceError(
                            "agent_image_mismatch",
                            "The Pod agent image version does not match the workspace.",
                            status_code=502,
                        )
                    readiness = health.readiness.model_dump()
                    state = (
                        WorkspaceState.READY
                        if health.status == "ready"
                        else WorkspaceState.STARTING
                    )
                except WorkspaceError as error:
                    state = WorkspaceState.STARTING
                    error_code = error.code
                    error_message = error.message
        return workspace.model_copy(
            update={
                "state": state,
                "updated_at": utc_now(),
                "readiness": readiness,
                "error_code": error_code,
                "error_message": error_message,
                "provider_resource_id": provider_resource_id,
            }
        )

    async def _api(self) -> RunPodApi:
        key = await self._vault.retrieve(self.PROVIDER_CREDENTIAL_ID)
        if not key:
            raise WorkspaceError(
                "credentials_required",
                "Connect a RunPod account before planning a workspace.",
                status_code=401,
            )
        return self._api_factory(key)

    @staticmethod
    def _gpu_option(item: RunPodGpuType) -> GpuOption:
        secure_price = item.secure_price.uninterruptible_price if item.secure_price else None
        community_price = (
            item.community_price.uninterruptible_price if item.community_price else None
        )
        advertised = [price for price in (secure_price, community_price) if price is not None]
        return GpuOption(
            id=item.id,
            display_name=item.display_name,
            memory_gb=item.memory_gb,
            secure_available=(
                item.secure_cloud
                and item.secure_price is not None
                and item.secure_price.one_gpu_available
            ),
            community_available=(
                item.community_cloud
                and item.community_price is not None
                and item.community_price.one_gpu_available
            ),
            on_demand_price_per_hour=min(advertised, default=0.0),
            secure_on_demand_price_per_hour=secure_price,
            community_on_demand_price_per_hour=community_price,
            available=(
                item.secure_cloud
                and item.secure_price is not None
                and item.secure_price.one_gpu_available
            )
            or (
                item.community_cloud
                and item.community_price is not None
                and item.community_price.one_gpu_available
            ),
        )

    @staticmethod
    def _cloud_available(gpu: GpuOption, cloud_type: CloudType) -> bool:
        return gpu.secure_available if cloud_type == CloudType.SECURE else gpu.community_available

    @staticmethod
    def _price(gpu: GpuOption, cloud_type: CloudType, interruptible: bool) -> float | None:
        if cloud_type == CloudType.SECURE:
            return (
                gpu.secure_interruptible_price_per_hour
                if interruptible
                else gpu.secure_on_demand_price_per_hour
            )
        return (
            gpu.community_interruptible_price_per_hour
            if interruptible
            else gpu.community_on_demand_price_per_hour
        )

    def _create_payload(
        self,
        workspace_id: str,
        name: str,
        plan: WorkspacePlan,
        agent_secret: str,
        *,
        network_volume_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": f"k2lab-{workspace_id[:8]}-{name}"[:191],
            "imageName": self._image_digest,
            "cloudType": plan.request.cloud_type.value.upper(),
            "computeType": "GPU",
            "gpuTypeIds": plan.provider_gpu_priority_ids,
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "containerDiskInGb": plan.request.container_disk_gb,
            "volumeMountPath": "/workspace",
            "interruptible": plan.request.interruptible,
            "locked": False,
            "ports": ["8080/http"],
            "env": {
                "K2LAB_AGENT_SESSION_TOKEN": agent_secret,
                "K2LAB_WORKSPACE_ID": workspace_id,
                "K2LAB_IMAGE_VERSION": self._image_version,
            },
        }
        if plan.request.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            if not network_volume_id or not plan.selected_datacenter_id:
                raise WorkspaceError(
                    "portable_plan_invalid",
                    "The portable workspace plan is missing its network volume.",
                    status_code=409,
                )
            payload.update(
                {
                    "networkVolumeId": network_volume_id,
                    "dataCenterIds": [plan.selected_datacenter_id],
                    "dataCenterPriority": "custom",
                }
            )
        else:
            payload["volumeInGb"] = plan.request.workspace_disk_gb
        return payload

    def _portable_create_payload(
        self, workspace: WorkspaceRecord, agent_secret: str
    ) -> dict[str, Any]:
        if not workspace.network_volume_id or not workspace.datacenter_id:
            raise WorkspaceError(
                "portable_workspace_invalid",
                "The portable workspace has no attached network volume.",
                status_code=409,
            )
        return {
            "name": f"k2lab-{workspace.id[:8]}-{workspace.name}"[:191],
            "imageName": self._image_digest,
            "cloudType": "SECURE",
            "computeType": "GPU",
            "gpuTypeIds": workspace.gpu_priority_ids or [workspace.gpu.id],
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "containerDiskInGb": workspace.container_disk_gb,
            "networkVolumeId": workspace.network_volume_id,
            "dataCenterIds": [workspace.datacenter_id],
            "dataCenterPriority": "custom",
            "volumeMountPath": "/workspace",
            "interruptible": workspace.interruptible,
            "locked": False,
            "ports": ["8080/http"],
            "env": {
                "K2LAB_AGENT_SESSION_TOKEN": agent_secret,
                "K2LAB_WORKSPACE_ID": workspace.id,
                "K2LAB_IMAGE_VERSION": self._image_version,
            },
        }

    def _portable_candidates(
        self,
        request: WorkspacePlanRequest,
        options: dict[str, GpuOption],
        datacenters: list[DatacenterOption],
        fixed_datacenter_id: str | None,
    ) -> list[tuple[GpuOption, str]]:
        by_id = {item.id: item for item in datacenters}
        datacenter_ids = (
            [fixed_datacenter_id]
            if fixed_datacenter_id
            else request.datacenter_priority_ids or sorted(by_id)
        )

        def available(datacenter_id: str, gpu_id: str) -> bool:
            datacenter = by_id.get(datacenter_id)
            if datacenter is None:
                return False
            return any(
                item.gpu_type_id == gpu_id
                and item.stock_status.casefold() not in {"", "none", "unavailable"}
                for item in datacenter.gpu_availability
            )

        selected_datacenter = fixed_datacenter_id
        if selected_datacenter is None:
            for gpu_id in request.gpu_priority_ids:
                if selected_datacenter is not None:
                    break
                for datacenter_id in datacenter_ids:
                    if available(datacenter_id, gpu_id):
                        selected_datacenter = datacenter_id
                        break
        if selected_datacenter is None:
            return []
        return [
            (options[gpu_id], selected_datacenter)
            for gpu_id in request.gpu_priority_ids
            if gpu_id in options
            and available(selected_datacenter, gpu_id)
            and options[gpu_id].secure_available
            and self._price(options[gpu_id], CloudType.SECURE, request.interruptible) is not None
        ]

    def _storage_price(self, request: WorkspacePlanRequest) -> float:
        if request.mode == WorkspaceMode.PORTABLE_WORKSPACE:
            rate = (
                self.NETWORK_VOLUME_PRICE_OVER_1TB
                if request.workspace_disk_gb > 1_000
                else self.NETWORK_VOLUME_PRICE_UNDER_1TB
            )
            return round(request.workspace_disk_gb * rate, 2)
        return round(request.workspace_disk_gb * self.STORAGE_PRICE_PER_GB_MONTH, 2)

    @staticmethod
    def _state_from_provider(status: str) -> WorkspaceState:
        if status == "RUNNING":
            return WorkspaceState.STARTING
        if status == "EXITED":
            return WorkspaceState.STOPPED
        if status == "TERMINATED":
            return WorkspaceState.DELETED
        return WorkspaceState.ERROR

    @staticmethod
    def _readiness(status: str) -> dict[str, bool]:
        container = status == "RUNNING"
        return {
            "container": container,
            "agent": False,
            "storage": False,
            "models": False,
            "worker": False,
        }

    async def _workspace(self, workspace_id: str) -> WorkspaceRecord:
        workspace = await self.state_store.get_workspace(workspace_id)
        if workspace is None:
            raise WorkspaceError(
                "workspace_not_found",
                "The requested workspace does not exist.",
                status_code=404,
            )
        return workspace

    async def _migration(self, workspace_id: str, migration_id: str) -> WorkspaceMigrationRecord:
        migration = await self.state_store.get_migration(migration_id)
        if migration is None or migration.workspace_id != workspace_id:
            raise WorkspaceError(
                "migration_not_found",
                "The requested workspace migration does not exist.",
                status_code=404,
            )
        return migration

    async def _ensure_no_active_migration(self, workspace_id: str) -> None:
        active = {
            MigrationState.PREPARING,
            MigrationState.COPYING,
            MigrationState.VERIFYING,
        }
        if any(
            migration.state in active
            for migration in await self.state_store.list_migrations(workspace_id)
        ):
            raise WorkspaceError(
                "migration_in_progress",
                "Finish the active workspace migration before changing lifecycle state.",
                status_code=409,
            )

    async def _abort_active_migrations(self, workspace_id: str) -> None:
        active = {
            MigrationState.PREPARING,
            MigrationState.COPYING,
            MigrationState.VERIFYING,
        }
        for migration in await self.state_store.list_migrations(workspace_id):
            if migration.state not in active:
                continue
            source_secret = await self._vault.retrieve(f"agent:{workspace_id}")
            if source_secret:
                await self._agent_factory(
                    migration.source_provider_resource_id, source_secret
                ).unseal_after_migration()
            if migration.target_provider_resource_id:
                await (await self._api()).delete_pod(migration.target_provider_resource_id)
            await self._vault.delete(f"migration-agent:{migration.id}")
            failed = migration.model_copy(
                update={
                    "state": MigrationState.FAILED,
                    "error_code": "migration_aborted_by_stop",
                    "error_message": (
                        "The migration was stopped; its network volume was retained."
                    ),
                    "updated_at": utc_now(),
                }
            )
            await self.state_store.save_migration(failed)
            if migration.operation_id:
                await self.state_store.update_operation(migration.operation_id, state="failed")
            await self.state_store.append_audit(
                action="runpod.workspace.migrate.abort",
                result="success",
                workspace_id=workspace_id,
                context={
                    "migration_id": migration.id,
                    "retained_network_volume_id": migration.target_network_volume_id,
                },
            )

    @staticmethod
    def _manifests_match(source: WorkspaceManifest, target: WorkspaceManifest) -> bool:
        return (
            source.layout_version == target.layout_version
            and source.file_count == target.file_count
            and source.total_bytes == target.total_bytes
            and source.root_sha256 == target.root_sha256
            and [item.model_dump() for item in source.files]
            == [item.model_dump() for item in target.files]
        )

    async def _record_provider_failure(
        self, workspace: WorkspaceRecord, action: str, error: Exception
    ) -> None:
        failed = workspace.model_copy(
            update={
                "state": WorkspaceState.ERROR,
                "updated_at": utc_now(),
                "error_code": getattr(error, "code", "provider_operation_failed"),
                "error_message": str(error),
            }
        )
        await self.state_store.save_workspace(failed, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action=action,
            result="failure",
            workspace_id=workspace.id,
            context={"error_type": type(error).__name__},
        )

    @staticmethod
    def _provider_id(workspace: WorkspaceRecord) -> str:
        if not workspace.provider_resource_id:
            raise WorkspaceError(
                "provider_resource_missing",
                "The workspace no longer has a RunPod Pod.",
                status_code=409,
            )
        return workspace.provider_resource_id

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise WorkspaceError(
                "provider_response_invalid",
                "RunPod returned an incomplete Pod response.",
                status_code=502,
            )
        return value
