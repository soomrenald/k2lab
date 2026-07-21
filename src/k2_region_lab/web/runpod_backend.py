from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from uuid import uuid4

from k2_region_lab.agent.domain import (
    ChunkReceipt,
    FileKind,
    FilePage,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadSession,
)
from k2_region_lab.web.agent_client import WorkspaceAgentApi, WorkspaceAgentClient
from k2_region_lab.web.credential_vault import CredentialVault
from k2_region_lab.web.domain import (
    CloudType,
    CostSnapshot,
    CredentialStatus,
    GpuOption,
    WorkspaceCreateRequest,
    WorkspaceError,
    WorkspaceMode,
    WorkspacePlan,
    WorkspacePlanRequest,
    WorkspaceRecord,
    WorkspaceState,
    utc_now,
)
from k2_region_lab.web.runpod_api import RunPodApi, RunPodApiClient, RunPodGpuType
from k2_region_lab.web.state_store import RunPodStateStore


class RunPodPersistentPodBackend:
    """Phase-one RunPod lifecycle backend.

    Provider credentials and agent secrets are held by the injected encrypted vault. Plans
    and workspace records remain process-local in this milestone; therefore production
    activation is explicit and documented as experimental until the durable repository and
    reconciler land.
    """

    PROVIDER_CREDENTIAL_ID = "provider:runpod"
    STORAGE_PRICE_PER_GB_MONTH = 0.10

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
        await self.state_store.append_audit(
            action="runpod.credentials.validate", result="success"
        )
        return status

    async def clear_credentials(self) -> CredentialStatus:
        active = any(
            workspace.state != WorkspaceState.DELETED and workspace.provider_resource_id
            for workspace in await self.state_store.list_workspaces()
        )
        if active:
            raise WorkspaceError(
                "credentials_in_use",
                "Delete every RunPod workspace before disconnecting the account.",
                status_code=409,
            )
        await self._vault.delete(self.PROVIDER_CREDENTIAL_ID)
        await self.state_store.append_audit(
            action="runpod.credentials.revoke", result="success"
        )
        return CredentialStatus(configured=False)

    async def list_gpu_options(self) -> list[GpuOption]:
        api = await self._api()
        inventory = await api.list_gpu_types()
        options = [self._gpu_option(item) for item in inventory if item.memory_gb >= 24]
        return sorted(options, key=lambda item: (-item.memory_gb, item.display_name))

    async def plan_workspace(self, request: WorkspacePlanRequest) -> WorkspacePlan:
        if request.mode != WorkspaceMode.PERSISTENT_POD:
            raise WorkspaceError(
                "workspace_mode_unavailable",
                "Portable workspaces are reserved for phase two.",
            )
        options = {item.id: item for item in await self.list_gpu_options()}
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
            estimated_storage_per_month=(
                request.workspace_disk_gb * self.STORAGE_PRICE_PER_GB_MONTH
            ),
            image_digest=self._image_digest,
            provider_gpu_priority_ids=[item.id for item in eligible],
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
        secret_id = f"agent:{workspace_id}"
        agent_secret = secrets.token_urlsafe(32)
        await self._vault.store(secret_id, agent_secret)
        api = await self._api()
        payload = self._create_payload(workspace_id, request.name.strip(), plan, agent_secret)
        try:
            provider = await api.create_pod(payload)
            provider_id = self._required_string(provider, "id")
        except Exception:
            await self._vault.delete(secret_id)
            await self.state_store.append_audit(
                action="runpod.workspace.create",
                result="failure",
                workspace_id=workspace_id,
                context={"plan_id": plan.id},
            )
            raise

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
        )
        await self.state_store.save_workspace(workspace, image_digest=self._image_digest)
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
        if workspace.state == WorkspaceState.DELETED:
            return workspace.model_copy(deep=True)
        provider_id = self._provider_id(workspace)
        provider = await (await self._api()).get_pod(provider_id)
        updated = await self._workspace_from_provider(workspace, provider)
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        return updated.model_copy(deep=True)

    async def start_workspace(self, workspace_id: str) -> WorkspaceRecord:
        workspace = await self.state_store.claim_workspace_transition(
            workspace_id,
            allowed_states={WorkspaceState.STOPPED, WorkspaceState.ERROR},
            claimed_state=WorkspaceState.STARTING,
        )
        try:
            provider = await (await self._api()).start_pod(self._provider_id(workspace))
        except Exception as error:
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
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action="runpod.workspace.start", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def stop_workspace(self, workspace_id: str) -> WorkspaceRecord:
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
            }
        )
        await self.state_store.save_workspace(updated, image_digest=self._image_digest)
        await self.state_store.append_audit(
            action="runpod.workspace.stop", result="success", workspace_id=workspace_id
        )
        return updated.model_copy(deep=True)

    async def terminate_workspace(
        self, workspace_id: str, confirmation: str
    ) -> WorkspaceRecord:
        workspace = await self._workspace(workspace_id)
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
        try:
            await (await self._api()).delete_pod(self._provider_id(workspace))
        except Exception as error:
            await self._record_provider_failure(workspace, "runpod.workspace.delete", error)
            raise
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
        return CostSnapshot(
            workspace_id=workspace.id,
            state=workspace.state,
            compute_per_hour=workspace.estimated_compute_per_hour if running else 0.0,
            storage_per_month=(
                0.0
                if workspace.state == WorkspaceState.DELETED
                else workspace.estimated_storage_per_month
            ),
            accrued_compute_estimate=(
                elapsed / 3600 * workspace.estimated_compute_per_hour if running else 0.0
            ),
            observed_at=utc_now(),
        )

    async def reconcile_workspaces(self) -> list[WorkspaceRecord]:
        """Refresh every durable provider resource after control-plane startup."""
        if not (await self.credential_status()).configured:
            return []
        reconciled: list[WorkspaceRecord] = []
        api = await self._api()
        for workspace in await self.state_store.list_workspaces():
            if workspace.state == WorkspaceState.DELETED or not workspace.provider_resource_id:
                continue
            try:
                provider = await api.get_pod(workspace.provider_resource_id)
                status = str(provider.get("desiredStatus", ""))
                updated = await self._workspace_from_provider(workspace, provider)
                state = updated.state
                if state == WorkspaceState.DELETED:
                    updated = updated.model_copy(update={"provider_resource_id": None})
                    await self._vault.delete(f"agent:{workspace.id}")
                await self.state_store.save_workspace(
                    updated, image_digest=self._image_digest
                )
                await self.state_store.append_audit(
                    action="runpod.workspace.reconcile",
                    result="success",
                    workspace_id=workspace.id,
                    context={"provider_state": status},
                )
                reconciled.append(updated)
            except Exception as error:
                await self._record_provider_failure(
                    workspace, "runpod.workspace.reconcile", error
                )
        return reconciled

    async def get_file_inventory(
        self, workspace_id: str, kind: FileKind, cursor: str | None = None
    ) -> FilePage:
        return await (await self._workspace_agent(workspace_id)).inventory(
            kind, cursor=cursor
        )

    async def create_upload(
        self, workspace_id: str, request: UploadCreateRequest
    ) -> UploadSession:
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

    async def complete_upload(
        self, workspace_id: str, upload_id: str
    ) -> UploadCompleteResponse:
        return await (await self._workspace_agent(workspace_id)).complete_upload(upload_id)

    async def cancel_upload(self, workspace_id: str, upload_id: str) -> None:
        await (await self._workspace_agent(workspace_id)).cancel_upload(upload_id)

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
        error_code = None
        error_message = None
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
    def _price(
        gpu: GpuOption, cloud_type: CloudType, interruptible: bool
    ) -> float | None:
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
    ) -> dict[str, Any]:
        return {
            "name": f"k2lab-{workspace_id[:8]}-{name}"[:191],
            "imageName": self._image_digest,
            "cloudType": plan.request.cloud_type.value.upper(),
            "computeType": "GPU",
            "gpuTypeIds": plan.provider_gpu_priority_ids,
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "containerDiskInGb": plan.request.container_disk_gb,
            "volumeInGb": plan.request.workspace_disk_gb,
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
