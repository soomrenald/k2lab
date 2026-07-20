from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import uuid4

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


class DevelopmentWorkspaceBackend:
    """Safe local backend for UI development; it never calls or bills RunPod."""

    IMAGE_DIGEST = "ghcr.io/k2-region-lab/runtime@sha256:development-placeholder"
    STORAGE_PRICE_PER_GB_MONTH = 0.10

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._credential = CredentialStatus(configured=False, development_only=True)
        self._plans: dict[str, WorkspacePlan] = {}
        self._workspaces: dict[str, WorkspaceRecord] = {}
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

    async def plan_workspace(self, request: WorkspacePlanRequest) -> WorkspacePlan:
        self._require_credentials()
        if request.mode != WorkspaceMode.PERSISTENT_POD:
            raise WorkspaceError(
                "workspace_mode_unavailable",
                "Portable workspaces are reserved for phase two.",
            )
        available = {gpu.id: gpu for gpu in self._gpus if gpu.available}
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
        warnings = [
            "Development preview only: no RunPod resource will be created.",
            "Stopping releases GPU compute but persistent storage continues to incur cost.",
        ]
        if request.interruptible:
            warnings.append("Interruptible Pods may stop without notice.")
        plan = WorkspacePlan(
            id=uuid4().hex,
            request=request,
            selected_gpu=selected,
            estimated_compute_per_hour=compute_price,
            estimated_storage_per_month=(
                request.workspace_disk_gb * self.STORAGE_PRICE_PER_GB_MONTH
            ),
            image_digest=self.IMAGE_DIGEST,
            provider_gpu_priority_ids=[selected.id],
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
                lease_expires_at=now
                + timedelta(seconds=plan.request.idle_timeout_seconds),
                hard_expires_at=now
                + timedelta(seconds=plan.request.hard_deadline_seconds),
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
                    "lease_expires_at": now
                    + timedelta(seconds=workspace.idle_timeout_seconds),
                    "hard_expires_at": now
                    + timedelta(seconds=workspace.hard_deadline_seconds),
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
            workspace = workspace.model_copy(
                update={"state": WorkspaceState.STOPPED, "updated_at": utc_now()}
            )
            self._workspaces[workspace_id] = workspace
        return workspace.model_copy(deep=True)

    async def terminate_workspace(
        self, workspace_id: str, confirmation: str
    ) -> WorkspaceRecord:
        async with self._lock:
            workspace = self._workspace(workspace_id)
            if workspace.state == WorkspaceState.DELETED:
                return workspace.model_copy(deep=True)
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
        return CostSnapshot(
            workspace_id=workspace.id,
            state=workspace.state,
            compute_per_hour=(
                workspace.estimated_compute_per_hour
                if workspace.state == WorkspaceState.READY
                else 0.0
            ),
            storage_per_month=(
                0.0
                if workspace.state == WorkspaceState.DELETED
                else workspace.estimated_storage_per_month
            ),
            accrued_compute_estimate=accrued,
            observed_at=utc_now(),
        )

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
