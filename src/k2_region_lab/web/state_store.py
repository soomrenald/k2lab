from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, LargeBinary, String, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from k2_region_lab.agent.domain import RemoteTransfer
from k2_region_lab.web.domain import (
    WorkspaceError,
    WorkspacePlan,
    WorkspaceRecord,
    WorkspaceState,
    utc_now,
)


@dataclass(frozen=True)
class StoredCredential:
    credential_id: str
    ciphertext: bytes
    key_hint: str | None
    validated_at: datetime | None


class RunPodStateStore(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def save_credential(self, credential: StoredCredential) -> None: ...

    async def get_credential(self, credential_id: str) -> StoredCredential | None: ...

    async def delete_credential(self, credential_id: str) -> None: ...

    async def save_plan(self, plan: WorkspacePlan) -> None: ...

    async def consume_plan(self, plan_id: str) -> WorkspacePlan | None: ...

    async def save_workspace(
        self, workspace: WorkspaceRecord, *, image_digest: str
    ) -> None: ...

    async def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None: ...

    async def list_workspaces(self) -> list[WorkspaceRecord]: ...

    async def claim_workspace_transition(
        self,
        workspace_id: str,
        *,
        allowed_states: Iterable[WorkspaceState],
        claimed_state: WorkspaceState,
    ) -> WorkspaceRecord: ...

    async def expired_workspace_ids(self, observed_at: datetime) -> list[str]: ...

    async def append_audit(
        self,
        *,
        action: str,
        result: str,
        workspace_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None: ...

    async def save_transfer(
        self, workspace_id: str, transfer: RemoteTransfer
    ) -> None: ...

    async def get_transfer(
        self, transfer_id: str
    ) -> tuple[str, RemoteTransfer] | None: ...


class Base(DeclarativeBase):
    pass


class SchemaVersionEntity(Base):
    __tablename__ = "schema_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)


class ProviderCredentialEntity(Base):
    __tablename__ = "provider_credentials"

    id: Mapped[str] = mapped_column(String(191), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    encrypted_secret: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_hint: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkspacePlanEntity(Base):
    __tablename__ = "workspace_plans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WorkspaceEntity(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    image_digest: Mapped[str] = mapped_column(String(512), nullable=False)
    provider_resource_id: Mapped[str | None] = mapped_column(String(191), index=True)
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    hard_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunPodResourceEntity(Base):
    __tablename__ = "runpod_resources"

    workspace_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    pod_id: Mapped[str | None] = mapped_column(String(191), unique=True)
    volume_kind: Mapped[str] = mapped_column(String(32), default="pod_volume")
    volume_id: Mapped[str | None] = mapped_column(String(191))
    datacenter_id: Mapped[str | None] = mapped_column(String(80))
    gpu_type_id: Mapped[str] = mapped_column(String(191), nullable=False)
    cloud_type: Mapped[str] = mapped_column(String(32), nullable=False)
    container_disk_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    workspace_disk_gb: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_per_hour: Mapped[float] = mapped_column(nullable=False)
    desired_state: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_state: Mapped[str] = mapped_column(String(32), nullable=False)


class LeaseEntity(Base):
    __tablename__ = "leases"

    workspace_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hard_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str] = mapped_column(String(80), nullable=False)


class OperationJournalEntity(Base):
    __tablename__ = "operation_journal"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    operation: Mapped[str] = mapped_column(String(80), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    redacted_context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TransferEntity(Base):
    __tablename__ = "transfers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(2048), nullable=False)
    destination_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    bytes_total: Mapped[int | None] = mapped_column()
    bytes_complete: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(80))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditEventEntity(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str | None] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    redacted_context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SqlRunPodStateStore:
    """Transactional state store for PostgreSQL, with SQLite support for local tests."""

    SCHEMA_VERSION = 1

    def __init__(self, database_url: str, *, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(database_url, pool_pre_ping=True)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialization_lock:
            if self._initialized:
                return
            async with self._engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
            async with self._sessions.begin() as session:
                version = await session.get(SchemaVersionEntity, 1)
                if version is None:
                    session.add(SchemaVersionEntity(id=1, version=self.SCHEMA_VERSION))
                elif version.version != self.SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported control-plane schema version {version.version}; "
                        f"expected {self.SCHEMA_VERSION}"
                    )
            self._initialized = True

    async def close(self) -> None:
        await self._engine.dispose()

    async def save_credential(self, credential: StoredCredential) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            entity = await session.get(ProviderCredentialEntity, credential.credential_id)
            if entity is None:
                entity = ProviderCredentialEntity(
                    id=credential.credential_id,
                    provider=credential.credential_id.split(":", 1)[-1],
                    encrypted_secret=credential.ciphertext,
                    key_hint=credential.key_hint,
                    created_at=utc_now(),
                    last_validated_at=credential.validated_at,
                    revoked_at=None,
                )
                session.add(entity)
            else:
                entity.encrypted_secret = credential.ciphertext
                entity.key_hint = credential.key_hint
                entity.last_validated_at = credential.validated_at
                entity.revoked_at = None

    async def get_credential(self, credential_id: str) -> StoredCredential | None:
        await self.initialize()
        async with self._sessions() as session:
            entity = await session.get(ProviderCredentialEntity, credential_id)
            if entity is None or entity.revoked_at is not None:
                return None
            return StoredCredential(
                credential_id=entity.id,
                ciphertext=entity.encrypted_secret,
                key_hint=entity.key_hint,
                validated_at=entity.last_validated_at,
            )

    async def delete_credential(self, credential_id: str) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            entity = await session.get(ProviderCredentialEntity, credential_id)
            if entity is not None:
                entity.revoked_at = utc_now()
                entity.encrypted_secret = b"revoked"

    async def save_plan(self, plan: WorkspacePlan) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            session.add(
                WorkspacePlanEntity(
                    id=plan.id,
                    payload=plan.model_dump(mode="json"),
                    created_at=plan.created_at,
                )
            )

    async def consume_plan(self, plan_id: str) -> WorkspacePlan | None:
        await self.initialize()
        async with self._sessions.begin() as session:
            result = await session.execute(
                select(WorkspacePlanEntity)
                .where(WorkspacePlanEntity.id == plan_id)
                .with_for_update()
            )
            entity = result.scalar_one_or_none()
            if entity is None:
                return None
            plan = WorkspacePlan.model_validate(entity.payload)
            await session.delete(entity)
            return plan

    async def save_workspace(
        self, workspace: WorkspaceRecord, *, image_digest: str
    ) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            entity = await session.get(WorkspaceEntity, workspace.id)
            if entity is None:
                entity = WorkspaceEntity(id=workspace.id)
                session.add(entity)
            self._update_workspace_entity(entity, workspace, image_digest)
            await session.flush()
            await self._upsert_resource(session, workspace)
            await self._upsert_lease(session, workspace)

    async def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        await self.initialize()
        async with self._sessions() as session:
            entity = await session.get(WorkspaceEntity, workspace_id)
            return self._record(entity) if entity else None

    async def list_workspaces(self) -> list[WorkspaceRecord]:
        await self.initialize()
        async with self._sessions() as session:
            result = await session.scalars(
                select(WorkspaceEntity).order_by(WorkspaceEntity.created_at)
            )
            return [self._record(entity) for entity in result]

    async def claim_workspace_transition(
        self,
        workspace_id: str,
        *,
        allowed_states: Iterable[WorkspaceState],
        claimed_state: WorkspaceState,
    ) -> WorkspaceRecord:
        await self.initialize()
        allowed = set(allowed_states)
        async with self._sessions.begin() as session:
            result = await session.execute(
                select(WorkspaceEntity)
                .where(WorkspaceEntity.id == workspace_id)
                .with_for_update()
            )
            entity = result.scalar_one_or_none()
            if entity is None:
                raise WorkspaceError(
                    "workspace_not_found",
                    "The requested workspace does not exist.",
                    status_code=404,
                )
            workspace = self._record(entity)
            if workspace.state not in allowed:
                raise WorkspaceError(
                    "invalid_workspace_transition",
                    f"A {workspace.state.value} workspace cannot enter {claimed_state.value}.",
                    status_code=409,
                )
            claimed = workspace.model_copy(
                update={"state": claimed_state, "updated_at": utc_now()}
            )
            self._update_workspace_entity(entity, claimed, entity.image_digest)
            return claimed

    async def expired_workspace_ids(self, observed_at: datetime) -> list[str]:
        await self.initialize()
        active_states = [
            WorkspaceState.PROVISIONING.value,
            WorkspaceState.STARTING.value,
            WorkspaceState.READY.value,
            WorkspaceState.ERROR.value,
        ]
        async with self._sessions() as session:
            result = await session.scalars(
                select(WorkspaceEntity.id).where(
                    WorkspaceEntity.state.in_(active_states),
                    WorkspaceEntity.lease_expires_at <= observed_at,
                )
            )
            return list(result)

    async def append_audit(
        self,
        *,
        action: str,
        result: str,
        workspace_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            session.add(
                AuditEventEntity(
                    id=uuid4().hex,
                    workspace_id=workspace_id,
                    action=action,
                    result=result,
                    redacted_context=context or {},
                    created_at=utc_now(),
                )
            )

    async def save_transfer(
        self, workspace_id: str, transfer: RemoteTransfer
    ) -> None:
        await self.initialize()
        async with self._sessions.begin() as session:
            entity = await session.get(TransferEntity, transfer.id)
            if entity is None:
                entity = TransferEntity(id=transfer.id, workspace_id=workspace_id)
                session.add(entity)
            elif entity.workspace_id != workspace_id:
                raise WorkspaceError(
                    "transfer_workspace_mismatch",
                    "The transfer belongs to another workspace.",
                    status_code=409,
                )
            entity.kind = transfer.provider.value
            entity.source = transfer.source_url
            entity.destination_kind = transfer.destination_kind.value
            entity.state = transfer.state.value
            entity.bytes_total = transfer.bytes_total
            entity.bytes_complete = transfer.bytes_complete
            entity.sha256 = transfer.sha256
            entity.error_code = transfer.error_code
            entity.payload = transfer.model_dump(mode="json")
            entity.created_at = transfer.created_at
            entity.updated_at = transfer.updated_at

    async def get_transfer(
        self, transfer_id: str
    ) -> tuple[str, RemoteTransfer] | None:
        await self.initialize()
        async with self._sessions() as session:
            entity = await session.get(TransferEntity, transfer_id)
            if entity is None:
                return None
            return entity.workspace_id, RemoteTransfer.model_validate(entity.payload)

    async def audit_events(self) -> list[dict[str, Any]]:
        await self.initialize()
        async with self._sessions() as session:
            result = await session.scalars(
                select(AuditEventEntity).order_by(AuditEventEntity.created_at)
            )
            return [
                {
                    "workspace_id": entity.workspace_id,
                    "action": entity.action,
                    "result": entity.result,
                    "context": entity.redacted_context,
                }
                for entity in result
            ]

    @staticmethod
    def _record(entity: WorkspaceEntity) -> WorkspaceRecord:
        return WorkspaceRecord.model_validate(entity.payload)

    @staticmethod
    def _update_workspace_entity(
        entity: WorkspaceEntity, workspace: WorkspaceRecord, image_digest: str
    ) -> None:
        entity.state = workspace.state.value
        entity.mode = workspace.mode.value
        entity.image_digest = image_digest
        entity.provider_resource_id = workspace.provider_resource_id
        entity.lease_expires_at = workspace.lease_expires_at
        entity.hard_expires_at = workspace.hard_expires_at
        entity.payload = workspace.model_dump(mode="json")
        entity.created_at = workspace.created_at
        entity.updated_at = workspace.updated_at

    @staticmethod
    async def _upsert_resource(
        session: AsyncSession, workspace: WorkspaceRecord
    ) -> None:
        resource = await session.get(RunPodResourceEntity, workspace.id)
        if resource is None:
            resource = RunPodResourceEntity(workspace_id=workspace.id)
            session.add(resource)
        resource.pod_id = workspace.provider_resource_id
        resource.volume_kind = "pod_volume"
        resource.gpu_type_id = workspace.gpu.id
        resource.cloud_type = workspace.cloud_type.value
        resource.container_disk_gb = workspace.container_disk_gb
        resource.workspace_disk_gb = workspace.workspace_disk_gb
        resource.cost_per_hour = workspace.estimated_compute_per_hour
        resource.desired_state = workspace.state.value
        resource.observed_state = workspace.state.value

    @staticmethod
    async def _upsert_lease(session: AsyncSession, workspace: WorkspaceRecord) -> None:
        lease = await session.get(LeaseEntity, workspace.id)
        if lease is None:
            lease = LeaseEntity(workspace_id=workspace.id)
            session.add(lease)
        lease.expires_at = workspace.lease_expires_at
        lease.hard_expires_at = workspace.hard_expires_at
        lease.last_activity_at = workspace.updated_at
        lease.reason = "workspace_activity"
