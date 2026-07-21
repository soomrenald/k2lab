from __future__ import annotations

import asyncio
import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cryptography.fernet import Fernet

from k2_region_lab.agent.domain import FileKind, UploadCreateRequest
from k2_region_lab.web.credential_vault import DatabaseCredentialVault
from k2_region_lab.web.domain import (
    CloudType,
    WorkspaceCreateRequest,
    WorkspaceMode,
    WorkspacePlanRequest,
    WorkspaceState,
)
from k2_region_lab.web.runpod_backend import RunPodPersistentPodBackend
from k2_region_lab.web.state_store import SqlRunPodStateStore


LIVE_SENTINEL = "I_ACCEPT_BILLING_AND_DELETION"


@unittest.skipUnless(
    LIVE_ENABLED,
)
class RunPodLiveAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Opt-in destructive acceptance test; never runs from the ordinary test suite."""

    async def asyncSetUp(self) -> None:
        required = {
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            self.fail(f"Live acceptance configuration is missing: {', '.join(missing)}")
        self.temporary_directory = TemporaryDirectory()
        database = Path(self.temporary_directory.name) / "acceptance.sqlite3"
        self.store = SqlRunPodStateStore(f"sqlite+aiosqlite:///{database}")
        self.backend = RunPodPersistentPodBackend(
            credential_vault=DatabaseCredentialVault(self.store, Fernet.generate_key()),
            state_store=self.store,
        )
        self.workspace = None
        self.network_volume_ids: set[str] = set()

    async def asyncTearDown(self) -> None:
        try:
            if self.workspace is not None and self.workspace.state != WorkspaceState.DELETED:
                self.workspace = await self.backend.terminate_workspace(
                    self.workspace.id, self.workspace.name
                )
        finally:
            try:
                api = await self.backend._api()
                for volume_id in self.network_volume_ids:
                    await api.delete_network_volume(volume_id)
            finally:
                await self.store.close()
                self.temporary_directory.cleanup()

    async def test_disposable_pod_ready_stop_restart_volume_and_delete(self) -> None:
        await self.backend.validate_credentials(self.api_key)
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=[self.gpu_id],
                cloud_type=CloudType.SECURE,
                container_disk_gb=50,
                workspace_disk_gb=50,
                idle_timeout_seconds=300,
                hard_deadline_seconds=1800,
            )
        )
        self.workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="K2 disposable acceptance")
        )
        self.workspace = await self._wait_for(WorkspaceState.READY)

        content = b"k2-live-acceptance" * 64
        digest = hashlib.sha256(content).hexdigest()
        upload = await self.backend.create_upload(
            self.workspace.id,
            UploadCreateRequest(
                filename="acceptance.bin",
                destination_kind=FileKind.INPUTS,
                size_bytes=len(content),
                sha256=digest,
                chunk_size_bytes=1024,
            ),
        )
        await self.backend.write_upload_chunk(
            self.workspace.id,
            upload.id,
            0,
            content[:1024],
            hashlib.sha256(content[:1024]).hexdigest(),
        )
        await self.backend.write_upload_chunk(
            self.workspace.id,
            upload.id,
            1,
            content[1024:],
            hashlib.sha256(content[1024:]).hexdigest(),
        )
        completed = await self.backend.complete_upload(self.workspace.id, upload.id)

        self.workspace = await self.backend.stop_workspace(self.workspace.id)
        self.assertEqual(self.workspace.state, WorkspaceState.STOPPED)
        self.workspace = await self.backend.start_workspace(self.workspace.id)
        self.workspace = await self._wait_for(WorkspaceState.READY)
        inventory = await self.backend.get_file_inventory(self.workspace.id, FileKind.INPUTS)
        self.assertIn(completed.file.id, [item.id for item in inventory.items])

        self.workspace = await self.backend.terminate_workspace(
            self.workspace.id, self.workspace.name
        )
        self.assertEqual(self.workspace.state, WorkspaceState.DELETED)

    async def test_portable_pod_recreation_preserves_verified_manifest(self) -> None:
        await self.backend.validate_credentials(self.api_key)
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                mode=WorkspaceMode.PORTABLE_WORKSPACE,
                gpu_priority_ids=[self.gpu_id],
                cloud_type=CloudType.SECURE,
                container_disk_gb=50,
                workspace_disk_gb=50,
                idle_timeout_seconds=300,
                hard_deadline_seconds=1800,
            )
        )
        self.workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="K2 portable acceptance")
        )
        assert self.workspace.network_volume_id is not None
        self.network_volume_ids.add(self.workspace.network_volume_id)
        self.workspace = await self._wait_for(WorkspaceState.READY)

        content = b"portable-manifest-acceptance" * 64
        digest = hashlib.sha256(content).hexdigest()
        upload = await self.backend.create_upload(
            self.workspace.id,
            UploadCreateRequest(
                filename="portable-acceptance.bin",
                destination_kind=FileKind.INPUTS,
                size_bytes=len(content),
                sha256=digest,
                chunk_size_bytes=len(content),
            ),
        )
        await self.backend.write_upload_chunk(self.workspace.id, upload.id, 0, content, digest)
        await self.backend.complete_upload(self.workspace.id, upload.id)
        source_agent = await self.backend._workspace_agent(self.workspace.id)
        source_manifest = await source_agent.seal_for_migration()
        await source_agent.unseal_after_migration()

        self.workspace = await self.backend.stop_workspace(self.workspace.id)
        self.assertIsNone(self.workspace.provider_resource_id)
        self.workspace = await self.backend.start_workspace(self.workspace.id)
        self.workspace = await self._wait_for(WorkspaceState.READY)
        recreated_agent = await self.backend._workspace_agent(self.workspace.id)
        recreated_manifest = await recreated_agent.seal_for_migration()
        await recreated_agent.unseal_after_migration()
        self.assertEqual(recreated_manifest.root_sha256, source_manifest.root_sha256)
        self.assertEqual(recreated_manifest.files, source_manifest.files)

        self.workspace = await self.backend.terminate_workspace(
            self.workspace.id, self.workspace.name
        )
        self.assertEqual(self.workspace.state, WorkspaceState.DELETED)

    async def _wait_for(self, state: WorkspaceState):
        assert self.workspace is not None
        for _attempt in range(120):
            self.workspace = await self.backend.get_workspace_status(self.workspace.id)
            if self.workspace.state == state:
                return self.workspace
            if self.workspace.state == WorkspaceState.ERROR:
                self.fail(
                    f"Workspace failed: {self.workspace.error_code}: {self.workspace.error_message}"
                )
            await asyncio.sleep(5)
        self.fail(f"Workspace did not reach {state.value} within ten minutes")
