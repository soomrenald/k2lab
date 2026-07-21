from __future__ import annotations

import json
import hashlib
import importlib.util
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


WEB_PROVIDER_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("aiosqlite", "cryptography", "fastapi", "httpx", "sqlalchemy")
)

if WEB_PROVIDER_AVAILABLE:
    import httpx
    from cryptography.fernet import Fernet

    from k2_region_lab.agent.domain import (
        AgentHealth,
        FileKind,
        GenerationJob,
        JobEvent,
        JobKind,
        JobState,
        ManifestEntry,
        MigrationChunkReceipt,
        RemoteProvider,
        RemoteTransfer,
        TransferState,
        WorkspaceManifest,
    )
    from k2_region_lab.web.credential_vault import (
        DatabaseCredentialVault,
        EncryptedMemoryCredentialVault,
    )
    from k2_region_lab.web.domain import (
        CloudType,
        WorkspaceCreateRequest,
        WorkspaceMigrationCreateRequest,
        WorkspaceMode,
        WorkspacePlanRequest,
    )
    from k2_region_lab.web.runpod_api import (
        RunPodApiClient,
        RunPodDatacenter,
        RunPodGpuType,
        RunPodNetworkVolume,
    )
    from k2_region_lab.web.runpod_backend import RunPodPersistentPodBackend
    from k2_region_lab.web.lease_reaper import WorkspaceLeaseReaper
    from k2_region_lab.web.state_store import SqlRunPodStateStore

    GPU_FIXTURE = RunPodGpuType.model_validate(
        {
            "id": "NVIDIA RTX A6000",
            "displayName": "RTX A6000",
            "memoryInGb": 48,
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": {
                "stockStatus": "High",
                "uninterruptablePrice": 0.6,
                "availableGpuCounts": [1, 2],
            },
            "communityPrice": {
                "stockStatus": "Medium",
                "uninterruptablePrice": 0.4,
                "availableGpuCounts": [1],
            },
        }
    )


class FakeRunPodApi:
    def __init__(self) -> None:
        self.validated = False
        self.create_requests: list[dict[str, Any]] = []
        self.deleted_pods: list[str] = []
        self.stopped_pods: list[str] = []
        self.created_volumes: list[str] = []
        self.status = "RUNNING"
        self.network_volumes = {
            "volume-existing": RunPodNetworkVolume.model_validate(
                {
                    "id": "volume-existing",
                    "name": "Existing volume",
                    "size": 200,
                    "dataCenterId": "US-GA-2",
                }
            )
        }

    async def validate_credentials(self) -> None:
        self.validated = True

    async def list_gpu_types(self) -> list[RunPodGpuType]:
        return [GPU_FIXTURE]

    async def list_datacenters(self) -> list[RunPodDatacenter]:
        return [
            RunPodDatacenter.model_validate(
                {
                    "id": "US-GA-2",
                    "name": "US-GA-2",
                    "location": "United States",
                    "gpuAvailability": [
                        {
                            "gpuTypeId": GPU_FIXTURE.id,
                            "displayName": GPU_FIXTURE.display_name,
                            "stockStatus": "High",
                        }
                    ],
                }
            )
        ]

    async def list_network_volumes(self) -> list[RunPodNetworkVolume]:
        return list(self.network_volumes.values())

    async def get_network_volume(self, volume_id: str) -> RunPodNetworkVolume:
        return self.network_volumes[volume_id]

    async def create_network_volume(
        self, *, name: str, size_gb: int, datacenter_id: str
    ) -> RunPodNetworkVolume:
        volume = RunPodNetworkVolume.model_validate(
            {
                "id": "volume-created",
                "name": name,
                "size": size_gb,
                "dataCenterId": datacenter_id,
            }
        )
        self.network_volumes[volume.id] = volume
        self.created_volumes.append(volume.id)
        return volume

    async def delete_network_volume(self, volume_id: str) -> None:
        self.network_volumes.pop(volume_id, None)

    async def create_pod(self, request: dict[str, Any]) -> dict[str, Any]:
        self.create_requests.append(request)
        self.status = "RUNNING"
        pod_id = f"pod-{len(self.create_requests)}"
        return {
            "id": pod_id,
            "desiredStatus": self.status,
            "adjustedCostPerHr": 0.4,
        }

    async def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {"id": pod_id, "desiredStatus": self.status}

    async def start_pod(self, _pod_id: str) -> dict[str, Any]:
        self.status = "RUNNING"
        return {"id": _pod_id, "desiredStatus": self.status}

    async def stop_pod(self, _pod_id: str) -> dict[str, Any]:
        self.stopped_pods.append(_pod_id)
        self.status = "EXITED"
        return {"id": _pod_id, "desiredStatus": self.status}

    async def delete_pod(self, _pod_id: str) -> None:
        self.deleted_pods.append(_pod_id)
        self.status = "TERMINATED"


class FakeAgentApi:
    def __init__(self, workspace_id: str, image_version: str = "0.1.0") -> None:
        self.workspace_id = workspace_id
        self.image_version = image_version

    async def health(self) -> AgentHealth:
        return AgentHealth.model_validate(
            {
                "status": "ready",
                "workspace_id": self.workspace_id,
                "image_version": self.image_version,
                "readiness": {
                    "container": True,
                    "agent": True,
                    "storage": True,
                    "models": False,
                    "worker": False,
                },
                "observed_at": "2026-07-20T12:00:00Z",
            }
        )

    async def capabilities(self):
        raise NotImplementedError

    async def storage(self):
        raise NotImplementedError


class FakeMigrationAgent(FakeAgentApi):
    def __init__(
        self,
        workspace_id: str,
        files: dict[str, bytes] | None = None,
        *,
        corrupt_manifest: bool = False,
    ) -> None:
        super().__init__(workspace_id)
        self.files = dict(files or {})
        self.corrupt_manifest = corrupt_manifest
        self.sealed = False
        self.generation = 0

    async def seal_for_migration(self) -> WorkspaceManifest:
        self.sealed = True
        return self._manifest()

    async def unseal_after_migration(self) -> None:
        self.sealed = False

    async def create_migration_manifest(self) -> WorkspaceManifest:
        return self._manifest()

    async def migration_file(
        self, generation: int, relative_path: str, *, start: int, end: int
    ) -> bytes:
        del generation
        return self.files[relative_path][start : end + 1]

    async def import_migration_chunk(
        self,
        migration_id: str,
        relative_path: str,
        *,
        offset: int,
        total_size: int,
        file_sha256: str,
        content: bytes,
    ) -> MigrationChunkReceipt:
        del migration_id
        existing = self.files.get(relative_path, b"")
        if len(existing) > offset:
            self.assert_bytes(existing[offset : offset + len(content)], content)
        else:
            self.assert_bytes(len(existing), offset)
            self.files[relative_path] = existing + content
        completed = len(self.files.get(relative_path, b"")) == total_size
        if completed:
            self.assert_bytes(hashlib.sha256(self.files[relative_path]).hexdigest(), file_sha256)
        return MigrationChunkReceipt(
            path=relative_path,
            next_offset=len(self.files.get(relative_path, b"")),
            completed=completed,
        )

    @staticmethod
    def assert_bytes(actual: object, expected: object) -> None:
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")

    def _manifest(self) -> WorkspaceManifest:
        self.generation += 1
        entries = [
            ManifestEntry(
                path=path,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for path, content in sorted(self.files.items())
        ]
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(entry.path.encode())
            digest.update(b"\0")
            digest.update(str(entry.size_bytes).encode())
            digest.update(b"\0")
            digest.update(entry.sha256.encode())
            digest.update(b"\n")
        root_sha256 = digest.hexdigest()
        if self.corrupt_manifest:
            root_sha256 = "f" * 64
        return WorkspaceManifest(
            generation=self.generation,
            layout_version=1,
            files=entries,
            file_count=len(entries),
            total_bytes=sum(item.size_bytes for item in entries),
            root_sha256=root_sha256,
            created_at="2026-07-20T12:00:00Z",
        )


@unittest.skipUnless(WEB_PROVIDER_AVAILABLE, "web provider dependencies are not installed")
class RunPodApiClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_authorization_header_and_never_places_key_in_url(self) -> None:
        observed: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["url"] = str(request.url)
            observed["authorization"] = request.headers["Authorization"]
            return httpx.Response(200, json=[])

        client = RunPodApiClient(
            "secret-runpod-key",
            transport=httpx.MockTransport(handler),
        )
        await client.validate_credentials()
        self.assertNotIn("secret-runpod-key", observed["url"])
        self.assertEqual(observed["authorization"], "Bearer secret-runpod-key")

    async def test_parses_gpu_inventory_for_both_clouds(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertIn("securePrice", body["query"])
            return httpx.Response(
                200,
                json={"data": {"gpuTypes": [GPU_FIXTURE.model_dump(by_alias=True)]}},
            )

        client = RunPodApiClient("secret-runpod-key", transport=httpx.MockTransport(handler))
        inventory = await client.list_gpu_types()
        self.assertEqual(inventory[0].secure_price.uninterruptible_price, 0.6)
        self.assertTrue(inventory[0].community_price.one_gpu_available)

    async def test_provider_errors_do_not_echo_provider_body(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "secret-runpod-key"})

        client = RunPodApiClient("secret-runpod-key", transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(Exception, "RunPod rejected") as caught:
            await client.validate_credentials()
        self.assertNotIn("secret-runpod-key", str(caught.exception))

    async def test_network_volume_and_datacenter_contracts(self) -> None:
        observed: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            observed.append((request.method, request.url.path))
            if request.url.host == "api.runpod.io":
                self.assertIn("gpuAvailability", json.loads(request.content)["query"])
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "dataCenters": [
                                {
                                    "id": "US-GA-2",
                                    "name": "US-GA-2",
                                    "location": "United States",
                                    "gpuAvailability": [
                                        {
                                            "gpuTypeId": "NVIDIA A40",
                                            "displayName": "A40",
                                            "stockStatus": "High",
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                )
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "volume-1",
                            "name": "Portable",
                            "size": 100,
                            "dataCenterId": "US-GA-2",
                        }
                    ],
                )
            if request.method == "POST":
                body = json.loads(request.content)
                self.assertEqual(body["dataCenterId"], "US-GA-2")
                return httpx.Response(
                    200,
                    json={"id": "volume-2", **body},
                )
            return httpx.Response(204)

        client = RunPodApiClient("secret-runpod-key", transport=httpx.MockTransport(handler))
        datacenters = await client.list_datacenters()
        volumes = await client.list_network_volumes()
        created = await client.create_network_volume(
            name="New portable", size_gb=100, datacenter_id="US-GA-2"
        )
        await client.delete_network_volume(created.id)

        self.assertEqual(datacenters[0].gpu_availability[0].gpu_type_id, "NVIDIA A40")
        self.assertEqual(volumes[0].size_gb, 100)
        self.assertIn(("DELETE", "/v1/networkvolumes/volume-2"), observed)


@unittest.skipUnless(WEB_PROVIDER_AVAILABLE, "web provider dependencies are not installed")
class RunPodBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        database = Path(self.temporary_directory.name) / "control-plane.sqlite3"
        self.database_url = f"sqlite+aiosqlite:///{database}"
        self.state_store = SqlRunPodStateStore(self.database_url)
        self.vault = EncryptedMemoryCredentialVault(Fernet.generate_key())
        self.api = FakeRunPodApi()
        self.agent_workspace_id = "unused"
        self.migration_agents: dict[str, FakeMigrationAgent] = {}
        self.backend = RunPodPersistentPodBackend(
            credential_vault=self.vault,
            state_store=self.state_store,
            image_digest="ghcr.io/example/k2lab@sha256:" + "a" * 64,
            image_version="0.1.0",
            api_factory=lambda _key: self.api,
            agent_factory=lambda pod_id, _token: self.migration_agents.get(
                pod_id, FakeAgentApi(self.agent_workspace_id)
            ),
        )

    async def asyncTearDown(self) -> None:
        await self.state_store.close()
        self.temporary_directory.cleanup()

    async def test_validates_before_storing_and_returns_only_hint(self) -> None:
        status = await self.backend.validate_credentials("secret-runpod-key")
        self.assertTrue(self.api.validated)
        self.assertEqual(status.key_hint, "••••-key")
        self.assertEqual(
            await self.vault.retrieve(self.backend.PROVIDER_CREDENTIAL_ID),
            "secret-runpod-key",
        )

    async def test_download_provider_token_is_encrypted_and_audited_without_secret(self) -> None:
        status = await self.backend.store_download_credential(
            RemoteProvider.HUGGINGFACE, "hf_read_secret_token"
        )
        self.assertEqual(status.key_hint, "••••oken")
        self.assertEqual(
            await self.vault.retrieve("provider:huggingface"),
            "hf_read_secret_token",
        )
        events = await self.state_store.audit_events()
        self.assertEqual(events[-1]["context"], {"provider": "huggingface"})
        self.assertNotIn("hf_read_secret_token", json.dumps(events))

    async def test_audit_store_redacts_nested_secrets_and_authenticated_urls(self) -> None:
        await self.state_store.append_audit(
            action="security.redaction.test",
            result="success",
            context={
                "nested": {"api_key": "runpod-secret", "safe": "retained"},
                "source": "https://example.test/model?token=url-secret&revision=main#fragment",
                "prompt_text": "private portrait prompt",
            },
        )
        event = (await self.state_store.audit_events())[-1]
        serialized = json.dumps(event)
        self.assertEqual(event["context"]["nested"]["safe"], "retained")
        self.assertEqual(event["context"]["nested"]["api_key"], "[redacted]")
        self.assertNotIn("runpod-secret", serialized)
        self.assertNotIn("url-secret", serialized)
        self.assertNotIn("private portrait prompt", serialized)
        self.assertNotIn("fragment", serialized)

    async def test_plan_uses_selected_cloud_price(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.COMMUNITY,
            )
        )
        self.assertEqual(plan.estimated_compute_per_hour, 0.4)

    async def test_create_lifecycle_and_delete_keep_secrets_out_of_records(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.COMMUNITY,
            )
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Portrait lab")
        )
        self.assertEqual(workspace.state, "starting")
        self.assertEqual(workspace.provider_resource_id, "pod-1")
        self.assertNotIn("K2LAB_AGENT_SESSION_TOKEN", workspace.model_dump_json())

        self.agent_workspace_id = workspace.id
        ready = await self.backend.get_workspace_status(workspace.id)
        self.assertEqual(ready.state, "ready")
        self.assertTrue(ready.readiness["agent"])

        request = self.api.create_requests[0]
        self.assertEqual(request["gpuTypePriority"], "custom")
        self.assertEqual(request["gpuTypeIds"], ["NVIDIA RTX A6000"])
        self.assertEqual(request["volumeMountPath"], "/workspace")
        self.assertEqual(request["cloudType"], "COMMUNITY")
        self.assertGreaterEqual(len(request["env"]["K2LAB_AGENT_SESSION_TOKEN"]), 32)

        stopped = await self.backend.stop_workspace(workspace.id)
        self.assertEqual(stopped.state, "stopped")
        started = await self.backend.start_workspace(workspace.id)
        self.assertEqual(started.state, "starting")
        deleted = await self.backend.terminate_workspace(workspace.id, "Portrait lab")
        self.assertEqual(deleted.state, "deleted")
        self.assertIsNone(deleted.provider_resource_id)

    async def test_workspace_state_and_audit_are_durable(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
            )
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Durable lab")
        )
        transfer = RemoteTransfer(
            id="transfer-123",
            provider=RemoteProvider.CIVITAI,
            source_url="https://civitai.com/models/123",
            destination_kind=FileKind.LORAS,
            state=TransferState.DOWNLOADING,
            bytes_total=1024,
            bytes_complete=512,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )
        await self.state_store.save_transfer(workspace.id, transfer)
        job = GenerationJob(
            id="job-123",
            command_id="command-123",
            kind=JobKind.GENERATE,
            project_id="project-123",
            state=JobState.RUNNING,
            progress_current=1,
            progress_total=8,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
        )
        await self.state_store.save_generation_job(workspace.id, job)
        await self.state_store.save_job_events(
            job.id,
            [
                JobEvent(
                    sequence=0,
                    state="running",
                    message="Denoising step 1/8",
                    payload={"step": 1, "total_steps": 8},
                    created_at=workspace.updated_at,
                )
            ],
        )

        reopened_store = SqlRunPodStateStore(self.database_url)
        try:
            restored = await reopened_store.get_workspace(workspace.id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.provider_resource_id, "pod-1")
            events = await reopened_store.audit_events()
            self.assertEqual(events[-1]["action"], "runpod.workspace.create")
            self.assertNotIn("secret-runpod-key", json.dumps(events))
            restored_transfer = await reopened_store.get_transfer(transfer.id)
            self.assertEqual(restored_transfer, (workspace.id, transfer))
            restored_job = await reopened_store.get_generation_job(job.id)
            self.assertEqual(restored_job, (workspace.id, job))
        finally:
            await reopened_store.close()

    async def test_expired_lease_is_discoverable_for_reaper(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(gpu_priority_ids=["NVIDIA RTX A6000"])
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Idle lab")
        )
        expired = workspace.model_copy(
            update={"lease_expires_at": workspace.created_at - timedelta(seconds=1)}
        )
        await self.state_store.save_workspace(
            expired,
            image_digest="ghcr.io/example/k2lab@sha256:" + "a" * 64,
        )
        self.assertEqual(
            await self.state_store.expired_workspace_ids(workspace.created_at),
            [workspace.id],
        )
        stopped = await WorkspaceLeaseReaper(self.backend).run_once()
        self.assertEqual(stopped, [workspace.id])
        restored = await self.state_store.get_workspace(workspace.id)
        self.assertEqual(restored.state, "stopped")

    async def test_database_vault_survives_backend_reconstruction(self) -> None:
        encryption_key = Fernet.generate_key()
        vault = DatabaseCredentialVault(self.state_store, encryption_key)
        await vault.store(
            "provider:runpod",
            "secret-runpod-key",
            key_hint="••••-key",
        )
        reopened_store = SqlRunPodStateStore(self.database_url)
        try:
            reopened_vault = DatabaseCredentialVault(reopened_store, encryption_key)
            self.assertEqual(
                await reopened_vault.retrieve("provider:runpod"),
                "secret-runpod-key",
            )
            self.assertEqual(
                (await reopened_vault.status("provider:runpod")).key_hint,
                "••••-key",
            )
        finally:
            await reopened_store.close()

    async def test_startup_reconciliation_refreshes_durable_provider_state(self) -> None:
        encryption_key = Fernet.generate_key()
        vault = DatabaseCredentialVault(self.state_store, encryption_key)
        backend = RunPodPersistentPodBackend(
            credential_vault=vault,
            state_store=self.state_store,
            image_digest="ghcr.io/example/k2lab@sha256:" + "a" * 64,
            image_version="0.1.0",
            api_factory=lambda _key: self.api,
            agent_factory=lambda _pod_id, _token: FakeAgentApi("unused"),
        )
        await backend.validate_credentials("secret-runpod-key")
        plan = await backend.plan_workspace(
            WorkspacePlanRequest(gpu_priority_ids=["NVIDIA RTX A6000"])
        )
        workspace = await backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Reconciled lab")
        )
        self.api.status = "EXITED"

        reopened_store = SqlRunPodStateStore(self.database_url)
        try:
            reopened_backend = RunPodPersistentPodBackend(
                credential_vault=DatabaseCredentialVault(reopened_store, encryption_key),
                state_store=reopened_store,
                image_digest="ghcr.io/example/k2lab@sha256:" + "a" * 64,
                image_version="0.1.0",
                api_factory=lambda _key: self.api,
                agent_factory=lambda _pod_id, _token: FakeAgentApi(workspace.id),
            )
            reconciled = await reopened_backend.reconcile_workspaces()
            self.assertEqual(reconciled[0].id, workspace.id)
            self.assertEqual(reconciled[0].state, "stopped")
        finally:
            await reopened_store.close()

    async def test_startup_reconciliation_stops_journaled_orphan_pod(self) -> None:
        await self.vault.store(
            self.backend.PROVIDER_CREDENTIAL_ID,
            "secret-runpod-key",
        )
        await self.vault.store("agent:orphan-workspace", "a" * 43)
        operation_id = await self.state_store.begin_operation(
            operation="runpod.workspace.create",
            workspace_id="orphan-workspace",
            context={"plan_id": "consumed-plan"},
        )
        await self.state_store.update_operation(
            operation_id,
            state="provider_created",
            context={"provider_resource_id": "pod-orphan"},
        )

        reconciled = await self.backend.reconcile_workspaces()

        self.assertEqual(reconciled, [])
        self.assertEqual(self.api.status, "EXITED")
        self.assertIsNone(await self.vault.retrieve("agent:orphan-workspace"))
        self.assertEqual(await self.state_store.incomplete_operations(), [])
        events = await self.state_store.audit_events()
        self.assertEqual(events[-1]["action"], "runpod.workspace.orphan_stop")

    async def test_portable_workspace_creates_volume_and_recreates_ephemeral_pod(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                mode=WorkspaceMode.PORTABLE_WORKSPACE,
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
                workspace_disk_gb=100,
                datacenter_priority_ids=["US-GA-2"],
            )
        )
        self.assertTrue(plan.create_network_volume)
        self.assertEqual(plan.selected_datacenter_id, "US-GA-2")
        self.assertEqual(plan.estimated_storage_per_month, 7.0)

        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Portable lab")
        )
        self.assertEqual(workspace.network_volume_id, "volume-created")
        self.assertEqual(workspace.provider_resource_id, "pod-1")
        self.assertTrue(workspace.owns_network_volume)
        payload = self.api.create_requests[-1]
        self.assertEqual(payload["networkVolumeId"], "volume-created")
        self.assertEqual(payload["dataCenterIds"], ["US-GA-2"])
        self.assertNotIn("volumeInGb", payload)

        stopped = await self.backend.stop_workspace(workspace.id)
        self.assertEqual(stopped.state, "stopped")
        self.assertIsNone(stopped.provider_resource_id)
        self.assertEqual(self.api.deleted_pods, ["pod-1"])
        self.assertIn("volume-created", self.api.network_volumes)

        restarted = await self.backend.start_workspace(workspace.id)
        self.assertEqual(restarted.provider_resource_id, "pod-2")
        self.assertEqual(self.api.create_requests[-1]["networkVolumeId"], "volume-created")
        deleted = await self.backend.terminate_workspace(workspace.id, "Portable lab")
        self.assertEqual(deleted.state, "deleted")
        self.assertIn("volume-created", self.api.network_volumes)
        cost = await self.backend.get_cost_snapshot(workspace.id)
        self.assertEqual(cost.storage_per_month, 7.0)

    async def test_portable_workspace_uses_existing_volume_and_its_datacenter(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                mode=WorkspaceMode.PORTABLE_WORKSPACE,
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
                workspace_disk_gb=100,
                network_volume_id="volume-existing",
            )
        )
        self.assertFalse(plan.create_network_volume)
        self.assertEqual(plan.selected_datacenter_id, "US-GA-2")
        self.assertEqual(plan.request.workspace_disk_gb, 200)
        self.assertEqual(plan.estimated_storage_per_month, 14.0)
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Existing portable")
        )
        self.assertEqual(workspace.network_volume_id, "volume-existing")
        self.assertFalse(workspace.owns_network_volume)
        self.assertEqual(self.api.created_volumes, [])

    async def test_portable_workspace_rejects_community_cloud(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        with self.assertRaisesRegex(Exception, "Secure Cloud"):
            await self.backend.plan_workspace(
                WorkspacePlanRequest(
                    mode=WorkspaceMode.PORTABLE_WORKSPACE,
                    gpu_priority_ids=["NVIDIA RTX A6000"],
                    cloud_type=CloudType.COMMUNITY,
                )
            )

    async def test_portable_orphan_pod_is_terminated_during_reconciliation(self) -> None:
        await self.vault.store(self.backend.PROVIDER_CREDENTIAL_ID, "secret-runpod-key")
        operation_id = await self.state_store.begin_operation(
            operation="runpod.workspace.create",
            workspace_id="portable-orphan",
            context={"mode": WorkspaceMode.PORTABLE_WORKSPACE.value},
        )
        await self.state_store.update_operation(
            operation_id,
            state="provider_created",
            context={"provider_resource_id": "pod-portable-orphan"},
        )

        await self.backend.reconcile_workspaces()

        self.assertEqual(self.api.deleted_pods, ["pod-portable-orphan"])
        events = await self.state_store.audit_events()
        self.assertEqual(events[-1]["action"], "runpod.workspace.orphan_delete")

    async def test_verified_migration_switches_then_retains_original_until_confirmation(
        self,
    ) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
            )
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Migrated lab")
        )
        self.agent_workspace_id = workspace.id
        source = FakeMigrationAgent(
            workspace.id,
            {"projects/portrait.json": b'{"schema":"k2"}'},
        )
        self.migration_agents["pod-1"] = source
        ready = await self.backend.get_workspace_status(workspace.id)
        self.assertEqual(ready.state, "ready")

        migration = await self.backend.create_workspace_migration(
            workspace.id,
            WorkspaceMigrationCreateRequest(
                workspace_disk_gb=100, datacenter_priority_ids=["US-GA-2"]
            ),
        )
        self.assertEqual(migration.state, "preparing")
        self.assertTrue(source.sealed)
        self.assertEqual(migration.target_provider_resource_id, "pod-2")
        migrating_cost = await self.backend.get_cost_snapshot(workspace.id)
        self.assertEqual(migrating_cost.compute_per_hour, 1.0)
        self.assertEqual(migrating_cost.storage_per_month, 27.0)
        target = FakeMigrationAgent(workspace.id)
        self.migration_agents["pod-2"] = target

        self.backend.MIGRATION_CHUNK_SIZE = 4
        self.backend.MIGRATION_COPY_BUDGET = 4
        partial = await self.backend.resume_workspace_migration(workspace.id, migration.id)
        self.assertEqual(partial.state, "copying")
        self.assertEqual(partial.bytes_copied, 4)
        restored_partial = await self.state_store.get_migration(migration.id)
        self.assertEqual(restored_partial.current_file_offset, 4)
        verified = partial
        while verified.state in {"preparing", "copying", "verifying"}:
            verified = await self.backend.resume_workspace_migration(workspace.id, migration.id)
        self.assertEqual(verified.state, "awaiting_confirmation")
        self.assertEqual(target.files, source.files)
        switched = await self.state_store.get_workspace(workspace.id)
        self.assertEqual(switched.mode, "portable_workspace")
        self.assertEqual(switched.provider_resource_id, "pod-2")
        self.assertEqual(switched.retained_original_provider_resource_id, "pod-1")
        self.assertEqual(self.api.stopped_pods, ["pod-1"])
        retained_cost = await self.backend.get_cost_snapshot(workspace.id)
        self.assertEqual(retained_cost.compute_per_hour, 0.6)
        self.assertEqual(retained_cost.storage_per_month, 27.0)

        with self.assertRaisesRegex(Exception, "Confirm the migrated workspace"):
            await self.backend.terminate_workspace(workspace.id, "Migrated lab")
        with self.assertRaisesRegex(Exception, "Type the workspace name"):
            await self.backend.confirm_workspace_migration(workspace.id, migration.id, "wrong")
        completed = await self.backend.confirm_workspace_migration(
            workspace.id, migration.id, "Migrated lab"
        )
        self.assertEqual(completed.state, "completed")
        restored = await self.state_store.get_workspace(workspace.id)
        self.assertIsNone(restored.retained_original_provider_resource_id)
        self.assertIn("pod-1", self.api.deleted_pods)
        final_cost = await self.backend.get_cost_snapshot(workspace.id)
        self.assertEqual(final_cost.storage_per_month, 7.0)
        self.assertEqual(await self.state_store.incomplete_operations(), [])

    async def test_manifest_mismatch_keeps_original_and_releases_target_compute(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
            )
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Mismatch lab")
        )
        self.agent_workspace_id = workspace.id
        source = FakeMigrationAgent(workspace.id, {"outputs/result.png": b"verified-output"})
        self.migration_agents["pod-1"] = source
        await self.backend.get_workspace_status(workspace.id)
        migration = await self.backend.create_workspace_migration(
            workspace.id,
            WorkspaceMigrationCreateRequest(datacenter_priority_ids=["US-GA-2"]),
        )
        self.migration_agents["pod-2"] = FakeMigrationAgent(workspace.id, corrupt_manifest=True)

        failed = await self.backend.resume_workspace_migration(workspace.id, migration.id)

        self.assertEqual(failed.state, "failed")
        self.assertEqual(failed.error_code, "migration_manifest_mismatch")
        self.assertFalse(source.sealed)
        self.assertIn("pod-2", self.api.deleted_pods)
        original = await self.state_store.get_workspace(workspace.id)
        self.assertEqual(original.mode, "persistent_pod")
        self.assertEqual(original.provider_resource_id, "pod-1")

    async def test_stop_aborts_active_migration_and_retains_target_volume(self) -> None:
        await self.backend.validate_credentials("secret-runpod-key")
        plan = await self.backend.plan_workspace(
            WorkspacePlanRequest(
                gpu_priority_ids=["NVIDIA RTX A6000"],
                cloud_type=CloudType.SECURE,
            )
        )
        workspace = await self.backend.create_workspace(
            WorkspaceCreateRequest(plan_id=plan.id, name="Abort migration")
        )
        self.agent_workspace_id = workspace.id
        source = FakeMigrationAgent(workspace.id, {"projects/project.json": b"durable"})
        self.migration_agents["pod-1"] = source
        await self.backend.get_workspace_status(workspace.id)
        migration = await self.backend.create_workspace_migration(
            workspace.id,
            WorkspaceMigrationCreateRequest(datacenter_priority_ids=["US-GA-2"]),
        )

        stopped = await self.backend.stop_workspace(workspace.id)

        self.assertEqual(stopped.state, "stopped")
        self.assertFalse(source.sealed)
        self.assertIn("pod-2", self.api.deleted_pods)
        self.assertIn(migration.target_network_volume_id, self.api.network_volumes)
        failed = await self.backend.get_workspace_migration(workspace.id, migration.id)
        self.assertEqual(failed.error_code, "migration_aborted_by_stop")
        self.assertEqual(await self.state_store.incomplete_operations(), [])


if __name__ == "__main__":
    unittest.main()
