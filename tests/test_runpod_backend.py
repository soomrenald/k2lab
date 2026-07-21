from __future__ import annotations

import json
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

    from k2_region_lab.agent.domain import AgentHealth
    from k2_region_lab.web.credential_vault import (
        DatabaseCredentialVault,
        EncryptedMemoryCredentialVault,
    )
    from k2_region_lab.web.domain import (
        CloudType,
        WorkspaceCreateRequest,
        WorkspacePlanRequest,
    )
    from k2_region_lab.web.runpod_api import RunPodApiClient, RunPodGpuType
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
        self.status = "RUNNING"

    async def validate_credentials(self) -> None:
        self.validated = True

    async def list_gpu_types(self) -> list[RunPodGpuType]:
        return [GPU_FIXTURE]

    async def create_pod(self, request: dict[str, Any]) -> dict[str, Any]:
        self.create_requests.append(request)
        self.status = "RUNNING"
        return {
            "id": "pod-123",
            "desiredStatus": self.status,
            "adjustedCostPerHr": 0.4,
        }

    async def get_pod(self, _pod_id: str) -> dict[str, Any]:
        return {"id": "pod-123", "desiredStatus": self.status}

    async def start_pod(self, _pod_id: str) -> dict[str, Any]:
        self.status = "RUNNING"
        return {"id": "pod-123", "desiredStatus": self.status}

    async def stop_pod(self, _pod_id: str) -> dict[str, Any]:
        self.status = "EXITED"
        return {"id": "pod-123", "desiredStatus": self.status}

    async def delete_pod(self, _pod_id: str) -> None:
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
        self.backend = RunPodPersistentPodBackend(
            credential_vault=self.vault,
            state_store=self.state_store,
            image_digest="ghcr.io/example/k2lab@sha256:" + "a" * 64,
            image_version="0.1.0",
            api_factory=lambda _key: self.api,
            agent_factory=lambda _pod_id, _token: FakeAgentApi(self.agent_workspace_id),
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
        self.assertEqual(workspace.provider_resource_id, "pod-123")
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

        reopened_store = SqlRunPodStateStore(self.database_url)
        try:
            restored = await reopened_store.get_workspace(workspace.id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.provider_resource_id, "pod-123")
            events = await reopened_store.audit_events()
            self.assertEqual(events[-1]["action"], "runpod.workspace.create")
            self.assertNotIn("secret-runpod-key", json.dumps(events))
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


if __name__ == "__main__":
    unittest.main()
