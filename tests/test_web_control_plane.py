from __future__ import annotations

import importlib.util
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from httpx import ASGITransport, AsyncClient

    from k2_region_lab.web.app import create_app
    from k2_region_lab.web.development_backend import DevelopmentWorkspaceBackend


@unittest.skipUnless(FASTAPI_AVAILABLE, "web dependencies are not installed")
class WebControlPlaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.backend = DevelopmentWorkspaceBackend()
        self.client = AsyncClient(
            transport=ASGITransport(app=create_app(self.backend)),
            base_url="http://test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def connect(self) -> None:
        response = await self.client.post(
            "/api/v1/credentials/runpod", json={"api_key": "development-key"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["key_hint"], "••••-key")
        self.assertNotIn("development-key", response.text)

    async def plan(self) -> dict:
        response = await self.client.post(
            "/api/v1/workspace-plans",
            json={
                "gpu_priority_ids": ["NVIDIA RTX A6000", "NVIDIA A40"],
                "cloud_type": "secure",
                "container_disk_gb": 50,
                "workspace_disk_gb": 200,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_capabilities_are_explicitly_development_only(self) -> None:
        response = await self.client.get("/api/v1/capabilities")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["development_backend"])
        self.assertEqual(response.json()["workspace_modes"], ["persistent_pod"])

    async def test_credentials_are_required_and_never_echoed(self) -> None:
        blocked = await self.client.get("/api/v1/gpus")
        self.assertEqual(blocked.status_code, 401)
        self.assertEqual(blocked.json()["code"], "credentials_required")

        await self.connect()
        available = await self.client.get("/api/v1/gpus")
        self.assertEqual(available.status_code, 200)
        self.assertGreaterEqual(len(available.json()), 3)

    async def test_plan_respects_order_cloud_and_storage_cost(self) -> None:
        await self.connect()
        plan = await self.plan()
        self.assertEqual(plan["selected_gpu"]["id"], "NVIDIA RTX A6000")
        self.assertEqual(plan["estimated_storage_per_month"], 20.0)
        self.assertTrue(any("no RunPod" in item for item in plan["warnings"]))

    async def test_workspace_lifecycle_keeps_storage_cost_while_stopped(self) -> None:
        await self.connect()
        plan = await self.plan()
        created = await self.client.post(
            "/api/v1/workspaces",
            json={"plan_id": plan["id"], "name": "Portrait lab"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        workspace = created.json()
        self.assertEqual(workspace["state"], "ready")
        self.assertTrue(workspace["readiness"]["storage"])

        inventory = await self.client.get(
            f"/api/v1/workspaces/{workspace['id']}/files?kind=inputs"
        )
        self.assertEqual(inventory.status_code, 200)
        self.assertEqual(inventory.json(), {"items": [], "next_cursor": None})
        unavailable_upload = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/uploads",
            json={
                "filename": "test.bin",
                "destination_kind": "inputs",
                "size_bytes": 1024,
                "sha256": "0" * 64,
                "chunk_size_bytes": 1024,
            },
        )
        self.assertEqual(unavailable_upload.status_code, 501)
        self.assertEqual(
            unavailable_upload.json()["code"], "development_feature_unavailable"
        )
        provider_status = await self.client.get(
            "/api/v1/credentials/downloads/huggingface"
        )
        self.assertEqual(provider_status.status_code, 200)
        self.assertFalse(provider_status.json()["configured"])
        rejected_token = await self.client.post(
            "/api/v1/credentials/downloads/huggingface",
            json={"token": "hf_read_test_token"},
        )
        self.assertEqual(rejected_token.status_code, 501)
        rejected_download = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/downloads/civitai/preview",
            json={"source_url": "https://civitai.com/models/123"},
        )
        self.assertEqual(rejected_download.status_code, 501)
        rejected_job = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/jobs",
            json={
                "command_id": "dev-job",
                "kind": "generate",
                "project_id": "dev-project",
                "project": {
                    "schema": "k2-region-lab-project",
                    "version": 18,
                    "canvas": {"width": 1024, "height": 1024},
                },
            },
        )
        self.assertEqual(rejected_job.status_code, 501)

        stopped = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/stop"
        )
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.json()["state"], "stopped")
        cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(cost.json()["compute_per_hour"], 0.0)
        self.assertEqual(cost.json()["storage_per_month"], 20.0)

        started = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/start"
        )
        self.assertEqual(started.status_code, 200)
        self.assertEqual(started.json()["state"], "ready")

        rejected = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/terminate",
            json={"confirmation": "wrong name"},
        )
        self.assertEqual(rejected.status_code, 409)
        deleted = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/terminate",
            json={"confirmation": "Portrait lab"},
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["state"], "deleted")
        final_cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(final_cost.json()["storage_per_month"], 0.0)

    async def test_workspace_plan_is_single_use(self) -> None:
        await self.connect()
        plan = await self.plan()
        payload = {"plan_id": plan["id"], "name": "One workspace"}
        created = await self.client.post("/api/v1/workspaces", json=payload)
        self.assertEqual(created.status_code, 200)
        duplicate = await self.client.post("/api/v1/workspaces", json=payload)
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(duplicate.json()["code"], "workspace_plan_missing")


if __name__ == "__main__":
    unittest.main()
