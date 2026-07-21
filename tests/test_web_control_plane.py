from __future__ import annotations

import importlib.util
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from httpx import ASGITransport, AsyncClient

    from k2_region_lab.web.app import create_app
    from k2_region_lab.web.development_backend import DevelopmentWorkspaceBackend
    from k2_region_lab.web.security import ControlPlaneSecuritySettings


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
        self.assertEqual(
            response.json()["workspace_modes"],
            ["persistent_pod", "portable_workspace"],
        )

    async def test_unified_prompt_preview_uses_project_order_roles_and_identity(self) -> None:
        project = {
            "schema": "k2-region-lab-project",
            "version": 18,
            "canvas": {"width": 1024, "height": 1024},
            "generation": {
                "global_prompt": "a studio scene",
                "regional_subject_fill": True,
                "regional_relaxation": False,
                "regional_late_step_scale": 0.2,
            },
            "regions": [
                {
                    "id": "person",
                    "name": "Person",
                    "box": {"x0": 300, "y0": 100, "x1": 700, "y1": 950},
                    "prompt": "a smiling woman",
                    "face_identity_prompt": "brown hair and green eyes",
                    "enabled": True,
                    "priority": 2,
                    "spatial_role": "subject",
                },
                {
                    "id": "wall",
                    "name": "Back wall",
                    "box": {"x0": 0, "y0": 0, "x1": 1024, "y1": 1024},
                    "prompt": "a brick wall",
                    "face_identity_prompt": "",
                    "enabled": True,
                    "priority": 1,
                    "spatial_role": "background",
                },
            ],
            "loras": [
                {
                    "path": "character.safetensors",
                    "global": False,
                    "region_ids": ["person"],
                    "strength": 1.0,
                    "routing_mode": "character_identity",
                    "trigger_phrase": "lface",
                }
            ],
            "image_edit": {},
        }
        response = await self.client.post(
            "/api/v1/projects/unified-prompt-preview", json={"project": project}
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            [(item["id"], item["spatial_role"]) for item in body["regions"]],
            [("person", "subject"), ("wall", "background")],
        )
        self.assertLess(
            body["prompt"].index("a smiling woman"), body["prompt"].index("a brick wall")
        )
        self.assertIn("brown hair and green eyes", body["prompt"])
        self.assertIn("lface", body["prompt"])

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

        inventory = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/files?kind=inputs")
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
        self.assertEqual(unavailable_upload.json()["code"], "development_feature_unavailable")
        provider_status = await self.client.get("/api/v1/credentials/downloads/huggingface")
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

        stopped = await self.client.post(f"/api/v1/workspaces/{workspace['id']}/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertEqual(stopped.json()["state"], "stopped")
        cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(cost.json()["compute_per_hour"], 0.0)
        self.assertEqual(cost.json()["storage_per_month"], 20.0)

        started = await self.client.post(f"/api/v1/workspaces/{workspace['id']}/start")
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

    async def test_portable_workspace_inventory_and_ephemeral_lifecycle(self) -> None:
        await self.connect()
        datacenters = await self.client.get("/api/v1/datacenters")
        volumes = await self.client.get("/api/v1/network-volumes")
        self.assertEqual(datacenters.status_code, 200)
        self.assertEqual(volumes.status_code, 200)
        self.assertEqual(volumes.json()[0]["datacenter_id"], "US-GA-2")

        planned = await self.client.post(
            "/api/v1/workspace-plans",
            json={
                "mode": "portable_workspace",
                "gpu_priority_ids": ["NVIDIA RTX A6000"],
                "cloud_type": "secure",
                "workspace_disk_gb": 100,
                "datacenter_priority_ids": ["US-GA-2"],
            },
        )
        self.assertEqual(planned.status_code, 200, planned.text)
        self.assertTrue(planned.json()["create_network_volume"])
        created = await self.client.post(
            "/api/v1/workspaces",
            json={"plan_id": planned.json()["id"], "name": "Portable preview"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        workspace = created.json()
        self.assertIsNotNone(workspace["network_volume_id"])

        stopped = await self.client.post(f"/api/v1/workspaces/{workspace['id']}/stop")
        self.assertIsNone(stopped.json()["provider_resource_id"])
        restarted = await self.client.post(f"/api/v1/workspaces/{workspace['id']}/start")
        self.assertIsNotNone(restarted.json()["provider_resource_id"])
        deleted = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/terminate",
            json={"confirmation": "Portable preview"},
        )
        self.assertEqual(deleted.status_code, 200)
        cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(cost.json()["storage_per_month"], 7.0)

    async def test_persistent_to_portable_migration_requires_verified_confirmation(self) -> None:
        await self.connect()
        plan = await self.plan()
        created = await self.client.post(
            "/api/v1/workspaces",
            json={"plan_id": plan["id"], "name": "Migration preview"},
        )
        workspace = created.json()
        started = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/migrations",
            json={"workspace_disk_gb": 200, "datacenter_priority_ids": ["US-GA-2"]},
        )
        self.assertEqual(started.status_code, 202, started.text)
        migration = started.json()
        migrating_cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(migrating_cost.json()["storage_per_month"], 34.0)
        listed = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/migrations")
        self.assertEqual(listed.json()[0]["id"], migration["id"])

        verified = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/migrations/{migration['id']}/resume"
        )
        self.assertEqual(verified.json()["state"], "awaiting_confirmation")
        switched = await self.client.get(f"/api/v1/workspaces/{workspace['id']}")
        self.assertEqual(switched.json()["mode"], "portable_workspace")
        self.assertIsNotNone(switched.json()["retained_original_provider_resource_id"])
        retained_cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(retained_cost.json()["storage_per_month"], 34.0)

        rejected = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/migrations/{migration['id']}/confirm",
            json={"confirmation": "wrong"},
        )
        self.assertEqual(rejected.status_code, 409)
        confirmed = await self.client.post(
            f"/api/v1/workspaces/{workspace['id']}/migrations/{migration['id']}/confirm",
            json={"confirmation": "Migration preview"},
        )
        self.assertEqual(confirmed.json()["state"], "completed")
        final_workspace = await self.client.get(f"/api/v1/workspaces/{workspace['id']}")
        self.assertIsNone(final_workspace.json()["retained_original_provider_resource_id"])
        final_cost = await self.client.get(f"/api/v1/workspaces/{workspace['id']}/cost")
        self.assertEqual(final_cost.json()["storage_per_month"], 14.0)


@unittest.skipUnless(FASTAPI_AVAILABLE, "web dependencies are not installed")
class HostedWebSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.origin = "https://studio.example"
        self.proxy_secret = "p" * 48
        self.security = ControlPlaneSecuritySettings(
            enabled=True,
            trusted_proxy_secret=self.proxy_secret,
            allowed_subject="user-123",
            allowed_origins=(self.origin,),
            provisioning_requests_per_minute=20,
        )
        self.client = AsyncClient(
            transport=ASGITransport(
                app=create_app(DevelopmentWorkspaceBackend(), security=self.security)
            ),
            base_url=self.origin,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def open_session(self) -> str:
        response = await self.client.post(
            "/api/v1/auth/session",
            headers={
                "X-K2-Proxy-Secret": self.proxy_secret,
                "X-K2-Authenticated-User": "user-123",
                "X-K2-Authenticated-MFA": "true",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["mfa_verified"])
        self.assertIn("HttpOnly", response.headers.get_list("set-cookie")[0])
        return self.client.cookies["k2lab-csrf"]

    async def test_hosted_session_requires_trusted_mfa_assertion(self) -> None:
        blocked = await self.client.get("/api/v1/capabilities")
        self.assertEqual(blocked.status_code, 401)
        self.assertEqual(blocked.json()["code"], "authentication_required")

        missing_assertion = await self.client.post("/api/v1/auth/session")
        self.assertEqual(missing_assertion.status_code, 401)

        missing_mfa = await self.client.post(
            "/api/v1/auth/session",
            headers={
                "X-K2-Proxy-Secret": self.proxy_secret,
                "X-K2-Authenticated-User": "user-123",
            },
        )
        self.assertEqual(missing_mfa.status_code, 403)
        self.assertEqual(missing_mfa.json()["code"], "mfa_required")

        await self.open_session()
        accepted = await self.client.get("/api/v1/capabilities")
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.headers["x-frame-options"], "DENY")
        self.assertEqual(accepted.headers["cache-control"], "no-store")

    async def test_hosted_mutations_require_origin_and_double_submit_csrf(self) -> None:
        csrf = await self.open_session()
        no_origin = await self.client.post(
            "/api/v1/credentials/runpod", json={"api_key": "development-key"}
        )
        self.assertEqual(no_origin.status_code, 403)
        self.assertEqual(no_origin.json()["code"], "origin_forbidden")

        no_csrf = await self.client.post(
            "/api/v1/credentials/runpod",
            headers={"Origin": self.origin},
            json={"api_key": "development-key"},
        )
        self.assertEqual(no_csrf.status_code, 403)
        self.assertEqual(no_csrf.json()["code"], "csrf_failed")

        connected = await self.client.post(
            "/api/v1/credentials/runpod",
            headers={"Origin": self.origin, "X-CSRF-Token": csrf},
            json={"api_key": "development-key"},
        )
        self.assertEqual(connected.status_code, 200, connected.text)

        signed_out = await self.client.delete(
            "/api/v1/auth/session",
            headers={"Origin": self.origin, "X-CSRF-Token": csrf},
        )
        self.assertEqual(signed_out.status_code, 204)
        blocked = await self.client.get("/api/v1/capabilities")
        self.assertEqual(blocked.status_code, 401)

    async def test_hosted_provisioning_rate_limit_is_stable(self) -> None:
        security = ControlPlaneSecuritySettings(
            enabled=True,
            trusted_proxy_secret=self.proxy_secret,
            allowed_subject="user-123",
            allowed_origins=(self.origin,),
            provisioning_requests_per_minute=1,
        )
        async with AsyncClient(
            transport=ASGITransport(
                app=create_app(DevelopmentWorkspaceBackend(), security=security)
            ),
            base_url=self.origin,
        ) as client:
            opened = await client.post(
                "/api/v1/auth/session",
                headers={
                    "X-K2-Proxy-Secret": self.proxy_secret,
                    "X-K2-Authenticated-User": "user-123",
                    "X-K2-Authenticated-MFA": "true",
                },
            )
            self.assertEqual(opened.status_code, 200)
            csrf = client.cookies["k2lab-csrf"]
            headers = {"Origin": self.origin, "X-CSRF-Token": csrf}
            first = await client.post("/api/v1/workspace-plans", headers=headers, json={})
            second = await client.post("/api/v1/workspace-plans", headers=headers, json={})
            self.assertEqual(first.status_code, 422)
            self.assertEqual(second.status_code, 429)
            self.assertEqual(second.json()["code"], "rate_limit_exceeded")
            self.assertIn("Retry-After", second.headers)


if __name__ == "__main__":
    unittest.main()
