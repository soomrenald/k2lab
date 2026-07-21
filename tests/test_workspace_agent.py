from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    import httpx
    from httpx import ASGITransport, AsyncClient

    from k2_region_lab.agent.app import AgentSettings, create_agent_app
    from k2_region_lab.agent.storage import LAYOUT_VERSION, WorkspaceLayout
    from k2_region_lab.web.agent_client import WorkspaceAgentClient


@unittest.skipUnless(FASTAPI_AVAILABLE, "web dependencies are not installed")
class WorkspaceAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "k2lab"
        self.settings = AgentSettings(
            session_token="a" * 43,
            workspace_id="workspace-123",
            image_version="0.1.0-test",
            workspace_root=self.root,
            worker_python=Path("/unavailable/worker-python"),
            cuda_version="12.8",
            pytorch_version="2.9.1",
        )
        self.app = create_agent_app(self.settings)
        self.app.state.layout.initialize()
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://agent.test",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.temporary_directory.cleanup()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.session_token}"}

    async def test_every_agent_endpoint_requires_bearer_authentication(self) -> None:
        for path in ("/v1/health", "/v1/capabilities", "/v1/storage"):
            response = await self.client.get(path)
            self.assertEqual(response.status_code, 401)
            self.assertNotIn(self.settings.session_token, response.text)

            accepted = await self.client.get(path, headers=self.headers)
            self.assertEqual(accepted.status_code, 200, accepted.text)

    async def test_health_reports_identity_and_staged_readiness(self) -> None:
        response = await self.client.get("/v1/health", headers=self.headers)
        body = response.json()
        self.assertEqual(body["workspace_id"], "workspace-123")
        self.assertEqual(body["image_version"], "0.1.0-test")
        self.assertTrue(body["readiness"]["container"])
        self.assertTrue(body["readiness"]["storage"])
        self.assertFalse(body["readiness"]["models"])
        self.assertFalse(body["readiness"]["worker"])
        self.assertEqual(body["status"], "ready")

    async def test_capabilities_are_versioned(self) -> None:
        response = await self.client.get("/v1/capabilities", headers=self.headers)
        body = response.json()
        self.assertEqual(body["api_version"], "v1")
        self.assertEqual(body["project_schema"], "k2-region-lab-project")
        self.assertEqual(body["project_schema_version"], 18)
        self.assertEqual(body["workspace_layout_version"], LAYOUT_VERSION)
        self.assertEqual(body["cuda_version"], "12.8")
        self.assertEqual(body["pytorch_version"], "2.9.1")

    async def test_layout_is_idempotent_and_marks_its_version(self) -> None:
        layout = WorkspaceLayout(self.root)
        layout.initialize()
        layout.initialize()
        marker = json.loads(layout.marker_path.read_text(encoding="utf-8"))
        self.assertEqual(marker, {"layout_version": LAYOUT_VERSION})
        self.assertTrue((self.root / "downloads" / "incomplete").is_dir())
        self.assertTrue((self.root / "models" / "face_detection").is_dir())

    async def test_path_resolution_rejects_traversal_absolute_and_symlink(self) -> None:
        layout = WorkspaceLayout(self.root)
        for unsafe in ("../escape", "/etc/passwd", "nested/file", ".."):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                layout.resolve_child("inputs", unsafe)

        target = self.root / "inputs" / "target.png"
        target.write_bytes(b"image")
        link = self.root / "inputs" / "link.png"
        os.symlink(target, link)
        with self.assertRaises(ValueError):
            layout.resolve_child("inputs", "link.png")

    async def test_storage_response_does_not_expose_host_test_path(self) -> None:
        response = await self.client.get("/v1/storage", headers=self.headers)
        body = response.json()
        self.assertEqual(body["root"], "/workspace/k2lab")
        self.assertNotIn(self.temporary_directory.name, response.text)
        self.assertGreater(body["free_bytes"], 0)

    async def test_control_plane_agent_client_keeps_token_out_of_url(self) -> None:
        observed: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["url"] = str(request.url)
            observed["authorization"] = request.headers["Authorization"]
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "workspace_id": "workspace-123",
                    "image_version": "0.1.0-test",
                    "readiness": {
                        "container": True,
                        "agent": True,
                        "storage": True,
                        "models": False,
                        "worker": False,
                    },
                    "observed_at": "2026-07-20T12:00:00Z",
                },
            )

        client = WorkspaceAgentClient(
            "pod-123",
            self.settings.session_token,
            transport=httpx.MockTransport(handler),
        )
        health = await client.health()
        self.assertEqual(health.workspace_id, "workspace-123")
        self.assertNotIn(self.settings.session_token, observed["url"])
        self.assertEqual(
            observed["authorization"],
            f"Bearer {self.settings.session_token}",
        )


if __name__ == "__main__":
    unittest.main()
