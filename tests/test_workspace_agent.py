from __future__ import annotations

import importlib.util
import hashlib
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
    from k2_region_lab.agent.transfers import TransferManager
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

    async def test_chunked_upload_resumes_verifies_and_updates_inventory(self) -> None:
        content = bytes(range(256)) * 8 + b"xx"
        digest = hashlib.sha256(content).hexdigest()
        created = await self.client.post(
            "/v1/uploads",
            headers=self.headers,
            json={
                "filename": "portrait.bin",
                "destination_kind": "inputs",
                "size_bytes": len(content),
                "sha256": digest,
                "chunk_size_bytes": 1024,
            },
        )
        self.assertEqual(created.status_code, 201, created.text)
        upload = created.json()
        self.assertEqual(upload["chunk_count"], 3)

        rejected = await self.client.put(
            f"/v1/uploads/{upload['id']}/chunks/0",
            headers={**self.headers, "X-Chunk-SHA256": "0" * 64},
            content=content[:1024],
        )
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(rejected.json()["code"], "chunk_hash_mismatch")

        for index, chunk in ((1, content[1024:2048]), (0, content[:1024])):
            response = await self.client.put(
                f"/v1/uploads/{upload['id']}/chunks/{index}",
                headers={
                    **self.headers,
                    "X-Chunk-SHA256": hashlib.sha256(chunk).hexdigest(),
                },
                content=chunk,
            )
            self.assertEqual(response.status_code, 200, response.text)

        resumed = await TransferManager(WorkspaceLayout(self.root)).get_upload(upload["id"])
        self.assertEqual(resumed.completed_chunks, [0, 1])
        incomplete = await self.client.post(
            f"/v1/uploads/{upload['id']}/complete", headers=self.headers
        )
        self.assertEqual(incomplete.status_code, 409)

        final_chunk = content[2048:]
        accepted = await self.client.put(
            f"/v1/uploads/{upload['id']}/chunks/2",
            headers={
                **self.headers,
                "X-Chunk-SHA256": hashlib.sha256(final_chunk).hexdigest(),
            },
            content=final_chunk,
        )
        self.assertEqual(accepted.status_code, 200)
        completed = await self.client.post(
            f"/v1/uploads/{upload['id']}/complete", headers=self.headers
        )
        self.assertEqual(completed.status_code, 200, completed.text)
        file_record = completed.json()["file"]
        self.assertEqual(file_record["sha256"], digest)
        self.assertEqual((self.root / "inputs" / "portrait.bin").read_bytes(), content)

        inventory = await self.client.get(
            "/v1/files?kind=inputs", headers=self.headers
        )
        self.assertEqual(inventory.status_code, 200)
        self.assertEqual(inventory.json()["items"][0]["id"], file_record["id"])

    async def test_duplicate_upload_returns_existing_opaque_file(self) -> None:
        content = b"same-content" * 86
        content = content[:1024]
        digest = hashlib.sha256(content).hexdigest()
        (self.root / "inputs" / "first.bin").write_bytes(content)
        await self.client.get("/v1/files?kind=inputs", headers=self.headers)
        created = await self.client.post(
            "/v1/uploads",
            headers=self.headers,
            json={
                "filename": "second.bin",
                "destination_kind": "inputs",
                "size_bytes": len(content),
                "sha256": digest,
                "chunk_size_bytes": 1024,
            },
        )
        upload_id = created.json()["id"]
        await self.client.put(
            f"/v1/uploads/{upload_id}/chunks/0",
            headers={**self.headers, "X-Chunk-SHA256": digest},
            content=content,
        )
        completed = await self.client.post(
            f"/v1/uploads/{upload_id}/complete", headers=self.headers
        )
        self.assertTrue(completed.json()["duplicate"])
        self.assertEqual(completed.json()["file"]["display_name"], "first.bin")
        self.assertFalse((self.root / "inputs" / "second.bin").exists())

    async def test_upload_rejects_unsafe_name_and_can_be_cancelled(self) -> None:
        unsafe = await self.client.post(
            "/v1/uploads",
            headers=self.headers,
            json={
                "filename": "../escape.bin",
                "destination_kind": "inputs",
                "size_bytes": 1024,
                "sha256": hashlib.sha256(b"x" * 1024).hexdigest(),
                "chunk_size_bytes": 1024,
            },
        )
        self.assertEqual(unsafe.status_code, 400)
        self.assertEqual(unsafe.json()["code"], "unsafe_filename")

        created = await self.client.post(
            "/v1/uploads",
            headers=self.headers,
            json={
                "filename": "cancel.bin",
                "destination_kind": "inputs",
                "size_bytes": 1024,
                "sha256": hashlib.sha256(b"x" * 1024).hexdigest(),
                "chunk_size_bytes": 1024,
            },
        )
        upload_id = created.json()["id"]
        cancelled = await self.client.delete(
            f"/v1/uploads/{upload_id}", headers=self.headers
        )
        self.assertEqual(cancelled.status_code, 204)
        missing = await self.client.get(
            f"/v1/uploads/{upload_id}", headers=self.headers
        )
        self.assertEqual(missing.status_code, 404)

    async def test_output_download_supports_authentication_and_byte_ranges(self) -> None:
        content = b"0123456789"
        (self.root / "outputs" / "render image.bin").write_bytes(content)
        inventory = await self.client.get(
            "/v1/files?kind=outputs", headers=self.headers
        )
        file_id = inventory.json()["items"][0]["id"]

        unauthenticated = await self.client.get(f"/v1/outputs/{file_id}")
        self.assertEqual(unauthenticated.status_code, 401)

        partial = await self.client.get(
            f"/v1/outputs/{file_id}",
            headers={**self.headers, "Range": "bytes=2-5"},
        )
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(partial.content, b"2345")
        self.assertEqual(partial.headers["accept-ranges"], "bytes")
        self.assertEqual(partial.headers["content-range"], "bytes 2-5/10")
        self.assertIn("render%20image.bin", partial.headers["content-disposition"])

        invalid = await self.client.get(
            f"/v1/outputs/{file_id}",
            headers={**self.headers, "Range": "bytes=20-30"},
        )
        self.assertEqual(invalid.status_code, 416)
        self.assertEqual(invalid.json()["code"], "invalid_range")


if __name__ == "__main__":
    unittest.main()
