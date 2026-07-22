from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import json
import os
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    import httpx
    from httpx import ASGITransport, AsyncClient

    from k2_region_lab.agent.app import AgentSettings, create_agent_app
    from k2_region_lab.agent.domain import FaceDetectionRequest, FileKind, JobSubmitRequest
    from k2_region_lab.agent.downloads import parse_civitai_url, parse_huggingface_url
    from k2_region_lab.agent.storage import LAYOUT_VERSION, WorkspaceLayout
    from k2_region_lab.agent.transfers import TransferError, TransferManager
    from k2_region_lab.project import project_state
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

    async def test_agent_rate_limit_and_security_headers(self) -> None:
        settings = AgentSettings(
            session_token=self.settings.session_token,
            workspace_id=self.settings.workspace_id,
            image_version=self.settings.image_version,
            workspace_root=self.root,
            worker_python=self.settings.worker_python,
            read_requests_per_minute=2,
        )
        app = create_agent_app(settings)
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            first = await client.get("/v1/health", headers=self.headers)
            second = await client.get("/v1/health", headers=self.headers)
            blocked = await client.get("/v1/health", headers=self.headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.headers["x-content-type-options"], "nosniff")
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["code"], "rate_limit_exceeded")
        self.assertIn("Retry-After", blocked.headers)

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

    async def test_generation_payload_disables_late_relaxation_exactly_like_desktop(
        self,
    ) -> None:
        document = self._project_document("portrait")
        document["generation"].update(
            {
                "regional_relaxation": False,
                "regional_late_step_scale": 0.2,
            }
        )
        request = JobSubmitRequest.model_validate(
            {
                "command_id": "relaxation-contract",
                "kind": "generate",
                "project_id": "relaxation-project",
                "project": document,
            }
        )
        payload = await self.app.state.job_manager._job_payload(
            "job-id", request, project_state(document), document
        )
        self.assertEqual(payload["regional_late_step_scale"], 1.0)

    async def test_generation_payload_resolves_selected_models_and_output_prefix(self) -> None:
        selections: dict[FileKind, str] = {}
        filenames = {
            FileKind.DIFFUSION_MODELS: "chosen-transformer.safetensors",
            FileKind.TEXT_ENCODERS: "chosen-text.safetensors",
            FileKind.VAE: "chosen-vae.safetensors",
            FileKind.FACE_DETECTION: "chosen-detector.onnx",
        }
        for kind, filename in filenames.items():
            path = self.app.state.layout.destination(kind.value) / filename
            path.write_bytes(b"selected model")
            record = await self.app.state.transfer_manager.index_existing_file(kind, path)
            selections[kind] = record.id
        document = self._project_document("portrait")
        request = JobSubmitRequest.model_validate(
            {
                "command_id": "model-selection-contract",
                "kind": "generate",
                "project_id": "model-selection-project",
                "project": document,
                "diffusion_model_file_id": selections[FileKind.DIFFUSION_MODELS],
                "text_encoder_file_id": selections[FileKind.TEXT_ENCODERS],
                "vae_file_id": selections[FileKind.VAE],
                "face_detector_file_id": selections[FileKind.FACE_DETECTION],
                "filename_prefix": "portrait study",
            }
        )
        payload = await self.app.state.job_manager._job_payload(
            "job-id", request, project_state(document), document
        )
        self.assertEqual(payload["filename_prefix"], "portrait study")
        self.assertEqual(Path(payload["diffusion_model_file"]).name, filenames[FileKind.DIFFUSION_MODELS])
        self.assertEqual(Path(payload["text_encoder_file"]).name, filenames[FileKind.TEXT_ENCODERS])
        self.assertEqual(Path(payload["vae_file"]).name, filenames[FileKind.VAE])
        self.assertEqual(Path(payload["face_detector_path"]).name, filenames[FileKind.FACE_DETECTION])

        invalid = request.model_copy(update={"filename_prefix": "../escape"})
        with self.assertRaisesRegex(Exception, "filename"):
            await self.app.state.job_manager._job_payload(
                "job-id", invalid, project_state(document), document
            )

    async def test_cloud_project_save_overwrites_and_content_is_safely_readable(self) -> None:
        document = self._project_document("first prompt")
        saved = await asyncio.wait_for(
            self.client.put(
                "/v1/projects/portrait.k2lab.json",
                headers=self.headers,
                json={"project": document},
            ),
            timeout=2,
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        file_id = saved.json()["id"]
        content = await self.client.get(
            f"/v1/files/{file_id}/content", headers=self.headers
        )
        self.assertEqual(content.status_code, 200, content.text)
        self.assertEqual(content.json()["generation"]["global_prompt"], "first prompt")

        document["generation"]["global_prompt"] = "updated prompt"
        overwritten = await self.client.put(
            "/v1/projects/portrait.k2lab.json",
            headers=self.headers,
            json={"project": document},
        )
        latest = await self.client.get(
            f"/v1/files/{overwritten.json()['id']}/content", headers=self.headers
        )
        self.assertEqual(latest.json()["generation"]["global_prompt"], "updated prompt")

        model = self.root / "models" / "loras" / "private.safetensors"
        model.write_bytes(b"private model")
        model_record = await self.app.state.transfer_manager.index_existing_file(
            FileKind.LORAS, model
        )
        blocked = await self.client.get(
            f"/v1/files/{model_record.id}/content", headers=self.headers
        )
        self.assertEqual(blocked.status_code, 404)

    async def test_face_detection_indexes_boxes_and_uses_opaque_input(self) -> None:
        observed: dict[str, object] = {}

        async def runner(command: list[str], environment: dict[str, str]):
            observed["command"] = command
            observed["environment"] = environment
            return (
                0,
                json.dumps(
                    {
                        "width": 768,
                        "height": 1024,
                        "execution_provider": "CPUExecutionProvider",
                        "faces": [
                            {"box": [12.0, 20.0, 112.0, 150.0], "score": 0.93},
                            {"box": [300.0, 40.0, 390.0, 160.0], "score": 0.88},
                        ],
                    }
                ).encode(),
                b"",
            )

        source = self.root / "inputs" / "portrait.png"
        source.write_bytes(b"test image placeholder")
        app = create_agent_app(self.settings, face_detection_runner=runner)
        app.state.layout.initialize()
        record = await app.state.transfer_manager.index_existing_file(FileKind.INPUTS, source)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            response = await client.post(
                "/v1/faces/detect",
                headers=self.headers,
                json={
                    "input_file_id": record.id,
                    "threshold": 0.2,
                    "provider": "cpu",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([face["index"] for face in response.json()["faces"]], [0, 1])
        command = observed["command"]
        self.assertIn(str(source), command)
        self.assertNotIn(record.id, command)

    async def test_agent_client_proxies_face_detection_contract(self) -> None:
        observed: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["path"] = request.url.path
            observed["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "width": 512,
                    "height": 512,
                    "execution_provider": "CPUExecutionProvider",
                    "faces": [{"index": 0, "box": [1, 2, 30, 40], "score": 0.9}],
                },
            )

        client = WorkspaceAgentClient(
            "pod-123",
            self.settings.session_token,
            transport=httpx.MockTransport(handler),
        )
        result = await client.detect_faces(
            FaceDetectionRequest(input_file_id="opaque-input", threshold=0.25, provider="cpu")
        )
        self.assertEqual(observed["path"], "/v1/faces/detect")
        self.assertEqual(observed["body"]["input_file_id"], "opaque-input")
        self.assertEqual(result.faces[0].index, 0)

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

    async def test_workspace_manifest_is_allowlisted_deterministic_and_copyable(self) -> None:
        project = self.root / "projects" / "portrait.json"
        project.write_bytes(b'{"project":1}')
        model = self.root / "models" / "loras" / "portrait.safetensors"
        model.write_bytes(b"model-weights")
        (self.root / "cache" / "huggingface" / "credential-like-cache").write_text(
            "excluded", encoding="utf-8"
        )
        (self.root / "downloads" / "incomplete" / "partial.bin").write_bytes(b"excluded")

        sealed = await self.client.post("/v1/migrations/seal", headers=self.headers)
        self.assertEqual(sealed.status_code, 200, sealed.text)
        source_manifest = sealed.json()
        paths = {item["path"] for item in source_manifest["files"]}
        self.assertIn("projects/portrait.json", paths)
        self.assertIn("models/loras/portrait.safetensors", paths)
        self.assertIn("state/layout.json", paths)
        self.assertFalse(any(path.startswith("cache/") for path in paths))
        self.assertFalse(any(path.startswith("downloads/incomplete/") for path in paths))
        blocked = await self.client.post(
            "/v1/uploads",
            headers=self.headers,
            json={
                "filename": "blocked.bin",
                "destination_kind": "inputs",
                "size_bytes": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
                "chunk_size_bytes": 1024,
            },
        )
        self.assertEqual(blocked.status_code, 423)
        self.assertEqual(blocked.json()["code"], "workspace_sealed_for_migration")

        target_root = Path(self.temporary_directory.name) / "target-k2lab"
        target_settings = AgentSettings(
            session_token=self.settings.session_token,
            workspace_id=self.settings.workspace_id,
            image_version=self.settings.image_version,
            workspace_root=target_root,
            worker_python=self.settings.worker_python,
        )
        target_app = create_agent_app(target_settings)
        target_app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=target_app), base_url="http://target-agent.test"
        ) as target:
            target_sealed = await target.post("/v1/migrations/seal", headers=self.headers)
            self.assertEqual(target_sealed.status_code, 200, target_sealed.text)
            for entry in source_manifest["files"]:
                source = await self.client.get(
                    f"/v1/migrations/files/{entry['path']}",
                    params={"generation": source_manifest["generation"]},
                    headers={**self.headers, "Range": f"bytes=0-{entry['size_bytes'] - 1}"},
                )
                self.assertEqual(source.status_code, 206, source.text)
                content = source.content
                split = max(1, len(content) // 2)
                chunks = [content[:split], content[split:]]
                offset = 0
                for index, chunk in enumerate(chunks):
                    if not chunk:
                        continue
                    headers = {
                        **self.headers,
                        "X-Migration-ID": "migration123",
                        "X-File-Offset": str(offset),
                        "X-File-Size": str(len(content)),
                        "X-File-SHA256": entry["sha256"],
                        "X-Chunk-SHA256": hashlib.sha256(chunk).hexdigest(),
                    }
                    imported = await target.put(
                        f"/v1/migrations/files/{entry['path']}",
                        headers=headers,
                        content=chunk,
                    )
                    self.assertEqual(imported.status_code, 200, imported.text)
                    if index == 0 and len(chunks) > 1:
                        retried = await target.put(
                            f"/v1/migrations/files/{entry['path']}",
                            headers=headers,
                            content=chunk,
                        )
                        self.assertEqual(retried.status_code, 200, retried.text)
                    offset += len(chunk)
                retried_complete = await target.put(
                    f"/v1/migrations/files/{entry['path']}",
                    headers={
                        **self.headers,
                        "X-Migration-ID": "migration123",
                        "X-File-Offset": "0",
                        "X-File-Size": str(len(content)),
                        "X-File-SHA256": entry["sha256"],
                        "X-Chunk-SHA256": hashlib.sha256(content).hexdigest(),
                    },
                    content=content,
                )
                self.assertEqual(retried_complete.status_code, 200)
                self.assertTrue(retried_complete.json()["completed"])
            target_manifest = await target.post("/v1/migrations/manifests", headers=self.headers)
            self.assertEqual(target_manifest.status_code, 200, target_manifest.text)
            self.assertEqual(target_manifest.json()["root_sha256"], source_manifest["root_sha256"])
            self.assertEqual(target_manifest.json()["total_bytes"], source_manifest["total_bytes"])

        unsealed = await self.client.delete("/v1/migrations/seal", headers=self.headers)
        self.assertEqual(unsealed.status_code, 204)

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

    async def test_control_plane_output_proxy_forwards_range_and_auth_header(self) -> None:
        observed: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            observed["url"] = str(request.url)
            observed["authorization"] = request.headers["Authorization"]
            observed["range"] = request.headers["Range"]
            return httpx.Response(
                206,
                content=b"2345",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": "4",
                    "Content-Range": "bytes 2-5/10",
                    "Content-Type": "image/png",
                    "X-Agent-Internal": "must-not-be-forwarded",
                },
            )

        client = WorkspaceAgentClient(
            "pod-123",
            self.settings.session_token,
            transport=httpx.MockTransport(handler),
        )
        output = await client.output("opaque-output", range_header="bytes=2-5")

        self.assertEqual(output.status_code, 206)
        self.assertEqual(output.content, b"2345")
        self.assertEqual(output.headers["content-range"], "bytes 2-5/10")
        self.assertNotIn("x-agent-internal", output.headers)
        self.assertEqual(observed["range"], "bytes=2-5")
        self.assertEqual(observed["authorization"], f"Bearer {self.settings.session_token}")
        self.assertNotIn(self.settings.session_token, observed["url"])

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

        inventory = await self.client.get("/v1/files?kind=inputs", headers=self.headers)
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
        cancelled = await self.client.delete(f"/v1/uploads/{upload_id}", headers=self.headers)
        self.assertEqual(cancelled.status_code, 204)
        missing = await self.client.get(f"/v1/uploads/{upload_id}", headers=self.headers)
        self.assertEqual(missing.status_code, 404)

    async def test_output_download_supports_authentication_and_byte_ranges(self) -> None:
        content = b"0123456789"
        (self.root / "outputs" / "render image.bin").write_bytes(content)
        inventory = await self.client.get("/v1/files?kind=outputs", headers=self.headers)
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

    async def test_remote_url_parsers_reject_untrusted_hosts_and_embedded_tokens(self) -> None:
        civitai = parse_civitai_url("https://civitai.com/models/123/name?modelVersionId=456")
        self.assertEqual(civitai.model_id, "123")
        self.assertEqual(civitai.version_id, "456")
        huggingface = parse_huggingface_url(
            "https://huggingface.co/owner/repo/blob/revision/models/model.safetensors"
        )
        self.assertEqual(huggingface.repo_id, "owner/repo")
        self.assertEqual(huggingface.filename, "models/model.safetensors")

        for unsafe in (
            "http://civitai.com/models/123",
            "https://evil.example/models/123",
            "https://civitai.com/models/123?token=secret",
            "https://huggingface.co/owner/repo?api_key=secret",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(TransferError):
                if "huggingface" in unsafe:
                    parse_huggingface_url(unsafe)
                else:
                    parse_civitai_url(unsafe)

    async def test_civitai_download_uses_header_verifies_and_installs(self) -> None:
        payload = self._safetensors_payload()
        digest = hashlib.sha256(payload).hexdigest()
        observed: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            observed.append(request)
            if "/api/v1/model-versions/456" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "id": 456,
                        "name": "Version One",
                        "baseModel": "Krea",
                        "trainedWords": ["portrait"],
                        "model": {"id": 123, "name": "Portrait", "type": "LORA"},
                        "files": [
                            {
                                "id": 789,
                                "name": "portrait.safetensors",
                                "sizeKB": len(payload) / 1024,
                                "primary": True,
                                "downloadUrl": "https://civitai.com/api/download/models/456",
                                "hashes": {"SHA256": digest.upper()},
                                "pickleScanResult": "Success",
                                "virusScanResult": "Success",
                                "metadata": {"format": "SafeTensor"},
                            }
                        ],
                    },
                )
            if "/api/download/models/456" in request.url.path:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/octet-stream"},
                    content=payload,
                )
            return httpx.Response(404)

        app = create_agent_app(self.settings, download_transport=httpx.MockTransport(handler))
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            preview = await client.post(
                "/v1/downloads/civitai/preview",
                headers={**self.headers, "X-Provider-Token": "secret-download-token"},
                json={"source_url": "https://civitai.com/models/123?modelVersionId=456"},
            )
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(preview.json()["files"][0]["sha256"], digest)
            started = await client.post(
                "/v1/downloads/civitai",
                headers={**self.headers, "X-Provider-Token": "secret-download-token"},
                json={
                    "source_url": "https://civitai.com/models/123?modelVersionId=456",
                    "file_id": "789",
                    "destination_kind": "loras",
                },
            )
            self.assertEqual(started.status_code, 202, started.text)
            transfer = await self._wait_for_transfer(client, started.json()["id"])
            self.assertEqual(transfer["state"], "completed", transfer)
            self.assertNotIn("secret-download-token", json.dumps(transfer))
            self.assertEqual(
                (self.root / "models" / "loras" / "portrait.safetensors").read_bytes(),
                payload,
            )
        self.assertTrue(observed)
        self.assertTrue(
            all(
                request.headers.get("Authorization") == "Bearer secret-download-token"
                for request in observed
            )
        )
        self.assertTrue(
            all("secret-download-token" not in str(request.url) for request in observed)
        )

    async def test_huggingface_file_download_uses_cache_metadata_and_nested_destination(
        self,
    ) -> None:
        payload = self._safetensors_payload()
        cached = self.root / "cache" / "huggingface" / "downloaded.safetensors"
        captured: dict[str, object] = {}

        def repo_info(**kwargs):
            captured["metadata_token"] = kwargs["token"]
            return SimpleNamespace(
                siblings=[SimpleNamespace(rfilename="nested/model.safetensors", size=len(payload))]
            )

        def file_download(**kwargs):
            captured["download_token"] = kwargs["token"]
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
            return str(cached)

        app = create_agent_app(
            self.settings,
            hf_repo_info=repo_info,
            hf_file_download=file_download,
            hf_snapshot_download=lambda **_kwargs: "unused",
        )
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            preview = await client.post(
                "/v1/downloads/huggingface/preview",
                headers={**self.headers, "X-Provider-Token": "hf_read_secret"},
                json={
                    "source_url": (
                        "https://huggingface.co/owner/repo/resolve/main/nested/model.safetensors"
                    )
                },
            )
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(preview.json()["required_bytes"], len(payload))
            started = await client.post(
                "/v1/downloads/huggingface",
                headers={**self.headers, "X-Provider-Token": "hf_read_secret"},
                json={
                    "source_url": (
                        "https://huggingface.co/owner/repo/resolve/main/nested/model.safetensors"
                    ),
                    "destination_kind": "diffusion_models",
                },
            )
            transfer = await self._wait_for_transfer(client, started.json()["id"])
            self.assertEqual(transfer["state"], "completed", transfer)
            inventory = await client.get("/v1/files?kind=diffusion_models", headers=self.headers)
            self.assertEqual(
                inventory.json()["items"][0]["display_name"],
                "nested/model.safetensors",
            )
        self.assertEqual(captured["metadata_token"], "hf_read_secret")
        self.assertEqual(captured["download_token"], "hf_read_secret")

    async def test_civitai_download_resumes_with_http_range_after_disconnect(self) -> None:
        payload = self._safetensors_payload() + b"x" * (2 * 1024 * 1024)
        split = 1024 * 1024
        digest = hashlib.sha256(payload).hexdigest()
        downloads = 0

        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield payload[:split]
                raise httpx.ReadError("simulated disconnect")

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal downloads
            if "/api/v1/model-versions/456" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "id": 456,
                        "name": "Resume",
                        "model": {"id": 123, "name": "Resume", "type": "LORA"},
                        "files": [
                            {
                                "id": 789,
                                "name": "resume.safetensors",
                                "sizeKB": len(payload) / 1024,
                                "downloadUrl": "https://civitai.com/api/download/models/456",
                                "hashes": {"SHA256": digest},
                            }
                        ],
                    },
                )
            downloads += 1
            if downloads == 1:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/octet-stream"},
                    stream=BrokenStream(),
                )
            self.assertEqual(request.headers["Range"], f"bytes={split}-")
            return httpx.Response(
                206,
                headers={"Content-Type": "application/octet-stream"},
                content=payload[split:],
            )

        app = create_agent_app(self.settings, download_transport=httpx.MockTransport(handler))
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            request = {
                "source_url": "https://civitai.com/api/download/models/456",
                "file_id": "789",
                "destination_kind": "loras",
            }
            started = await client.post("/v1/downloads/civitai", headers=self.headers, json=request)
            failed = await self._wait_for_transfer(client, started.json()["id"])
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["bytes_complete"], split)
            resumed = await client.post(
                "/v1/downloads/civitai",
                headers=self.headers,
                json={**request, "resume_transfer_id": started.json()["id"]},
            )
            completed = await self._wait_for_transfer(client, resumed.json()["id"])
            self.assertEqual(completed["state"], "completed", completed)
        self.assertEqual(downloads, 2)

    async def test_huggingface_repository_mirror_preserves_safe_relative_paths(self) -> None:
        payload = self._safetensors_payload()
        snapshot_root = self.root / "cache" / "mock-snapshot"

        def repo_info(**_kwargs):
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="nested/model.safetensors", size=len(payload)),
                    SimpleNamespace(rfilename="config.json", size=2),
                ]
            )

        def snapshot_download(**kwargs):
            self.assertEqual(kwargs["allow_patterns"], ["*.safetensors", "*.json"])
            (snapshot_root / "nested").mkdir(parents=True, exist_ok=True)
            (snapshot_root / "nested" / "model.safetensors").write_bytes(payload)
            (snapshot_root / "config.json").write_text("{}", encoding="utf-8")
            return str(snapshot_root)

        app = create_agent_app(
            self.settings,
            hf_repo_info=repo_info,
            hf_file_download=lambda **_kwargs: "unused",
            hf_snapshot_download=snapshot_download,
        )
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            preview = await client.post(
                "/v1/downloads/huggingface/preview",
                headers=self.headers,
                json={
                    "source_url": "https://huggingface.co/owner/repo",
                    "allow_patterns": ["*.safetensors", "*.json"],
                },
            )
            self.assertTrue(preview.json()["mirror_repository"])
            started = await client.post(
                "/v1/downloads/huggingface",
                headers=self.headers,
                json={
                    "source_url": "https://huggingface.co/owner/repo",
                    "destination_kind": "diffusion_models",
                    "allow_patterns": ["*.safetensors", "*.json"],
                },
            )
            completed = await self._wait_for_transfer(client, started.json()["id"])
            self.assertEqual(completed["state"], "completed", completed)
            inventory = await client.get("/v1/files?kind=diffusion_models", headers=self.headers)
            self.assertEqual(
                [item["display_name"] for item in inventory.json()["items"]],
                ["config.json", "nested/model.safetensors"],
            )

    async def test_remote_job_is_idempotent_streams_events_and_indexes_output(self) -> None:
        observed_commands: list[dict[str, object]] = []
        executions = 0

        class FakeExecutor:
            async def run(self, commands, on_event):
                nonlocal executions
                executions += 1
                observed_commands.extend(commands)
                target = commands[-1]
                output = Path(target["payload"]["output_directory"]) / "result.png"
                output.write_bytes(b"fake-png")
                await on_event(
                    {
                        "command_id": target["command_id"],
                        "state": "running",
                        "message": "Denoising step 1/2",
                        "payload": {
                            "step": 1,
                            "total_steps": 2,
                            "prompt": "must not reach event storage",
                        },
                    }
                )
                await on_event(
                    {
                        "command_id": target["command_id"],
                        "state": "running",
                        "message": "Preparing decoder",
                        "payload": {},
                    }
                )
                await on_event(
                    {
                        "command_id": target["command_id"],
                        "state": "ready",
                        "message": "Generation complete",
                        "payload": {"image_path": str(output)},
                    }
                )
                return 0

            async def cancel(self):
                return None

        app = create_agent_app(self.settings, job_executor_factory=lambda: FakeExecutor())
        app.state.layout.initialize()
        request = {
            "command_id": "browser-command-1",
            "kind": "generate",
            "project_id": "portrait-project",
            "project": self._project_document("private portrait prompt"),
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            unauthorized = await client.post("/v1/jobs", json=request)
            self.assertEqual(unauthorized.status_code, 401)
            submitted = await client.post("/v1/jobs", headers=self.headers, json=request)
            self.assertEqual(submitted.status_code, 202, submitted.text)
            duplicate = await client.post("/v1/jobs", headers=self.headers, json=request)
            self.assertEqual(duplicate.json()["id"], submitted.json()["id"])
            job = await self._wait_for_job(client, submitted.json()["id"])
            self.assertEqual(job["state"], "completed", job)
            self.assertEqual(job["progress_current"], 1)
            self.assertEqual(job["progress_total"], 2)
            self.assertEqual(len(job["output_file_ids"]), 1)
            page = await client.get(f"/v1/jobs/{job['id']}/events", headers=self.headers)
            self.assertEqual(page.status_code, 200, page.text)
            self.assertNotIn("private portrait prompt", page.text)
            self.assertNotIn(str(self.root), page.text)
            cursor = page.json()["next_cursor"]
            empty = await client.get(
                f"/v1/jobs/{job['id']}/events?cursor={cursor}", headers=self.headers
            )
            self.assertEqual(empty.json()["items"], [])
            inventory = await client.get("/v1/files?kind=outputs", headers=self.headers)
            self.assertEqual(inventory.json()["items"][0]["id"], job["output_file_ids"][0])
        self.assertEqual(executions, 1)
        self.assertEqual(
            [command["kind"] for command in observed_commands],
            ["probe", "load_model", "generate_baseline"],
        )
        saved_project = json.loads(
            (self.root / "projects" / "portrait-project.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_project["schema"], "k2-region-lab-project")

    async def test_remote_job_cancellation_stops_isolated_executor(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingExecutor:
            async def run(self, commands, on_event):
                await on_event(
                    {
                        "command_id": commands[-1]["command_id"],
                        "state": "running",
                        "message": "Generation started",
                        "payload": {},
                    }
                )
                started.set()
                await cancelled.wait()
                return -15

            async def cancel(self):
                cancelled.set()

        app = create_agent_app(self.settings, job_executor_factory=lambda: BlockingExecutor())
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            submitted = await client.post(
                "/v1/jobs",
                headers=self.headers,
                json={
                    "command_id": "cancel-command",
                    "kind": "generate",
                    "project_id": "cancel-project",
                    "project": self._project_document("cancel me"),
                },
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            response = await client.post(
                f"/v1/jobs/{submitted.json()['id']}/cancel", headers=self.headers
            )
            self.assertEqual(response.json()["state"], "cancelled")
            self.assertTrue(cancelled.is_set())
            await asyncio.sleep(0)
            status = await client.get(f"/v1/jobs/{submitted.json()['id']}", headers=self.headers)
            self.assertEqual(status.json()["state"], "cancelled")

    async def test_worker_memory_release_cancels_active_executor(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingExecutor:
            async def run(self, commands, on_event):
                await on_event(
                    {
                        "command_id": commands[-1]["command_id"],
                        "state": "running",
                        "message": "Generation started",
                        "payload": {},
                    }
                )
                started.set()
                await cancelled.wait()
                return -15

            async def cancel(self):
                cancelled.set()

        app = create_agent_app(self.settings, job_executor_factory=lambda: BlockingExecutor())
        app.state.layout.initialize()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            submitted = await client.post(
                "/v1/jobs",
                headers=self.headers,
                json={
                    "command_id": "release-command",
                    "kind": "generate",
                    "project_id": "release-project",
                    "project": self._project_document("release me"),
                },
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            response = await client.post("/v1/worker/release", headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["released"])
            self.assertEqual(response.json()["cancelled_job_ids"], [submitted.json()["id"]])
            self.assertTrue(cancelled.is_set())
            status = await client.get(f"/v1/jobs/{submitted.json()['id']}", headers=self.headers)
            self.assertEqual(status.json()["state"], "cancelled")

    async def test_remote_job_reports_missing_worker_runtime(self) -> None:
        submitted = await self.client.post(
            "/v1/jobs",
            headers=self.headers,
            json={
                "command_id": "missing-worker-command",
                "kind": "generate",
                "project_id": "missing-worker-project",
                "project": self._project_document("safe prompt"),
            },
        )
        self.assertEqual(submitted.status_code, 202, submitted.text)
        job = await self._wait_for_job(self.client, submitted.json()["id"])
        self.assertEqual(job["state"], "failed")
        self.assertEqual(job["error_code"], "worker_unavailable")

    async def _wait_for_transfer(self, client: AsyncClient, transfer_id: str) -> dict[str, object]:
        for _attempt in range(100):
            response = await client.get(f"/v1/transfers/{transfer_id}", headers=self.headers)
            body = response.json()
            if body["state"] in {"completed", "failed", "cancelled", "paused"}:
                return body
            await asyncio.sleep(0.01)
        self.fail("transfer did not reach a terminal state")

    async def _wait_for_job(self, client: AsyncClient, job_id: str) -> dict[str, object]:
        for _attempt in range(100):
            response = await client.get(f"/v1/jobs/{job_id}", headers=self.headers)
            body = response.json()
            if body["state"] in {"completed", "failed", "cancelled"}:
                return body
            await asyncio.sleep(0.01)
        self.fail("job did not reach a terminal state")

    @staticmethod
    def _project_document(prompt: str) -> dict[str, object]:
        return {
            "schema": "k2-region-lab-project",
            "version": 18,
            "canvas": {"width": 1024, "height": 1024},
            "generation": {
                "global_prompt": prompt,
                "steps": 8,
                "sampler": "euler",
                "scheduler": "simple",
                "seed": 42,
            },
            "regions": [],
            "loras": [],
            "image_edit": {},
            "runtime": {},
        }

    @staticmethod
    def _safetensors_payload() -> bytes:
        header = json.dumps(
            {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        ).encode("utf-8")
        return struct.pack("<Q", len(header)) + header + b"\0\0\0\0"


if __name__ == "__main__":
    unittest.main()
