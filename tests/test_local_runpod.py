from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

try:
    from httpx import ASGITransport, AsyncClient

    from k2_region_lab.web.app import create_app
    from k2_region_lab.web.development_backend import DevelopmentWorkspaceBackend
    from k2_region_lab.web.local_runpod import (
        prepare_local_environment,
        validate_image_digest,
    )
    from k2_region_lab.web.security import ControlPlaneSecuritySettings

    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


IMAGE = f"ghcr.io/example/k2-region-lab-workspace@sha256:{'a' * 64}"


@unittest.skipUnless(WEB_AVAILABLE, "web dependencies are not installed")
class LocalRunPodConfigurationTests(unittest.TestCase):
    def test_first_run_creates_private_durable_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_directory = Path(temporary) / "state"
            environment, changed = prepare_local_environment(
                state_directory=state_directory,
                image_digest=IMAGE,
                image_version="1.2.3",
                port=8123,
                environment={},
                interactive=False,
            )

            self.assertTrue(changed)
            self.assertEqual(environment["K2LAB_WEB_BACKEND"], "runpod")
            self.assertEqual(environment["K2LAB_LOCAL_PORT"], "8123")
            self.assertEqual(
                environment["K2LAB_DATABASE_URL"],
                f"sqlite+aiosqlite:///{(state_directory / 'state.sqlite3').as_posix()}",
            )
            self.assertEqual(
                stat.S_IMODE(state_directory.stat().st_mode),
                0o700,
            )
            for filename in ("config.json", "credential.key"):
                self.assertEqual(
                    stat.S_IMODE((state_directory / filename).stat().st_mode),
                    0o600,
                )

    def test_next_run_reuses_image_and_credential_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_directory = Path(temporary) / "state"
            first, _changed = prepare_local_environment(
                state_directory=state_directory,
                image_digest=IMAGE,
                image_version=None,
                port=8000,
                environment={},
                interactive=False,
            )
            second, changed = prepare_local_environment(
                state_directory=state_directory,
                image_digest=None,
                image_version=None,
                port=8000,
                environment={},
                interactive=False,
            )

            self.assertFalse(changed)
            self.assertEqual(
                first["K2LAB_CREDENTIAL_FERNET_KEY"],
                second["K2LAB_CREDENTIAL_FERNET_KEY"],
            )
            stored = json.loads((state_directory / "config.json").read_text())
            self.assertEqual(stored["workspace_image_digest"], IMAGE)

    def test_requires_an_immutable_image_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_image_digest("ghcr.io/example/workspace:latest")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Pass --image"):
                prepare_local_environment(
                    state_directory=Path(temporary) / "state",
                    image_digest=None,
                    image_version=None,
                    port=8000,
                    environment={},
                    interactive=False,
                )


@unittest.skipUnless(WEB_AVAILABLE, "web dependencies are not installed")
class LocalRunPodSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_mode_accepts_only_loopback_same_origin_mutations(self) -> None:
        security = ControlPlaneSecuritySettings.local_single_user(port=8000)
        application = create_app(DevelopmentWorkspaceBackend(), security=security)
        async with AsyncClient(
            transport=ASGITransport(app=application, client=("127.0.0.1", 12345)),
            base_url="http://127.0.0.1:8000",
        ) as client:
            accepted = await client.post(
                "/api/v1/credentials/runpod",
                headers={"Origin": "http://127.0.0.1:8000"},
                json={"api_key": "development-key"},
            )
            self.assertEqual(accepted.status_code, 200, accepted.text)

            rejected = await client.post(
                "/api/v1/credentials/runpod",
                headers={"Origin": "https://malicious.example"},
                json={"api_key": "development-key"},
            )
            self.assertEqual(rejected.status_code, 403)
            self.assertEqual(rejected.json()["code"], "origin_forbidden")

    async def test_local_mode_rejects_remote_clients_and_dns_rebinding_hosts(self) -> None:
        security = ControlPlaneSecuritySettings.local_single_user(port=8000)
        application = create_app(DevelopmentWorkspaceBackend(), security=security)
        async with AsyncClient(
            transport=ASGITransport(app=application, client=("198.51.100.2", 12345)),
            base_url="http://127.0.0.1:8000",
        ) as remote:
            response = await remote.get("/api/v1/health")
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["code"], "loopback_required")

        async with AsyncClient(
            transport=ASGITransport(app=application, client=("127.0.0.1", 12345)),
            base_url="http://attacker.example:8000",
        ) as rebound:
            response = await rebound.get("/api/v1/health")
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["code"], "loopback_required")

    async def test_bundled_interface_can_be_served_by_the_control_plane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            static_directory = Path(temporary)
            (static_directory / "index.html").write_text("<h1>K2 local</h1>", encoding="utf-8")
            application = create_app(
                DevelopmentWorkspaceBackend(),
                security=ControlPlaneSecuritySettings.local_single_user(port=8000),
                static_directory=static_directory,
            )
            async with AsyncClient(
                transport=ASGITransport(app=application, client=("127.0.0.1", 12345)),
                base_url="http://127.0.0.1:8000",
            ) as client:
                response = await client.get("/")
                self.assertEqual(response.status_code, 200)
                self.assertIn("K2 local", response.text)
                self.assertEqual(response.headers["x-frame-options"], "DENY")


if __name__ == "__main__":
    unittest.main()
