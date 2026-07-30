from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent.parent
FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "device"
    / "gate12_a40_100_job_soak.json"
)


class NativeReleaseFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_soak_uses_pinned_source_models_and_release_workload(self) -> None:
        self.assertEqual(
            self.fixture["schema_version"],
            "k2lab-gate12-release-evidence/1",
        )
        implementation = self.fixture["implementation"]
        self.assertIn(
            implementation["k2core_commit"],
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        self.assertEqual(len(implementation["runpod_commit"]), 40)
        self.assertEqual(self.fixture["hardware"]["gpu"], "NVIDIA A40")
        for digest in self.fixture["model_hashes"].values():
            self.assertEqual(len(digest), 64)

        scope = self.fixture["test_scope"]
        self.assertEqual(scope["generation_jobs"], 100)
        self.assertEqual((scope["width"], scope["height"], scope["steps"]), (512, 512, 8))
        self.assertEqual((scope["sampler"], scope["scheduler"]), ("euler", "simple"))

    def test_soak_passes_determinism_growth_and_cleanup_contracts(self) -> None:
        results = self.fixture["results"]
        self.assertEqual(results["status"], "passed")
        self.assertEqual(results["completed_jobs"], 100)
        self.assertTrue(results["pixel_exact"])
        self.assertEqual(results["unique_pixel_hashes"], 1)
        self.assertEqual(len(results["pixel_sha256"]), 64)

        memory = results["memory"]
        for summary in (
            memory["nvidia_used_mib"],
            memory["worker_rss_bytes"],
        ):
            self.assertTrue(summary["passed"])
            self.assertLessEqual(summary["growth"], summary["tolerance"])
        self.assertTrue(results["cleanup"]["passed"])
        self.assertEqual(results["cleanup"]["initial_gpu_used_mib"], 0)
        self.assertEqual(results["cleanup"]["terminal_gpu_used_mib"], 0)

    def test_cost_is_recomputed_and_release_blockers_remain_explicit(self) -> None:
        execution = self.fixture["execution"]
        recomputed_cost = (
            execution["session_seconds"]
            / 3600
            * execution["published_a40_rate_usd_per_hour"]
        )
        self.assertAlmostEqual(
            recomputed_cost,
            execution["estimated_session_cost_usd"],
        )
        self.assertLess(execution["estimated_session_cost_usd"], 0.13)
        self.assertEqual(len(self.fixture["raw_evidence"]["state_sha256"]), 64)

        readiness = self.fixture["release_readiness"]
        self.assertFalse(readiness["native_backend_default_changed"])
        self.assertTrue(readiness["comfyui_fallback_retained"])
        self.assertFalse(readiness["clean_native_image_built_and_booted"])
        self.assertFalse(readiness["first_party_and_model_license_review_complete"])
        self.assertFalse(readiness["representative_output_human_approval_complete"])


if __name__ == "__main__":
    unittest.main()
