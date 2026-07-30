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
    / "gate10_a40_48gb.json"
)


class NativeDeviceFixtureTests(unittest.TestCase):
    def test_a40_fixture_uses_the_pinned_core_and_approved_models(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

        self.assertEqual(
            fixture["schema_version"],
            "k2lab-gate10-device-evidence/1",
        )
        core_commit = fixture["implementation"]["k2core_commit"]
        self.assertIn(
            core_commit,
            (ROOT / "docs" / "native_backend" / "gate_10_report.md").read_text(
                encoding="utf-8"
            ),
        )
        self.assertRegex(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
            r"k2core\.git@[0-9a-f]{40}",
        )
        self.assertEqual(fixture["hardware"]["gpu"], "NVIDIA A40")
        self.assertGreaterEqual(fixture["hardware"]["gpu_total_bytes"], 40 * 1024**3)
        for digest in fixture["model_hashes"].values():
            self.assertEqual(len(digest), 64)

    def test_a40_matrix_passes_cleanup_and_recovery_contracts(self) -> None:
        results = json.loads(FIXTURE.read_text(encoding="utf-8"))["results"]

        for result in results.values():
            self.assertTrue(result["passed"])
        self.assertTrue(results["load_unload"]["all_post_unload_allocated_bytes"] == 0)
        self.assertTrue(results["load_unload"]["all_post_unload_reserved_bytes"] == 0)
        self.assertTrue(results["sequential_soak"]["pixel_exact"])
        self.assertEqual(results["sequential_soak"]["post_unload_allocated_bytes"], 0)
        self.assertEqual(results["sequential_soak"]["post_unload_reserved_bytes"], 0)
        self.assertEqual(results["intentional_oom"]["category"], "OutOfMemoryError")
        self.assertTrue(results["intentional_oom"]["retry_safe"])
        self.assertFalse(results["intentional_oom"]["gpu_work_started"])
        self.assertEqual(results["cancellation"]["steps_seen"], [1])
        self.assertEqual(results["cancellation"]["category"], "CancellationError")
        self.assertTrue(results["cancellation"]["gpu_work_started"])
        self.assertEqual(results["cancellation"]["post_unload_allocated_bytes"], 0)
        self.assertEqual(results["cancellation"]["post_unload_reserved_bytes"], 0)

    def test_a40_matrix_stayed_inside_the_declared_cost_ceiling(self) -> None:
        execution = json.loads(FIXTURE.read_text(encoding="utf-8"))["execution"]
        matrix_cost = (
            execution["matrix_elapsed_seconds"]
            / 3600
            * execution["published_a40_rate_usd_per_hour"]
        )

        self.assertLessEqual(matrix_cost, execution["approved_cost_ceiling_usd"])
        self.assertEqual(execution["terminal_nvidia_smi_memory_used_mib"], 0)
        self.assertEqual(len(execution["raw_jsonl_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
