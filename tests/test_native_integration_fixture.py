from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent.parent
EVIDENCE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "integration"
    / "gate11_desktop_runpod.json"
)
REQUEST_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gate11"
    / "native_clean_generation.json"
)


class NativeIntegrationFixtureTests(unittest.TestCase):
    def test_gate11_uses_the_pinned_core_and_canonical_request(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

        self.assertEqual(
            evidence["schema_version"],
            "k2lab-gate11-integration-evidence/1",
        )
        core_commit = evidence["implementation"]["k2core_commit"]
        self.assertIn(
            core_commit,
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        observed_hash = hashlib.sha256(REQUEST_FIXTURE.read_bytes()).hexdigest()
        self.assertEqual(
            observed_hash,
            evidence["request_fixture"]["sha256"],
        )
        self.assertTrue(evidence["entrypoints"]["desktop"]["passed"])
        self.assertTrue(evidence["entrypoints"]["runpod_worker"]["passed"])
        self.assertTrue(evidence["entrypoints"]["runpod_agent"]["passed"])

    def test_gate11_runpod_entrypoints_are_pixel_exact(self) -> None:
        entrypoints = json.loads(
            EVIDENCE.read_text(encoding="utf-8")
        )["entrypoints"]

        self.assertEqual(
            entrypoints["runpod_worker"]["rgb_sha256"],
            entrypoints["runpod_agent"]["rgb_sha256"],
        )
        self.assertEqual(
            entrypoints["runpod_agent"]["job_id"],
            entrypoints["runpod_agent"]["duplicate_job_id"],
        )
        self.assertEqual(
            entrypoints["runpod_agent"]["job_id"],
            entrypoints["runpod_agent"]["post_reconnect_replay_job_id"],
        )
        self.assertEqual(
            entrypoints["runpod_agent"]["output_file_id"],
            entrypoints["runpod_agent"][
                "post_reconnect_replay_output_file_id"
            ],
        )
        self.assertEqual(
            entrypoints["runpod_agent"]["reconnected_state"],
            "completed",
        )
        self.assertEqual(
            entrypoints["runpod_agent"]["resumed_event_count"],
            0,
        )

    def test_gate11_resilience_and_cost_contracts_pass(self) -> None:
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

        self.assertTrue(all(evidence["resilience"].values()))
        execution = evidence["execution"]
        calculated_cost = (
            execution["billable_probe_seconds"]
            / 3600
            * execution["published_a40_rate_usd_per_hour"]
        )
        self.assertAlmostEqual(
            calculated_cost,
            execution["estimated_incremental_compute_usd"],
        )
        self.assertLess(execution["estimated_incremental_compute_usd"], 0.02)
        self.assertEqual(
            execution["terminal_nvidia_smi_memory_used_mib"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
