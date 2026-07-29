from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "regional_lora"
    / "krea2_two_vases_regional_lora_512.json"
)


class RegionalLoraParityFixtureTests(unittest.TestCase):
    def test_every_cross_backend_case_meets_its_recorded_thresholds(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schema_version"], "k2lab-parity-fixture/1")
        self.assertEqual(fixture["status"], "PASS WITH APPROVED DIFFERENCE")
        self.assertEqual(fixture["artifact_policy"]["storage"], "outside_git")

        for case in fixture["cases"].values():
            measured = case["measurements"]
            thresholds = case["thresholds"]
            self.assertGreaterEqual(measured["cosine"], thresholds["minimum_cosine"])
            self.assertLessEqual(
                measured["mean_absolute_error"],
                thresholds["maximum_mean_absolute_error"],
            )
            self.assertLessEqual(measured["rmse"], thresholds["maximum_rmse"])
            self.assertLessEqual(
                measured["maximum_absolute_error"],
                thresholds["maximum_absolute_error"],
            )
            self.assertGreaterEqual(measured["psnr_db"], thresholds["minimum_psnr_db"])
            self.assertIn("no unexplained artifact", case["human_review"].casefold())

    def test_fixture_records_locality_and_disabled_path_regressions(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        leakage = fixture["leakage_and_instrumentation"]
        regressions = fixture["regressions"]

        self.assertEqual(leakage["maximum_outside_route_across_all_records"], 0.0)
        self.assertEqual(leakage["comfy_outside_gate_delta_rms"], 0.0)
        self.assertFalse(leakage["instrumentation_default_enabled"])
        self.assertTrue(regressions["instrumentation_on_off_pixel_exact"])
        self.assertTrue(regressions["no_lora_matches_gate_5"])
        self.assertTrue(regressions["regional_disabled_matches_gate_6_ordinary_lora"])
        self.assertEqual(
            fixture["semantic_checkpoints"]["standard_five_target_locality_skipped"],
            ["blocks.0.attn.wk", "blocks.0.attn.wv"],
        )


if __name__ == "__main__":
    unittest.main()
