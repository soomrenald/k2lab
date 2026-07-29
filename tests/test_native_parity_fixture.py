from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "parity"
    / "clean_t2i"
    / "krea2_turbo_teapot_512.json"
)


class NativeParityFixtureTests(unittest.TestCase):
    def test_clean_fixture_is_content_addressed_and_meets_recorded_thresholds(
        self,
    ) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["schema_version"], "k2lab-parity-fixture/1")
        self.assertEqual(fixture["status"], "PASS WITH APPROVED DIFFERENCE")
        self.assertEqual(fixture["artifact_policy"]["storage"], "outside_git")
        for component in fixture["components"].values():
            self.assertEqual(len(component["sha256"]), 64)

        measured = fixture["measurements"]
        thresholds = fixture["thresholds"]
        self.assertEqual(measured["reference_size"], measured["candidate_size"])
        self.assertEqual(measured["reference_mode"], measured["candidate_mode"])
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
        self.assertTrue(fixture["repeatability"]["pixel_exact"])
        sequential = fixture["sequential_reliability"]
        self.assertEqual(sequential["runs"], 10)
        self.assertEqual(sequential["unique_pixel_hashes"], 1)
        self.assertEqual(
            sequential["first_post_allocated_bytes"],
            sequential["last_post_allocated_bytes"],
        )
        self.assertEqual(
            sequential["first_post_reserved_bytes"],
            sequential["last_post_reserved_bytes"],
        )

    def test_clean_fixture_records_exact_current_turbo_contract(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        request = fixture["request"]
        self.assertEqual(
            (request["steps"], request["sampler"], request["scheduler"], request["cfg"]),
            (8, "euler", "simple", 1.0),
        )
        self.assertEqual(fixture["exact_checkpoints"]["sigma_count"], 9)
        self.assertEqual(fixture["exact_checkpoints"]["sigmas"][0], 1.0)
        self.assertEqual(fixture["exact_checkpoints"]["sigmas"][-1], 0.0)


if __name__ == "__main__":
    unittest.main()
