from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from k2_region_lab.desktop.issue_report import (
    SCHEMA_VERSION,
    create_issue_report_bundle,
)
from k2core.inference import BackendName


class IssueReportTests(unittest.TestCase):
    def test_bundle_is_bounded_prompt_safe_and_created_below_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = "private prompt about a real person"
            path = create_issue_report_bundle(
                data_directory=root,
                backend_name=BackendName.NATIVE,
                diagnostics=[
                    {"label": "Selected backend", "value": "native"},
                    {"label": "Model hashes", "value": "transformer " + "a" * 64},
                    {"label": "Loaded LoRAs", "value": "private-person-name"},
                    {"label": "Prompt", "value": secret},
                ],
            )

            self.assertEqual(path.parent, root / "issue-reports")
            self.assertTrue(path.is_file())
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"README.txt", "issue-report.json"},
                )
                report_bytes = archive.read("issue-report.json")
                report = json.loads(report_bytes)
                readme = archive.read("README.txt")

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["backend"], "native")
        self.assertEqual(
            report["diagnostics"],
            [
                {"label": "Selected backend", "value": "native"},
                {"label": "Model hashes", "value": "transformer " + "a" * 64},
            ],
        )
        self.assertFalse(report["privacy"]["full_prompts_included"])
        self.assertFalse(report["privacy"]["lora_names_included"])
        self.assertNotIn(secret.encode(), report_bytes)
        self.assertNotIn(b"private-person-name", report_bytes)
        self.assertIn(b"Review issue-report.json before sharing", readme)


if __name__ == "__main__":
    unittest.main()
