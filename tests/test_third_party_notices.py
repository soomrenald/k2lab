from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ThirdPartyNoticeTests(unittest.TestCase):
    def test_notice_records_runtime_and_unresolved_release_boundaries(self) -> None:
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        for dependency in (
            "NumPy",
            "Pillow",
            "Pydantic",
            "PySide6",
            "safetensors",
            "ONNX Runtime",
            "Diffusers",
            "PyTorch",
            "Transformers",
        ):
            self.assertIn(dependency, notice)
        self.assertIn("does not contain copied or vendored ComfyUI source", notice)
        self.assertIn("no declared project license", notice)
        self.assertIn("SHA-256 registry entry proves file identity, not permission", notice)
