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
        self.assertIn("Krea 2 Community License Agreement v1", notice)
        self.assertIn("content-filter", notice)
        self.assertIn(
            "7ea8de1da304c1459a11f637798bb1140805365aeb3cf6637ca6d61909720aec",
            notice,
        )
        self.assertIn("FantasyPortrait detector's source and file identity are", notice)
        self.assertIn("official Qwen3-VL-4B-Instruct repository", notice)
        self.assertIn("official Qwen-Image repository and VAE", notice)
