from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.config import ModelDirectories
from k2_region_lab.model import (
    discover_krea_transformers,
    discover_model_artifacts,
    read_safetensors_summary,
)


def write_header(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    header = dict(tensors)
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


class ModelArtifactTests(unittest.TestCase):
    def test_summary_reads_only_header_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_header(
                path,
                {"weight": {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]}},
                {"format": "pt", "_quantization_metadata": "{}"},
            )
            summary = read_safetensors_summary(path)
            self.assertEqual(summary.tensor_count, 1)
            self.assertEqual(summary.format_name, "pt")
            self.assertTrue(summary.quantized)

    def test_discovery_matches_comfyui_style_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusion = root / "diffusion_models"
            text = root / "text_encoders"
            vae = root / "vae"
            diffusion.mkdir()
            text.mkdir()
            vae.mkdir()
            tensor = {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
            write_header(diffusion / "krea2_turbo_fp8_scaled.safetensors", tensor)
            write_header(text / "qwen3vl_4b_fp8_scaled.safetensors", tensor)
            write_header(vae / "qwen_image_vae.safetensors", tensor)

            artifacts = discover_model_artifacts(ModelDirectories(diffusion, text, vae))
            self.assertTrue(artifacts.complete)
            self.assertEqual(artifacts.transformer.path.name, "krea2_turbo_fp8_scaled.safetensors")

    def test_krea_transformer_selector_lists_raw_and_turbo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = {"weight": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}}
            write_header(root / "krea2_raw_int8_convrot.safetensors", tensor)
            write_header(root / "krea2_turbo_fp8_scaled.safetensors", tensor)
            write_header(root / "unrelated_model.safetensors", tensor)

            candidates = discover_krea_transformers(root)

            self.assertEqual(
                [candidate.path.name for candidate in candidates],
                [
                    "krea2_raw_int8_convrot.safetensors",
                    "krea2_turbo_fp8_scaled.safetensors",
                ],
            )
            self.assertTrue(all(candidate.summary.quantized for candidate in candidates))

    def test_explicit_configured_files_override_name_based_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffusion = root / "diffusion_models"
            text = root / "text_encoders"
            vae = root / "vae"
            diffusion.mkdir()
            text.mkdir()
            vae.mkdir()
            tensor = {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
            explicit_diffusion = diffusion / "custom-transformer.safetensors"
            explicit_text = text / "custom-encoder.safetensors"
            explicit_vae = vae / "custom-decoder.safetensors"
            write_header(explicit_diffusion, tensor)
            write_header(explicit_text, tensor)
            write_header(explicit_vae, tensor)

            artifacts = discover_model_artifacts(
                ModelDirectories(
                    diffusion,
                    text,
                    vae,
                    diffusion_model_file=explicit_diffusion,
                    text_encoder_file=explicit_text,
                    vae_file=explicit_vae,
                )
            )

            self.assertEqual(artifacts.transformer.path, explicit_diffusion)
            self.assertEqual(artifacts.text_encoder.path, explicit_text)
            self.assertEqual(artifacts.vae.path, explicit_vae)


if __name__ == "__main__":
    unittest.main()
