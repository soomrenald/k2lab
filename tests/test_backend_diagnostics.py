from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from k2_region_lab.config import AppSettings, ModelDirectories
from k2_region_lab.desktop.backend_diagnostics import backend_diagnostic_rows
from k2core.inference import BackendName
from k2core.model import (
    ComponentReference,
    ModelRegistry,
    RegisteredModel,
    TokenizerReference,
)


class BackendDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _settings(root: Path, registry_path: Path | None = None) -> AppSettings:
        return AppSettings(
            model_directories=ModelDirectories(
                diffusion_models=root / "diffusion",
                text_encoders=root / "text",
                vae=root / "vae",
                loras=root / "loras",
                upscale_models=root / "upscale",
            ),
            data_directory=root / "data",
            model_registry_path=registry_path,
            registered_model_name="diagnostic-model" if registry_path else "",
        )

    @staticmethod
    def _by_label(rows: list[dict[str, str]]) -> dict[str, str]:
        return {row["label"]: row["value"] for row in rows}

    def test_default_comfyui_diagnostics_do_not_claim_native_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = self._by_label(
                backend_diagnostic_rows(
                    backend_name=BackendName.COMFYUI,
                    settings=self._settings(Path(directory)),
                )
            )

        self.assertEqual(rows["Selected backend"], "comfyui")
        self.assertIn("ComfyUI runtime", rows["Device placement"])
        self.assertEqual(rows["Loaded LoRAs"], "None")
        self.assertEqual(rows["Last parity status"], "Reference production backend")

    def test_native_diagnostics_show_registry_hashes_and_live_device_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "models.toml"
            registry = ModelRegistry(
                models=(
                    RegisteredModel(
                        name="diagnostic-model",
                        architecture="krea2",
                        transformer=ComponentReference(
                            root / "transformer.safetensors", "1" * 64
                        ),
                        text_encoder=ComponentReference(
                            root / "text.safetensors", "2" * 64
                        ),
                        vae=ComponentReference(
                            root / "vae.safetensors", "3" * 64
                        ),
                        tokenizer=TokenizerReference(
                            root / "tokenizer", "4" * 64
                        ),
                    ),
                )
            )
            registry_path.write_text(registry.to_toml(), encoding="utf-8")
            rows = self._by_label(
                backend_diagnostic_rows(
                    backend_name=BackendName.NATIVE,
                    settings=self._settings(root, registry_path),
                    load_payload={
                        "device_plan": {
                            "transformer_device": "cuda:0",
                            "text_encoder_device": "cuda:0",
                            "vae_device": "cpu",
                            "compute_dtype": "bfloat16",
                            "weight_dtype": "auto",
                        }
                    },
                    loaded_loras=("portrait", "style"),
                    memory_text="VRAM 31.0/44.4 GiB free",
                )
            )

        self.assertEqual(rows["Selected backend"], "native")
        self.assertIn("transformer " + "1" * 64, rows["Model hashes"])
        self.assertIn("tokenizer " + "4" * 64, rows["Model hashes"])
        self.assertEqual(
            rows["Device placement"],
            "transformer=cuda:0, text=cuda:0, VAE=cpu",
        )
        self.assertEqual(rows["Dtype"], "compute=bfloat16, weights=auto")
        self.assertEqual(rows["Loaded LoRAs"], "portrait, style")
        self.assertIn("Gate 11 PASS", rows["Last parity status"])
        self.assertIn("face refinement", rows["Unsupported"])

