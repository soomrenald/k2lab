from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from k2_region_lab.config import AppSettings, discover_worker_python


class ConfigTests(unittest.TestCase):
    def test_toml_config_sets_model_paths_and_generation_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "settings.toml"
            config.write_text(
                """
[paths]
comfyui_root = "comfy"
worker_python = "worker/bin/python"
data_directory = "data"
output_directory = "renders"
diffusion_models = "models/diffusion"
text_encoders = "models/text"
vae = "models/vae"
loras = "models/loras"
upscale_models = "models/upscale"

[models]
diffusion_model = "models/diffusion/krea.safetensors"
text_encoder = "models/text/qwen.safetensors"
vae = "models/vae/qwen_vae.safetensors"
face_detector = "models/detectors/face.onnx"
upscale_model = "models/upscale/4x.pth"

[runtime]
auto_start_worker = false
memory_policy = "balanced"
reserve_vram_gb = 3.5
minimum_system_ram_gb = 12.0
cpu_vae = true
oom_recovery = false

[generation]
width = 768
height = 1152
steps = 12
sampler = "heun"
scheduler = "normal"
seed = 42
seed_mode = "increment"
filename_prefix = "configured"
""".strip(),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"K2LAB_CONFIG_FILE": str(config)}, clear=True
            ):
                settings = AppSettings.from_environment()

            models = settings.model_directories
            self.assertEqual(models.diffusion_models, root / "models/diffusion")
            self.assertEqual(models.text_encoders, root / "models/text")
            self.assertEqual(models.vae, root / "models/vae")
            self.assertEqual(models.loras, root / "models/loras")
            self.assertEqual(models.upscale_models, root / "models/upscale")
            self.assertEqual(
                models.diffusion_model_file,
                root / "models/diffusion/krea.safetensors",
            )
            self.assertEqual(settings.face_detector_path, root / "models/detectors/face.onnx")
            self.assertEqual(settings.default_upscale_model, root / "models/upscale/4x.pth")
            self.assertEqual(settings.output_directory, root / "renders")
            self.assertFalse(settings.auto_start_worker)
            self.assertEqual(settings.reserve_vram_gb, 3.5)
            self.assertTrue(settings.cpu_vae)
            self.assertFalse(settings.oom_recovery)
            self.assertEqual(settings.default_width, 768)
            self.assertEqual(settings.default_height, 1152)
            self.assertEqual(settings.default_steps, 12)
            self.assertEqual(settings.default_sampler, "heun")
            self.assertEqual(settings.default_scheduler, "normal")
            self.assertEqual(settings.default_seed, 42)
            self.assertEqual(settings.default_seed_mode, "increment")
            self.assertEqual(settings.filename_prefix, "configured")
            self.assertEqual(settings.config_file, config)

    def test_worker_auto_discovery_prefers_vendor_neutral_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cuda_python = root / ".venv" / "bin" / "python"
            rocm_python = root / "venv_rocm7" / "bin" / "python"
            cuda_python.parent.mkdir(parents=True)
            rocm_python.parent.mkdir(parents=True)
            cuda_python.touch()
            rocm_python.touch()

            self.assertEqual(discover_worker_python(root), cuda_python)

    def test_worker_auto_discovery_supports_rocm_named_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rocm_python = root / "venv_rocm7" / "bin" / "python"
            rocm_python.parent.mkdir(parents=True)
            rocm_python.touch()

            self.assertEqual(discover_worker_python(root), rocm_python)

    def test_emergency_memory_policy_supplies_safe_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {"K2LAB_MEMORY_POLICY": "emergency"},
            clear=True,
        ):
            settings = AppSettings.from_environment()

        self.assertEqual(settings.reserve_vram_gb, 5.5)
        self.assertEqual(settings.minimum_system_ram_gb, 16.0)
        self.assertTrue(settings.cpu_vae)
        self.assertFalse(settings.oom_recovery)

    def test_worker_python_preserves_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system_python = root / "python-system"
            system_python.touch()
            virtual_environment = root / "worker-venv" / "bin"
            virtual_environment.mkdir(parents=True)
            worker_python = virtual_environment / "python"
            worker_python.symlink_to(system_python)

            with patch.dict(
                os.environ,
                {"K2LAB_WORKER_PYTHON": str(worker_python)},
                clear=False,
            ):
                settings = AppSettings.from_environment()

            self.assertEqual(settings.worker_python, worker_python)
            self.assertNotEqual(settings.worker_python, system_python)

    def test_output_environment_settings_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "renders"
            with patch.dict(
                os.environ,
                {
                    "K2LAB_OUTPUT_DIRECTORY": str(output),
                    "K2LAB_FILENAME_PREFIX": "beach study",
                },
                clear=False,
            ):
                settings = AppSettings.from_environment()

            self.assertEqual(settings.output_directory, output.resolve())
            self.assertEqual(settings.filename_prefix, "beach study")


if __name__ == "__main__":
    unittest.main()
