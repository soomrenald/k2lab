from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from k2_region_lab.config import AppSettings


class ConfigTests(unittest.TestCase):
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
