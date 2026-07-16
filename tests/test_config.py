from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from k2_region_lab.config import AppSettings


class ConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
