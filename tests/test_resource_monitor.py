from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from k2_region_lab.desktop.resource_monitor import (
    discover_gpu_device,
    read_resource_sample,
)


class ResourceMonitorTests(unittest.TestCase):
    def test_largest_vram_device_is_monitored_with_system_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "card0" / "device"
            large = root / "card1" / "device"
            small.mkdir(parents=True)
            large.mkdir(parents=True)
            (small / "mem_info_vram_total").write_text("2000\n", encoding="ascii")
            (large / "mem_info_vram_total").write_text("16000\n", encoding="ascii")
            (large / "mem_info_vram_used").write_text("4000\n", encoding="ascii")
            (large / "gpu_busy_percent").write_text("37\n", encoding="ascii")
            meminfo = root / "meminfo"
            meminfo.write_text(
                "MemTotal: 64000 kB\nMemAvailable: 16000 kB\n",
                encoding="ascii",
            )

            device = discover_gpu_device(root)
            sample = read_resource_sample(device, meminfo)

            self.assertEqual(device, large)
            self.assertEqual(sample.gpu_memory_percent, 25.0)
            self.assertEqual(sample.ram_percent, 75.0)
            self.assertEqual(sample.gpu_busy_percent, 37.0)

    def test_nvidia_smi_supplies_cuda_telemetry(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout="1024, 8192, 25\n2048, 24576, 40\n",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "k2_region_lab.desktop.resource_monitor.subprocess.run",
                return_value=completed,
            ),
        ):
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(
                "MemTotal: 64000 kB\nMemAvailable: 32000 kB\n",
                encoding="ascii",
            )
            sample = read_resource_sample(None, meminfo, nvidia_smi="nvidia-smi")

        self.assertEqual(sample.gpu_used_bytes, 2048 * 1024**2)
        self.assertEqual(sample.gpu_total_bytes, 24576 * 1024**2)
        self.assertEqual(sample.gpu_busy_percent, 40.0)


if __name__ == "__main__":
    unittest.main()
