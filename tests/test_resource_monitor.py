from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
