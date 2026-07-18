from __future__ import annotations

import unittest

from k2_region_lab.worker.runtime import (
    accelerator_backend,
    native_scaled_fp8_supported,
)


class AcceleratorCapabilityTests(unittest.TestCase):
    def test_backend_distinguishes_rocm_cuda_and_cpu_builds(self) -> None:
        self.assertEqual(accelerator_backend("7.1", None), "rocm")
        self.assertEqual(accelerator_backend(None, "12.8"), "cuda")
        self.assertEqual(accelerator_backend(None, None), "unknown")

    def test_native_fp8_accepts_supported_rocm_and_nvidia_devices(self) -> None:
        self.assertTrue(native_scaled_fp8_supported("rocm", "7.1", []))
        self.assertFalse(native_scaled_fp8_supported("rocm", "6.4", []))
        self.assertTrue(
            native_scaled_fp8_supported(
                "cuda", "12.8", [{"major": 8, "minor": 9}]
            )
        )
        self.assertTrue(
            native_scaled_fp8_supported(
                "cuda", "12.8", [{"major": 9, "minor": 0}]
            )
        )
        self.assertFalse(
            native_scaled_fp8_supported(
                "cuda", "12.8", [{"major": 8, "minor": 6}]
            )
        )


if __name__ == "__main__":
    unittest.main()
