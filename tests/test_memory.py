from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from k2_region_lab.memory import (
    MEMORY_POLICIES,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
)
from k2_region_lab.worker.runtime import CriticalGpuMemoryPressure, ComfyBaselineRuntime


class MemoryPolicyTests(unittest.TestCase):
    def test_safe_16gb_policy_keeps_four_gib_free(self) -> None:
        policy = memory_policy("safe_16gb")
        self.assertEqual(policy.reserve_vram_gb, 4.0)
        self.assertEqual(policy.minimum_system_ram_gb, 14.0)
        self.assertTrue(policy.oom_recovery)

    def test_policy_keys_are_unique(self) -> None:
        keys = [policy.key for policy in MEMORY_POLICIES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_saved_value_cannot_weaken_policy_floor(self) -> None:
        self.assertEqual(effective_reserve_vram_gb("safe_16gb", 2.0), 4.0)
        self.assertEqual(effective_reserve_vram_gb("emergency", 4.0), 5.5)
        self.assertEqual(effective_reserve_vram_gb("balanced", 3.5), 3.5)
        self.assertEqual(effective_minimum_system_ram_gb("safe_16gb", 12.0), 14.0)
        self.assertEqual(effective_minimum_system_ram_gb("emergency", 14.0), 16.0)
        self.assertEqual(effective_minimum_system_ram_gb("balanced", 13.0), 13.0)

    def test_critical_pressure_uses_the_single_oom_recovery_path(self) -> None:
        self.assertTrue(
            ComfyBaselineRuntime._is_oom(CriticalGpuMemoryPressure("1.4 GiB free"))
        )

    def test_generation_retries_only_once_with_same_request_after_oom(self) -> None:
        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.model = object()
        runtime.clip = object()
        runtime.vae = object()
        runtime.oom_recovery = True
        expected = {"image_path": "/tmp/result.png", "oom_recovered": True}
        runtime._generate_once = Mock(
            side_effect=[RuntimeError("HIP out of memory"), expected]
        )
        runtime._recover_from_oom = Mock()
        runtime.memory_snapshot = Mock(return_value={"stage": "OOM detected"})

        with tempfile.TemporaryDirectory() as directory:
            result = runtime.generate(
                prompt="a teapot",
                width=256,
                height=256,
                steps=1,
                seed=42,
                output_directory=Path(directory),
            )

        self.assertEqual(result, expected)
        self.assertEqual(runtime._generate_once.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in runtime._generate_once.call_args_list],
            [42, 42],
        )
        runtime._recover_from_oom.assert_called_once()

    def test_vae_decode_stays_inside_torch_inference_mode(self) -> None:
        active = False

        class InferenceMode:
            def __enter__(self):
                nonlocal active
                active = True

            def __exit__(self, *_):
                nonlocal active
                active = False

        def decode(samples):
            self.assertTrue(active)
            return samples

        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.vae = SimpleNamespace(decode=decode)
        fake_torch = SimpleNamespace(inference_mode=InferenceMode)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = runtime._decode_vae("latent")

        self.assertEqual(result, "latent")
        self.assertFalse(active)

    def test_vae_encode_disables_gradient_tracking(self) -> None:
        active = False

        class NoGrad:
            def __enter__(self):
                nonlocal active
                active = True

            def __exit__(self, *_):
                nonlocal active
                active = False

        def encode(pixels):
            self.assertTrue(active)
            return pixels

        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.vae = SimpleNamespace(encode=encode)
        fake_torch = SimpleNamespace(no_grad=NoGrad)

        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = runtime._encode_vae("pixels")

        self.assertEqual(result, "pixels")
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
