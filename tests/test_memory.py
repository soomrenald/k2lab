from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from k2_region_lab.memory import MEMORY_POLICIES, memory_policy
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


class MemoryPolicyTests(unittest.TestCase):
    def test_safe_16gb_policy_keeps_four_gib_free(self) -> None:
        policy = memory_policy("safe_16gb")
        self.assertEqual(policy.reserve_vram_gb, 4.0)
        self.assertEqual(policy.minimum_system_ram_gb, 14.0)
        self.assertTrue(policy.oom_recovery)

    def test_policy_keys_are_unique(self) -> None:
        keys = [policy.key for policy in MEMORY_POLICIES]
        self.assertEqual(len(keys), len(set(keys)))

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


if __name__ == "__main__":
    unittest.main()
