from __future__ import annotations

import unittest

from k2_region_lab.sampling import (
    COMFYUI_SAMPLERS,
    COMFYUI_SCHEDULERS,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    validate_sampler,
    validate_scheduler,
)


class SamplingOptionsTests(unittest.TestCase):
    def test_current_comfyui_standard_lists_include_recent_options(self) -> None:
        self.assertEqual(COMFYUI_SAMPLERS[0], "euler")
        self.assertIn("euler_cfg_pp", COMFYUI_SAMPLERS)
        self.assertIn("exp_heun_2_x0_sde", COMFYUI_SAMPLERS)
        self.assertIn("res_multistep_ancestral_cfg_pp", COMFYUI_SAMPLERS)
        self.assertIn("sa_solver_pece", COMFYUI_SAMPLERS)
        self.assertEqual(COMFYUI_SAMPLERS[-3:], ("ddim", "uni_pc", "uni_pc_bh2"))
        self.assertEqual(
            COMFYUI_SCHEDULERS,
            (
                "simple",
                "sgm_uniform",
                "karras",
                "exponential",
                "ddim_uniform",
                "beta",
                "normal",
                "linear_quadratic",
                "kl_optimal",
            ),
        )

    def test_defaults_and_validation(self) -> None:
        self.assertEqual(validate_sampler(DEFAULT_SAMPLER), "euler")
        self.assertEqual(validate_scheduler(DEFAULT_SCHEDULER), "simple")
        with self.assertRaisesRegex(ValueError, "unsupported ComfyUI sampler"):
            validate_sampler("not-a-sampler")
        with self.assertRaisesRegex(ValueError, "unsupported ComfyUI scheduler"):
            validate_scheduler("not-a-scheduler")


if __name__ == "__main__":
    unittest.main()
