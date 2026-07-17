from __future__ import annotations

import unittest

from k2_region_lab.regional_prompting import (
    compile_regional_prompt_plan,
    region_definitions_from_payload,
)
from k2_region_lab.regions import PixelBox, RegionDefinition


class RegionalPromptingTests(unittest.TestCase):
    def test_pixel_box_compiles_to_fractional_latent_and_image_token_masks(self) -> None:
        region = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(0, 0, 17, 16),
            prompt="a red glass sculpture",
        )

        plan = compile_regional_prompt_plan(
            64, 64, (region,), strength=1.5, feather_pixels=0
        )
        compiled = plan.regions[0]

        self.assertEqual((plan.latent_width, plan.latent_height), (8, 8))
        self.assertEqual(compiled.area, (2, 3, 0, 0))
        self.assertAlmostEqual(sum(compiled.latent_mask), 4.25)
        self.assertAlmostEqual(sum(compiled.image_token_mask), 1.0625)
        self.assertEqual(plan.summary()["backend"], "comfy-latent-area-v1")

    def test_default_feather_reduces_influence_at_box_edge(self) -> None:
        region = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(0, 0, 64, 64),
            prompt="a red glass sculpture",
        )

        hard = compile_regional_prompt_plan(
            64, 64, (region,), feather_pixels=0
        ).regions[0].latent_mask
        feathered = compile_regional_prompt_plan(
            64, 64, (region,), feather_pixels=16
        ).regions[0].latent_mask

        self.assertEqual(hard[0], 1.0)
        self.assertLess(feathered[0], hard[0])
        self.assertEqual(feathered[3 * 8 + 3], 1.0)

    def test_disabled_and_empty_prompt_regions_do_not_compile(self) -> None:
        disabled = RegionDefinition(
            "disabled", "Disabled", PixelBox(0, 0, 16, 16), "subject", enabled=False
        )
        empty = RegionDefinition("empty", "Empty", PixelBox(16, 0, 32, 16), "")

        plan = compile_regional_prompt_plan(64, 64, (disabled, empty))

        self.assertEqual(plan.regions, ())

    def test_worker_payload_conversion_preserves_generic_region_fields(self) -> None:
        regions = region_definitions_from_payload(
            [
                {
                    "id": "anything",
                    "name": "Anything",
                    "box": {"x0": 4, "y0": 8, "x1": 40, "y1": 56},
                    "prompt": "a small tree",
                    "negative_prompt": "building",
                    "enabled": True,
                    "priority": 3,
                }
            ]
        )

        self.assertEqual(regions[0].name, "Anything")
        self.assertEqual(regions[0].negative_prompt, "building")
        self.assertEqual(regions[0].priority, 3)


if __name__ == "__main__":
    unittest.main()
