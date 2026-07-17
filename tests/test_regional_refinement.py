from __future__ import annotations

import unittest

from k2_region_lab.regional_refinement import (
    compile_refinement_crops,
    latent_blend_mask,
)
from k2_region_lab.regions import PixelBox, RegionDefinition


class RegionalRefinementTests(unittest.TestCase):
    def test_only_active_subjects_receive_aligned_padded_crops(self) -> None:
        regions = (
            RegionDefinition(
                "sky",
                "Sky",
                PixelBox(0, 0, 1024, 320),
                "blue sky",
                spatial_role="background",
            ),
            RegionDefinition(
                "face",
                "Face",
                PixelBox(20, 300, 180, 820),
                "a detailed person",
                spatial_role="subject",
            ),
            RegionDefinition(
                "empty", "Empty", PixelBox(700, 400, 900, 800), ""
            ),
        )

        crops = compile_refinement_crops(1024, 1024, regions, scale=1.5)

        self.assertEqual(len(crops), 1)
        crop = crops[0]
        self.assertEqual(crop.region_id, "face")
        self.assertEqual(crop.crop.x0, 0)
        self.assertEqual(crop.width % 16, 0)
        self.assertEqual(crop.height % 16, 0)
        self.assertGreater(crop.internal_width, crop.width)
        self.assertLessEqual(max(crop.internal_width, crop.internal_height), 1024)

    def test_large_crop_scale_is_capped_to_memory_bound(self) -> None:
        region = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(100, 100, 900, 900),
            "portrait",
            spatial_role="subject",
        )

        crop = compile_refinement_crops(1024, 1024, (region,), scale=2.0)[0]

        self.assertEqual(max(crop.internal_width, crop.internal_height), 1024)
        self.assertGreaterEqual(crop.scale, 1.0)

    def test_blend_mask_is_zero_outside_and_feathered_inside_target(self) -> None:
        region = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(128, 128, 384, 384),
            "portrait",
            spatial_role="subject",
        )
        crop = compile_refinement_crops(
            512, 512, (region,), scale=1.5, padding_pixels=64
        )[0]

        mask = latent_blend_mask(crop, feather_pixels=48)
        latent_width = crop.width // 8
        outside = mask[0]
        center_row = int((256 - crop.crop.y0) // 8)
        center_column = int((256 - crop.crop.x0) // 8)
        center = mask[center_row * latent_width + center_column]

        self.assertEqual(outside, 0.0)
        self.assertEqual(center, 1.0)
        self.assertTrue(any(0.0 < value < 1.0 for value in mask))


if __name__ == "__main__":
    unittest.main()
