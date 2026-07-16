from __future__ import annotations

import unittest

from k2_region_lab.regions import (
    CanvasGeometry,
    PixelBox,
    RegionDefinition,
    align_up,
    compile_spatial_layout,
)


class GeometryTests(unittest.TestCase):
    def test_alignment_and_krea_grid(self) -> None:
        self.assertEqual(align_up(1000, 16), 1008)
        geometry = CanvasGeometry.resolve(1024, 1024)
        self.assertEqual(geometry.patch_width, 64)
        self.assertEqual(geometry.patch_height, 64)
        self.assertEqual(geometry.image_lane_count, 4096)
        self.assertEqual(geometry.image_lane_index(1, 1), 65)

    def test_half_open_box_maps_to_exact_token(self) -> None:
        geometry = CanvasGeometry.resolve(64, 64)
        mask = geometry.rasterize_box(PixelBox(16, 16, 32, 32))
        self.assertEqual(sum(mask), 1.0)
        self.assertEqual(mask[geometry.image_lane_index(1, 1)], 1.0)

    def test_fractional_coverage_does_not_extend_outside_box(self) -> None:
        geometry = CanvasGeometry.resolve(64, 64)
        mask = geometry.rasterize_box(PixelBox(0, 0, 17, 16))
        self.assertEqual(mask[0], 1.0)
        self.assertAlmostEqual(mask[1], 1.0 / 16.0)
        self.assertAlmostEqual(sum(mask), 1.0 + 1.0 / 16.0)
        self.assertTrue(all(value == 0.0 for value in mask[2:]))

    def test_clipping_and_non_square_canvas(self) -> None:
        geometry = CanvasGeometry.resolve(31, 17)
        self.assertEqual((geometry.aligned_width, geometry.aligned_height), (32, 32))
        mask = geometry.rasterize_box(PixelBox(-4, -4, 8, 8))
        self.assertAlmostEqual(mask[0], 0.25)

    def test_generic_layout_uses_one_shared_region_mask(self) -> None:
        geometry = CanvasGeometry.resolve(64, 64)
        region = RegionDefinition(
            region_id="subject-left",
            name="Any prompted content",
            box=PixelBox(0, 0, 32, 64),
            prompt="a glass sculpture",
        )
        layout = compile_spatial_layout(geometry, (region,))
        mask = layout.mask_for("subject-left")
        self.assertEqual(sum(mask), 8.0)
        self.assertEqual(len(mask), geometry.image_lane_count)

    def test_duplicate_region_ids_are_rejected(self) -> None:
        geometry = CanvasGeometry.resolve(64, 64)
        region = RegionDefinition("same", "One", PixelBox(0, 0, 16, 16))
        with self.assertRaises(ValueError):
            compile_spatial_layout(geometry, (region, region))


if __name__ == "__main__":
    unittest.main()
