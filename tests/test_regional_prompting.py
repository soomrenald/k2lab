from __future__ import annotations

import json
import unittest
from pathlib import Path

from k2_region_lab.project import project_state
from k2_region_lab.regional_prompting import (
    BACKEND,
    compile_regional_prompt_plan,
    krea_prompt_token_count,
    region_definitions_from_payload,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.spatial_attention import spatial_pair_bias


class RegionalPromptingTests(unittest.TestCase):
    def test_testone_compiles_one_scene_ordered_prompt(self) -> None:
        document = json.loads(
            (Path(__file__).parents[1] / "testone.k2lab.json").read_text(
                encoding="utf-8"
            )
        )
        state = project_state(document)

        plan = compile_regional_prompt_plan(
            state.canvas_width,
            state.canvas_height,
            state.global_prompt,
            state.regions,
        )

        self.assertTrue(plan.prompt.startswith(state.global_prompt.strip()))
        ordered_names = [region.name for region in plan.regions]
        self.assertEqual(
            ordered_names,
            ["sky", "ocean", "sand", "red bikini woman", "lface", "dog"],
        )
        self.assertIn("centered about 17% across and 64% down", plan.prompt)
        self.assertIn(
            "From left to right, the subjects are red bikini woman, dog, and lface",
            plan.prompt,
        )
        self.assertEqual(plan.backend, BACKEND)
        for region in plan.regions:
            start, end = region.character_span
            self.assertEqual(plan.prompt[start:end], region.clause)

    def test_soft_field_has_full_box_core_and_smooth_outside_falloff(self) -> None:
        region = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(16, 16, 32, 32),
            prompt="a red glass sculpture",
        )

        plan = compile_regional_prompt_plan(
            64, 64, "gallery interior", (region,), falloff_pixels=24
        )
        field = plan.regions[0].image_token_field

        self.assertEqual((plan.image_token_width, plan.image_token_height), (4, 4))
        self.assertEqual(field[1 * 4 + 1], 1.0)
        self.assertGreater(field[1 * 4 + 2], 0.0)
        self.assertLess(field[1 * 4 + 2], 1.0)
        self.assertEqual(field[3 * 4 + 3], 0.0)

    def test_character_spans_bind_to_one_text_sequence(self) -> None:
        regions = (
            RegionDefinition("left", "Left", PixelBox(0, 0, 32, 64), "red vase"),
            RegionDefinition("right", "Right", PixelBox(32, 0, 64, 64), "blue vase"),
        )
        plan = compile_regional_prompt_plan(64, 64, "two objects", regions)

        bound = plan.bind_tokens(lambda prefix: len(prefix.split()))

        self.assertEqual(bound.text_token_count, len(plan.prompt.split()))
        self.assertEqual(len(bound.spans), 2)
        self.assertLess(bound.spans[0].start, bound.spans[0].end)
        self.assertLessEqual(bound.spans[0].end, bound.spans[1].start)
        self.assertEqual(bound.image_token_count, 16)

    def test_spatial_pair_bias_boosts_core_and_only_softly_penalizes_far_field(self) -> None:
        values = spatial_pair_bias((1.0, 0.5, 0.0), 2.0)

        self.assertEqual(values[0], 2.0)
        self.assertEqual(values[1], 0.75)
        self.assertEqual(values[2], -0.5)

    def test_krea_prompt_token_count_excludes_fixed_wrapper_and_suffix(self) -> None:
        tokenized = {
            "qwen3vl_4b": [
                [
                    (151644, 1.0),
                    (999, 1.0),
                    (151645, 1.0),
                    (151644, 1.0),
                    (872, 1.0),
                    (198, 1.0),
                    (11, 1.0),
                    (12, 1.0),
                    (151645, 1.0),
                    (198, 1.0),
                    (151644, 1.0),
                ]
            ]
        }

        self.assertEqual(krea_prompt_token_count(tokenized), 2)

    def test_disabled_and_empty_prompt_regions_do_not_compile(self) -> None:
        disabled = RegionDefinition(
            "disabled", "Disabled", PixelBox(0, 0, 16, 16), "subject", enabled=False
        )
        empty = RegionDefinition("empty", "Empty", PixelBox(16, 0, 32, 16), "")

        plan = compile_regional_prompt_plan(64, 64, "scene", (disabled, empty))

        self.assertEqual(plan.regions, ())
        self.assertEqual(plan.prompt, "scene.")

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
