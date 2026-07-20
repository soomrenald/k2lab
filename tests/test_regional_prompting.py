from __future__ import annotations

import unittest

from k2_region_lab.project import PROJECT_VERSION, project_document, project_state
from k2_region_lab.regional_prompting import (
    BACKEND,
    GLOBAL_EMPHASIS_SCOPE,
    PromptEmphasis,
    compile_regional_prompt_plan,
    krea_prompt_token_count,
    region_definitions_from_payload,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.spatial_attention import (
    KreaSpatialAttentionOverride,
    spatial_pair_bias,
    text_region_ownership,
)


class RegionalPromptingTests(unittest.TestCase):
    def test_scene_compiles_one_scene_ordered_prompt(self) -> None:
        regions = (
            RegionDefinition(
                "dog", "dog", PixelBox(440, 800, 600, 980), "a small brown dog"
            ),
            RegionDefinition(
                "sand", "sand", PixelBox(0, 680, 1024, 1024), "white sand"
            ),
            RegionDefinition(
                "left", "left subject", PixelBox(100, 410, 250, 900), "a woman in red"
            ),
            RegionDefinition(
                "sky", "sky", PixelBox(0, 0, 1024, 320), "clear blue sky"
            ),
            RegionDefinition(
                "right",
                "right subject",
                PixelBox(650, 390, 850, 910),
                "a woman in blue",
            ),
            RegionDefinition(
                "ocean", "ocean", PixelBox(0, 320, 1024, 680), "tropical ocean"
            ),
        )
        global_prompt = "photorealistic beach scene"

        plan = compile_regional_prompt_plan(
            1024,
            1024,
            global_prompt,
            regions,
        )

        self.assertTrue(plan.prompt.startswith(global_prompt))
        ordered_names = [region.name for region in plan.regions]
        subjects = sorted(
            (region for region in regions if region.box.width < 0.70 * 1024),
            key=lambda region: (region.box.x0 + region.box.x1) / 2.0,
        )
        self.assertEqual(
            ordered_names,
            [region.name for region in regions],
        )
        self.assertIn("In the lower portion far-left side", plan.prompt)
        self.assertIn("visible subject itself should fill most", plan.prompt)
        self.assertIn("without drawing guides, borders, coordinates", plan.prompt)
        self.assertIn("prominent medium-to-large subject", plan.prompt)
        expected_order = (
            "From left to right, the subjects are "
            + ", ".join(region.name for region in subjects[:-1])
            + f", and {subjects[-1].name}"
        )
        self.assertIn(expected_order, plan.prompt)
        self.assertIn(
            "left subject and right subject are equally large, at the same camera distance",
            plan.prompt,
        )
        self.assertEqual(plan.backend, BACKEND)
        self.assertEqual(
            [region.spatial_role for region in plan.regions],
            ["subject", "background", "subject", "background", "subject", "background"],
        )
        for region in plan.regions:
            start, end = region.character_span
            self.assertEqual(plan.prompt[start:end], region.clause)

    def test_overlapping_subjects_follow_front_to_back_region_order(self) -> None:
        front = RegionDefinition(
            "person",
            "sface",
            PixelBox(20, 10, 80, 100),
            "a standing person",
            priority=2,
            spatial_role="subject",
        )
        behind = RegionDefinition(
            "dog",
            "dog",
            PixelBox(50, 65, 95, 105),
            "a small brown dog",
            priority=1,
            spatial_role="subject",
        )

        plan = compile_regional_prompt_plan(128, 128, "outdoor scene", (front, behind))

        self.assertEqual([region.name for region in plan.regions], ["sface", "dog"])
        self.assertIn(
            "sface appears in front of dog where their target boxes overlap",
            plan.prompt,
        )
        self.assertIn("both occupy the shared image area as distinct subjects", plan.prompt)
        self.assertIn("dog naturally and partially occluded behind sface", plan.prompt)

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

        stronger_outside = spatial_pair_bias(
            (1.0, 0.5, 0.0), 2.0, outside_penalty=1.5
        )
        self.assertEqual(stronger_outside, (2.0, 0.25, -1.5))

    def test_attention_guidance_stays_strong_early_and_relaxes_late(self) -> None:
        region = RegionDefinition(
            "subject", "Subject", PixelBox(0, 0, 32, 32), "red vase"
        )
        plan = compile_regional_prompt_plan(
            64, 64, "gallery", (region,), late_step_scale=0.35
        )
        override = KreaSpatialAttentionOverride(
            plan.bind_tokens(lambda prefix: len(prefix.split()))
        )

        override.set_denoising_progress(4, 8)
        self.assertEqual(override.step_scale, 1.0)
        override.set_denoising_progress(8, 8)
        self.assertAlmostEqual(override.step_scale, 0.35)

    def test_subject_text_attention_is_partitioned_between_regions(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Torch is exercised in the configured ComfyUI worker environment")

        regions = (
            RegionDefinition("left", "Left", PixelBox(0, 0, 16, 16), "red coat"),
            RegionDefinition("right", "Right", PixelBox(16, 0, 32, 16), "blue coat"),
        )
        plan = compile_regional_prompt_plan(32, 16, "portrait", regions)
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
        override = KreaSpatialAttentionOverride(bound)
        owners = torch.tensor(text_region_ownership(bound), dtype=torch.int16)
        scores = torch.zeros((1, 1, bound.text_token_count, bound.text_token_count))

        override._partition_regional_text(
            scores, 0, bound.text_token_count, owners
        )

        left, right = bound.spans
        self.assertTrue(torch.isneginf(scores[0, 0, left.start, right.start]))
        self.assertTrue(torch.isneginf(scores[0, 0, right.start, left.start]))
        self.assertTrue(torch.isneginf(scores[0, 0, 0, left.start]))
        self.assertEqual(float(scores[0, 0, left.start, left.start]), 0.0)
        self.assertEqual(float(scores[0, 0, left.start, 0]), 0.0)

    def test_edit_clause_is_spatial_but_does_not_take_subject_ownership(self) -> None:
        regions = (
            RegionDefinition(
                "person",
                "Person",
                PixelBox(0, 0, 48, 64),
                "the same woman",
                spatial_role="subject",
            ),
            RegionDefinition(
                "edit",
                "Jacket edit",
                PixelBox(8, 20, 40, 55),
                "a green jacket",
                spatial_role="edit",
            ),
        )
        plan = compile_regional_prompt_plan(64, 64, "portrait", regions)
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
        owners = text_region_ownership(bound)
        subject, edit = bound.spans

        self.assertTrue(all(owner > 0 for owner in owners[subject.start : subject.end]))
        self.assertTrue(all(owner == 0 for owner in owners[edit.start : edit.end]))
        self.assertIn("desired final appearance", plan.regions[1].clause)

    def test_cross_modal_partition_preserves_image_to_image_attention(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("Torch is exercised in the configured ComfyUI worker environment")

        regions = (
            RegionDefinition("left", "Left", PixelBox(0, 0, 16, 16), "red coat"),
            RegionDefinition("right", "Right", PixelBox(16, 0, 32, 16), "blue coat"),
        )
        plan = compile_regional_prompt_plan(
            32, 16, "portrait", regions, falloff_pixels=0.0
        )
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
        override = KreaSpatialAttentionOverride(bound)
        reference = torch.zeros((1, 1, bound.text_token_count + 2, 1))
        _fields, _emphases, text_owners, image_owners = override._pair_fields(reference)
        scores = torch.zeros((1, 1, bound.text_token_count + 2, bound.text_token_count + 2))

        override._partition_regional_stream(
            scores,
            0,
            bound.text_token_count + 2,
            text_owners,
            image_owners,
        )

        left, right = bound.spans
        left_image = bound.text_token_count
        right_image = left_image + 1
        self.assertTrue(torch.isneginf(scores[0, 0, left.start, right_image]))
        self.assertTrue(torch.isneginf(scores[0, 0, right_image, left.start]))
        self.assertEqual(float(scores[0, 0, left_image, right_image]), 0.0)
        self.assertEqual(float(scores[0, 0, right_image, left_image]), 0.0)
        self.assertTrue(torch.isneginf(scores[0, 0, 0, left_image]))
        self.assertEqual(float(scores[0, 0, left.start, left_image]), 0.0)
        self.assertEqual(float(scores[0, 0, left_image, left.start]), 0.0)
        self.assertEqual(float(scores[0, 0, left_image, 0]), 0.0)

    def test_lora_delta_adaptation_uses_bounded_region_scales(self) -> None:
        region = RegionDefinition(
            "subject", "Subject", PixelBox(0, 0, 32, 32), "red vase"
        )
        plan = compile_regional_prompt_plan(64, 64, "gallery", (region,))
        override = KreaSpatialAttentionOverride(
            plan.bind_tokens(lambda prefix: len(prefix.split())),
            lora_delta_adaptation=True,
            lora_delta_adaptation_gain=0.5,
        )

        override.set_lora_delta_scales({"subject": 3.0, "unknown": 0.0})

        self.assertEqual(override.region_scales, {"subject": 1.5})
        summary = override.summary()
        self.assertTrue(summary["lora_delta_adaptation"])
        self.assertEqual(summary["final_region_scales"], {"subject": 1.5})

    def test_prompt_emphasis_binds_global_and_region_phrases_to_tokens(self) -> None:
        region = RegionDefinition(
            "subject", "Subject", PixelBox(0, 0, 32, 64), "a red glass vase"
        )
        plan = compile_regional_prompt_plan(
            64,
            64,
            "two distinct people in a gallery",
            (region,),
            emphases=(
                PromptEmphasis(GLOBAL_EMPHASIS_SCOPE, "distinct people", 0.5),
                PromptEmphasis("subject", "red glass", 0.4),
            ),
        )

        bound = plan.bind_tokens(lambda prefix: len(prefix.split()))

        self.assertEqual(len(bound.emphases), 2)
        self.assertEqual(bound.emphases[0].phrase, "distinct people")
        self.assertEqual(bound.emphases[0].image_token_field, (1.0,) * 16)
        self.assertEqual(bound.emphases[1].phrase, "red glass")
        self.assertEqual(
            bound.emphases[1].image_token_field, plan.regions[0].image_token_field
        )
        self.assertLess(bound.emphases[0].start, bound.emphases[0].end)
        self.assertLess(bound.emphases[1].start, bound.emphases[1].end)

    def test_prompt_emphasis_excludes_tokenized_leading_space_from_start(self) -> None:
        region = RegionDefinition(
            "subject", "Subject", PixelBox(0, 0, 32, 64), "woman in foreground"
        )
        plan = compile_regional_prompt_plan(
            64,
            64,
            "gallery",
            (region,),
            emphases=(PromptEmphasis("subject", "foreground", 0.8),),
        )

        def qwen_like_prefix_count(prefix: str) -> int:
            words = len(prefix.rstrip().split())
            return words + int(bool(prefix) and prefix[-1].isspace())

        bound = plan.bind_tokens(qwen_like_prefix_count)

        self.assertEqual(bound.emphases[0].end - bound.emphases[0].start, 1)

    def test_subject_field_peaks_at_center_while_background_fills_its_box(self) -> None:
        subject = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(16, 16, 80, 80),
            "a person",
            spatial_role="subject",
        )
        background = RegionDefinition(
            "background",
            "Background",
            PixelBox(16, 16, 80, 80),
            "a wall",
            spatial_role="background",
        )

        subject_plan = compile_regional_prompt_plan(
            96, 96, "scene", (subject,), falloff_pixels=16
        )
        background_plan = compile_regional_prompt_plan(
            96, 96, "scene", (background,), falloff_pixels=16
        )
        center = 2 * 6 + 2
        near_edge = 1 * 6 + 1

        self.assertGreater(
            subject_plan.regions[0].image_token_field[center],
            subject_plan.regions[0].image_token_field[near_edge],
        )
        self.assertEqual(
            background_plan.regions[0].image_token_field[center], 1.0
        )
        self.assertEqual(
            background_plan.regions[0].image_token_field[near_edge], 1.0
        )

    def test_subject_fill_strengthens_box_edges_and_can_be_disabled(self) -> None:
        subject = RegionDefinition(
            "subject",
            "Subject",
            PixelBox(16, 16, 80, 80),
            "a standing person",
            spatial_role="subject",
        )
        filled = compile_regional_prompt_plan(
            96, 96, "scene", (subject,), falloff_pixels=16, subject_fill=True
        )
        positioned = compile_regional_prompt_plan(
            96, 96, "scene", (subject,), falloff_pixels=16, subject_fill=False
        )
        near_edge = 1 * 6 + 1

        self.assertGreater(
            filled.regions[0].image_token_field[near_edge],
            positioned.regions[0].image_token_field[near_edge],
        )
        self.assertIn("minimal empty margin", filled.prompt)
        self.assertNotIn("minimal empty margin", positioned.prompt)
        self.assertTrue(filled.summary()["subject_fill"])

    def test_overlapping_subjects_compete_without_changing_background_field(self) -> None:
        regions = (
            RegionDefinition(
                "left", "Left", PixelBox(8, 8, 56, 56), "red vase", spatial_role="subject"
            ),
            RegionDefinition(
                "right", "Right", PixelBox(24, 8, 72, 56), "blue vase", spatial_role="subject"
            ),
            RegionDefinition(
                "wall", "Wall", PixelBox(0, 0, 96, 96), "white wall", spatial_role="background"
            ),
        )
        raw = compile_regional_prompt_plan(
            96, 96, "gallery", regions, subject_competition=False
        )
        competed = compile_regional_prompt_plan(
            96, 96, "gallery", regions, subject_competition=True
        )
        raw_by_id = {region.region_id: region for region in raw.regions}
        competed_by_id = {region.region_id: region for region in competed.regions}

        self.assertTrue(
            any(
                competed_value < raw_value
                for competed_value, raw_value in zip(
                    competed_by_id["left"].image_token_field,
                    raw_by_id["left"].image_token_field,
                    strict=True,
                )
            )
        )
        self.assertEqual(
            competed_by_id["wall"].image_token_field,
            raw_by_id["wall"].image_token_field,
        )

    def test_v1_project_migrates_to_current_schema_on_save(self) -> None:
        old_document = {
            "schema": "k2-region-lab-project",
            "version": 1,
            "canvas": {"width": 512, "height": 512},
            "generation": {},
            "regions": [
                {
                    "id": "legacy",
                    "name": "Legacy",
                    "box": {"x0": 0, "y0": 0, "x1": 256, "y1": 256},
                    "prompt": "a tree",
                }
            ],
        }

        state = project_state(old_document)

        self.assertEqual(state.regions[0].spatial_role, "auto")
        self.assertEqual(state.regional_outside_penalty, 1.0)
        self.assertTrue(state.regional_subject_competition)
        self.assertTrue(state.regional_subject_fill)
        self.assertEqual(project_document(state)["version"], PROJECT_VERSION)

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

    def test_character_identity_anchor_is_added_and_bound_to_trigger_tokens(self) -> None:
        region = RegionDefinition(
            "person",
            "Person",
            PixelBox(0, 0, 32, 64),
            "lface, an adult woman",
            spatial_role="subject",
        )

        plan = compile_regional_prompt_plan(
            64,
            64,
            "portrait",
            (region,),
            character_identity_triggers={"person": ("lface",)},
        )
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))

        self.assertIn("lface identifies the person in this region", plan.prompt)
        self.assertIn("Generate this person's face and facial identity", plan.prompt)
        self.assertEqual(len(plan.character_identities), 1)
        identity = plan.character_identities[0]
        self.assertEqual(identity.region_id, "person")
        self.assertEqual(identity.trigger_phrase, "lface")
        self.assertEqual(
            [plan.prompt[start:end] for start, end in identity.character_spans],
            ["lface", "lface"],
        )
        self.assertEqual(
            bound.character_identities[0].token_spans,
            identity.character_spans,
        )

    def test_face_identity_prompt_is_compiled_and_bound_separately(self) -> None:
        region = RegionDefinition(
            "person",
            "Person",
            PixelBox(0, 0, 32, 64),
            "standing beside a window",
            face_identity_prompt=(
                "lface, a specific woman with brown hair and an oval face"
            ),
            spatial_role="subject",
        )

        plan = compile_regional_prompt_plan(64, 64, "portrait", (region,))
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))

        self.assertIn(
            "lface, a specific woman with brown hair and an oval face. "
            "standing beside a window",
            plan.prompt,
        )
        self.assertEqual(len(plan.face_identities), 1)
        identity = plan.face_identities[0]
        self.assertEqual(
            plan.prompt[slice(*identity.character_span)],
            "lface, a specific woman with brown hair and an oval face",
        )
        self.assertEqual(
            (bound.face_identities[0].start, bound.face_identities[0].end),
            identity.character_span,
        )

    def test_worker_payload_conversion_preserves_generic_region_fields(self) -> None:
        regions = region_definitions_from_payload(
            [
                {
                    "id": "anything",
                    "name": "Anything",
                    "box": {"x0": 4, "y0": 8, "x1": 40, "y1": 56},
                    "prompt": "a small tree",
                    "negative_prompt": "building",
                    "face_identity_prompt": "treeface, an old oak",
                    "enabled": True,
                    "priority": 3,
                    "spatial_role": "subject",
                }
            ]
        )

        self.assertEqual(regions[0].name, "Anything")
        self.assertEqual(regions[0].negative_prompt, "building")
        self.assertEqual(regions[0].face_identity_prompt, "treeface, an old oak")
        self.assertEqual(regions[0].priority, 3)
        self.assertEqual(regions[0].spatial_role, "subject")


if __name__ == "__main__":
    unittest.main()
