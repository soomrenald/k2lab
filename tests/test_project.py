from __future__ import annotations

import unittest

from pathlib import Path

from k2_region_lab.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    STANDARD_LORA_ROUTING,
)
from k2_region_lab.project import (
    PROJECT_VERSION,
    ProjectState,
    SavedLora,
    project_document,
    project_state,
)
from k2_region_lab.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis
from k2_region_lab.regions import PixelBox, RegionDefinition


class ProjectStateTests(unittest.TestCase):
    def test_late_step_scale_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            regional_late_step_scale=0.8,
            regional_lora_delta_adaptation=True,
            regional_lora_delta_adaptation_gain=0.6,
            strict_regional_lora_isolation=False,
            face_detail_seed=123,
            face_detail_steps=10,
            face_detail_denoise=0.25,
            face_detail_crop_size=768,
            face_detail_padding=1.8,
            face_detail_feather=0.16,
            face_detail_blend=0.4,
            face_detail_lora_scale=1.2,
            face_detail_detector_threshold=0.35,
            projector_identity_protection=0.65,
            prompt_emphases=(
                PromptEmphasis(GLOBAL_EMPHASIS_SCOPE, "two distinct people", 0.5),
            ),
        )

        document = project_document(state)

        self.assertEqual(document["version"], PROJECT_VERSION)
        self.assertEqual(document["generation"]["regional_late_step_scale"], 0.8)
        self.assertEqual(project_state(document).regional_late_step_scale, 0.8)
        self.assertTrue(project_state(document).regional_lora_delta_adaptation)
        self.assertEqual(
            project_state(document).regional_lora_delta_adaptation_gain, 0.6
        )
        self.assertFalse(project_state(document).strict_regional_lora_isolation)
        self.assertEqual(project_state(document).prompt_emphases[0].phrase, "two distinct people")
        restored = project_state(document)
        self.assertEqual(restored.face_detail_seed, 123)
        self.assertEqual(restored.face_detail_steps, 10)
        self.assertEqual(restored.face_detail_denoise, 0.25)
        self.assertEqual(restored.face_detail_crop_size, 768)
        self.assertEqual(restored.face_detail_padding, 1.8)
        self.assertEqual(restored.face_detail_feather, 0.16)
        self.assertEqual(restored.face_detail_blend, 0.4)
        self.assertEqual(restored.face_detail_lora_scale, 1.2)
        self.assertEqual(restored.face_detail_detector_threshold, 0.35)
        self.assertEqual(restored.projector_identity_protection, 0.65)

    def test_character_identity_lora_routing_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            regions=(
                RegionDefinition(
                    "person",
                    "Person",
                    PixelBox(0, 0, 512, 1024),
                    "lface, an adult woman",
                    face_identity_prompt="lface, a specific woman with an oval face",
                ),
            ),
            loras=(
                SavedLora(
                    Path("lface.safetensors"),
                    global_scope=False,
                    region_ids=("person",),
                    strength=1.5,
                    routing_mode=CHARACTER_IDENTITY_LORA_ROUTING,
                    trigger_phrase="lface",
                ),
            ),
        )

        document = project_document(state)
        restored = project_state(document)

        self.assertEqual(
            document["loras"][0]["routing_mode"],
            CHARACTER_IDENTITY_LORA_ROUTING,
        )
        self.assertEqual(document["loras"][0]["trigger_phrase"], "lface")
        self.assertEqual(
            restored.loras[0].routing_mode,
            CHARACTER_IDENTITY_LORA_ROUTING,
        )
        self.assertEqual(restored.loras[0].trigger_phrase, "lface")
        self.assertEqual(
            restored.regions[0].face_identity_prompt,
            "lface, a specific woman with an oval face",
        )

    def test_version_twelve_lora_uses_standard_routing_defaults(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 12
        document["loras"] = [{"path": "style.safetensors", "global": True}]

        restored = project_state(document)

        self.assertEqual(restored.loras[0].routing_mode, STANDARD_LORA_ROUTING)
        self.assertEqual(restored.loras[0].trigger_phrase, "")

    def test_legacy_project_uses_existing_relaxation_default(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 8
        document["generation"].pop("regional_late_step_scale")

        self.assertEqual(project_state(document).regional_late_step_scale, 0.35)

    def test_version_eleven_project_uses_safe_face_refinement_defaults(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 11
        for key in tuple(document["generation"]):
            if key.startswith("face_detail_"):
                document["generation"].pop(key)

        restored = project_state(document)

        self.assertEqual(restored.face_detail_seed, 0)
        self.assertEqual(restored.face_detail_denoise, 0.15)
        self.assertEqual(restored.face_detail_blend, 0.5)


if __name__ == "__main__":
    unittest.main()
