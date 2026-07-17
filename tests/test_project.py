from __future__ import annotations

import unittest

from k2_region_lab.project import ProjectState, project_document, project_state
from k2_region_lab.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis


class ProjectStateTests(unittest.TestCase):
    def test_late_step_scale_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            regional_late_step_scale=0.8,
            regional_lora_delta_adaptation=True,
            regional_lora_delta_adaptation_gain=0.6,
            prompt_emphases=(
                PromptEmphasis(GLOBAL_EMPHASIS_SCOPE, "two distinct people", 0.5),
            ),
        )

        document = project_document(state)

        self.assertEqual(document["version"], 11)
        self.assertEqual(document["generation"]["regional_late_step_scale"], 0.8)
        self.assertEqual(project_state(document).regional_late_step_scale, 0.8)
        self.assertTrue(project_state(document).regional_lora_delta_adaptation)
        self.assertEqual(
            project_state(document).regional_lora_delta_adaptation_gain, 0.6
        )
        self.assertEqual(project_state(document).prompt_emphases[0].phrase, "two distinct people")

    def test_legacy_project_uses_existing_relaxation_default(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 8
        document["generation"].pop("regional_late_step_scale")

        self.assertEqual(project_state(document).regional_late_step_scale, 0.35)


if __name__ == "__main__":
    unittest.main()
