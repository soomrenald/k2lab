from __future__ import annotations

import unittest

from k2_region_lab.project import ProjectState, project_document, project_state


class ProjectStateTests(unittest.TestCase):
    def test_late_step_scale_round_trips(self) -> None:
        state = ProjectState(
            canvas_width=1024,
            canvas_height=1024,
            regional_late_step_scale=0.8,
        )

        document = project_document(state)

        self.assertEqual(document["version"], 9)
        self.assertEqual(document["generation"]["regional_late_step_scale"], 0.8)
        self.assertEqual(project_state(document).regional_late_step_scale, 0.8)

    def test_legacy_project_uses_existing_relaxation_default(self) -> None:
        document = project_document(ProjectState(canvas_width=1024, canvas_height=1024))
        document["version"] = 8
        document["generation"].pop("regional_late_step_scale")

        self.assertEqual(project_state(document).regional_late_step_scale, 0.35)


if __name__ == "__main__":
    unittest.main()
