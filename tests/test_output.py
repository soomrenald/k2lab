from __future__ import annotations

import unittest

from k2_region_lab.output import (
    default_output_directory,
    default_prompt_directory,
    project_workspace_directory,
    validate_filename_prefix,
)


class OutputSettingsTests(unittest.TestCase):
    def test_repository_prompt_and_output_defaults_are_siblings(self) -> None:
        root = project_workspace_directory()

        self.assertEqual(default_prompt_directory(), root / "prompts")
        self.assertEqual(default_output_directory(), root / "outputs")

    def test_prefix_is_trimmed_but_human_readable_names_are_allowed(self) -> None:
        self.assertEqual(validate_filename_prefix("  beach study  "), "beach study")

    def test_prefix_cannot_escape_the_output_directory(self) -> None:
        for prefix in ("", ".", "..", "../escape", "nested/name", "nested\\name"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                validate_filename_prefix(prefix)


if __name__ == "__main__":
    unittest.main()
