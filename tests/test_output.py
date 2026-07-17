from __future__ import annotations

import unittest

from k2_region_lab.output import validate_filename_prefix


class OutputSettingsTests(unittest.TestCase):
    def test_prefix_is_trimmed_but_human_readable_names_are_allowed(self) -> None:
        self.assertEqual(validate_filename_prefix("  beach study  "), "beach study")

    def test_prefix_cannot_escape_the_output_directory(self) -> None:
        for prefix in ("", ".", "..", "../escape", "nested/name", "nested\\name"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                validate_filename_prefix(prefix)


if __name__ == "__main__":
    unittest.main()
