from __future__ import annotations

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from k2_region_lab.debug import configure_debug_logging


class DebugLoggingTests(unittest.TestCase):
    def test_debug_one_writes_bounded_component_log(self) -> None:
        logger = logging.getLogger("k2_region_lab")
        previous_handlers = tuple(logger.handlers)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"DEBUG": "1"}, clear=False):
                log_path = configure_debug_logging("test", Path(directory))
                logging.getLogger("k2_region_lab.test").debug("diagnostic marker")
                for handler in logger.handlers:
                    handler.flush()

            self.assertIsNotNone(log_path)
            self.assertIn("diagnostic marker", log_path.read_text(encoding="utf-8"))

        for handler in tuple(logger.handlers):
            if handler not in previous_handlers:
                logger.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()
