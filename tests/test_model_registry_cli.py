from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from k2_region_lab import app
from k2core.model import ModelRegistry


class ModelRegistryCliTests(unittest.TestCase):
    def test_validate_parse_failure_is_clean_and_does_not_construct_desktop_settings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.toml"
            path.write_text("not valid TOML [", encoding="utf-8")
            error = io.StringIO()

            with (
                patch.object(app.AppSettings, "from_environment") as settings,
                redirect_stderr(error),
            ):
                result = app.main(["--validate-model-registry", str(path)])

            self.assertEqual(result, 2)
            self.assertIn("model registry could not be read", error.getvalue())
            settings.assert_not_called()

    def test_scan_prints_generated_registry_without_launching(self) -> None:
        settings = SimpleNamespace(model_directories=object(), data_directory=Path("/tmp"))
        registry = Mock(spec=ModelRegistry)
        registry.to_toml.return_value = 'schema_version = "k2lab-model-registry/1"\n'
        output = io.StringIO()

        with (
            patch.object(app.AppSettings, "from_environment", return_value=settings),
            patch.object(app, "configure_debug_logging") as logging,
            patch.object(app, "scan_legacy_comfyui_models", return_value=registry) as scan,
            patch.object(app, "launch_desktop") as launch,
            redirect_stdout(output),
        ):
            result = app.main(["--scan-comfyui-models"])

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), registry.to_toml.return_value)
        scan.assert_called_once_with(settings.model_directories)
        logging.assert_not_called()
        launch.assert_not_called()

    def test_scan_failure_has_nonzero_exit_and_no_partial_config(self) -> None:
        settings = SimpleNamespace(model_directories=object(), data_directory=Path("/tmp"))
        output = io.StringIO()
        error = io.StringIO()

        with (
            patch.object(app.AppSettings, "from_environment", return_value=settings),
            patch.object(app, "configure_debug_logging") as logging,
            patch.object(
                app,
                "scan_legacy_comfyui_models",
                side_effect=FileNotFoundError("no Krea transformer"),
            ),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            result = app.main(["--scan-comfyui-models"])

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("no Krea transformer", error.getvalue())
        logging.assert_not_called()


if __name__ == "__main__":
    unittest.main()
