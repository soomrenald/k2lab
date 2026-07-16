from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from k2_region_lab.config import AppSettings
from k2_region_lab.debug import configure_debug_logging
from k2_region_lab.model import discover_model_artifacts


def _format_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.2f} GiB"


def check_models(settings: AppSettings) -> int:
    artifacts = discover_model_artifacts(settings.model_directories)
    for artifact in artifacts.present():
        dtypes = ", ".join(f"{name}:{count}" for name, count in artifact.summary.dtypes)
        quantized = " quantized" if artifact.summary.quantized else ""
        print(
            f"{artifact.kind.value}: {artifact.path} "
            f"({_format_size(artifact.size_bytes)}, {artifact.summary.tensor_count} tensors, "
            f"{dtypes}{quantized})"
        )
    missing = 3 - len(artifacts.present())
    if missing:
        print(f"model set incomplete: {missing} component(s) missing", file=sys.stderr)
        return 1
    print("model set complete")
    return 0


def launch_desktop(settings: AppSettings) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is not installed. Install the project desktop dependencies before "
            "launching, or use --check-models for the headless foundation check.",
            file=sys.stderr,
        )
        return 2

    from k2_region_lab.desktop.main_window import MainWindow

    application = QApplication(sys.argv)
    application.setApplicationName("K2 Region Lab")
    window = MainWindow(settings)
    window.show()
    return application.exec()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k2lab")
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="inspect configured safetensors headers without starting Qt",
    )
    args = parser.parse_args(argv)
    settings = AppSettings.from_environment()
    log_path = configure_debug_logging("desktop", settings.data_directory)
    if log_path is not None:
        logging.getLogger(__name__).debug("application settings: %r", settings)
        print(f"K2 Region Lab debug log: {log_path}", file=sys.stderr)
    if args.check_models:
        return check_models(settings)
    return launch_desktop(settings)
