from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def debug_enabled() -> bool:
    return os.environ.get("DEBUG", "").strip() == "1"


def configure_debug_logging(component: str, data_directory: Path | None = None) -> Path | None:
    """Enable a bounded component log for every application process."""

    root = data_directory or Path(
        os.environ.get("K2LAB_DATA_DIR", "~/.local/share/k2-region-lab")
    ).expanduser()
    log_directory = root.resolve() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{component}-debug.log"

    logger = logging.getLogger("k2_region_lab")
    logger.setLevel(logging.DEBUG)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.debug("%s logging initialized at %s", component, log_path)
    return log_path
