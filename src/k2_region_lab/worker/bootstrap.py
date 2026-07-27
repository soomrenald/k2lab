"""Expose the installed k2core package to an isolated GPU worker."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


K2CORE_PACKAGE_ENV = "K2LAB_K2CORE_PACKAGE"


def bootstrap_k2core(package_directory: str | Path | None = None) -> Path:
    """Import k2core from one exact package directory.

    Only the ``k2core`` package is borrowed from the lightweight desktop
    environment.  Its site-packages directory is removed from ``sys.path``
    immediately after the import, so NumPy, Pillow, ONNX Runtime, Torch, and
    ComfyUI continue to come from the selected GPU-worker environment.
    """

    configured = package_directory or os.environ.get(K2CORE_PACKAGE_ENV, "")
    package_path = Path(configured).expanduser().resolve() if configured else None
    if package_path is None:
        module = importlib.import_module("k2core")
        return Path(module.__file__).resolve().parent
    if package_path.name != "k2core" or not (package_path / "__init__.py").is_file():
        raise RuntimeError(
            f"{K2CORE_PACKAGE_ENV} must name an installed k2core package directory: "
            f"{package_path}"
        )

    existing = sys.modules.get("k2core")
    if existing is not None:
        loaded_path = Path(existing.__file__).resolve().parent
        if loaded_path != package_path:
            raise RuntimeError(
                f"k2core was loaded from {loaded_path}, expected {package_path}"
            )
        return loaded_path

    package_parent = str(package_path.parent)
    sys.path.insert(0, package_parent)
    try:
        module = importlib.import_module("k2core")
    finally:
        try:
            sys.path.remove(package_parent)
        except ValueError:
            pass
    loaded_path = Path(module.__file__).resolve().parent
    if loaded_path != package_path:
        raise RuntimeError(f"k2core loaded from {loaded_path}, expected {package_path}")
    return loaded_path


def main() -> int:
    bootstrap_k2core()
    from k2_region_lab.worker.entrypoint import main as worker_main

    return worker_main()


if __name__ == "__main__":
    raise SystemExit(main())
