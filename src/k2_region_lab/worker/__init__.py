"""Isolated GPU worker protocol.

Keep this package import dependency-free.  The external worker first imports
``k2_region_lab.worker.bootstrap`` with the ComfyUI interpreter, before that
interpreter can see the desktop environment's installed ``k2core`` package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CommandKind", "WorkerCommand", "WorkerEvent", "WorkerState"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from k2_region_lab.worker import protocol

    return getattr(protocol, name)
