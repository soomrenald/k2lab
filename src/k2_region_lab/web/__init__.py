"""Provider-neutral web control plane for K2 Region Lab."""

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name != "create_app":
        raise AttributeError(name)
    from k2_region_lab.web.app import create_app

    return create_app
