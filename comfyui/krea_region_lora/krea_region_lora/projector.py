from __future__ import annotations

from typing import Any

import torch

from .types import K2ProjectorSettings


PROJECTOR_TARGETS = (
    "diffusion_model.txtfusion.projector.weight",
    "txtfusion.projector.weight",
    "model.diffusion_model.txtfusion.projector.weight",
)


def apply_projector_settings(model: Any, settings: K2ProjectorSettings) -> tuple[Any, str]:
    values = tuple(float(value) * float(settings.multiplier) for value in settings.values)
    if len(values) != 12:
        raise ValueError("Krea projector vector must contain 12 values")
    if not settings.enabled or not any(values):
        return model, f"Projector disabled (preset={settings.preset})."
    state_owner = getattr(model, "model", model)
    state = state_owner.state_dict() if hasattr(state_owner, "state_dict") else {}
    target = next((name for name in PROJECTOR_TARGETS if name in state), None)
    if target is None:
        available = [name for name in state if "txtfusion.projector" in str(name)]
        if available:
            target = str(available[0])
        else:
            raise RuntimeError(
                "This MODEL does not expose Krea's txtfusion.projector.weight; "
                "the projector controls only apply to a Krea 2 diffusion model."
            )
    weight = state[target]
    if tuple(weight.shape) != (1, 12):
        raise RuntimeError(f"unexpected Krea projector shape {tuple(weight.shape)}")
    delta = torch.tensor((values,), dtype=torch.float32)
    patched = model.clone()
    patched_keys = patched.add_patches({target: ("diff", (delta,))})
    if target not in patched_keys:
        raise RuntimeError(f"ComfyUI rejected the Krea projector patch for {target}")
    report = (
        f"Applied projector preset={settings.preset} multiplier={settings.multiplier:.4f} "
        f"target={target}; identity_protection={settings.identity_protection:.2f}. "
        "Identity prompts are encoded as separate regional conditionings in ComfyUI."
    )
    return patched, report
