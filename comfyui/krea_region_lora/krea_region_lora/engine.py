from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .types import K2RegionalLora, K2RegionalLoraStack, OverlapMode


PredictFn = Callable[[str, torch.Tensor, float, Any, Any], torch.Tensor]
StepFn = Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor]


@dataclass
class RegionalSamplerDebug:
    outside_equal_after_step: list[bool] = field(default_factory=list)
    active_counts: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        pieces = [
            f"steps={len(self.active_counts)}",
            f"active_counts={self.active_counts}",
            f"outside_pinned={all(self.outside_equal_after_step) if self.outside_equal_after_step else True}",
        ]
        pieces.extend(self.notes)
        return "; ".join(pieces)


def euler_velocity_step(
    x: torch.Tensor, velocity: torch.Tensor, t_current: float, t_prev: float
) -> torch.Tensor:
    return x + (float(t_prev) - float(t_current)) * velocity


def run_regional_velocity_sampler(
    *,
    initial: torch.Tensor,
    stack: K2RegionalLoraStack,
    base_positive: Any,
    base_negative: Any,
    cfg: float,
    schedule: Sequence[float],
    predict: PredictFn,
    step_fn: StepFn = euler_velocity_step,
    pin_outside_regions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RegionalSamplerDebug]:
    x_base = initial.clone()
    x_regional = initial.clone()
    debug = RegionalSamplerDebug()
    if len(schedule) < 2:
        return x_regional, x_base, torch.zeros_like(initial[:, :1]), debug

    enabled = stack.enabled_regions
    union_mask = _union_mask_for(enabled, initial)

    for step_index in range(len(schedule) - 1):
        denom = max(1, len(schedule) - 2)
        step_percent = step_index / denom
        t_current = float(schedule[step_index])
        t_prev = float(schedule[step_index + 1])
        active = [regional for regional in enabled if regional.active_at(step_percent)]
        debug.active_counts.append(len(active))

        v_base_for_base = predict("base", x_base, t_current, base_positive, base_negative)
        x_base_next = step_fn(x_base, v_base_for_base, t_current, t_prev)

        if not active:
            x_regional_next = (
                x_base_next.clone()
                if pin_outside_regions
                else step_fn(
                    x_regional,
                    predict("base", x_regional, t_current, base_positive, base_negative),
                    t_current,
                    t_prev,
                )
            )
        else:
            v_base_for_region = predict("base", x_regional, t_current, base_positive, base_negative)
            region_deltas: list[tuple[torch.Tensor, torch.Tensor]] = []
            for regional in active:
                v_lora = predict(
                    regional.lora_name, x_regional, t_current, regional.positive, base_negative
                )
                delta = (
                    (v_lora - v_base_for_region)
                    * float(regional.delta_strength)
                    * float(regional.lora_strength)
                )
                mask = regional.region.mask_for(v_base_for_region)
                region_deltas.append((mask, delta))
            regional_delta = resolve_overlaps(region_deltas, stack.overlap_mode, v_base_for_region)
            x_regional_next = step_fn(
                x_regional, v_base_for_region + regional_delta, t_current, t_prev
            )

        if pin_outside_regions:
            persist_mask = union_mask.to(device=x_regional_next.device, dtype=x_regional_next.dtype)
            x_regional = persist_mask * x_regional_next + (1.0 - persist_mask) * x_base_next
            outside_equal = torch.equal(
                ((1.0 - persist_mask) * x_regional), ((1.0 - persist_mask) * x_base_next)
            )
            debug.outside_equal_after_step.append(bool(outside_equal))
        else:
            x_regional = x_regional_next
            debug.outside_equal_after_step.append(False)
        x_base = x_base_next

    return x_regional, x_base, union_mask, debug


def resolve_overlaps(
    region_deltas: Sequence[tuple[torch.Tensor, torch.Tensor]],
    overlap_mode: OverlapMode,
    reference: torch.Tensor,
) -> torch.Tensor:
    if not region_deltas:
        return torch.zeros_like(reference)
    if overlap_mode == "normalize":
        delta_accum = torch.zeros_like(reference)
        weight_accum = torch.zeros_like(reference[:, :1])
        for mask, delta in region_deltas:
            delta_accum = delta_accum + mask * delta
            weight_accum = weight_accum + mask
        return delta_accum / weight_accum.clamp_min(1.0)
    if overlap_mode == "add_clamped":
        delta_accum = torch.zeros_like(reference)
        for mask, delta in region_deltas:
            delta_accum = delta_accum + mask * delta
        return delta_accum.clamp(-1.0, 1.0)
    if overlap_mode in ("priority_1", "priority_3"):
        ordered = list(region_deltas)
        if overlap_mode == "priority_3":
            ordered = list(reversed(ordered))
        out = torch.zeros_like(reference)
        claimed = torch.zeros_like(reference[:, :1])
        for mask, delta in ordered:
            usable = (mask * (1.0 - claimed)).clamp(0.0, 1.0)
            out = out + usable * delta
            claimed = torch.maximum(claimed, mask)
        return out
    raise ValueError(f"Unsupported overlap_mode {overlap_mode}")


def _union_mask_for(regions: Sequence[K2RegionalLora], target: torch.Tensor) -> torch.Tensor:
    if not regions:
        return torch.zeros(
            (target.shape[0], 1, *target.shape[-2:]), device=target.device, dtype=target.dtype
        )
    union = torch.zeros(
        (target.shape[0], 1, *target.shape[-2:]), device=target.device, dtype=target.dtype
    )
    for regional in regions:
        union = torch.maximum(
            union, regional.region.mask_for(target).to(device=target.device, dtype=target.dtype)
        )
    return union.clamp(0.0, 1.0)
