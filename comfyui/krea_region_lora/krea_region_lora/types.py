from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


BBoxFormat = Literal["xywh", "xyxy"]
BatchMode = Literal["single", "repeat", "per_batch"]
OverlapMode = Literal["normalize", "priority_1", "priority_3", "add_clamped"]
SpatialRole = Literal["auto", "subject", "background"]
LoraRoutingMode = Literal["standard", "character_identity"]


@dataclass(frozen=True)
class K2Region:
    pixel_bbox: tuple[int, int, int, int]
    image_size: tuple[int, int]
    pixel_mask: torch.Tensor
    latent_mask: torch.Tensor
    token_mask: torch.Tensor
    bbox_format: BBoxFormat = "xywh"
    bbox_index: int = 0
    batch_mode: BatchMode = "repeat"
    metadata: dict[str, Any] = field(default_factory=dict)

    def mask_for(self, target: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        shape = tuple(target.shape) if isinstance(target, torch.Tensor) else tuple(target)
        if len(shape) == 4:
            mask = self.latent_mask
            return _fit_mask_batch(mask, shape[0]).to(
                device=target.device if isinstance(target, torch.Tensor) else mask.device,
                dtype=target.dtype if isinstance(target, torch.Tensor) else mask.dtype,
            )
        if len(shape) == 3:
            mask = self.token_mask
            return _fit_mask_batch(mask, shape[0]).to(
                device=target.device if isinstance(target, torch.Tensor) else mask.device,
                dtype=target.dtype if isinstance(target, torch.Tensor) else mask.dtype,
            )
        if len(shape) == 2:
            mask = self.pixel_mask
            return _fit_mask_batch(mask, shape[0]).to(
                device=target.device if isinstance(target, torch.Tensor) else mask.device,
                dtype=target.dtype if isinstance(target, torch.Tensor) else mask.dtype,
            )
        raise ValueError(f"Cannot build a region mask for shape {shape}")


@dataclass(frozen=True)
class K2RegionSpec:
    """A named drawable region and its native ComfyUI conditionings."""

    region: K2Region
    name: str
    prompt: str = ""
    negative_prompt: str = ""
    face_identity_prompt: str = ""
    enabled: bool = True
    priority: int = 0
    spatial_role: SpatialRole = "auto"
    positive: Any = None
    negative: Any = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("region names must not be empty")
        if self.spatial_role not in ("auto", "subject", "background"):
            raise ValueError(f"unsupported spatial role {self.spatial_role!r}")

    @property
    def region_id(self) -> str:
        return str(self.region.metadata.get("region_id", self.name))


@dataclass(frozen=True)
class K2PromptEmphasis:
    scope_id: str
    phrase: str
    strength: float = 0.5
    occurrence: int = 0


@dataclass(frozen=True)
class K2RegionalSettings:
    regional_prompting: bool = True
    inside_strength: float = 1.0
    outside_penalty: float = 1.0
    feather_pixels: int = 128
    subject_competition: bool = True
    subject_fill: bool = True
    relaxation: bool = True
    late_step_scale: float = 0.35
    lora_delta_adaptation: bool = False
    lora_delta_adaptation_gain: float = 0.35


@dataclass(frozen=True)
class K2ProjectorSettings:
    enabled: bool = False
    preset: str = "filter_bypass2"
    values: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.5117,
        -0.8906,
        0.0,
        0.0,
    )
    multiplier: float = 1.0
    identity_protection: float = 1.0


@dataclass(frozen=True)
class K2RegionLayout:
    width: int
    height: int
    regions: tuple[K2RegionSpec, ...]
    global_prompt: str = ""
    global_negative: str = ""
    global_positive: Any = None
    global_negative_conditioning: Any = None
    emphases: tuple[K2PromptEmphasis, ...] = ()
    regional: K2RegionalSettings = field(default_factory=K2RegionalSettings)
    projector: K2ProjectorSettings = field(default_factory=K2ProjectorSettings)
    source_document: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("layout dimensions must be at least 16 pixels")
        names = [region.name.casefold() for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("region names must be unique")

    def named_region(self, name_or_id: str) -> K2RegionSpec | None:
        needle = name_or_id.strip().casefold()
        return next(
            (
                region
                for region in self.regions
                if region.name.casefold() == needle or region.region_id.casefold() == needle
            ),
            None,
        )


@dataclass(frozen=True)
class K2LoraReference:
    lora_name: str
    strength: float = 1.0
    routing_mode: LoraRoutingMode = "standard"
    trigger_phrase: str = ""
    enabled: bool = True
    start_percent: float = 0.0
    end_percent: float = 1.0

    def __post_init__(self) -> None:
        if self.routing_mode not in ("standard", "character_identity"):
            raise ValueError(f"unsupported LoRA routing mode {self.routing_mode!r}")
        if self.routing_mode == "character_identity" and not self.trigger_phrase.strip():
            raise ValueError("character identity LoRAs require a trigger phrase")


@dataclass(frozen=True)
class K2RegionalLora:
    region: K2Region
    positive: Any
    negative: Any
    lora_name: str
    lora_strength: float = 1.0
    delta_strength: float = 1.0
    start_percent: float = 0.10
    end_percent: float = 0.95
    enabled: bool = True
    attention_only_filter: bool = True
    ignore_text_encoder_lora: bool = True
    routing_mode: LoraRoutingMode = "standard"
    trigger_phrase: str = ""
    region_name: str = ""

    def active_at(self, step_percent: float) -> bool:
        return self.enabled and self.start_percent <= step_percent <= self.end_percent


@dataclass(frozen=True)
class K2RegionalLoraStack:
    regions: tuple[K2RegionalLora, ...]
    overlap_mode: OverlapMode = "normalize"

    @property
    def enabled_regions(self) -> tuple[K2RegionalLora, ...]:
        return tuple(r for r in self.regions if r.enabled)


def _fit_mask_batch(mask: torch.Tensor, batch: int) -> torch.Tensor:
    if mask.shape[0] == batch:
        return mask
    if mask.shape[0] == 1:
        reps = [batch] + [1] * (mask.ndim - 1)
        return mask.repeat(*reps)
    if batch == 1:
        return mask[:1]
    raise ValueError(f"Mask batch {mask.shape[0]} does not match target batch {batch}")
