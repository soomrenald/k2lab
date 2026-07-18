from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from .masks import debug_bbox_image, region_from_bbox
from .types import (
    K2LoraReference,
    K2ProjectorSettings,
    K2PromptEmphasis,
    K2RegionLayout,
    K2RegionSpec,
    K2RegionalLora,
    K2RegionalLoraStack,
    K2RegionalSettings,
)


PROJECTOR_PRESETS: dict[str, tuple[float, ...]] = {
    "filter_bypass2": (0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, 0, 0),
    "filter_bypass3": (0, 0, 0, 0, 0, 0, 0, 0, -0.5117, -0.8906, -0.6094, 0),
    "skc3vo": (
        -5.44,
        -16.11,
        -37.11,
        -50.39,
        -70.70,
        -39.45,
        -39.84,
        -143.7511,
        -51.17,
        -89.06,
        -60.94,
        -11.28,
    ),
    "z0jglf": (
        -13.6,
        -40.275,
        -92.775,
        -159.75,
        -176.75,
        -98.625,
        -99.6,
        -359.3778,
        -127.925,
        -222.65,
        -152.35,
        -28.2,
    ),
}


def default_project_document(width: int = 1024, height: int = 1024) -> dict[str, Any]:
    return {
        "schema": "k2-region-lab-comfy",
        "version": 1,
        "canvas": {"width": int(width), "height": int(height)},
        "global_prompt": "",
        "global_negative": "",
        "regions": [],
        "emphases": [],
        "regional": {
            "enabled": True,
            "inside_strength": 1.0,
            "outside_penalty": 1.0,
            "feather_pixels": 128,
            "subject_competition": True,
            "subject_fill": True,
            "relaxation": True,
            "late_step_scale": 0.35,
            "lora_delta_adaptation": False,
            "lora_delta_adaptation_gain": 0.35,
        },
        "projector": {
            "enabled": False,
            "preset": "filter_bypass2",
            "values": list(PROJECTOR_PRESETS["filter_bypass2"]),
            "multiplier": 1.0,
            "identity_protection": 1.0,
        },
        "lora_assignments": {},
    }


def parse_project_document(payload: str | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None or payload == "":
        return default_project_document()
    document = json.loads(payload) if isinstance(payload, str) else copy.deepcopy(payload)
    if not isinstance(document, dict):
        raise ValueError("K2 project JSON must contain an object")
    # Desktop .k2lab.json files are accepted directly for round-trip portability.
    if document.get("schema") == "k2-region-lab-project":
        generation = document.get("generation", {})
        return {
            "schema": "k2-region-lab-comfy",
            "version": 1,
            "canvas": document.get("canvas", {}),
            "global_prompt": generation.get("global_prompt", ""),
            "global_negative": generation.get("global_negative", ""),
            "regions": document.get("regions", []),
            "emphases": generation.get("prompt_emphases", []),
            "regional": {
                "enabled": generation.get("regional_prompting", True),
                "inside_strength": generation.get("regional_prompt_strength", 1.0),
                "outside_penalty": generation.get("regional_outside_penalty", 1.0),
                "feather_pixels": generation.get("regional_feather_pixels", 128),
                "subject_competition": generation.get("regional_subject_competition", True),
                "subject_fill": generation.get("regional_subject_fill", True),
                "relaxation": generation.get("regional_relaxation", True),
                "late_step_scale": generation.get("regional_late_step_scale", 0.35),
                "lora_delta_adaptation": generation.get("regional_lora_delta_adaptation", False),
                "lora_delta_adaptation_gain": generation.get(
                    "regional_lora_delta_adaptation_gain", 0.35
                ),
            },
            "projector": {
                "enabled": generation.get("projector_enabled", False),
                "preset": generation.get("projector_preset", "filter_bypass2"),
                "values": generation.get("projector_values", PROJECTOR_PRESETS["filter_bypass2"]),
                "multiplier": generation.get("projector_multiplier", 1.0),
                "identity_protection": generation.get("projector_identity_protection", 1.0),
            },
            "lora_assignments": document.get("lora_assignments", {}),
        }
    defaults = default_project_document()
    defaults.update(document)
    return defaults


def serialize_project_document(document: dict[str, Any]) -> str:
    return json.dumps(document, separators=(",", ":"), sort_keys=True)


def apply_prompt_emphases(
    prompt: str,
    emphases: Iterable[K2PromptEmphasis],
    scope_id: str,
) -> str:
    rendered = prompt
    for emphasis in emphases:
        if emphasis.scope_id not in (scope_id, "global" if scope_id == "global" else ""):
            continue
        phrase = emphasis.phrase.strip()
        if not phrase:
            continue
        matches = list(re.finditer(re.escape(phrase), rendered, flags=re.IGNORECASE))
        if not matches:
            continue
        index = min(max(0, int(emphasis.occurrence)), len(matches) - 1)
        match = matches[index]
        weighted = f"({match.group(0)}:{1.0 + float(emphasis.strength):.3f})"
        rendered = rendered[: match.start()] + weighted + rendered[match.end() :]
    return rendered


def encode_prompt(clip: Any, prompt: str) -> Any:
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def layout_from_document(
    payload: str | dict[str, Any],
    *,
    clip: Any | None = None,
) -> K2RegionLayout:
    document = parse_project_document(payload)
    canvas = document.get("canvas", {})
    width = int(canvas.get("width", 1024))
    height = int(canvas.get("height", 1024))
    if not 16 <= width <= 16384 or not 16 <= height <= 16384:
        raise ValueError("canvas dimensions must be between 16 and 16384 pixels")
    regional_payload = document.get("regional", {})
    settings = K2RegionalSettings(
        regional_prompting=bool(regional_payload.get("enabled", True)),
        inside_strength=float(regional_payload.get("inside_strength", 1.0)),
        outside_penalty=float(regional_payload.get("outside_penalty", 1.0)),
        feather_pixels=int(regional_payload.get("feather_pixels", 128)),
        subject_competition=bool(regional_payload.get("subject_competition", True)),
        subject_fill=bool(regional_payload.get("subject_fill", True)),
        relaxation=bool(regional_payload.get("relaxation", True)),
        late_step_scale=float(regional_payload.get("late_step_scale", 0.35)),
        lora_delta_adaptation=bool(regional_payload.get("lora_delta_adaptation", False)),
        lora_delta_adaptation_gain=float(regional_payload.get("lora_delta_adaptation_gain", 0.35)),
    )
    projector_payload = document.get("projector", {})
    preset = str(projector_payload.get("preset", "filter_bypass2"))
    values = tuple(
        float(value)
        for value in projector_payload.get(
            "values", PROJECTOR_PRESETS.get(preset, PROJECTOR_PRESETS["filter_bypass2"])
        )
    )
    if len(values) != 12:
        raise ValueError("projector vector must have exactly 12 values")
    projector = K2ProjectorSettings(
        enabled=bool(projector_payload.get("enabled", False)),
        preset=preset,
        values=values,
        multiplier=float(projector_payload.get("multiplier", 1.0)),
        identity_protection=float(projector_payload.get("identity_protection", 1.0)),
    )
    emphases = tuple(
        K2PromptEmphasis(
            scope_id=str(item.get("scope_id", "global")),
            phrase=str(item.get("phrase", "")),
            strength=float(item.get("strength", 0.5)),
            occurrence=int(item.get("occurrence", 0)),
        )
        for item in document.get("emphases", [])
    )
    specs: list[K2RegionSpec] = []
    for index, item in enumerate(document.get("regions", [])):
        box = item.get("box", item.get("bbox", {}))
        if isinstance(box, dict):
            xyxy = (
                float(box.get("x0", box.get("x", 0))),
                float(box.get("y0", box.get("y", 0))),
                float(box.get("x1", float(box.get("x", 0)) + float(box.get("width", 0)))),
                float(box.get("y1", float(box.get("y", 0)) + float(box.get("height", 0)))),
            )
        else:
            xyxy = tuple(float(value) for value in box[:4])
        name = str(item.get("name", f"Region {index + 1}")).strip()
        region_id = str(item.get("id", f"region-{index + 1}"))
        region = region_from_bbox(
            [xyxy],
            width=width,
            height=height,
            bbox_format="xyxy",
            # LoRA routing uses the exact region. Prompt falloff is derived
            # separately when native ComfyUI conditioning masks are compiled.
            feather_px=0,
            snap_to_krea_token_grid=True,
        )
        metadata = dict(region.metadata)
        metadata.update({"region_id": region_id, "name": name, "source": "bbox"})
        region = replace(region, metadata=metadata)
        prompt = str(item.get("prompt", ""))
        negative_prompt = str(item.get("negative_prompt", ""))
        identity_prompt = str(item.get("face_identity_prompt", ""))
        scoped_prompt = apply_prompt_emphases(prompt, emphases, region_id)
        if identity_prompt.strip():
            scoped_prompt = f"{identity_prompt.strip()}, {scoped_prompt}".strip(", ")
        specs.append(
            K2RegionSpec(
                region=region,
                name=name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                face_identity_prompt=identity_prompt,
                enabled=bool(item.get("enabled", True)),
                priority=int(item.get("priority", len(document.get("regions", [])) - index)),
                spatial_role=str(item.get("spatial_role", "auto")),
                positive=encode_prompt(clip, scoped_prompt) if clip is not None else None,
                negative=encode_prompt(clip, negative_prompt) if clip is not None else None,
            )
        )
    global_prompt = str(document.get("global_prompt", ""))
    global_negative = str(document.get("global_negative", ""))
    return K2RegionLayout(
        width=width,
        height=height,
        regions=tuple(specs),
        global_prompt=global_prompt,
        global_negative=global_negative,
        global_positive=encode_prompt(
            clip, apply_prompt_emphases(global_prompt, emphases, "global")
        )
        if clip is not None
        else None,
        global_negative_conditioning=encode_prompt(clip, global_negative)
        if clip is not None
        else None,
        emphases=emphases,
        regional=settings,
        projector=projector,
        source_document=document,
    )


def stack_layout_regions(
    regions: Sequence[K2RegionSpec | None],
    *,
    width: int,
    height: int,
    global_positive: Any = None,
    global_negative: Any = None,
) -> K2RegionLayout:
    active = tuple(region for region in regions if region is not None)
    return K2RegionLayout(
        width=int(width),
        height=int(height),
        regions=active,
        global_positive=global_positive,
        global_negative_conditioning=global_negative,
    )


def _masked_conditioning(
    conditioning: Any,
    mask: torch.Tensor,
    strength: float,
    *,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
) -> list[Any]:
    if not conditioning:
        return []
    output = []
    for tensor, metadata in conditioning:
        values = dict(metadata)
        values.update(
            {
                "mask": mask,
                "mask_strength": float(strength),
                "set_area_to_bounds": True,
                "k2_region_lab_spatial": True,
                "start_percent": float(start_percent),
                "end_percent": float(end_percent),
            }
        )
        output.append([tensor, values])
    return output


def _spatial_mask(layout: K2RegionLayout, spec: K2RegionSpec, role: str) -> torch.Tensor:
    base = spec.region.pixel_mask
    if spec.region.metadata.get("source") == "bbox":
        x0, y0, x1, y1 = spec.region.pixel_bbox
        yy = torch.arange(layout.height, dtype=torch.float32).view(1, layout.height, 1)
        xx = torch.arange(layout.width, dtype=torch.float32).view(1, 1, layout.width)
        dx = torch.maximum(
            torch.maximum(torch.tensor(float(x0)) - xx, xx - float(x1 - 1)), torch.zeros_like(xx)
        )
        dy = torch.maximum(
            torch.maximum(torch.tensor(float(y0)) - yy, yy - float(y1 - 1)), torch.zeros_like(yy)
        )
        feather = int(layout.regional.feather_pixels)
        if feather > 0:
            outside = (1.0 - torch.sqrt(dx * dx + dy * dy) / float(feather)).clamp(0.0, 1.0)
            base = torch.maximum(base, outside)
        if role == "subject" and not layout.regional.subject_fill and x1 > x0 and y1 > y0:
            center_x = (x0 + x1 - 1) / 2.0
            center_y = (y0 + y1 - 1) / 2.0
            nx = (xx - center_x).abs() / max((x1 - x0) / 2.0, 1.0)
            ny = (yy - center_y).abs() / max((y1 - y0) / 2.0, 1.0)
            center_weight = (1.0 - 0.5 * torch.maximum(nx, ny)).clamp(0.25, 1.0)
            base = base * torch.where(spec.region.pixel_mask > 0, center_weight, 1.0)
    return base.clamp(0.0, 1.0)


def compile_spatial_conditioning(
    layout: K2RegionLayout,
    positive: Any = None,
    negative: Any = None,
) -> tuple[list[Any], list[Any]]:
    if positive is not None and any(
        bool(metadata.get("k2_region_lab_spatial")) for _tensor, metadata in positive
    ):
        return list(positive), list(negative or [])
    positive_out = list(positive if positive is not None else (layout.global_positive or []))
    negative_out = list(
        negative if negative is not None else (layout.global_negative_conditioning or [])
    )
    if not layout.regional.regional_prompting:
        return positive_out, negative_out
    ordered = sorted(
        (item for item in layout.regions if item.enabled), key=lambda item: -item.priority
    )
    claimed_subject = torch.zeros((1, layout.height, layout.width), dtype=torch.float32)
    for spec in ordered:
        role = spec.spatial_role
        if role == "auto":
            role = (
                "background"
                if spec.region.pixel_bbox[2] - spec.region.pixel_bbox[0] >= layout.width * 0.72
                else "subject"
            )
        strength = layout.regional.inside_strength * (0.75 if role == "background" else 1.0)
        spatial_mask = _spatial_mask(layout, spec, role)
        if role == "subject" and layout.regional.subject_competition:
            spatial_mask = spatial_mask * (1.0 - claimed_subject)
            claimed_subject = torch.maximum(claimed_subject, spec.region.pixel_mask[:1])
        schedules = ((0.0, 1.0, 1.0),)
        if layout.regional.relaxation:
            schedules = ((0.0, 0.5, 1.0), (0.5, 1.0, layout.regional.late_step_scale))
        for start, end, scale in schedules:
            positive_out.extend(
                _masked_conditioning(
                    spec.positive,
                    spatial_mask,
                    strength * scale,
                    start_percent=start,
                    end_percent=end,
                )
            )
            negative_out.extend(
                _masked_conditioning(
                    spec.negative,
                    spatial_mask,
                    strength * scale,
                    start_percent=start,
                    end_percent=end,
                )
            )
        if layout.regional.outside_penalty > 0 and spec.positive:
            inverse = (1.0 - spatial_mask).clamp(0.0, 1.0)
            penalty = layout.regional.outside_penalty * (0.25 if role == "background" else 1.0)
            for start, end, scale in schedules:
                negative_out.extend(
                    _masked_conditioning(
                        spec.positive,
                        inverse,
                        penalty * scale,
                        start_percent=start,
                        end_percent=end,
                    )
                )
    return positive_out, negative_out


def bind_lora_reference(
    layout: K2RegionLayout,
    reference: K2LoraReference,
    targets: str | Sequence[str],
) -> tuple[K2RegionalLora, ...]:
    names = (
        [item.strip() for item in targets.split(",")]
        if isinstance(targets, str)
        else [str(item).strip() for item in targets]
    )
    if any(name.casefold() == "global" for name in names):
        raise ValueError(
            "global LoRAs should use ComfyUI's native Load LoRA node before the K2 sampler"
        )
    output = []
    for name in names:
        spec = layout.named_region(name)
        if spec is None:
            raise ValueError(f"unknown LoRA target region {name!r}")
        if (
            reference.routing_mode == "character_identity"
            and spec.face_identity_prompt.strip() == ""
        ):
            raise ValueError(
                f"character identity LoRA target {spec.name!r} needs a face identity prompt"
            )
        output.append(
            K2RegionalLora(
                region=spec.region,
                positive=spec.positive,
                negative=spec.negative,
                lora_name=reference.lora_name,
                lora_strength=reference.strength,
                start_percent=reference.start_percent,
                end_percent=reference.end_percent,
                enabled=reference.enabled and spec.enabled,
                routing_mode=reference.routing_mode,
                trigger_phrase=reference.trigger_phrase,
                region_name=spec.name,
            )
        )
    return tuple(output)


def stack_regional_loras(
    existing: K2RegionalLoraStack | None,
    additions: Iterable[K2RegionalLora],
    overlap_mode: str = "normalize",
) -> K2RegionalLoraStack:
    regions = tuple(existing.regions if existing is not None else ()) + tuple(additions)
    return K2RegionalLoraStack(regions=regions, overlap_mode=overlap_mode)


def layout_preview(
    layout: K2RegionLayout, background_image: torch.Tensor | None = None
) -> torch.Tensor:
    if background_image is None:
        image = torch.full((1, layout.height, layout.width, 3), 0.08, dtype=torch.float32)
    else:
        image = background_image[:1, ..., :3].detach().float().cpu()
        if image.shape[1:3] != (layout.height, layout.width):
            image = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(layout.height, layout.width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
    palette = (
        (0.94, 0.31, 0.31),
        (0.25, 0.72, 0.98),
        (0.35, 0.86, 0.48),
        (0.95, 0.70, 0.25),
        (0.72, 0.43, 0.96),
        (0.20, 0.84, 0.78),
    )
    for index, spec in enumerate(layout.regions):
        if not spec.enabled:
            continue
        color = torch.tensor(palette[index % len(palette)], dtype=image.dtype).view(1, 1, 1, 3)
        mask = spec.region.pixel_mask[:1].unsqueeze(-1)
        image = image * (1.0 - mask * 0.22) + color * mask * 0.22
        x0, y0, x1, y1 = spec.region.pixel_bbox
        if x1 > x0 and y1 > y0:
            image[:, y0 : min(y0 + 3, y1), x0:x1] = color
            image[:, max(y1 - 3, y0) : y1, x0:x1] = color
            image[:, y0:y1, x0 : min(x0 + 3, x1)] = color
            image[:, y0:y1, max(x1 - 3, x0) : x1] = color
    return image.clamp(0.0, 1.0)


def preview_region(spec: K2RegionSpec) -> torch.Tensor:
    return debug_bbox_image(spec.region)
