from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .lora import make_lora_branch_model
from .masks import coerce_bbox_list
from .types import K2RegionLayout, K2RegionalLoraStack


def coerce_face_boxes(
    *,
    bboxes: Any = None,
    segs: Any = None,
    width: int,
    height: int,
    bbox_format: str = "xywh",
) -> list[tuple[int, int, int, int]]:
    raw: list[tuple[float, float, float, float, str]] = []
    if segs is not None and isinstance(segs, (tuple, list)) and len(segs) >= 2:
        for segment in segs[1]:
            bbox = getattr(segment, "bbox", None) or getattr(segment, "crop_region", None)
            if bbox is not None:
                raw.extend((*item, "xyxy") for item in coerce_bbox_list(bbox))
    raw.extend((*item, bbox_format) for item in coerce_bbox_list(bboxes))
    boxes: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1, item_format in raw:
        if item_format == "xywh":
            x1 += x0
            y1 += y0
        elif item_format != "xyxy":
            raise ValueError(f"unsupported face bbox format {item_format!r}")
        ix0 = max(0, min(width, int(round(x0))))
        iy0 = max(0, min(height, int(round(y0))))
        ix1 = max(ix0, min(width, int(round(x1))))
        iy1 = max(iy0, min(height, int(round(y1))))
        if ix1 - ix0 >= 4 and iy1 - iy0 >= 4:
            boxes.append((ix0, iy0, ix1, iy1))
    return boxes


def expanded_square_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    side = min(max(x1 - x0, y1 - y0) * float(padding), width, height)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = min(max(cx - side / 2.0, 0.0), width - side)
    top = min(max(cy - side / 2.0, 0.0), height - side)
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + side)),
        int(round(top + side)),
    )


def face_blend_mask(height: int, width: int, feather: float, blend: float) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("face crop has no area")
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    distance = torch.minimum(
        torch.minimum(xx.expand(height, width), (width - 1 - xx).expand(height, width)),
        torch.minimum(yy.expand(height, width), (height - 1 - yy).expand(height, width)),
    )
    feather_pixels = max(1.0, min(height, width) * float(feather))
    alpha = (distance / feather_pixels).clamp(0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha * float(blend)


def _region_for_box(layout: K2RegionLayout | None, box: tuple[int, int, int, int]):
    if layout is None:
        return None
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    candidates = []
    for spec in layout.regions:
        x0, y0, x1, y1 = spec.region.pixel_bbox
        if spec.enabled and spec.spatial_role != "background" and x0 <= cx <= x1 and y0 <= cy <= y1:
            candidates.append(spec)
    return max(candidates, key=lambda item: item.priority) if candidates else None


def _crop_conditioning(conditioning: Any) -> Any:
    output = []
    for tensor, metadata in conditioning or []:
        values = {
            key: value
            for key, value in metadata.items()
            if key not in {"mask", "mask_strength", "area", "set_area_to_bounds"}
        }
        output.append([tensor, values])
    return output


def _regional_crop_model(
    model: Any,
    stack: K2RegionalLoraStack | None,
    region_id: str | None,
    region_name: str | None,
    lora_scale: float,
) -> tuple[Any, list[str]]:
    current = model
    applied: list[str] = []
    if stack is None or region_id is None:
        return current, applied
    for binding in stack.enabled_regions:
        binding_id = str(binding.region.metadata.get("region_id", ""))
        if binding_id != region_id and binding.region_name != region_name:
            continue
        current = make_lora_branch_model(
            current,
            binding.lora_name,
            strength_model=float(binding.lora_strength) * float(lora_scale),
            attention_only_filter=binding.attention_only_filter,
            ignore_text_encoder_lora=binding.ignore_text_encoder_lora,
        )
        applied.append(binding.lora_name)
    return current, applied


def detail_faces(
    *,
    image: torch.Tensor,
    model: Any,
    vae: Any,
    positive: Any,
    negative: Any,
    sampler: Any,
    bboxes: Any = None,
    segs: Any = None,
    layout: K2RegionLayout | None = None,
    regional_lora_stack: K2RegionalLoraStack | None = None,
    seed: int = 0,
    steps: int = 8,
    cfg: float = 1.0,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 0.15,
    crop_size: int = 512,
    padding: float = 2.0,
    feather: float = 0.12,
    blend: float = 0.5,
    lora_scale: float = 0.5,
    bbox_format: str = "xywh",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if image.ndim != 4:
        raise ValueError("face detailer expects an IMAGE batch")
    batch, height, width, _channels = image.shape
    boxes = coerce_face_boxes(
        bboxes=bboxes,
        segs=segs,
        width=width,
        height=height,
        bbox_format=bbox_format,
    )
    if not boxes:
        empty = torch.zeros((batch, height, width), dtype=image.dtype, device=image.device)
        return image, empty, "No face boxes received; connect SEGS or BOUNDING_BOX from a detector."
    output = image.clone()
    combined_mask = torch.zeros((batch, height, width), dtype=image.dtype, device=image.device)
    report: list[str] = []
    for batch_index in range(batch):
        for face_index, face_box in enumerate(boxes):
            crop_box = expanded_square_box(face_box, width, height, padding)
            x0, y0, x1, y1 = crop_box
            source = output[batch_index : batch_index + 1, y0:y1, x0:x1, :3]
            work = (
                F.interpolate(
                    source.permute(0, 3, 1, 2),
                    size=(int(crop_size), int(crop_size)),
                    mode="bicubic",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .clamp(0.0, 1.0)
            )
            spec = _region_for_box(layout, face_box)
            cond_positive = _crop_conditioning(
                spec.positive if spec is not None and spec.positive else positive
            )
            cond_negative = _crop_conditioning(
                spec.negative if spec is not None and spec.negative else negative
            )
            crop_model, loras = _regional_crop_model(
                model,
                regional_lora_stack,
                spec.region_id if spec is not None else None,
                spec.name if spec is not None else None,
                lora_scale,
            )
            latent = {"samples": vae.encode(work[:, :, :, :3])}
            sampled = sampler(
                crop_model,
                int(seed) + face_index + batch_index * len(boxes),
                int(steps),
                float(cfg),
                sampler_name,
                scheduler,
                cond_positive,
                cond_negative,
                latent,
                float(denoise),
            )
            refined = vae.decode(sampled["samples"])
            refined = (
                F.interpolate(
                    refined.permute(0, 3, 1, 2),
                    size=(y1 - y0, x1 - x0),
                    mode="bicubic",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .clamp(0.0, 1.0)
            )
            alpha = face_blend_mask(y1 - y0, x1 - x0, feather, blend).to(
                device=output.device, dtype=output.dtype
            )
            output[batch_index, y0:y1, x0:x1, :3] = refined[0].to(
                device=output.device, dtype=output.dtype
            ) * alpha.unsqueeze(-1) + source[0] * (1.0 - alpha.unsqueeze(-1))
            combined_mask[batch_index, y0:y1, x0:x1] = torch.maximum(
                combined_mask[batch_index, y0:y1, x0:x1], alpha
            )
            report.append(
                f"face {face_index + 1}: region={spec.name if spec else 'global'} "
                f"crop={crop_box} loras={','.join(loras) if loras else 'none'}"
            )
    return output.clamp(0.0, 1.0), combined_mask, "\n".join(report)
