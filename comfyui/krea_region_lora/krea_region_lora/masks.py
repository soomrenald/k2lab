from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .types import BatchMode, BBoxFormat, K2Region

KREA_VAE_SCALE = 8
KREA_TOKEN_PIXELS = 16


def coerce_bbox_list(bboxes: Any, bbox_index: int = 0) -> list[tuple[float, float, float, float]]:
    if bboxes is None:
        return []
    if isinstance(bboxes, torch.Tensor):
        data = bboxes.detach().cpu().tolist()
    else:
        data = bboxes
    if isinstance(data, dict):
        for keys in (("x_min", "y_min", "width", "height"), ("x0", "y0", "x1", "y1")):
            if all(k in data for k in keys):
                return [tuple(float(data[k]) for k in keys)]  # type: ignore[arg-type]
        for keys in (("x", "y", "width", "height"), ("x", "y", "w", "h")):
            if all(k in data for k in keys):
                return [tuple(float(data[k]) for k in keys)]  # type: ignore[arg-type]
        if "bbox" in data:
            return coerce_bbox_list(data["bbox"], bbox_index)
        if "bboxes" in data:
            return coerce_bbox_list(data["bboxes"], bbox_index)
    if _is_bbox(data):
        return [tuple(float(v) for v in data[:4])]  # type: ignore[index]
    if isinstance(data, Sequence):
        out: list[tuple[float, float, float, float]] = []
        for item in data:
            if isinstance(item, dict):
                out.extend(coerce_bbox_list(item, bbox_index))
            elif _is_bbox(item):
                out.append(tuple(float(v) for v in item[:4]))  # type: ignore[index]
        return out
    return []


def region_from_bbox(
    bboxes: Any,
    *,
    width: int,
    height: int,
    bbox_format: BBoxFormat = "xywh",
    bbox_index: int = 0,
    grow_px: int = 0,
    feather_px: int = 32,
    snap_to_krea_token_grid: bool = True,
    batch_mode: BatchMode = "repeat",
    batch_size: int = 1,
) -> K2Region:
    bbox_list = coerce_bbox_list(bboxes, bbox_index)
    if not bbox_list:
        pixel_bbox = (0, 0, 0, 0)
    else:
        index = max(0, min(int(bbox_index), len(bbox_list) - 1))
        pixel_bbox = normalize_bbox(
            bbox_list[index],
            width=width,
            height=height,
            bbox_format=bbox_format,
            grow_px=grow_px,
            snap_to_krea_token_grid=snap_to_krea_token_grid,
        )

    mask_batch = batch_size if batch_mode == "per_batch" else 1
    pixel_mask = make_pixel_mask(
        pixel_bbox,
        width=width,
        height=height,
        feather_px=feather_px,
        batch_size=mask_batch,
    )
    latent_mask = pixel_to_latent_mask(pixel_mask)
    token_mask = pixel_to_token_mask(pixel_mask)
    return K2Region(
        pixel_bbox=pixel_bbox,
        image_size=(int(width), int(height)),
        pixel_mask=pixel_mask,
        latent_mask=latent_mask,
        token_mask=token_mask,
        bbox_format=bbox_format,
        bbox_index=int(bbox_index),
        batch_mode=batch_mode,
        metadata={
            "grow_px": int(grow_px),
            "feather_px": int(feather_px),
            "snap_to_krea_token_grid": bool(snap_to_krea_token_grid),
        },
    )


def region_from_mask(
    mask: torch.Tensor,
    *,
    width: int | None = None,
    height: int | None = None,
    feather_px: int = 0,
    batch_mode: BatchMode = "repeat",
    metadata: dict[str, Any] | None = None,
) -> K2Region:
    """Build a K2 region from ComfyUI's native MASK socket value."""

    if not torch.is_tensor(mask):
        raise TypeError("region mask must be a torch tensor")
    current = mask.detach().float().cpu()
    if current.ndim == 2:
        current = current.unsqueeze(0)
    if current.ndim == 4 and current.shape[1] == 1:
        current = current[:, 0]
    if current.ndim != 3:
        raise ValueError(f"expected MASK shape [B,H,W], received {tuple(current.shape)}")
    output_height = int(height if height is not None else current.shape[-2])
    output_width = int(width if width is not None else current.shape[-1])
    if current.shape[-2:] != (output_height, output_width):
        current = F.interpolate(
            current.unsqueeze(1),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    current = current.clamp(0.0, 1.0)
    if feather_px > 0:
        current = _feather_inside(current, int(feather_px))
    occupied = torch.nonzero(current.amax(dim=0) > 0.001, as_tuple=False)
    if occupied.numel() == 0:
        bbox = (0, 0, 0, 0)
    else:
        y0, x0 = occupied.amin(dim=0).tolist()
        y1, x1 = occupied.amax(dim=0).tolist()
        bbox = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)
    return K2Region(
        pixel_bbox=bbox,
        image_size=(output_width, output_height),
        pixel_mask=current,
        latent_mask=pixel_to_latent_mask(current),
        token_mask=pixel_to_token_mask(current),
        bbox_format="xyxy",
        batch_mode=batch_mode,
        metadata={"source": "native_mask", **dict(metadata or {})},
    )


def infer_image_size(
    latent: dict[str, Any] | None, width: int | None, height: int | None
) -> tuple[int, int, int]:
    if latent is not None and "samples" in latent:
        samples = latent["samples"]
        if isinstance(samples, torch.Tensor) and samples.ndim >= 4:
            batch = int(samples.shape[0])
            return (
                int(samples.shape[-1] * KREA_VAE_SCALE),
                int(samples.shape[-2] * KREA_VAE_SCALE),
                batch,
            )
    if width is None or height is None:
        raise ValueError("Either latent or width/height must be provided")
    return int(width), int(height), 1


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    bbox_format: BBoxFormat,
    grow_px: int,
    snap_to_krea_token_grid: bool,
) -> tuple[int, int, int, int]:
    x0, y0, a, b = bbox
    if max(abs(x0), abs(y0), abs(a), abs(b)) <= 1.0:
        x0 *= width
        a *= width
        y0 *= height
        b *= height
    if bbox_format == "xywh":
        x1 = x0 + max(0.0, a)
        y1 = y0 + max(0.0, b)
    elif bbox_format == "xyxy":
        x1 = a
        y1 = b
    else:
        raise ValueError(f"Unsupported bbox_format {bbox_format}")

    grow = int(grow_px)
    x0 -= grow
    y0 -= grow
    x1 += grow
    y1 += grow

    if snap_to_krea_token_grid:
        grid = KREA_TOKEN_PIXELS
        x0 = math.floor(x0 / grid) * grid
        y0 = math.floor(y0 / grid) * grid
        x1 = math.ceil(x1 / grid) * grid
        y1 = math.ceil(y1 / grid) * grid

    ix0 = max(0, min(int(math.floor(x0)), int(width)))
    iy0 = max(0, min(int(math.floor(y0)), int(height)))
    ix1 = max(0, min(int(math.ceil(x1)), int(width)))
    iy1 = max(0, min(int(math.ceil(y1)), int(height)))
    if ix1 <= ix0 or iy1 <= iy0:
        return (0, 0, 0, 0)
    return (ix0, iy0, ix1, iy1)


def make_pixel_mask(
    pixel_bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    feather_px: int,
    batch_size: int,
) -> torch.Tensor:
    x0, y0, x1, y1 = pixel_bbox
    mask = torch.zeros((1, int(height), int(width)), dtype=torch.float32)
    if x1 <= x0 or y1 <= y0:
        return mask.repeat(max(1, int(batch_size)), 1, 1)
    mask[:, y0:y1, x0:x1] = 1.0
    feather = max(0, int(feather_px))
    if feather > 0:
        mask = _feather_inside(mask, feather)
    return mask.repeat(max(1, int(batch_size)), 1, 1)


def pixel_to_latent_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    mask = pixel_mask.unsqueeze(1)
    h = max(1, pixel_mask.shape[-2] // KREA_VAE_SCALE)
    w = max(1, pixel_mask.shape[-1] // KREA_VAE_SCALE)
    return F.interpolate(mask, size=(h, w), mode="area").clamp(0.0, 1.0)


def pixel_to_token_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    h = max(1, pixel_mask.shape[-2] // KREA_TOKEN_PIXELS)
    w = max(1, pixel_mask.shape[-1] // KREA_TOKEN_PIXELS)
    token_grid = F.interpolate(pixel_mask.unsqueeze(1), size=(h, w), mode="area").clamp(0.0, 1.0)
    return token_grid.flatten(2).transpose(1, 2)


def union_pixel_mask(regions: Sequence[K2Region], batch_size: int | None = None) -> torch.Tensor:
    if not regions:
        b = batch_size or 1
        return torch.zeros((b, 1, 1), dtype=torch.float32)
    mask = None
    for region in regions:
        current = region.pixel_mask
        if batch_size is not None and current.shape[0] == 1 and batch_size > 1:
            current = current.repeat(batch_size, 1, 1)
        mask = current if mask is None else torch.maximum(mask, current)
    return mask.clamp(0.0, 1.0)


def debug_bbox_image(region: K2Region) -> torch.Tensor:
    width, height = region.image_size
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    mask = region.pixel_mask[:1]
    image[..., 1] = mask
    x0, y0, x1, y1 = region.pixel_bbox
    if x1 > x0 and y1 > y0:
        image[:, y0:y1, x0 : min(x0 + 2, x1), 0] = 1.0
        image[:, y0:y1, max(x0, x1 - 2) : x1, 0] = 1.0
        image[:, y0 : min(y0 + 2, y1), x0:x1, 0] = 1.0
        image[:, max(y0, y1 - 2) : y1, x0:x1, 0] = 1.0
    return image.clamp(0.0, 1.0)


def _feather_inside(mask: torch.Tensor, feather_px: int) -> torch.Tensor:
    kernel = 2 * feather_px + 1
    pooled = F.avg_pool2d(
        mask.unsqueeze(1),
        kernel_size=kernel,
        stride=1,
        padding=feather_px,
        count_include_pad=False,
    ).squeeze(1)
    return torch.minimum(mask, pooled * (kernel * kernel) / max(1, (feather_px + 1) ** 2)).clamp(
        0.0, 1.0
    )


def _is_bbox(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4
