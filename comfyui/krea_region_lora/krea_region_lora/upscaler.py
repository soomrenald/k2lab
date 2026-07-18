from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def lanczos_resize(image: torch.Tensor, scale: int) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("upscaler expects a ComfyUI IMAGE batch")
    if int(scale) not in (2, 4):
        raise ValueError("K2 post-upscale scale must be 2 or 4")
    output = []
    for item in image.detach().float().cpu():
        array = (item[..., :3].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        source = Image.fromarray(array, mode="RGB")
        resized = source.resize(
            (source.width * int(scale), source.height * int(scale)), Image.Resampling.LANCZOS
        )
        output.append(torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0))
    return torch.stack(output).to(device=image.device, dtype=image.dtype)


def neural_upscale(image: torch.Tensor, upscale_model: Any) -> torch.Tensor:
    if upscale_model is None:
        raise ValueError("connect a native ComfyUI UPSCALE_MODEL for neural upscaling")
    try:
        import comfy.model_management as model_management  # type: ignore
        import comfy.utils  # type: ignore
    except Exception as exc:  # pragma: no cover - only inside ComfyUI
        raise RuntimeError("ComfyUI tiled upscale APIs are unavailable") from exc
    device = model_management.get_torch_device()
    upscale_model.to(device)
    source = image.movedim(-1, -3).to(device)
    tile = 512
    overlap = 32
    try:
        while True:
            try:
                steps = source.shape[0] * comfy.utils.get_tiled_scale_steps(
                    source.shape[3], source.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                progress = comfy.utils.ProgressBar(steps)
                result = comfy.utils.tiled_scale(
                    source,
                    lambda value: upscale_model(value.float()),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=progress,
                    output_device=model_management.intermediate_device(),
                )
                break
            except Exception as error:
                model_management.raise_non_oom(error)
                tile //= 2
                if tile < 128:
                    raise
    finally:
        upscale_model.to("cpu")
    return result.movedim(-3, -1).clamp(0.0, 1.0).to(device=image.device, dtype=image.dtype)


def post_upscale(
    image: torch.Tensor,
    *,
    scale: int = 2,
    method: str = "lanczos",
    upscale_model: Any = None,
) -> tuple[torch.Tensor, str]:
    requested_scale = int(scale)
    if method == "lanczos":
        return lanczos_resize(image, requested_scale), f"CPU Lanczos {requested_scale}x"
    if method != "model":
        raise ValueError(f"unsupported post-upscale method {method!r}")
    native = neural_upscale(image, upscale_model)
    target_height = image.shape[1] * requested_scale
    target_width = image.shape[2] * requested_scale
    if native.shape[1:3] != (target_height, target_width):
        # Preserve the GUI's requested final scale even when a native x4 model is
        # connected for a x2 output.
        native = (
            torch.nn.functional.interpolate(
                native.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .clamp(0.0, 1.0)
        )
    return (
        native,
        f"Neural tiled upscale {requested_scale}x (native model scale={float(upscale_model.scale):g})",
    )
