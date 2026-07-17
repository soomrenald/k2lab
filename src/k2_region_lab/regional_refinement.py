from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from k2_region_lab.regions import PixelBox, RegionDefinition


ALIGNMENT = 16
BACKEND = "krea-regional-latent-refinement-v1"


@dataclass(frozen=True, slots=True)
class RefinementCrop:
    region_id: str
    region_name: str
    prompt: str
    crop: PixelBox
    target: PixelBox
    internal_width: int
    internal_height: int
    scale: float

    @property
    def width(self) -> int:
        return int(self.crop.width)

    @property
    def height(self) -> int:
        return int(self.crop.height)

    def summary(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "region_name": self.region_name,
            "crop_pixels": [self.crop.x0, self.crop.y0, self.crop.x1, self.crop.y1],
            "target_pixels": [
                self.target.x0,
                self.target.y0,
                self.target.x1,
                self.target.y1,
            ],
            "internal_size": [self.internal_width, self.internal_height],
            "effective_scale": self.scale,
        }


def compile_refinement_crops(
    width: int,
    height: int,
    regions: tuple[RegionDefinition, ...],
    *,
    scale: float = 1.5,
    padding_pixels: int = 96,
    minimum_crop_pixels: int = 256,
    maximum_internal_pixels: int = 1024,
) -> tuple[RefinementCrop, ...]:
    if width % ALIGNMENT or height % ALIGNMENT:
        raise ValueError("refinement canvas dimensions must be multiples of 16")
    if not 1.0 <= scale <= 2.0:
        raise ValueError("regional refinement scale must be between 1 and 2")
    if not 0 <= padding_pixels <= 512:
        raise ValueError("regional refinement padding must be between 0 and 512 pixels")
    if minimum_crop_pixels < 256 or minimum_crop_pixels % ALIGNMENT:
        raise ValueError("minimum refinement crop must be a multiple of 16 and at least 256")
    if maximum_internal_pixels < minimum_crop_pixels:
        raise ValueError("maximum internal size must cover the minimum crop")

    crops = []
    for region in regions:
        if not region.enabled or not region.prompt.strip():
            continue
        role = region.spatial_role
        if role == "auto":
            role = "background" if region.box.width >= 0.70 * width else "subject"
        if role != "subject":
            continue
        target = region.box.clipped(width, height)
        crop = _aligned_crop(
            target,
            width,
            height,
            padding_pixels=padding_pixels,
            minimum_crop_pixels=minimum_crop_pixels,
        )
        internal_width, internal_height, effective_scale = _internal_size(
            int(crop.width),
            int(crop.height),
            scale,
            maximum_internal_pixels,
        )
        crops.append(
            RefinementCrop(
                region_id=region.region_id,
                region_name=region.name,
                prompt=region.prompt.strip(),
                crop=crop,
                target=target,
                internal_width=internal_width,
                internal_height=internal_height,
                scale=effective_scale,
            )
        )
    return tuple(crops)


def latent_blend_mask(
    crop: RefinementCrop,
    *,
    feather_pixels: int = 48,
    latent_scale: int = 8,
) -> tuple[float, ...]:
    if feather_pixels < 0:
        raise ValueError("refinement feather must not be negative")
    latent_width = crop.width // latent_scale
    latent_height = crop.height // latent_scale
    effective_feather = min(
        float(feather_pixels), crop.target.width / 4.0, crop.target.height / 4.0
    )
    values = []
    for row in range(latent_height):
        y = crop.crop.y0 + (row + 0.5) * latent_scale
        for column in range(latent_width):
            x = crop.crop.x0 + (column + 0.5) * latent_scale
            if not (
                crop.target.x0 <= x < crop.target.x1
                and crop.target.y0 <= y < crop.target.y1
            ):
                values.append(0.0)
                continue
            if effective_feather == 0.0:
                values.append(1.0)
                continue
            distance = min(
                x - crop.target.x0,
                crop.target.x1 - x,
                y - crop.target.y0,
                crop.target.y1 - y,
            )
            position = min(1.0, max(0.0, distance / effective_feather))
            values.append(position * position * (3.0 - 2.0 * position))
    return tuple(values)


def _aligned_crop(
    target: PixelBox,
    width: int,
    height: int,
    *,
    padding_pixels: int,
    minimum_crop_pixels: int,
) -> PixelBox:
    desired_width = max(minimum_crop_pixels, int(ceil(target.width + 2 * padding_pixels)))
    desired_height = max(minimum_crop_pixels, int(ceil(target.height + 2 * padding_pixels)))
    desired_width = min(width, _align_up(desired_width))
    desired_height = min(height, _align_up(desired_height))
    center_x = (target.x0 + target.x1) / 2.0
    center_y = (target.y0 + target.y1) / 2.0
    x0 = _aligned_origin(center_x, desired_width, width)
    y0 = _aligned_origin(center_y, desired_height, height)
    return PixelBox(x0, y0, x0 + desired_width, y0 + desired_height)


def _aligned_origin(center: float, extent: int, limit: int) -> int:
    origin = int(floor((center - extent / 2.0) / ALIGNMENT)) * ALIGNMENT
    return min(max(0, origin), limit - extent)


def _internal_size(
    width: int, height: int, scale: float, maximum: int
) -> tuple[int, int, float]:
    effective = min(scale, maximum / max(width, height))
    effective = max(1.0, effective)
    internal_width = min(maximum, _align_up(int(round(width * effective))))
    internal_height = min(maximum, _align_up(int(round(height * effective))))
    return internal_width, internal_height, min(
        internal_width / width, internal_height / height
    )


def _align_up(value: int) -> int:
    return ((value + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
