from __future__ import annotations

from dataclasses import dataclass

from k2_region_lab.regions.geometry import CanvasGeometry, PixelBox


REGION_ROLES = ("auto", "subject", "background", "edit")


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """A generic region shared by prompt, LoRA, influence, and attention controls."""

    region_id: str
    name: str
    box: PixelBox
    prompt: str = ""
    negative_prompt: str = ""
    face_identity_prompt: str = ""
    enabled: bool = True
    priority: int = 0
    spatial_role: str = "auto"

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.spatial_role not in REGION_ROLES:
            raise ValueError(f"unsupported spatial role: {self.spatial_role!r}")


@dataclass(frozen=True, slots=True)
class SpatialLayout:
    geometry: CanvasGeometry
    regions: tuple[RegionDefinition, ...]
    region_masks: tuple[tuple[float, ...], ...]
    revision: str = "pixel-box-area-fraction-v1"

    def __post_init__(self) -> None:
        if len(self.regions) != len(self.region_masks):
            raise ValueError("each region must have exactly one image-lane mask")
        expected = self.geometry.image_lane_count
        if any(len(mask) != expected for mask in self.region_masks):
            raise ValueError("region mask length does not match the image-token grid")

    def mask_for(self, region_id: str) -> tuple[float, ...]:
        for region, mask in zip(self.regions, self.region_masks, strict=True):
            if region.region_id == region_id:
                return mask
        raise KeyError(region_id)


def compile_spatial_layout(
    geometry: CanvasGeometry, regions: tuple[RegionDefinition, ...]
) -> SpatialLayout:
    ids = [region.region_id for region in regions]
    if len(ids) != len(set(ids)):
        raise ValueError("region IDs must be unique")
    masks = tuple(
        geometry.rasterize_box(region.box)
        if region.enabled
        else (0.0,) * geometry.image_lane_count
        for region in regions
    )
    return SpatialLayout(geometry=geometry, regions=regions, region_masks=masks)
