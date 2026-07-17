from __future__ import annotations

from dataclasses import dataclass

from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition


@dataclass(frozen=True, slots=True)
class RegionalPromptArea:
    region_id: str
    name: str
    prompt: str
    negative_prompt: str
    area: tuple[int, int, int, int]  # latent height, width, y, x
    latent_mask: tuple[float, ...]
    image_token_mask: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegionalPromptPlan:
    width: int
    height: int
    latent_width: int
    latent_height: int
    strength: float
    feather_pixels: float
    regions: tuple[RegionalPromptArea, ...]
    backend: str = "comfy-latent-area-v1"

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "strength": self.strength,
            "feather_pixels": self.feather_pixels,
            "region_count": len(self.regions),
            "regions": [
                {
                    "id": region.region_id,
                    "name": region.name,
                    "area_latent": list(region.area),
                    "covered_image_tokens": sum(region.image_token_mask),
                }
                for region in self.regions
            ],
        }


def compile_regional_prompt_plan(
    width: int,
    height: int,
    regions: tuple[RegionDefinition, ...],
    *,
    strength: float = 1.0,
    feather_pixels: float = 32.0,
) -> RegionalPromptPlan:
    if not 0.0 < strength <= 10.0:
        raise ValueError("regional prompt strength must be in (0, 10]")
    if not 0.0 <= feather_pixels <= 1024.0:
        raise ValueError("regional feather must be between 0 and 1024 pixels")
    token_geometry = CanvasGeometry.resolve(width, height)
    latent_geometry = CanvasGeometry.resolve(width, height, patch_size=1)
    compiled: list[RegionalPromptArea] = []
    for region in regions:
        if not region.enabled or not region.prompt.strip():
            continue
        box = region.box.clipped(width, height)
        coverage_mask = latent_geometry.rasterize_box(box)
        latent_mask = _inward_feather(
            latent_geometry, box, coverage_mask, feather_pixels
        )
        nonzero = [index for index, value in enumerate(coverage_mask) if value > 0.0]
        if not nonzero:
            continue
        rows = [index // latent_geometry.patch_width for index in nonzero]
        columns = [index % latent_geometry.patch_width for index in nonzero]
        y = min(rows)
        x = min(columns)
        area_height = max(rows) - y + 1
        area_width = max(columns) - x + 1
        compiled.append(
            RegionalPromptArea(
                region_id=region.region_id,
                name=region.name,
                prompt=region.prompt.strip(),
                negative_prompt=region.negative_prompt.strip(),
                area=(area_height, area_width, y, x),
                latent_mask=latent_mask,
                image_token_mask=token_geometry.rasterize_box(box),
            )
        )
    return RegionalPromptPlan(
        width=token_geometry.aligned_width,
        height=token_geometry.aligned_height,
        latent_width=latent_geometry.patch_width,
        latent_height=latent_geometry.patch_height,
        strength=float(strength),
        feather_pixels=float(feather_pixels),
        regions=tuple(compiled),
    )


def _inward_feather(
    geometry: CanvasGeometry,
    box: PixelBox,
    coverage: tuple[float, ...],
    feather_pixels: float,
) -> tuple[float, ...]:
    effective = min(feather_pixels, box.width / 2.0, box.height / 2.0)
    if effective <= 0.0:
        return coverage
    size = geometry.output_pixels_per_image_token
    output = list(coverage)
    for index, value in enumerate(coverage):
        if value <= 0.0:
            continue
        row, column = divmod(index, geometry.patch_width)
        center_x = (column + 0.5) * size
        center_y = (row + 0.5) * size
        distance = min(
            center_x - box.x0,
            box.x1 - center_x,
            center_y - box.y0,
            box.y1 - center_y,
        )
        u = max(0.0, min(1.0, distance / effective))
        smoothstep = u * u * (3.0 - 2.0 * u)
        output[index] = value * smoothstep
    return tuple(output)


def region_definitions_from_payload(items: list[dict]) -> tuple[RegionDefinition, ...]:
    return tuple(
        RegionDefinition(
            region_id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            box=PixelBox(
                float(item["box"]["x0"]),
                float(item["box"]["y0"]),
                float(item["box"]["x1"]),
                float(item["box"]["y1"]),
            ),
            prompt=str(item.get("prompt", "")),
            negative_prompt=str(item.get("negative_prompt", "")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
        )
        for item in items
    )
