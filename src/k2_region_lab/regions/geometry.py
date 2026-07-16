from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def align_up(value: int, alignment: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Half-open output-pixel rectangle: [x0, x1) x [y0, y1)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("box coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def clipped(self, width: int, height: int) -> "PixelBox":
        if width <= 0 or height <= 0:
            raise ValueError("canvas dimensions must be positive")
        x0 = min(max(self.x0, 0.0), float(width))
        y0 = min(max(self.y0, 0.0), float(height))
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("box does not overlap the canvas")
        return PixelBox(x0, y0, x1, y1)


@dataclass(frozen=True, slots=True)
class CanvasGeometry:
    requested_width: int
    requested_height: int
    aligned_width: int
    aligned_height: int
    vae_scale: int = 8
    patch_size: int = 2

    @classmethod
    def resolve(
        cls,
        requested_width: int,
        requested_height: int,
        *,
        vae_scale: int = 8,
        patch_size: int = 2,
    ) -> "CanvasGeometry":
        alignment = vae_scale * patch_size
        return cls(
            requested_width=requested_width,
            requested_height=requested_height,
            aligned_width=align_up(requested_width, alignment),
            aligned_height=align_up(requested_height, alignment),
            vae_scale=vae_scale,
            patch_size=patch_size,
        )

    @property
    def output_pixels_per_image_token(self) -> int:
        return self.vae_scale * self.patch_size

    @property
    def patch_width(self) -> int:
        return self.aligned_width // self.output_pixels_per_image_token

    @property
    def patch_height(self) -> int:
        return self.aligned_height // self.output_pixels_per_image_token

    @property
    def image_lane_count(self) -> int:
        return self.patch_width * self.patch_height

    def image_lane_index(self, row: int, column: int) -> int:
        if not (0 <= row < self.patch_height and 0 <= column < self.patch_width):
            raise IndexError("image-token coordinate is outside the patch grid")
        return row * self.patch_width + column

    def token_box(self, row: int, column: int) -> PixelBox:
        self.image_lane_index(row, column)
        size = self.output_pixels_per_image_token
        return PixelBox(column * size, row * size, (column + 1) * size, (row + 1) * size)

    @staticmethod
    def overlap_fraction(token: PixelBox, region: PixelBox) -> float:
        overlap_width = max(0.0, min(token.x1, region.x1) - max(token.x0, region.x0))
        overlap_height = max(0.0, min(token.y1, region.y1) - max(token.y0, region.y0))
        return (overlap_width * overlap_height) / (token.width * token.height)

    def rasterize_box(self, box: PixelBox) -> tuple[float, ...]:
        clipped = box.clipped(self.aligned_width, self.aligned_height)
        values = [0.0] * self.image_lane_count
        size = self.output_pixels_per_image_token
        first_column = max(0, int(clipped.x0 // size))
        last_column = min(self.patch_width - 1, int((clipped.x1 - 1e-12) // size))
        first_row = max(0, int(clipped.y0 // size))
        last_row = min(self.patch_height - 1, int((clipped.y1 - 1e-12) // size))
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                index = self.image_lane_index(row, column)
                values[index] = self.overlap_fraction(self.token_box(row, column), clipped)
        return tuple(values)
