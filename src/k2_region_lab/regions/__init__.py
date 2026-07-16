"""Generic pixel-space region geometry and layouts."""

from k2_region_lab.regions.geometry import CanvasGeometry, PixelBox, align_up
from k2_region_lab.regions.layout import RegionDefinition, SpatialLayout, compile_spatial_layout

__all__ = [
    "CanvasGeometry",
    "PixelBox",
    "RegionDefinition",
    "SpatialLayout",
    "align_up",
    "compile_spatial_layout",
]
