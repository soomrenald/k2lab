from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k2_region_lab.projector import (
    CUSTOM_PROJECTOR_PRESET,
    DEFAULT_PROJECTOR_PRESET,
    PROJECTOR_PRESETS,
    validate_projector_values,
)
from k2_region_lab.regions import PixelBox, RegionDefinition


PROJECT_SCHEMA = "k2-region-lab-project"
PROJECT_VERSION = 8
SUPPORTED_PROJECT_VERSIONS = {1, 2, 3, 4, 5, 6, 7, PROJECT_VERSION}


@dataclass(frozen=True, slots=True)
class SavedLora:
    path: Path
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not -4.0 <= self.strength <= 4.0:
            raise ValueError("saved LoRA strength must be between -4 and 4")


@dataclass(frozen=True, slots=True)
class ProjectState:
    canvas_width: int
    canvas_height: int
    global_prompt: str = ""
    steps: int = 8
    seed: int = 0
    seed_mode: str = "fixed"
    regional_prompting: bool = True
    regional_prompt_strength: float = 1.0
    regional_outside_penalty: float = 1.0
    regional_feather_pixels: int = 128
    regional_subject_competition: bool = True
    regional_subject_fill: bool = True
    regional_relaxation: bool = True
    projector_enabled: bool = False
    projector_preset: str = DEFAULT_PROJECTOR_PRESET
    projector_values: tuple[float, ...] = PROJECTOR_PRESETS[DEFAULT_PROJECTOR_PRESET]
    projector_multiplier: float = 1.0
    post_upscale: bool = False
    upscale_scale: int = 2
    upscale_method: str = "lanczos"
    upscale_model: Path | None = None
    regions: tuple[RegionDefinition, ...] = ()
    loras: tuple[SavedLora, ...] = ()
    runtime: dict[str, Any] | None = None
    background_image: Path | None = None

    def __post_init__(self) -> None:
        if not 256 <= self.canvas_width <= 4096 or not 256 <= self.canvas_height <= 4096:
            raise ValueError("canvas dimensions must be between 256 and 4096 pixels")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.seed_mode not in {"fixed", "random", "increment"}:
            raise ValueError(f"unsupported seed mode: {self.seed_mode!r}")
        if not 0.0 < self.regional_prompt_strength <= 10.0:
            raise ValueError("regional prompt strength must be in (0, 10]")
        if not 0.0 <= self.regional_outside_penalty <= 10.0:
            raise ValueError("regional outside penalty must be between 0 and 10")
        if not 0 <= self.regional_feather_pixels <= 2048:
            raise ValueError("spatial falloff must be between 0 and 2048 pixels")
        if self.projector_preset not in {
            *PROJECTOR_PRESETS,
            CUSTOM_PROJECTOR_PRESET,
        }:
            raise ValueError(f"unsupported projector preset: {self.projector_preset!r}")
        validate_projector_values(self.projector_values)
        if not -20.0 <= self.projector_multiplier <= 20.0:
            raise ValueError("projector multiplier must be between -20 and 20")
        if self.upscale_scale not in {2, 4}:
            raise ValueError("post-upscale scale must be 2 or 4")
        if self.upscale_method not in {"lanczos", "model"}:
            raise ValueError(f"unsupported post-upscale method: {self.upscale_method!r}")
        if self.post_upscale and self.upscale_method == "model" and not self.upscale_model:
            raise ValueError("a neural upscaler model must be selected")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("project region IDs must be unique")
        names = [region.name.casefold() for region in self.regions]
        if any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise ValueError("project region names must be non-empty and unique")
        known_ids = set(region_ids)
        for region in self.regions:
            box = region.box
            if box.width < 16 or box.height < 16:
                raise ValueError("project region boxes must be at least 16×16 pixels")
            if (
                box.x0 < 0
                or box.y0 < 0
                or box.x1 > self.canvas_width
                or box.y1 > self.canvas_height
            ):
                raise ValueError("project region boxes must stay inside the canvas")
        for lora in self.loras:
            if not lora.global_scope and not lora.region_ids:
                raise ValueError("a regional LoRA must target at least one region")
            if lora.global_scope and lora.region_ids:
                raise ValueError("a global LoRA cannot also target regions")
            if not set(lora.region_ids).issubset(known_ids):
                raise ValueError("a LoRA references a region missing from the project")


def project_document(state: ProjectState) -> dict[str, Any]:
    return {
        "schema": PROJECT_SCHEMA,
        "version": PROJECT_VERSION,
        "canvas": {"width": state.canvas_width, "height": state.canvas_height},
        "generation": {
            "global_prompt": state.global_prompt,
            "steps": state.steps,
            "seed": state.seed,
            "seed_mode": state.seed_mode,
            "regional_prompting": state.regional_prompting,
            "regional_prompt_strength": state.regional_prompt_strength,
            "regional_outside_penalty": state.regional_outside_penalty,
            "regional_feather_pixels": state.regional_feather_pixels,
            "regional_subject_competition": state.regional_subject_competition,
            "regional_subject_fill": state.regional_subject_fill,
            "regional_relaxation": state.regional_relaxation,
            "projector_enabled": state.projector_enabled,
            "projector_preset": state.projector_preset,
            "projector_values": list(state.projector_values),
            "projector_multiplier": state.projector_multiplier,
            "post_upscale": state.post_upscale,
            "upscale_scale": state.upscale_scale,
            "upscale_method": state.upscale_method,
            "upscale_model": (
                str(state.upscale_model) if state.upscale_model is not None else None
            ),
        },
        "regions": [
            {
                "id": region.region_id,
                "name": region.name,
                "box": {
                    "x0": region.box.x0,
                    "y0": region.box.y0,
                    "x1": region.box.x1,
                    "y1": region.box.y1,
                },
                "prompt": region.prompt,
                "negative_prompt": region.negative_prompt,
                "enabled": region.enabled,
                "priority": region.priority,
                "spatial_role": region.spatial_role,
            }
            for region in state.regions
        ],
        "loras": [
            {
                "path": str(lora.path),
                "global": lora.global_scope,
                "region_ids": list(lora.region_ids),
                "strength": lora.strength,
            }
            for lora in state.loras
        ],
        "runtime": state.runtime or {},
        "background_image": str(state.background_image) if state.background_image else None,
    }


def project_state(document: dict[str, Any]) -> ProjectState:
    if document.get("schema") != PROJECT_SCHEMA:
        raise ValueError("not a K2 Region Lab project file")
    if document.get("version") not in SUPPORTED_PROJECT_VERSIONS:
        raise ValueError(f"unsupported project version: {document.get('version')!r}")
    canvas = document["canvas"]
    generation = document.get("generation", {})
    regions = tuple(
        RegionDefinition(
            region_id=str(item["id"]),
            name=str(item["name"]),
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
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in document.get("regions", [])
    )
    loras = tuple(
        SavedLora(
            path=Path(item["path"]).expanduser(),
            global_scope=bool(item.get("global", True)),
            region_ids=tuple(str(region_id) for region_id in item.get("region_ids", [])),
            strength=float(item.get("strength", 1.0)),
        )
        for item in document.get("loras", [])
    )
    background = document.get("background_image")
    return ProjectState(
        canvas_width=int(canvas["width"]),
        canvas_height=int(canvas["height"]),
        global_prompt=str(generation.get("global_prompt", "")),
        steps=int(generation.get("steps", 8)),
        seed=int(generation.get("seed", 0)),
        seed_mode=str(generation.get("seed_mode", "fixed")),
        regional_prompting=bool(generation.get("regional_prompting", True)),
        regional_prompt_strength=float(
            generation.get("regional_prompt_strength", 1.0)
        ),
        regional_outside_penalty=float(
            generation.get("regional_outside_penalty", 1.0)
        ),
        regional_feather_pixels=int(generation.get("regional_feather_pixels", 128)),
        regional_subject_competition=bool(
            generation.get("regional_subject_competition", True)
        ),
        regional_subject_fill=bool(generation.get("regional_subject_fill", True)),
        regional_relaxation=bool(generation.get("regional_relaxation", True)),
        projector_enabled=bool(generation.get("projector_enabled", False)),
        projector_preset=str(
            generation.get("projector_preset", DEFAULT_PROJECTOR_PRESET)
        ),
        projector_values=validate_projector_values(
            generation.get(
                "projector_values",
                PROJECTOR_PRESETS[DEFAULT_PROJECTOR_PRESET],
            )
        ),
        projector_multiplier=float(generation.get("projector_multiplier", 1.0)),
        post_upscale=bool(generation.get("post_upscale", False)),
        upscale_scale=int(generation.get("upscale_scale", 2)),
        upscale_method=str(generation.get("upscale_method", "lanczos")),
        upscale_model=(
            Path(generation["upscale_model"]).expanduser()
            if generation.get("upscale_model")
            else None
        ),
        regions=regions,
        loras=loras,
        runtime=dict(document.get("runtime", {})),
        background_image=Path(background).expanduser() if background else None,
    )


def save_project(path: Path, state: ProjectState) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(project_document(state), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_project(path: Path) -> ProjectState:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("project root must be a JSON object")
    return project_state(document)
