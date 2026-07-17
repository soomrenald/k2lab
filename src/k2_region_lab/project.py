from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k2_region_lab.regions import PixelBox, RegionDefinition


PROJECT_SCHEMA = "k2-region-lab-project"
PROJECT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SavedLora:
    path: Path
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectState:
    canvas_width: int
    canvas_height: int
    global_prompt: str = ""
    steps: int = 8
    seed: int = 0
    seed_mode: str = "fixed"
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
            }
            for region in state.regions
        ],
        "loras": [
            {
                "path": str(lora.path),
                "global": lora.global_scope,
                "region_ids": list(lora.region_ids),
            }
            for lora in state.loras
        ],
        "runtime": state.runtime or {},
        "background_image": str(state.background_image) if state.background_image else None,
    }


def project_state(document: dict[str, Any]) -> ProjectState:
    if document.get("schema") != PROJECT_SCHEMA:
        raise ValueError("not a K2 Region Lab project file")
    if document.get("version") != PROJECT_VERSION:
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
        )
        for item in document.get("regions", [])
    )
    loras = tuple(
        SavedLora(
            path=Path(item["path"]).expanduser(),
            global_scope=bool(item.get("global", True)),
            region_ids=tuple(str(region_id) for region_id in item.get("region_ids", [])),
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
