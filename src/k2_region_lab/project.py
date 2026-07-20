from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from k2_region_lab.image_edit import ImageEditState
from k2_region_lab.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    LORA_ROUTING_MODES,
    STANDARD_LORA_ROUTING,
)
from k2_region_lab.projector import (
    CUSTOM_PROJECTOR_PRESET,
    DEFAULT_PROJECTOR_PRESET,
    PROJECTOR_PRESETS,
    validate_projector_values,
)
from k2_region_lab.regional_prompting import (
    GLOBAL_EMPHASIS_SCOPE,
    PromptEmphasis,
    prompt_emphases_from_payload,
)
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.sampling import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    validate_sampler,
    validate_scheduler,
)


PROJECT_SCHEMA = "k2-region-lab-project"
PNG_PROJECT_KEY = "k2lab_project"
PROJECT_VERSION = 18
SUPPORTED_PROJECT_VERSIONS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    15,
    16,
    17,
    PROJECT_VERSION,
}


@dataclass(frozen=True, slots=True)
class SavedLora:
    path: Path
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()
    strength: float = 1.0
    routing_mode: str = STANDARD_LORA_ROUTING
    trigger_phrase: str = ""
    edit_enabled: bool = False
    edit_global_scope: bool = False
    edit_region_ids: tuple[str, ...] = ()
    edit_routing_mode: str = STANDARD_LORA_ROUTING
    edit_trigger_phrase: str = ""
    reference_enabled: bool = False
    reference_global_scope: bool = False
    reference_region_ids: tuple[str, ...] = ()
    reference_routing_mode: str = STANDARD_LORA_ROUTING
    reference_trigger_phrase: str = ""

    def __post_init__(self) -> None:
        if not -4.0 <= self.strength <= 4.0:
            raise ValueError("saved LoRA strength must be between -4 and 4")
        if self.routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(f"unsupported saved LoRA routing mode: {self.routing_mode!r}")
        if self.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING and not self.trigger_phrase.strip():
            raise ValueError("character identity routing requires a trigger phrase")
        if self.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING and self.global_scope:
            raise ValueError("character identity routing requires regional scope")
        if self.edit_routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(
                f"unsupported saved edit LoRA routing mode: {self.edit_routing_mode!r}"
            )
        if self.edit_enabled and self.edit_global_scope and self.edit_region_ids:
            raise ValueError("an edit LoRA cannot be global and region-scoped at the same time")
        if self.edit_enabled and not self.edit_global_scope and not self.edit_region_ids:
            raise ValueError("an enabled edit LoRA must have a global or regional scope")
        if (
            self.edit_enabled
            and self.edit_routing_mode == CHARACTER_IDENTITY_LORA_ROUTING
            and not self.edit_trigger_phrase.strip()
        ):
            raise ValueError("character identity edit routing requires a trigger phrase")
        if (
            self.edit_enabled
            and self.edit_routing_mode == CHARACTER_IDENTITY_LORA_ROUTING
            and self.edit_global_scope
        ):
            raise ValueError("character identity edit routing requires regional scope")
        if self.reference_routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(
                "unsupported saved reference LoRA routing mode: "
                f"{self.reference_routing_mode!r}"
            )
        if (
            self.reference_enabled
            and self.reference_global_scope
            and self.reference_region_ids
        ):
            raise ValueError(
                "a reference LoRA cannot be global and region-scoped at the same time"
            )
        if (
            self.reference_enabled
            and not self.reference_global_scope
            and not self.reference_region_ids
        ):
            raise ValueError("an enabled reference LoRA must have a scope")


@dataclass(frozen=True, slots=True)
class ProjectState:
    canvas_width: int
    canvas_height: int
    global_prompt: str = ""
    steps: int = 8
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    seed: int = 0
    seed_mode: str = "fixed"
    batch_mode: bool = False
    batch_count: int = 2
    regional_prompting: bool = True
    regional_prompt_strength: float = 1.0
    regional_outside_penalty: float = 1.0
    regional_feather_pixels: int = 128
    regional_subject_competition: bool = True
    regional_subject_fill: bool = True
    regional_relaxation: bool = True
    regional_late_step_scale: float = 0.35
    regional_lora_delta_adaptation: bool = False
    regional_lora_delta_adaptation_gain: float = 0.35
    prompt_emphases: tuple[PromptEmphasis, ...] = ()
    projector_enabled: bool = False
    projector_preset: str = DEFAULT_PROJECTOR_PRESET
    projector_values: tuple[float, ...] = PROJECTOR_PRESETS[DEFAULT_PROJECTOR_PRESET]
    projector_multiplier: float = 1.0
    projector_identity_protection: float = 1.0
    face_detail_seed: int = 0
    face_detail_steps: int = 8
    face_detail_denoise: float = 0.15
    face_detail_crop_size: int = 512
    face_detail_padding: float = 2.0
    face_detail_feather: float = 0.12
    face_detail_blend: float = 0.5
    face_detail_lora_scale: float = 0.5
    face_detail_detector_threshold: float = 0.15
    face_detail_detector_provider: str = "auto"
    post_upscale: bool = False
    upscale_scale: int = 2
    upscale_method: str = "lanczos"
    upscale_model: Path | None = None
    regions: tuple[RegionDefinition, ...] = ()
    loras: tuple[SavedLora, ...] = ()
    runtime: dict[str, Any] | None = None
    background_image: Path | None = None
    image_edit: ImageEditState = field(default_factory=ImageEditState)

    def __post_init__(self) -> None:
        if not 256 <= self.canvas_width <= 4096 or not 256 <= self.canvas_height <= 4096:
            raise ValueError("canvas dimensions must be between 256 and 4096 pixels")
        if not 1 <= self.steps <= 100:
            raise ValueError("steps must be between 1 and 100")
        validate_sampler(self.sampler)
        validate_scheduler(self.scheduler)
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.seed_mode not in {"fixed", "random", "increment"}:
            raise ValueError(f"unsupported seed mode: {self.seed_mode!r}")
        if self.batch_mode and self.seed_mode not in {"random", "increment"}:
            raise ValueError("batch mode requires random or increment seed behavior")
        if not 1 <= self.batch_count <= 100:
            raise ValueError("batch count must be between 1 and 100")
        if not 0.0 < self.regional_prompt_strength <= 10.0:
            raise ValueError("regional prompt strength must be in (0, 10]")
        if not 0.0 <= self.regional_outside_penalty <= 10.0:
            raise ValueError("regional outside penalty must be between 0 and 10")
        if not 0 <= self.regional_feather_pixels <= 2048:
            raise ValueError("spatial falloff must be between 0 and 2048 pixels")
        if not 0.0 <= self.regional_late_step_scale <= 1.0:
            raise ValueError("late-step spatial scale must be between 0 and 1")
        if not 0.0 <= self.regional_lora_delta_adaptation_gain <= 1.0:
            raise ValueError("LoRA delta adaptation gain must be between zero and one")
        if self.projector_preset not in {
            *PROJECTOR_PRESETS,
            CUSTOM_PROJECTOR_PRESET,
        }:
            raise ValueError(f"unsupported projector preset: {self.projector_preset!r}")
        validate_projector_values(self.projector_values)
        if not -20.0 <= self.projector_multiplier <= 20.0:
            raise ValueError("projector multiplier must be between -20 and 20")
        if not 0.0 <= self.projector_identity_protection <= 1.0:
            raise ValueError("projector identity protection must be between zero and one")
        if not 0 <= self.face_detail_seed <= 2_147_483_647:
            raise ValueError("face-detail seed must be between 0 and 2147483647")
        if not 1 <= self.face_detail_steps <= 100:
            raise ValueError("face-detail steps must be between 1 and 100")
        if not 0.0 < self.face_detail_denoise <= 1.0:
            raise ValueError("face-detail denoise must be in (0, 1]")
        if self.face_detail_crop_size not in {256, 512, 768, 1024}:
            raise ValueError("unsupported face-detail crop size")
        if not 1.0 <= self.face_detail_padding <= 4.0:
            raise ValueError("face-detail padding must be between 1 and 4")
        if not 0.0 <= self.face_detail_feather <= 0.5:
            raise ValueError("face-detail feather must be between zero and 0.5")
        if not 0.0 <= self.face_detail_blend <= 1.0:
            raise ValueError("face-detail blend must be between zero and one")
        if not 0.0 <= self.face_detail_lora_scale <= 4.0:
            raise ValueError("face-detail LoRA scale must be between zero and four")
        if not 0.0 < self.face_detail_detector_threshold < 1.0:
            raise ValueError("face detector threshold must be in (0, 1)")
        if self.face_detail_detector_provider not in {"auto", "cpu", "cuda"}:
            raise ValueError("face detector provider must be auto, cpu, or cuda")
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
        for emphasis in self.prompt_emphases:
            if emphasis.scope_id != GLOBAL_EMPHASIS_SCOPE and emphasis.scope_id not in known_ids:
                raise ValueError("a prompt emphasis references a region missing from the project")
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
            edit_ids = {region.region_id for region in self.image_edit.regions}
            if not set(lora.edit_region_ids).issubset(edit_ids):
                raise ValueError("an edit LoRA references a region missing from the edit setup")
            reference_ids = {
                region.region_id for region in self.image_edit.reference_regions
            }
            if not set(lora.reference_region_ids).issubset(reference_ids):
                raise ValueError(
                    "a reference LoRA references a region missing from the reference setup"
                )


def project_document(state: ProjectState) -> dict[str, Any]:
    return {
        "schema": PROJECT_SCHEMA,
        "version": PROJECT_VERSION,
        "canvas": {"width": state.canvas_width, "height": state.canvas_height},
        "generation": {
            "global_prompt": state.global_prompt,
            "steps": state.steps,
            "sampler": state.sampler,
            "scheduler": state.scheduler,
            "seed": state.seed,
            "seed_mode": state.seed_mode,
            "batch_mode": state.batch_mode,
            "batch_count": state.batch_count,
            "regional_prompting": state.regional_prompting,
            "regional_prompt_strength": state.regional_prompt_strength,
            "regional_outside_penalty": state.regional_outside_penalty,
            "regional_feather_pixels": state.regional_feather_pixels,
            "regional_subject_competition": state.regional_subject_competition,
            "regional_subject_fill": state.regional_subject_fill,
            "regional_relaxation": state.regional_relaxation,
            "regional_late_step_scale": state.regional_late_step_scale,
            "regional_lora_delta_adaptation": state.regional_lora_delta_adaptation,
            "regional_lora_delta_adaptation_gain": (
                state.regional_lora_delta_adaptation_gain
            ),
            "prompt_emphases": [
                {
                    "scope_id": emphasis.scope_id,
                    "phrase": emphasis.phrase,
                    "strength": emphasis.strength,
                    "occurrence": emphasis.occurrence,
                }
                for emphasis in state.prompt_emphases
            ],
            "projector_enabled": state.projector_enabled,
            "projector_preset": state.projector_preset,
            "projector_values": list(state.projector_values),
            "projector_multiplier": state.projector_multiplier,
            "projector_identity_protection": state.projector_identity_protection,
            "face_detail_seed": state.face_detail_seed,
            "face_detail_steps": state.face_detail_steps,
            "face_detail_denoise": state.face_detail_denoise,
            "face_detail_crop_size": state.face_detail_crop_size,
            "face_detail_padding": state.face_detail_padding,
            "face_detail_feather": state.face_detail_feather,
            "face_detail_blend": state.face_detail_blend,
            "face_detail_lora_scale": state.face_detail_lora_scale,
            "face_detail_detector_threshold": state.face_detail_detector_threshold,
            "face_detail_detector_provider": state.face_detail_detector_provider,
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
                "face_identity_prompt": region.face_identity_prompt,
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
                "routing_mode": lora.routing_mode,
                "trigger_phrase": lora.trigger_phrase,
                "image_edit": {
                    "enabled": lora.edit_enabled,
                    "global": lora.edit_global_scope,
                    "region_ids": list(lora.edit_region_ids),
                    "routing_mode": lora.edit_routing_mode,
                    "trigger_phrase": lora.edit_trigger_phrase,
                },
                "image_edit_reference": {
                    "enabled": lora.reference_enabled,
                    "global": lora.reference_global_scope,
                    "region_ids": list(lora.reference_region_ids),
                    "routing_mode": lora.reference_routing_mode,
                    "trigger_phrase": lora.reference_trigger_phrase,
                },
            }
            for lora in state.loras
        ],
        "image_edit": {
            "source_image": (
                str(state.image_edit.source_image)
                if state.image_edit.source_image is not None
                else None
            ),
            "associated_project": (
                str(state.image_edit.associated_project)
                if state.image_edit.associated_project is not None
                else None
            ),
            "width": state.image_edit.width,
            "height": state.image_edit.height,
            "reference_global_prompt": state.image_edit.reference_global_prompt,
            "reference_prompt_emphases": [
                {
                    "scope_id": emphasis.scope_id,
                    "phrase": emphasis.phrase,
                    "strength": emphasis.strength,
                    "occurrence": emphasis.occurrence,
                }
                for emphasis in state.image_edit.reference_prompt_emphases
            ],
            "reference_projector_enabled": (
                state.image_edit.reference_projector_enabled
            ),
            "reference_projector_preset": state.image_edit.reference_projector_preset,
            "reference_projector_values": list(
                state.image_edit.reference_projector_values
            ),
            "reference_projector_multiplier": (
                state.image_edit.reference_projector_multiplier
            ),
            "reference_projector_identity_protection": (
                state.image_edit.reference_projector_identity_protection
            ),
            "global_prompt": state.image_edit.global_prompt,
            "steps": state.image_edit.steps,
            "sampler": state.image_edit.sampler,
            "scheduler": state.image_edit.scheduler,
            "seed": state.image_edit.seed,
            "denoise": state.image_edit.denoise,
            "latent_feather_pixels": state.image_edit.latent_feather_pixels,
            "composite_feather_pixels": state.image_edit.composite_feather_pixels,
            "edit_entire_image": state.image_edit.edit_entire_image,
            "preserve_identity": state.image_edit.preserve_identity,
            "reference_description_retention": (
                state.image_edit.reference_description_retention
            ),
            "regional_prompt_strength": state.image_edit.regional_prompt_strength,
            "regional_outside_penalty": state.image_edit.regional_outside_penalty,
            "regional_feather_pixels": state.image_edit.regional_feather_pixels,
            "regional_subject_competition": (
                state.image_edit.regional_subject_competition
            ),
            "regional_subject_fill": state.image_edit.regional_subject_fill,
            "regional_late_step_scale": state.image_edit.regional_late_step_scale,
            "regional_lora_delta_adaptation": (
                state.image_edit.regional_lora_delta_adaptation
            ),
            "regional_lora_delta_adaptation_gain": (
                state.image_edit.regional_lora_delta_adaptation_gain
            ),
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
                    "face_identity_prompt": region.face_identity_prompt,
                    "enabled": region.enabled,
                    "priority": region.priority,
                    "spatial_role": region.spatial_role,
                }
                for region in state.image_edit.regions
            ],
            "reference_regions": [
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
                    "face_identity_prompt": region.face_identity_prompt,
                    "enabled": region.enabled,
                    "priority": region.priority,
                    "spatial_role": region.spatial_role,
                }
                for region in state.image_edit.reference_regions
            ],
        },
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
            face_identity_prompt=str(item.get("face_identity_prompt", "")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in document.get("regions", [])
    )
    edit_document = document.get("image_edit", {})
    edit_regions = tuple(
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
            face_identity_prompt=str(item.get("face_identity_prompt", "")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in edit_document.get("regions", [])
    )
    edit_reference_regions = tuple(
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
            face_identity_prompt=str(item.get("face_identity_prompt", "")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in edit_document.get("reference_regions", [])
    )
    loras = tuple(
        SavedLora(
            path=Path(item["path"]).expanduser(),
            global_scope=bool(item.get("global", True)),
            region_ids=tuple(str(region_id) for region_id in item.get("region_ids", [])),
            strength=float(item.get("strength", 1.0)),
            routing_mode=str(item.get("routing_mode", STANDARD_LORA_ROUTING)),
            trigger_phrase=str(item.get("trigger_phrase", "")),
            edit_enabled=bool(item.get("image_edit", {}).get("enabled", False)),
            edit_global_scope=bool(item.get("image_edit", {}).get("global", False)),
            edit_region_ids=tuple(
                str(region_id)
                for region_id in item.get("image_edit", {}).get("region_ids", [])
            ),
            edit_routing_mode=str(
                item.get("image_edit", {}).get("routing_mode", STANDARD_LORA_ROUTING)
            ),
            edit_trigger_phrase=str(
                item.get("image_edit", {}).get("trigger_phrase", "")
            ),
            reference_enabled=bool(
                item.get("image_edit_reference", {}).get("enabled", False)
            ),
            reference_global_scope=bool(
                item.get("image_edit_reference", {}).get("global", False)
            ),
            reference_region_ids=tuple(
                str(region_id)
                for region_id in item.get("image_edit_reference", {}).get(
                    "region_ids", []
                )
            ),
            reference_routing_mode=str(
                item.get("image_edit_reference", {}).get(
                    "routing_mode", STANDARD_LORA_ROUTING
                )
            ),
            reference_trigger_phrase=str(
                item.get("image_edit_reference", {}).get("trigger_phrase", "")
            ),
        )
        for item in document.get("loras", [])
    )
    background = document.get("background_image")
    return ProjectState(
        canvas_width=int(canvas["width"]),
        canvas_height=int(canvas["height"]),
        global_prompt=str(generation.get("global_prompt", "")),
        steps=int(generation.get("steps", 8)),
        sampler=str(generation.get("sampler", DEFAULT_SAMPLER)),
        scheduler=str(generation.get("scheduler", DEFAULT_SCHEDULER)),
        seed=int(generation.get("seed", 0)),
        seed_mode=str(generation.get("seed_mode", "fixed")),
        batch_mode=bool(generation.get("batch_mode", False)),
        batch_count=int(generation.get("batch_count", 2)),
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
        regional_late_step_scale=float(
            generation.get("regional_late_step_scale", 0.35)
        ),
        regional_lora_delta_adaptation=bool(
            generation.get("regional_lora_delta_adaptation", False)
        ),
        regional_lora_delta_adaptation_gain=float(
            generation.get("regional_lora_delta_adaptation_gain", 0.35)
        ),
        prompt_emphases=prompt_emphases_from_payload(
            generation.get("prompt_emphases", [])
        ),
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
        projector_identity_protection=float(
            generation.get("projector_identity_protection", 1.0)
        ),
        face_detail_seed=int(generation.get("face_detail_seed", 0)),
        face_detail_steps=int(generation.get("face_detail_steps", 8)),
        face_detail_denoise=float(generation.get("face_detail_denoise", 0.15)),
        face_detail_crop_size=int(generation.get("face_detail_crop_size", 512)),
        face_detail_padding=float(generation.get("face_detail_padding", 2.0)),
        face_detail_feather=float(generation.get("face_detail_feather", 0.12)),
        face_detail_blend=float(generation.get("face_detail_blend", 0.5)),
        face_detail_lora_scale=float(generation.get("face_detail_lora_scale", 0.5)),
        face_detail_detector_threshold=float(
            generation.get("face_detail_detector_threshold", 0.15)
        ),
        face_detail_detector_provider=str(
            generation.get("face_detail_detector_provider", "auto")
        ),
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
        image_edit=ImageEditState(
            source_image=(
                Path(edit_document["source_image"]).expanduser()
                if edit_document.get("source_image")
                else None
            ),
            associated_project=(
                Path(edit_document["associated_project"]).expanduser()
                if edit_document.get("associated_project")
                else None
            ),
            width=int(edit_document.get("width", 0)),
            height=int(edit_document.get("height", 0)),
            reference_global_prompt=str(
                edit_document.get("reference_global_prompt", "")
            ),
            reference_regions=edit_reference_regions,
            reference_prompt_emphases=prompt_emphases_from_payload(
                edit_document.get("reference_prompt_emphases", [])
            ),
            reference_projector_enabled=bool(
                edit_document.get("reference_projector_enabled", False)
            ),
            reference_projector_preset=str(
                edit_document.get("reference_projector_preset", DEFAULT_PROJECTOR_PRESET)
            ),
            reference_projector_values=validate_projector_values(
                edit_document.get(
                    "reference_projector_values",
                    PROJECTOR_PRESETS[DEFAULT_PROJECTOR_PRESET],
                )
            ),
            reference_projector_multiplier=float(
                edit_document.get("reference_projector_multiplier", 1.0)
            ),
            reference_projector_identity_protection=float(
                edit_document.get("reference_projector_identity_protection", 1.0)
            ),
            global_prompt=str(edit_document.get("global_prompt", "")),
            steps=int(edit_document.get("steps", 8)),
            sampler=str(edit_document.get("sampler", DEFAULT_SAMPLER)),
            scheduler=str(edit_document.get("scheduler", DEFAULT_SCHEDULER)),
            seed=int(edit_document.get("seed", 0)),
            denoise=float(edit_document.get("denoise", 0.15)),
            latent_feather_pixels=int(
                edit_document.get("latent_feather_pixels", 64)
            ),
            composite_feather_pixels=int(
                edit_document.get("composite_feather_pixels", 48)
            ),
            edit_entire_image=bool(edit_document.get("edit_entire_image", False)),
            preserve_identity=bool(edit_document.get("preserve_identity", True)),
            reference_description_retention=float(
                edit_document.get("reference_description_retention", 1.0)
            ),
            regional_prompt_strength=float(
                edit_document.get("regional_prompt_strength", 1.0)
            ),
            regional_outside_penalty=float(
                edit_document.get("regional_outside_penalty", 1.0)
            ),
            regional_feather_pixels=int(
                edit_document.get("regional_feather_pixels", 128)
            ),
            regional_subject_competition=bool(
                edit_document.get("regional_subject_competition", True)
            ),
            regional_subject_fill=bool(
                edit_document.get("regional_subject_fill", True)
            ),
            regional_late_step_scale=float(
                edit_document.get("regional_late_step_scale", 0.35)
            ),
            regional_lora_delta_adaptation=bool(
                edit_document.get("regional_lora_delta_adaptation", False)
            ),
            regional_lora_delta_adaptation_gain=float(
                edit_document.get("regional_lora_delta_adaptation_gain", 0.35)
            ),
            regions=edit_regions,
        ),
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


def load_project_image(path: Path) -> ProjectState:
    """Restore project metadata from an application-generated PNG."""

    from PIL import Image

    source = path.expanduser().resolve()
    with Image.open(source) as image:
        if image.format != "PNG":
            raise ValueError("project image must be a PNG file")
        encoded = image.info.get(PNG_PROJECT_KEY)
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError(
            f"PNG does not contain K2 Region Lab metadata ({PNG_PROJECT_KEY})"
        )
    document = json.loads(encoded)
    if not isinstance(document, dict):
        raise ValueError("embedded K2 project root must be a JSON object")
    return replace(project_state(document), background_image=source)


def load_associated_image_project(path: Path) -> tuple[ProjectState, Path] | None:
    """Load project metadata embedded in an image or from an exact sidecar name."""

    from PIL import Image

    source = path.expanduser().resolve()
    with Image.open(source) as image:
        encoded = image.info.get(PNG_PROJECT_KEY)
    if isinstance(encoded, str) and encoded.strip():
        document = json.loads(encoded)
        if not isinstance(document, dict):
            raise ValueError("embedded K2 project root must be a JSON object")
        return replace(project_state(document), background_image=source), source

    sidecars = (
        source.with_suffix(".k2lab.json"),
        source.with_name(source.name + ".k2lab.json"),
    )
    for sidecar in dict.fromkeys(sidecars):
        if sidecar.is_file():
            return load_project(sidecar), sidecar
    return None
