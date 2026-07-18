from __future__ import annotations

import json
from dataclasses import replace
import torch

from .krea_region_lora.face_detailer import detail_faces
from .krea_region_lora.masks import region_from_mask
from .krea_region_lora.projector import apply_projector_settings
from .krea_region_lora.types import (
    K2LoraReference,
    K2ProjectorSettings,
    K2PromptEmphasis,
    K2RegionLayout,
    K2RegionSpec,
    K2RegionalLoraStack,
    K2RegionalSettings,
)
from .krea_region_lora.upscaler import post_upscale
from .krea_region_lora.workflow import (
    PROJECTOR_PRESETS,
    apply_prompt_emphases,
    bind_lora_reference,
    compile_spatial_conditioning,
    layout_from_document,
    layout_preview,
    parse_project_document,
    serialize_project_document,
    stack_layout_regions,
    stack_regional_loras,
)
from .nodes import (
    K2RegionalAttentionLoRASampler,
    _run_comfy_base_sampler,
    _sampler_names,
    _scheduler_names,
)

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - unit tests run outside ComfyUI
    folder_paths = None  # type: ignore


CATEGORY = "Krea 2/Region Lab"


def _lora_names() -> list[str]:
    if folder_paths is None:
        return ["None"]
    return folder_paths.get_filename_list("loras") or ["None"]


def _empty_mask(layout: K2RegionLayout) -> torch.Tensor:
    if not layout.regions:
        return torch.zeros((1, layout.height, layout.width), dtype=torch.float32)
    mask = torch.zeros_like(layout.regions[0].region.pixel_mask)
    for spec in layout.regions:
        if spec.enabled:
            mask = torch.maximum(mask, spec.region.pixel_mask)
    return mask.clamp(0.0, 1.0)


class K2RegionFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "name": ("STRING", {"default": "Subject 1"}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt_text": ("STRING", {"default": "", "multiline": True}),
                "face_identity_prompt": ("STRING", {"default": "", "multiline": True}),
                "spatial_role": (["auto", "subject", "background"], {"default": "auto"}),
                "priority": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("K2REGION_SPEC", "K2REGION", "MASK", "IMAGE")
    RETURN_NAMES = ("region_spec", "region", "mask", "preview")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        mask,
        positive,
        negative,
        name,
        prompt_text="",
        negative_prompt_text="",
        face_identity_prompt="",
        spatial_role="auto",
        priority=0,
        enabled=True,
    ):
        region_id = "-".join(str(name).strip().lower().split()) or "region"
        region = region_from_mask(mask, metadata={"region_id": region_id, "name": str(name)})
        spec = K2RegionSpec(
            region=region,
            name=str(name),
            prompt=str(prompt_text),
            negative_prompt=str(negative_prompt_text),
            face_identity_prompt=str(face_identity_prompt),
            spatial_role=str(spatial_role),
            priority=int(priority),
            enabled=bool(enabled),
            positive=positive,
            negative=negative,
        )
        preview_layout = stack_layout_regions(
            [spec], width=region.image_size[0], height=region.image_size[1]
        )
        return (spec, region, region.pixel_mask, layout_preview(preview_layout))


class K2RegionStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_1": ("K2REGION_SPEC",),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "global_positive": ("CONDITIONING",),
                "global_negative": ("CONDITIONING",),
            },
            "optional": {f"region_{index}": ("K2REGION_SPEC",) for index in range(2, 9)},
        }

    RETURN_TYPES = ("K2REGION_LAYOUT", "MASK", "IMAGE")
    RETURN_NAMES = ("layout", "union_mask", "preview")
    FUNCTION = "stack"
    CATEGORY = CATEGORY

    def stack(self, region_1, width, height, global_positive, global_negative, **kwargs):
        regions = [region_1] + [kwargs.get(f"region_{index}") for index in range(2, 9)]
        layout = stack_layout_regions(
            regions,
            width=width,
            height=height,
            global_positive=global_positive,
            global_negative=global_negative,
        )
        return (layout, _empty_mask(layout), layout_preview(layout))


class K2RegionEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_json": (
                    "STRING",
                    {
                        "default": serialize_project_document(parse_project_document(None)),
                        "multiline": True,
                    },
                ),
            },
            "optional": {"background_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("K2REGION_LAYOUT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("layout", "union_mask", "preview", "project_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, project_json, background_image=None):
        layout = layout_from_document(project_json)
        canonical = serialize_project_document(layout.source_document)
        return (layout, _empty_mask(layout), layout_preview(layout, background_image), canonical)


class K2CompileRegionalPrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": ("K2REGION_LAYOUT",),
                "clip": ("CLIP",),
            },
            "optional": {
                "global_positive": ("CONDITIONING",),
                "global_negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("K2REGION_LAYOUT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("layout", "spatial_positive", "spatial_negative")
    FUNCTION = "compile"
    CATEGORY = CATEGORY

    def compile(self, layout, clip, global_positive=None, global_negative=None):
        if layout.source_document:
            compiled = layout_from_document(layout.source_document, clip=clip)
        else:
            compiled = layout
        positive, negative = compile_spatial_conditioning(
            compiled, global_positive, global_negative
        )
        return (compiled, positive, negative)


class K2PromptEmphasisText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "phrase": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "occurrence": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("weighted_prompt",)
    FUNCTION = "emphasize"
    CATEGORY = CATEGORY

    def emphasize(self, prompt, phrase, strength=0.5, occurrence=0):
        emphasis = K2PromptEmphasis("global", str(phrase), float(strength), int(occurrence))
        return (apply_prompt_emphases(str(prompt), (emphasis,), "global"),)


class K2RegionalControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": ("K2REGION_LAYOUT",),
                "enabled": ("BOOLEAN", {"default": True}),
                "inside_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "outside_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "feather_pixels": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 16}),
                "subject_competition": ("BOOLEAN", {"default": True}),
                "subject_fill": ("BOOLEAN", {"default": True}),
                "relaxation": ("BOOLEAN", {"default": True}),
                "late_step_scale": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "lora_delta_adaptation": ("BOOLEAN", {"default": False}),
                "lora_delta_adaptation_gain": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("K2REGION_LAYOUT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("layout", "spatial_positive", "spatial_negative")
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(self, layout, **values):
        settings = K2RegionalSettings(
            regional_prompting=bool(values["enabled"]),
            inside_strength=float(values["inside_strength"]),
            outside_penalty=float(values["outside_penalty"]),
            feather_pixels=int(values["feather_pixels"]),
            subject_competition=bool(values["subject_competition"]),
            subject_fill=bool(values["subject_fill"]),
            relaxation=bool(values["relaxation"]),
            late_step_scale=float(values["late_step_scale"]),
            lora_delta_adaptation=bool(values["lora_delta_adaptation"]),
            lora_delta_adaptation_gain=float(values["lora_delta_adaptation_gain"]),
        )
        configured = replace(layout, regional=settings)
        positive, negative = compile_spatial_conditioning(configured)
        return (configured, positive, negative)


class K2ProjectorControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": ("K2REGION_LAYOUT",),
                "enabled": ("BOOLEAN", {"default": False}),
                "preset": ([*PROJECTOR_PRESETS, "custom"], {"default": "filter_bypass2"}),
                "vector": (
                    "STRING",
                    {"default": json.dumps(PROJECTOR_PRESETS["filter_bypass2"]), "multiline": True},
                ),
                "multiplier": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1}),
                "identity_protection": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("K2REGION_LAYOUT",)
    FUNCTION = "configure"
    CATEGORY = CATEGORY

    def configure(self, layout, enabled, preset, vector, multiplier, identity_protection):
        values = PROJECTOR_PRESETS[preset] if preset != "custom" else tuple(json.loads(vector))
        settings = K2ProjectorSettings(
            enabled=bool(enabled),
            preset=str(preset),
            values=tuple(float(value) for value in values),
            multiplier=float(multiplier),
            identity_protection=float(identity_protection),
        )
        return (replace(layout, projector=settings),)


class K2ApplyProjector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "layout": ("K2REGION_LAYOUT",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, layout):
        return apply_projector_settings(model, layout.projector)


class K2LoRAReferenceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_lora_names(),),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "routing_mode": (["standard", "character_identity"], {"default": "standard"}),
                "trigger_phrase": ("STRING", {"default": ""}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("K2LORA_REFERENCE",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self, lora_name, strength, routing_mode, trigger_phrase, start_percent, end_percent, enabled
    ):
        return (
            K2LoraReference(
                lora_name=str(lora_name),
                strength=float(strength),
                routing_mode=str(routing_mode),
                trigger_phrase=str(trigger_phrase),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                enabled=bool(enabled),
            ),
        )


class K2ApplyLoRAToRegions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": ("K2REGION_LAYOUT",),
                "lora": ("K2LORA_REFERENCE",),
                "target_regions": ("STRING", {"default": "Subject 1", "multiline": True}),
                "overlap_mode": (
                    ["normalize", "priority_1", "priority_3", "add_clamped"],
                    {"default": "normalize"},
                ),
            },
            "optional": {"regional_lora_stack": ("K2REGIONAL_LORA_STACK",)},
        }

    RETURN_TYPES = ("K2REGIONAL_LORA_STACK", "STRING")
    RETURN_NAMES = ("regional_lora_stack", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(
        self, layout, lora, target_regions, overlap_mode="normalize", regional_lora_stack=None
    ):
        targets = [
            item.strip()
            for item in str(target_regions).replace("\n", ",").split(",")
            if item.strip()
        ]
        bindings = bind_lora_reference(layout, lora, targets)
        stack = stack_regional_loras(regional_lora_stack, bindings, overlap_mode)
        report = f"Assigned {lora.lora_name} to {', '.join(item.region_name for item in bindings)}"
        return (stack, report)


class K2RegionLabSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "layout": ("K2REGION_LAYOUT",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "execution_mode": (
                    ["auto", "strict_adapter", "layer_injection"],
                    {"default": "auto"},
                ),
            },
            "optional": {"regional_lora_stack": ("K2REGIONAL_LORA_STACK",)},
        }

    RETURN_TYPES = ("LATENT", "LATENT", "MASK", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = (
        "samples",
        "base_samples",
        "union_mask",
        "spatial_positive",
        "spatial_negative",
        "report",
    )
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        layout,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        execution_mode="auto",
        regional_lora_stack=None,
    ):
        spatial_positive, spatial_negative = compile_spatial_conditioning(
            layout, positive, negative
        )
        patched_model, projector_report = apply_projector_settings(model, layout.projector)
        if regional_lora_stack is None or not regional_lora_stack.enabled_regions:
            samples = _run_comfy_base_sampler(
                patched_model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                spatial_positive,
                spatial_negative,
                latent_image,
                denoise,
            )
            union = _empty_mask(layout)
            report = "No regional LoRAs connected; sampled native spatial conditionings."
            return (
                samples,
                samples.copy(),
                union,
                spatial_positive,
                spatial_negative,
                f"{projector_report}\n{report}",
            )
        result = K2RegionalAttentionLoRASampler().sample(
            patched_model,
            spatial_positive,
            spatial_negative,
            latent_image,
            regional_lora_stack,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            execution_mode=execution_mode,
        )
        samples, base, mask, lora_report = result
        settings_report = (
            f"Spatial: inside={layout.regional.inside_strength:.2f} outside={layout.regional.outside_penalty:.2f} "
            f"feather={layout.regional.feather_pixels}px late={layout.regional.late_step_scale:.2f} "
            f"delta_adaptation={layout.regional.lora_delta_adaptation}"
        )
        return (
            samples,
            base,
            mask,
            spatial_positive,
            spatial_negative,
            f"{projector_report}\n{settings_report}\n{lora_report}",
        )


class K2FaceDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 1.0, "step": 0.05}),
                "crop_size": ([256, 512, 768, 1024], {"default": 512}),
                "padding": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),
                "feather": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.02}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "regional_lora_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "bbox_format": (["xywh", "xyxy"], {"default": "xywh"}),
            },
            "optional": {
                "layout": ("K2REGION_LAYOUT",),
                "regional_lora_stack": ("K2REGIONAL_LORA_STACK",),
                "segs": ("SEGS",),
                "bboxes": ("BOUNDING_BOX",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "detail_mask", "report")
    FUNCTION = "detail"
    CATEGORY = CATEGORY

    def detail(
        self,
        image,
        model,
        vae,
        positive,
        negative,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        crop_size,
        padding,
        feather,
        blend,
        regional_lora_scale,
        bbox_format="xywh",
        layout=None,
        regional_lora_stack=None,
        segs=None,
        bboxes=None,
    ):
        return detail_faces(
            image=image,
            model=model,
            vae=vae,
            positive=positive,
            negative=negative,
            sampler=_run_comfy_base_sampler,
            bboxes=bboxes,
            segs=segs,
            layout=layout,
            regional_lora_stack=regional_lora_stack,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            crop_size=int(crop_size),
            padding=padding,
            feather=feather,
            blend=blend,
            lora_scale=regional_lora_scale,
            bbox_format=bbox_format,
        )


class K2PostUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ([2, 4], {"default": 2}),
                "method": (["lanczos", "model"], {"default": "lanczos"}),
            },
            "optional": {"upscale_model": ("UPSCALE_MODEL",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, image, scale=2, method="lanczos", upscale_model=None):
        return post_upscale(
            image, scale=int(scale), method=str(method), upscale_model=upscale_model
        )


class K2RegionLabApp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "project_json": (
                    "STRING",
                    {
                        "default": serialize_project_document(parse_project_document(None)),
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "background_image": ("IMAGE",),
                **{f"lora_{index}": ("K2LORA_REFERENCE",) for index in range(1, 5)},
            },
        }

    RETURN_TYPES = (
        "K2REGION_LAYOUT",
        "CONDITIONING",
        "CONDITIONING",
        "K2REGIONAL_LORA_STACK",
        "MASK",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "layout",
        "spatial_positive",
        "spatial_negative",
        "regional_lora_stack",
        "union_mask",
        "preview",
        "project_json",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, clip, project_json, background_image=None, **kwargs):
        layout = layout_from_document(project_json, clip=clip)
        positive, negative = compile_spatial_conditioning(layout)
        stack = K2RegionalLoraStack(())
        assignments = layout.source_document.get("lora_assignments", {})
        for index in range(1, 5):
            reference = kwargs.get(f"lora_{index}")
            targets = assignments.get(str(index), assignments.get(index, []))
            if reference is not None and targets:
                stack = stack_regional_loras(stack, bind_lora_reference(layout, reference, targets))
        return (
            layout,
            positive,
            negative,
            stack,
            _empty_mask(layout),
            layout_preview(layout, background_image),
            serialize_project_document(layout.source_document),
        )
