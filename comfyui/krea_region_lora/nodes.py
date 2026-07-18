from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .krea_region_lora.engine import run_regional_velocity_sampler
    from .krea_region_lora.layer_injection import build_layer_injection_model
    from .krea_region_lora.lora import make_lora_branch_model
    from .krea_region_lora.masks import debug_bbox_image, infer_image_size, region_from_bbox
    from .krea_region_lora.types import K2RegionalLora, K2RegionalLoraStack
except ImportError:
    from krea_region_lora.engine import run_regional_velocity_sampler
    from krea_region_lora.layer_injection import build_layer_injection_model
    from krea_region_lora.lora import make_lora_branch_model
    from krea_region_lora.masks import debug_bbox_image, infer_image_size, region_from_bbox
    from krea_region_lora.types import K2RegionalLora, K2RegionalLoraStack

try:
    import comfy.sample  # type: ignore
    import comfy.samplers  # type: ignore
    import comfy.utils  # type: ignore
    import folder_paths  # type: ignore
    import latent_preview  # type: ignore
except Exception:  # pragma: no cover - tests run without a full Comfy import path
    comfy = None  # type: ignore
    folder_paths = None  # type: ignore
    latent_preview = None  # type: ignore


def _lora_names() -> list[str]:
    if folder_paths is None:
        return ["None"]
    names = folder_paths.get_filename_list("loras")
    return names or ["None"]


def _sampler_names() -> list[str]:
    if comfy is None:
        return ["euler"]
    return comfy.samplers.KSampler.SAMPLERS


def _scheduler_names() -> list[str]:
    if comfy is None:
        return ["normal"]
    return comfy.samplers.KSampler.SCHEDULERS


class K2BBoxToRegionalMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "bbox_format": (["xywh", "xyxy"], {"default": "xywh"}),
                "bbox_index": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "grow_px": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "feather_px": ("INT", {"default": 32, "min": 0, "max": 2048}),
                "snap_to_krea_token_grid": ("BOOLEAN", {"default": True}),
                "batch_mode": (["single", "repeat", "per_batch"], {"default": "repeat"}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "kj_bboxes": ("BBOX",),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("MASK", "K2REGION", "IMAGE")
    RETURN_NAMES = ("region_mask", "region", "debug_bbox_image")
    FUNCTION = "build"
    CATEGORY = "Krea 2/Regional LoRA"

    def build(
        self,
        width,
        height,
        bbox_format="xywh",
        bbox_index=0,
        grow_px=0,
        feather_px=32,
        snap_to_krea_token_grid=True,
        batch_mode="repeat",
        bboxes=None,
        kj_bboxes=None,
        latent=None,
    ):
        image_w, image_h, batch = infer_image_size(latent, width, height)
        bbox_source = bboxes if bboxes is not None else kj_bboxes
        region = region_from_bbox(
            bbox_source,
            width=image_w,
            height=image_h,
            bbox_format=bbox_format,
            bbox_index=bbox_index,
            grow_px=grow_px,
            feather_px=feather_px,
            snap_to_krea_token_grid=snap_to_krea_token_grid,
            batch_mode=batch_mode,
            batch_size=batch,
        )
        return (region.pixel_mask, region, debug_bbox_image(region))


class K2RegionalCharacterLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region": ("K2REGION",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "lora_name": (_lora_names(),),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "delta_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
                "attention_only_filter": ("BOOLEAN", {"default": True}),
                "ignore_text_encoder_lora": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("K2REGIONAL_LORA",)
    RETURN_NAMES = ("regional_lora",)
    FUNCTION = "bind"
    CATEGORY = "Krea 2/Regional LoRA"

    def bind(
        self,
        region,
        positive,
        negative,
        lora_name,
        lora_strength=1.0,
        delta_strength=1.0,
        start_percent=0.10,
        end_percent=0.95,
        enabled=True,
        attention_only_filter=True,
        ignore_text_encoder_lora=True,
    ):
        return (
            K2RegionalLora(
                region=region,
                positive=positive,
                negative=negative,
                lora_name=lora_name,
                lora_strength=float(lora_strength),
                delta_strength=float(delta_strength),
                start_percent=float(start_percent),
                end_percent=float(end_percent),
                enabled=bool(enabled),
                attention_only_filter=bool(attention_only_filter),
                ignore_text_encoder_lora=bool(ignore_text_encoder_lora),
            ),
        )


class K2RegionalLoRAStack3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regional_lora_1": ("K2REGIONAL_LORA",),
                "overlap_mode": (
                    ["normalize", "priority_1", "priority_3", "add_clamped"],
                    {"default": "normalize"},
                ),
            },
            "optional": {
                "regional_lora_2": ("K2REGIONAL_LORA",),
                "regional_lora_3": ("K2REGIONAL_LORA",),
            },
        }

    RETURN_TYPES = ("K2REGIONAL_LORA_STACK",)
    RETURN_NAMES = ("regional_lora_stack",)
    FUNCTION = "stack"
    CATEGORY = "Krea 2/Regional LoRA"

    def stack(
        self, regional_lora_1, overlap_mode="normalize", regional_lora_2=None, regional_lora_3=None
    ):
        regions = tuple(
            r for r in (regional_lora_1, regional_lora_2, regional_lora_3) if r is not None
        )
        return (K2RegionalLoraStack(regions=regions, overlap_mode=overlap_mode),)


class K2RegionalAttentionLoRASampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "regional_lora_stack": ("K2REGIONAL_LORA_STACK",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "execution_mode": (
                    ["auto", "strict_adapter", "layer_injection"],
                    {"default": "auto"},
                ),
                "layer_injection_targets": (
                    ["attn_out_mlp", "attention_only", "all_matched_linears"],
                    {"default": "attn_out_mlp"},
                ),
                "layer_outside_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "layer_text_token_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "pin_outside_regions": ("BOOLEAN", {"default": True}),
                "final_latent_pin": ("BOOLEAN", {"default": True}),
                "post_decode_safe_mode": ("BOOLEAN", {"default": True}),
                "debug_return_base_latent": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "MASK", "STRING")
    RETURN_NAMES = ("samples", "base_samples", "union_mask", "debug_info")
    FUNCTION = "sample"
    CATEGORY = "Krea 2/Regional LoRA"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        regional_lora_stack,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        execution_mode="auto",
        layer_injection_targets="attn_out_mlp",
        layer_outside_strength=0.0,
        layer_text_token_strength=0.0,
        pin_outside_regions=True,
        final_latent_pin=True,
        post_decode_safe_mode=True,
        debug_return_base_latent=True,
    ):
        base = _run_comfy_base_sampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )
        active = regional_lora_stack.enabled_regions
        union_mask = _stack_union_mask(regional_lora_stack, latent_image["samples"])
        if not active:
            return (
                base,
                base.copy(),
                union_mask,
                "No enabled regional LoRAs; returned base sampler output.",
            )

        adapter = getattr(model, "k2_regional_velocity_predictor", None)
        if adapter is None and execution_mode == "strict_adapter":
            return (
                base,
                base.copy() if debug_return_base_latent else base,
                union_mask,
                "Strict adapter mode selected, but this model does not expose k2_regional_velocity_predictor; returned base samples.",
            )

        if adapter is not None and execution_mode in ("auto", "strict_adapter"):
            return _run_adapter_regional_sampler(
                adapter,
                model,
                positive,
                negative,
                latent_image,
                regional_lora_stack,
                seed,
                steps,
                cfg,
                pin_outside_regions,
            )

        return _run_layer_injection_fallback(
            model,
            base,
            positive,
            negative,
            latent_image,
            regional_lora_stack,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            layer_injection_targets,
            layer_outside_strength,
            layer_text_token_strength,
            final_latent_pin,
            debug_return_base_latent,
        )


class K2RegionalLayerLoRAApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regional_lora_stack": ("K2REGIONAL_LORA_STACK",),
                "layer_injection_targets": (
                    ["attn_out_mlp", "attention_only", "all_matched_linears"],
                    {"default": "attn_out_mlp"},
                ),
                "outside_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "text_token_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "debug_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "Krea 2/Regional LoRA"

    def apply(
        self,
        model,
        regional_lora_stack,
        layer_injection_targets="attn_out_mlp",
        outside_strength=0.0,
        text_token_strength=0.0,
        debug_logging=False,
    ):
        return build_layer_injection_model(
            model,
            regional_lora_stack,
            target_policy=layer_injection_targets,
            outside_strength=outside_strength,
            text_token_strength=text_token_strength,
            debug=debug_logging,
        )


class K2RegionalDecodeComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "regional_samples": ("LATENT",),
                "base_samples": ("LATENT",),
                "union_mask": ("MASK",),
                "feather_px": ("INT", {"default": 32, "min": 0, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "Krea 2/Regional LoRA"

    def composite(self, vae, regional_samples, base_samples, union_mask, feather_px=32):
        regional_image = vae.decode(regional_samples["samples"])
        base_image = vae.decode(base_samples["samples"])
        mask = union_mask.unsqueeze(-1)
        if mask.shape[1:3] != regional_image.shape[1:3]:
            mask = F.interpolate(
                mask.permute(0, 3, 1, 2),
                size=regional_image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            )
            mask = mask.permute(0, 2, 3, 1)
        if feather_px > 0:
            mask = F.avg_pool2d(
                mask.permute(0, 3, 1, 2),
                kernel_size=2 * int(feather_px) + 1,
                stride=1,
                padding=int(feather_px),
                count_include_pad=False,
            ).permute(0, 2, 3, 1)
        image = mask.clamp(0.0, 1.0) * regional_image + (1.0 - mask.clamp(0.0, 1.0)) * base_image
        return (image.clamp(0.0, 1.0),)


def _run_comfy_base_sampler(
    model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
):
    if comfy is None:
        raise RuntimeError("ComfyUI sampler APIs are not importable")
    latent = latent_image
    latent_samples = latent["samples"]
    latent_samples = comfy.sample.fix_empty_latent_channels(
        model,
        latent_samples,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps) if latent_preview is not None else None
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_samples,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _stack_union_mask(stack: K2RegionalLoraStack, target: torch.Tensor) -> torch.Tensor:
    if not stack.enabled_regions:
        width = int(target.shape[-1] * 8)
        height = int(target.shape[-2] * 8)
        return torch.zeros(
            (target.shape[0], height, width), dtype=target.dtype, device=target.device
        )
    union = None
    for regional in stack.enabled_regions:
        mask = regional.region.pixel_mask
        if mask.shape[0] == 1 and target.shape[0] > 1:
            mask = mask.repeat(target.shape[0], 1, 1)
        union = mask if union is None else torch.maximum(union, mask)
    return union.clamp(0.0, 1.0)


def _stack_latent_union_mask(stack: K2RegionalLoraStack, target: torch.Tensor) -> torch.Tensor:
    if not stack.enabled_regions:
        return torch.zeros(
            (target.shape[0], 1, target.shape[-2], target.shape[-1]),
            dtype=target.dtype,
            device=target.device,
        )
    union = None
    for regional in stack.enabled_regions:
        mask = regional.region.mask_for(target).to(device=target.device, dtype=target.dtype)
        union = mask if union is None else torch.maximum(union, mask)
    return union.clamp(0.0, 1.0)


def _pin_latent_to_base_outside(samples, base_samples, stack):
    mask = _stack_latent_union_mask(stack, samples["samples"])
    out = samples.copy()
    regional = samples["samples"]
    base = base_samples["samples"].to(device=regional.device, dtype=regional.dtype)
    out["samples"] = mask * regional + (1.0 - mask) * base
    return out


def _run_layer_injection_fallback(
    model,
    base,
    positive,
    negative,
    latent_image,
    stack,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    denoise,
    layer_injection_targets,
    layer_outside_strength,
    layer_text_token_strength,
    final_latent_pin,
    debug_return_base_latent,
):
    patched_model, report = build_layer_injection_model(
        model,
        stack,
        target_policy=layer_injection_targets,
        outside_strength=float(layer_outside_strength),
        text_token_strength=float(layer_text_token_strength),
        debug=False,
    )
    regional = _run_comfy_base_sampler(
        patched_model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
    )
    pinned = False
    if final_latent_pin:
        regional = _pin_latent_to_base_outside(regional, base, stack)
        pinned = True
    debug = (
        "Used layer-injection fallback: LoRA deltas are written only to masked regional token streams "
        "on cloned model layer hooks, then the final latent is pinned outside the union mask to base. "
        f"final_latent_pin={pinned}\n{report}"
    )
    return (
        regional,
        base.copy() if debug_return_base_latent else base,
        _stack_union_mask(stack, latent_image["samples"]),
        debug,
    )


def _run_adapter_regional_sampler(
    adapter,
    model,
    positive,
    negative,
    latent_image,
    stack,
    seed,
    steps,
    cfg,
    pin_outside_regions,
):
    initial = latent_image["samples"]
    branches = {
        regional.lora_name: make_lora_branch_model(
            model,
            regional.lora_name,
            strength_model=regional.lora_strength,
            attention_only_filter=regional.attention_only_filter,
            ignore_text_encoder_lora=regional.ignore_text_encoder_lora,
        )
        for regional in stack.enabled_regions
    }
    schedule = adapter.schedule(model=model, steps=steps, seed=seed)

    def predict(branch_name, x, sigma, cond, uncond):
        branch_model = model if branch_name == "base" else branches[branch_name]
        return adapter.guided_predict(branch_model, x, cond, uncond, cfg, sigma)

    regional, base, _latent_union, debug = run_regional_velocity_sampler(
        initial=initial,
        stack=stack,
        base_positive=positive,
        base_negative=negative,
        cfg=cfg,
        schedule=schedule,
        predict=predict,
        pin_outside_regions=pin_outside_regions,
    )
    out = latent_image.copy()
    out["samples"] = regional
    base_out = latent_image.copy()
    base_out["samples"] = base
    return (out, base_out, _stack_union_mask(stack, initial), debug.to_text())
