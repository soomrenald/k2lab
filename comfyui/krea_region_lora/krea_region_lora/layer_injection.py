from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .lora import filter_lora_state_dict
from .types import K2RegionalLora, K2RegionalLoraStack

LayerTargetPolicy = Literal["attn_out_mlp", "attention_only", "all_matched_linears"]


@dataclass
class LayerLoRAPatch:
    regional: K2RegionalLora
    layer_name: str
    source_key: str
    down: torch.Tensor
    up: torch.Tensor
    scale: float


@dataclass
class LayerInjectionReport:
    lines: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)

    def text(self) -> str:
        return "\n".join(self.lines)


class RegionalLayerInjectionState:
    def __init__(
        self,
        patches_by_layer: dict[str, list[LayerLoRAPatch]],
        *,
        outside_strength: float = 0.0,
        text_token_strength: float = 0.0,
        debug: bool = False,
    ):
        self.patches_by_layer = patches_by_layer
        self.outside_strength = float(outside_strength)
        self.text_token_strength = float(text_token_strength)
        self.debug = bool(debug)
        self._tensor_cache: dict[tuple[int, str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def wrapper(self, executor, *args, **kwargs):
        model_obj = getattr(executor, "class_obj", None)
        if model_obj is None:
            return executor(*args, **kwargs)
        handles = []
        self._tensor_cache = {}
        try:
            name_to_module = dict(model_obj.named_modules())
            for layer_name, patches in self.patches_by_layer.items():
                module = name_to_module.get(layer_name)
                if module is not None:
                    handles.append(module.register_forward_hook(self._make_hook(patches)))
            return executor(*args, **kwargs)
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    pass
            self._tensor_cache = {}

    def _make_hook(self, patches: list[LayerLoRAPatch]):
        def hook(_module, inputs, output):
            if not torch.is_tensor(output) or not inputs:
                return output
            x = inputs[0]
            if (
                not torch.is_tensor(x)
                or x.ndim < 3
                or output.ndim < 3
                or x.shape[:-1] != output.shape[:-1]
            ):
                return output
            out = output
            compute_dtype = _compute_dtype_for(x)
            for patch in patches:
                mask = _sequence_mask_for_region(
                    patch.regional,
                    x,
                    outside_strength=self.outside_strength,
                    text_token_strength=self.text_token_strength,
                )
                if mask is None:
                    continue
                down, up = self._matrices_on_device(patch, x.device, compute_dtype)
                xin = x.to(dtype=compute_dtype) if x.dtype != compute_dtype else x
                delta = F.linear(F.linear(xin, down), up)
                delta = (
                    delta
                    * float(patch.scale)
                    * float(patch.regional.lora_strength)
                    * float(patch.regional.delta_strength)
                )
                out = out + (delta * mask.to(device=delta.device, dtype=delta.dtype)).to(
                    dtype=out.dtype
                )
            return out

        return hook

    def _matrices_on_device(
        self, patch: LayerLoRAPatch, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (id(patch), str(device), str(dtype), patch.source_key)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached
        down = patch.down.to(device=device, dtype=dtype, non_blocking=True)
        up = patch.up.to(device=device, dtype=dtype, non_blocking=True)
        self._tensor_cache[key] = (down, up)
        return down, up


def build_layer_injection_model(
    model: Any,
    stack: K2RegionalLoraStack,
    *,
    target_policy: LayerTargetPolicy = "attn_out_mlp",
    outside_strength: float = 0.0,
    text_token_strength: float = 0.0,
    debug: bool = False,
) -> tuple[Any, str]:
    try:
        import comfy.lora  # type: ignore
        import comfy.lora_convert  # type: ignore
        import comfy.patcher_extension  # type: ignore
        import comfy.utils  # type: ignore
        import folder_paths  # type: ignore
    except Exception as exc:  # pragma: no cover - only exercised inside ComfyUI
        raise RuntimeError("ComfyUI patcher/LoRA APIs are not importable") from exc

    report = LayerInjectionReport()
    model_out = model.clone()
    diffusion_model = model_out.get_model_object("diffusion_model")
    key_map_model = getattr(model_out, "model", diffusion_model)
    name_to_module = _eligible_linears(diffusion_model, target_policy)
    normalized_names = {_normalize_name(name): name for name in name_to_module}
    key_map = comfy.lora.model_lora_keys_unet(key_map_model, {})

    patches_by_layer: dict[str, list[LayerLoRAPatch]] = {}
    report.add(
        f"Layer-injection fallback: target_policy={target_policy} eligible_linears={len(name_to_module)} "
        f"outside_strength={outside_strength:.3f} text_token_strength={text_token_strength:.3f}"
    )

    for regional_index, regional in enumerate(stack.enabled_regions, start=1):
        if regional.lora_name in ("", "None"):
            report.add(f"[{regional_index}] skipped empty LoRA name")
            continue
        lora_path = folder_paths.get_full_path_or_raise("loras", regional.lora_name)
        raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
        raw = filter_lora_state_dict(
            raw,
            attention_only_filter=False,
            ignore_text_encoder_lora=regional.ignore_text_encoder_lora,
        )
        converted = comfy.lora_convert.convert_lora(raw)
        loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
        matched = 0
        skipped = 0
        for target, patch_data in loaded.items():
            weight_key = _target_weight_key(target)
            if weight_key is None:
                skipped += 1
                continue
            layer_name = _module_name_from_weight_key(weight_key, name_to_module, normalized_names)
            if layer_name is None:
                skipped += 1
                continue
            matrices = _matrices_from_loaded_patch(patch_data)
            if matrices is None:
                skipped += 1
                continue
            down, up, scale = matrices
            module = name_to_module[layer_name]
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight) or weight.ndim != 2:
                skipped += 1
                continue
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            if (
                int(down.shape[1]) != in_features
                or int(up.shape[0]) != out_features
                or int(up.shape[1]) != int(down.shape[0])
            ):
                skipped += 1
                continue
            patches_by_layer.setdefault(layer_name, []).append(
                LayerLoRAPatch(
                    regional=regional,
                    layer_name=layer_name,
                    source_key=str(weight_key),
                    down=down.detach().cpu().contiguous(),
                    up=up.detach().cpu().contiguous(),
                    scale=scale,
                )
            )
            matched += 1
        report.add(
            f"[{regional_index}] {regional.lora_name}: loaded_targets={len(loaded)} matched_layers={matched} skipped={skipped}"
        )

    if not patches_by_layer:
        raise RuntimeError("No usable model-side LoRA layers matched for regional layer injection.")

    state = RegionalLayerInjectionState(
        patches_by_layer,
        outside_strength=outside_strength,
        text_token_strength=text_token_strength,
        debug=debug,
    )
    model_out.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "k2_regional_layer_lora_injection",
        state.wrapper,
    )
    report.add(f"Installed regional layer injection wrapper on {len(patches_by_layer)} layers.")
    return model_out, report.text()


def _eligible_linears(model_obj: Any, target_policy: LayerTargetPolicy) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, module in model_obj.named_modules():
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        lname = name.lower()
        if target_policy == "all_matched_linears":
            out[name] = module
        elif target_policy == "attention_only":
            if _is_attention_out(lname) or _is_attention_qkv_or_gate(lname):
                out[name] = module
        elif _is_attention_out(lname) or _is_mlp(lname):
            out[name] = module
    return out


def _sequence_mask_for_region(
    regional: K2RegionalLora,
    x: torch.Tensor,
    *,
    outside_strength: float,
    text_token_strength: float,
) -> torch.Tensor | None:
    seq_len = int(x.shape[-2])
    token_mask = regional.region.token_mask
    img_len = int(token_mask.shape[1])
    if img_len <= 0:
        return None
    if token_mask.shape[0] == 1 and int(x.shape[0]) > 1:
        token_mask = token_mask.repeat(int(x.shape[0]), 1, 1)
    elif token_mask.shape[0] != int(x.shape[0]):
        token_mask = token_mask[:1].repeat(int(x.shape[0]), 1, 1)
    token_mask = token_mask.to(device=x.device, dtype=x.dtype)
    if seq_len == img_len:
        full = token_mask
    elif seq_len > img_len:
        full = torch.full(
            (int(x.shape[0]), seq_len, 1),
            float(text_token_strength),
            device=x.device,
            dtype=x.dtype,
        )
        full[:, seq_len - img_len :, :] = token_mask
    else:
        resized = F.interpolate(
            token_mask.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        full = resized.clamp(0.0, 1.0)
    if outside_strength != 0.0:
        full = full + (1.0 - full) * float(outside_strength)
    while full.ndim < x.ndim:
        full = full.unsqueeze(1)
    return full


def _matrices_from_loaded_patch(patch_data: Any) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    if isinstance(patch_data, tuple) and len(patch_data) >= 2 and patch_data[0] == "lora":
        weights = patch_data[1]
    else:
        if getattr(patch_data, "name", None) != "lora":
            return None
        weights = getattr(patch_data, "weights", None)
    if not isinstance(weights, tuple) or len(weights) < 6:
        return None
    up, down, alpha, mid, dora_scale, reshape = weights[:6]
    if mid is not None or dora_scale is not None or reshape is not None:
        return None
    if not torch.is_tensor(up) or not torch.is_tensor(down) or up.ndim != 2 or down.ndim != 2:
        return None
    rank = max(1, int(down.shape[0]))
    scale = 1.0 if alpha is None else float(alpha) / rank
    return down, up, scale


def _target_weight_key(target: Any) -> str | None:
    if isinstance(target, str):
        return target
    if isinstance(target, tuple) and target and isinstance(target[0], str) and len(target) == 1:
        return target[0]
    return None


def _module_name_from_weight_key(
    weight_key: str, names: dict[str, Any], normalized_names: dict[str, str]
) -> str | None:
    if not weight_key.endswith(".weight"):
        return None
    candidates = [weight_key[: -len(".weight")]]
    for candidate in list(candidates):
        for prefix in ("diffusion_model.", "model.diffusion_model.", "model.", "base_model.model."):
            if candidate.startswith(prefix):
                candidates.append(candidate[len(prefix) :])
    for candidate in candidates:
        if candidate in names:
            return candidate
        normalized = normalized_names.get(_normalize_name(candidate))
        if normalized is not None:
            return normalized
    return None


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "").replace(".", "")


def _is_attention_out(name: str) -> bool:
    if "attn" not in name and "attention" not in name:
        return False
    qkv_fragments = ("wq", "wk", "wv", "q_proj", "k_proj", "v_proj", ".q.", ".k.", ".v.")
    if any(fragment in name for fragment in qkv_fragments):
        return False
    return any(fragment in name for fragment in ("wo", "to_out", "out_proj", "o_proj", "proj"))


def _is_attention_qkv_or_gate(name: str) -> bool:
    return ("attn" in name or "attention" in name) and any(
        fragment in name for fragment in ("wq", "wk", "wv", "q_proj", "k_proj", "v_proj", "gate")
    )


def _is_mlp(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "mlp",
            "ffn",
            "feed_forward",
            "feedforward",
            "fc1",
            "fc2",
            "up_proj",
            "down_proj",
            "gate_proj",
        )
    )


def _compute_dtype_for(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return x.dtype
    return torch.float32
