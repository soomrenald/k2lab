from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ATTENTION_TARGETS = ("wq", "wk", "wv", "gate", "wo")
ATTENTION_OUT_TARGETS = ("wo", "to_out", "out_proj", "o_proj", "proj")
MLP_TARGETS = (
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
TEXT_ENCODER_HINTS = (
    "clip",
    "text_encoder",
    "text.encoder",
    "cond_stage_model",
    "transformer.text",
    "llm",
)
MLP_HINTS = ("mlp", "ffn", "feed_forward", "feedforward", "fc1", "fc2", "proj_in", "proj_out")


def is_text_encoder_lora_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in TEXT_ENCODER_HINTS)


def is_krea_attention_lora_key(key: str) -> bool:
    lowered = key.lower()
    if any(hint in lowered for hint in MLP_HINTS):
        return False
    parts = lowered.replace("/", ".").split(".")
    if not any(part == "attn" or "attention" in part for part in parts):
        return False
    return any(
        target in parts or lowered.endswith(f".{target}.weight") or f".{target}." in lowered
        for target in ATTENTION_TARGETS
    )


def is_krea_attention_out_lora_key(key: str) -> bool:
    lowered = key.lower()
    parts = lowered.replace("/", ".").split(".")
    if not any(part == "attn" or "attention" in part for part in parts):
        return False
    if any(target in parts for target in ("wq", "wk", "wv", "q_proj", "k_proj", "v_proj")):
        return False
    return any(
        target in parts or lowered.endswith(f".{target}.weight") or f".{target}." in lowered
        for target in ATTENTION_OUT_TARGETS
    )


def is_krea_mlp_lora_key(key: str) -> bool:
    lowered = key.lower()
    parts = lowered.replace("/", ".").split(".")
    return any(target in parts or f".{target}." in lowered for target in MLP_TARGETS)


def is_krea_writeback_lora_key(key: str) -> bool:
    return is_krea_attention_out_lora_key(key) or is_krea_mlp_lora_key(key)


def filter_lora_state_dict(
    state_dict: Mapping[str, Any],
    *,
    attention_only_filter: bool = True,
    ignore_text_encoder_lora: bool = True,
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in state_dict.items():
        if ignore_text_encoder_lora and is_text_encoder_lora_key(key):
            continue
        if attention_only_filter and not is_krea_attention_lora_key(key):
            continue
        filtered[key] = value
    return filtered


def make_lora_branch_model(
    base_model: Any,
    lora_name: str,
    *,
    strength_model: float,
    attention_only_filter: bool = True,
    ignore_text_encoder_lora: bool = True,
) -> Any:
    try:
        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore
        import folder_paths  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only inside ComfyUI
        raise RuntimeError("ComfyUI LoRA loading APIs are not importable") from exc

    if strength_model == 0:
        return base_model
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    lora = filter_lora_state_dict(
        lora,
        attention_only_filter=attention_only_filter,
        ignore_text_encoder_lora=ignore_text_encoder_lora,
    )
    if not lora:
        raise RuntimeError(f"LoRA '{lora_name}' had no keys left after filtering")
    branch, _clip = comfy.sd.load_lora_for_models(
        base_model, None, lora, strength_model, 0.0, lora_metadata=metadata
    )
    return branch
