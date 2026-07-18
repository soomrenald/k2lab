from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TypeVar

from k2_region_lab.model import read_safetensors_header


_T = TypeVar("_T")
_LORA_PAIR_SUFFIXES = (
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.weight", ".lora_up.weight"),
)
_LOKR_COMPONENT_SUFFIXES = (
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
)
_ADAPTER_SUFFIXES = tuple(
    suffix for pair in _LORA_PAIR_SUFFIXES for suffix in pair
) + _LOKR_COMPONENT_SUFFIXES
_AUXILIARY_SUFFIXES = (".alpha", ".dora_scale")
_KREA_INTERNAL_PREFIXES = ("blocks.", "txtfusion.")


def normalize_krea_lora_key(key: str) -> str:
    """Normalize AI Toolkit Krea keys to ComfyUI's generic internal namespace."""
    prefix = "diffusion_model."
    if key.startswith(prefix) and key[len(prefix) :].startswith(_KREA_INTERNAL_PREFIXES):
        return key[len(prefix) :]
    return key


def normalize_krea_lora_state_dict(state: dict[str, _T]) -> dict[str, _T]:
    normalized: dict[str, _T] = {}
    for key, value in state.items():
        target = normalize_krea_lora_key(key)
        if target in normalized:
            raise ValueError(f"LoRA key normalization collision: {target}")
        normalized[target] = value
    return normalized


def _adapter_base(key: str) -> str | None:
    for suffix in (*_ADAPTER_SUFFIXES, *_AUXILIARY_SUFFIXES):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def align_krea_lora_state_dict(
    state: dict[str, _T], supported_prefixes: Iterable[str]
) -> dict[str, _T]:
    """Choose the original or normalized Krea namespace supported by this worker."""
    supported = set(supported_prefixes)
    aligned: dict[str, _T] = {}
    for key, value in state.items():
        original_base = _adapter_base(key)
        normalized_key = normalize_krea_lora_key(key)
        normalized_base = _adapter_base(normalized_key)
        if original_base in supported:
            target = key
        elif normalized_base in supported:
            target = normalized_key
        else:
            target = key
        if target in aligned:
            raise ValueError(f"LoRA key alignment collision: {target}")
        aligned[target] = value
    return aligned


def adapter_prefixes(keys: Iterable[str]) -> tuple[str, ...]:
    prefixes = set()
    for key in keys:
        for suffix in _ADAPTER_SUFFIXES:
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return tuple(sorted(prefixes))


def _has_complete_lora_pair(prefix: str, tensors: dict[str, Any]) -> bool:
    return any(
        f"{prefix}{down}" in tensors and f"{prefix}{up}" in tensors
        for down, up in _LORA_PAIR_SUFFIXES
    )


def _has_lokr_component(
    prefix: str,
    tensors: dict[str, Any],
    component: str,
) -> bool:
    whole = f"{prefix}.lokr_{component}"
    return whole in tensors or (
        f"{whole}_a" in tensors and f"{whole}_b" in tensors
    )


def _has_complete_lokr(prefix: str, tensors: dict[str, Any]) -> bool:
    return _has_lokr_component(prefix, tensors, "w1") and _has_lokr_component(
        prefix, tensors, "w2"
    )


def _adapter_type(prefix: str, tensors: dict[str, Any]) -> str:
    if any(f"{prefix}{suffix}" in tensors for suffix in _LOKR_COMPONENT_SUFFIXES):
        return "lokr"
    return "lora"


def inspect_lora_header(path: Path) -> dict[str, Any]:
    header = read_safetensors_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        metadata = {}
    tensors = {
        key: descriptor
        for key, descriptor in header.items()
        if key != "__metadata__" and isinstance(descriptor, dict)
    }
    prefixes = adapter_prefixes(tensors)
    ranks = Counter()
    complete_pairs = 0
    adapter_types = Counter()
    for prefix in prefixes:
        adapter_type = _adapter_type(prefix, tensors)
        adapter_types[adapter_type] += 1
        complete = (
            _has_complete_lokr(prefix, tensors)
            if adapter_type == "lokr"
            else _has_complete_lora_pair(prefix, tensors)
        )
        if complete:
            complete_pairs += 1
        if adapter_type == "lora":
            rank_tensor = next(
                (
                    tensors[f"{prefix}{down}"]
                    for down, _up in _LORA_PAIR_SUFFIXES
                    if f"{prefix}{down}" in tensors
                ),
                None,
            )
            shape = rank_tensor.get("shape", []) if rank_tensor is not None else []
            if shape:
                ranks[int(shape[0])] += 1
    namespaces = Counter(
        ".".join(normalize_krea_lora_key(prefix).split(".")[:1])
        for prefix in prefixes
    )
    return {
        "path": str(path.expanduser().resolve()),
        "tensor_count": len(tensors),
        "adapter_count": len(prefixes),
        "complete_adapter_pairs": complete_pairs,
        "adapter_types": dict(sorted(adapter_types.items())),
        "ranks": dict(sorted(ranks.items())),
        "namespaces": dict(sorted(namespaces.items())),
        "base_model": metadata.get("ss_base_model_version"),
        "name": metadata.get("name") or metadata.get("ss_output_name") or path.stem,
        "format": metadata.get("format"),
        "training_info": metadata.get("training_info"),
        "software": metadata.get("software"),
        "metadata_keys": sorted(metadata),
    }
