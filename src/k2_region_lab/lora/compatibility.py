from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TypeVar

from k2_region_lab.model import read_safetensors_header


_T = TypeVar("_T")
_ADAPTER_SUFFIXES = (
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
)
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


def adapter_prefixes(keys: Iterable[str]) -> tuple[str, ...]:
    prefixes = set()
    for key in keys:
        for suffix in _ADAPTER_SUFFIXES:
            if key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
                break
    return tuple(sorted(prefixes))


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
    for prefix in prefixes:
        a = tensors.get(f"{prefix}.lora_A.weight")
        b = tensors.get(f"{prefix}.lora_B.weight")
        if a is not None and b is not None:
            complete_pairs += 1
            shape = a.get("shape", [])
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
        "ranks": dict(sorted(ranks.items())),
        "namespaces": dict(sorted(namespaces.items())),
        "base_model": metadata.get("ss_base_model_version"),
        "name": metadata.get("name") or metadata.get("ss_output_name") or path.stem,
        "format": metadata.get("format"),
        "training_info": metadata.get("training_info"),
        "software": metadata.get("software"),
        "metadata_keys": sorted(metadata),
    }
