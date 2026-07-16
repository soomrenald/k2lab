from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from k2_region_lab.model import SafetensorsSummary, read_safetensors_summary


@dataclass(frozen=True, slots=True)
class LoraEntry:
    lora_id: str
    path: Path
    display_name: str
    size_bytes: int
    summary: SafetensorsSummary


@dataclass(frozen=True, slots=True)
class LoraBinding:
    """One LoRA's mutually exclusive global or multi-region scope."""

    lora_id: str
    global_scope: bool = True
    region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.global_scope and self.region_ids:
            raise ValueError("a LoRA cannot be global and region-scoped at the same time")


class LoraLibrary:
    def __init__(self) -> None:
        self._entries: dict[str, LoraEntry] = {}
        self._path_index: dict[Path, str] = {}
        self._bindings: dict[str, LoraBinding] = {}

    def entries(self) -> tuple[LoraEntry, ...]:
        return tuple(self._entries.values())

    def get(self, lora_id: str) -> LoraEntry:
        return self._entries[lora_id]

    def add(self, path: Path) -> LoraEntry:
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() != ".safetensors":
            raise ValueError("LoRA files must use the .safetensors format")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        existing_id = self._path_index.get(resolved)
        if existing_id is not None:
            return self._entries[existing_id]

        lora_id = uuid5(NAMESPACE_URL, resolved.as_uri()).hex
        entry = LoraEntry(
            lora_id=lora_id,
            path=resolved,
            display_name=resolved.stem,
            size_bytes=resolved.stat().st_size,
            summary=read_safetensors_summary(resolved),
        )
        self._entries[lora_id] = entry
        self._path_index[resolved] = lora_id
        self._bindings[lora_id] = LoraBinding(lora_id=lora_id)
        return entry

    def remove(self, lora_id: str) -> None:
        entry = self._entries.pop(lora_id)
        self._path_index.pop(entry.path, None)
        self._bindings.pop(lora_id, None)

    def binding_for(self, lora_id: str) -> LoraBinding:
        return self._bindings[lora_id]

    def assign_global(self, lora_id: str) -> LoraBinding:
        binding = replace(self.binding_for(lora_id), global_scope=True, region_ids=())
        self._bindings[lora_id] = binding
        return binding

    def assign_regions(self, lora_id: str, region_ids: tuple[str, ...]) -> LoraBinding:
        unique_ids = tuple(dict.fromkeys(region_ids))
        if not unique_ids:
            return self.assign_global(lora_id)
        binding = replace(
            self.binding_for(lora_id), global_scope=False, region_ids=unique_ids
        )
        self._bindings[lora_id] = binding
        return binding

    def drop_region(self, region_id: str) -> None:
        for lora_id, binding in tuple(self._bindings.items()):
            if region_id not in binding.region_ids:
                continue
            remaining = tuple(item for item in binding.region_ids if item != region_id)
            if remaining:
                self._bindings[lora_id] = replace(binding, region_ids=remaining)
            else:
                self._bindings[lora_id] = replace(
                    binding, global_scope=True, region_ids=()
                )

    def bindings(self) -> tuple[LoraBinding, ...]:
        return tuple(self._bindings.values())
