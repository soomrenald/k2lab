from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from k2_region_lab.model import SafetensorsSummary, read_safetensors_summary


STANDARD_LORA_ROUTING = "standard"
CHARACTER_IDENTITY_LORA_ROUTING = "character_identity"
LORA_ROUTING_MODES = {
    STANDARD_LORA_ROUTING,
    CHARACTER_IDENTITY_LORA_ROUTING,
}


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
    strength: float = 1.0
    routing_mode: str = STANDARD_LORA_ROUTING
    trigger_phrase: str = ""

    def __post_init__(self) -> None:
        if self.global_scope and self.region_ids:
            raise ValueError("a LoRA cannot be global and region-scoped at the same time")
        if not -4.0 <= self.strength <= 4.0:
            raise ValueError("LoRA strength must be between -4 and 4")
        if self.routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(f"unsupported LoRA routing mode: {self.routing_mode!r}")
        if self.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING and not self.trigger_phrase.strip():
            raise ValueError("character identity routing requires a trigger phrase")


class LoraLibrary:
    def __init__(self) -> None:
        self._entries: dict[str, LoraEntry] = {}
        self._path_instance_counts: dict[Path, int] = {}
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
        size_bytes = resolved.stat().st_size
        summary = read_safetensors_summary(resolved)

        instance_number = self._path_instance_counts.get(resolved, 0) + 1
        self._path_instance_counts[resolved] = instance_number
        display_name = resolved.stem
        if instance_number > 1:
            display_name = f"{display_name} #{instance_number}"

        lora_id = uuid4().hex
        entry = LoraEntry(
            lora_id=lora_id,
            path=resolved,
            display_name=display_name,
            size_bytes=size_bytes,
            summary=summary,
        )
        self._entries[lora_id] = entry
        self._bindings[lora_id] = LoraBinding(
            lora_id=lora_id,
            trigger_phrase=resolved.stem,
        )
        return entry

    def remove(self, lora_id: str) -> None:
        self._entries.pop(lora_id)
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
        binding = replace(self.binding_for(lora_id), global_scope=False, region_ids=unique_ids)
        self._bindings[lora_id] = binding
        return binding

    def set_strength(self, lora_id: str, strength: float) -> LoraBinding:
        binding = replace(self.binding_for(lora_id), strength=float(strength))
        self._bindings[lora_id] = binding
        return binding

    def set_routing_mode(self, lora_id: str, routing_mode: str) -> LoraBinding:
        binding = replace(self.binding_for(lora_id), routing_mode=str(routing_mode))
        self._bindings[lora_id] = binding
        return binding

    def set_trigger_phrase(self, lora_id: str, trigger_phrase: str) -> LoraBinding:
        phrase = str(trigger_phrase).strip()
        binding = replace(self.binding_for(lora_id), trigger_phrase=phrase)
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
                self._bindings[lora_id] = replace(binding, global_scope=True, region_ids=())

    def bindings(self) -> tuple[LoraBinding, ...]:
        return tuple(self._bindings.values())
