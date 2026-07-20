from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from k2_region_lab.lora.library import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    LORA_ROUTING_MODES,
    STANDARD_LORA_ROUTING,
)
from k2_region_lab.regional_prompting import (
    BoundRegionalPromptPlan,
    RegionalPromptPlan,
)
from k2_region_lab.regions import CanvasGeometry


BACKEND = "krea-regional-lora-delta-gating-v3"


@dataclass(frozen=True, slots=True)
class LoraDeltaRoute:
    lora_id: str
    display_name: str
    strength: float
    global_scope: bool
    region_ids: tuple[str, ...]
    region_names: tuple[str, ...]
    text_token_mask: tuple[float, ...]
    image_token_mask: tuple[float, ...]
    routing_mode: str = STANDARD_LORA_ROUTING
    trigger_phrase: str = ""
    backend: str = BACKEND

    def sequence_mask(self, sequence_length: int, *, text_fusion: bool) -> tuple[float, ...]:
        text_count = len(self.text_token_mask)
        image_count = len(self.image_token_mask)
        if text_fusion and sequence_length == text_count:
            return self.text_token_mask
        if not text_fusion and sequence_length == image_count:
            return self.image_token_mask
        if not text_fusion and sequence_length == text_count + image_count:
            return self.text_token_mask + self.image_token_mask
        raise ValueError(
            f"LoRA route {self.display_name!r} expected "
            f"{text_count} text, {image_count} image, or "
            f"{text_count + image_count} combined tokens, "
            f"received {sequence_length}"
        )

    def layerwise_text_batch_mask(self, batch_size: int) -> tuple[float, ...]:
        text_count = len(self.text_token_mask)
        if batch_size <= 0 or batch_size % text_count:
            raise ValueError(
                f"LoRA route {self.display_name!r} expected a folded text batch "
                f"divisible by {text_count}, received {batch_size}"
            )
        return self.text_token_mask * (batch_size // text_count)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "global": self.global_scope,
            "region_ids": list(self.region_ids),
            "region_names": list(self.region_names),
            "text_tokens_enabled": sum(value > 0.0 for value in self.text_token_mask),
            "image_tokens_enabled": sum(value > 0.0 for value in self.image_token_mask),
            "image_mask_coverage": (
                sum(self.image_token_mask) / len(self.image_token_mask)
                if self.image_token_mask
                else 0.0
            ),
            "routing_mode": self.routing_mode,
            "trigger_phrase": self.trigger_phrase,
        }


def route_allows_adapter_target(route: LoraDeltaRoute, target: str) -> bool:
    """Keep standard regional deltas on token-local transformer paths.

    Regional text-fusion adapters are protected by the text partition and spatial
    attention router. Main-stream key/value deltas are consumed by every image
    query, so standard regional routes omit those targets. Character-identity
    routing retains its established anchored behavior; global routes remain
    unrestricted.
    """

    if route.global_scope:
        return True
    lowered = str(target).casefold()
    # Krea stores this modulation matrix as a bare Parameter rather than a
    # forward module. It can be patched globally, but there is no token-local
    # forward invocation on which a regional mask could be applied.
    if lowered.endswith(".last.modulation.lin") or lowered.endswith(
        ".last.modulation.lin.weight"
    ):
        return False
    if route.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING:
        return True
    if ".txtfusion." in lowered or ".txtmlp." in lowered:
        return True
    parts = lowered.split(".")
    module_name = parts[-2] if parts and parts[-1] == "weight" else parts[-1]
    return module_name not in {"wk", "wv", "k_proj", "v_proj"}


def character_identity_triggers(
    specifications: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Collect saved identity triggers by regional target for prompt compilation."""
    collected: dict[str, list[str]] = {}
    for specification in specifications:
        routing_mode = str(
            specification.get("routing_mode", STANDARD_LORA_ROUTING)
        )
        if routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(f"unsupported LoRA routing mode: {routing_mode!r}")
        if routing_mode != CHARACTER_IDENTITY_LORA_ROUTING:
            continue
        trigger_phrase = str(specification.get("trigger_phrase", "")).strip()
        if not trigger_phrase:
            raise ValueError("character identity routing requires a trigger phrase")
        if bool(specification.get("global", True)):
            raise ValueError("character identity routing requires regional scope")
        for region_id in map(str, specification.get("region_ids", ())):
            triggers = collected.setdefault(region_id, [])
            if trigger_phrase not in triggers:
                triggers.append(trigger_phrase)
    return {region_id: tuple(triggers) for region_id, triggers in collected.items()}


def compile_lora_delta_routes(
    specifications: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    text_token_count: int,
    regional_plan: RegionalPromptPlan | None,
    bound_plan: BoundRegionalPromptPlan | None,
) -> tuple[LoraDeltaRoute, ...]:
    """Compile exact text/image token gates for every active LoRA.

    Prompt attention may use soft falloff fields, but LoRA image masks are always
    rasterized directly from the pixel boxes so their parameter deltas are zero
    outside the assigned regions.
    """

    if text_token_count <= 0:
        raise ValueError("LoRA routing requires a positive text token count")
    geometry = CanvasGeometry.resolve(width, height)
    all_text = (1.0,) * text_token_count
    all_image = (1.0,) * geometry.image_lane_count
    active_regions = {
        region.region_id: region for region in (regional_plan.regions if regional_plan else ())
    }
    token_spans = {span.region_id: span for span in (bound_plan.spans if bound_plan else ())}

    routes: list[LoraDeltaRoute] = []
    for specification in specifications:
        strength = float(specification.get("strength", 1.0))
        if strength == 0.0:
            continue
        if not -4.0 <= strength <= 4.0:
            raise ValueError("LoRA strength must be between -4 and 4")
        lora_id = str(specification.get("id", specification.get("name", "LoRA")))
        display_name = str(specification.get("name", lora_id))
        global_scope = bool(specification.get("global", True))
        routing_mode = str(
            specification.get("routing_mode", STANDARD_LORA_ROUTING)
        )
        if routing_mode not in LORA_ROUTING_MODES:
            raise ValueError(f"unsupported LoRA routing mode: {routing_mode!r}")
        trigger_phrase = str(specification.get("trigger_phrase", "")).strip()
        if routing_mode == CHARACTER_IDENTITY_LORA_ROUTING and not trigger_phrase:
            raise ValueError("character identity routing requires a trigger phrase")
        region_ids = tuple(dict.fromkeys(map(str, specification.get("region_ids", ()))))
        if global_scope:
            routes.append(
                LoraDeltaRoute(
                    lora_id=lora_id,
                    display_name=display_name,
                    strength=strength,
                    global_scope=True,
                    region_ids=(),
                    region_names=(),
                    text_token_mask=all_text,
                    image_token_mask=all_image,
                    routing_mode=STANDARD_LORA_ROUTING,
                    trigger_phrase=trigger_phrase,
                )
            )
            continue
        if not region_ids:
            raise ValueError(f"regional LoRA {display_name!r} has no assigned regions")
        missing = [
            region_id
            for region_id in region_ids
            if region_id not in active_regions or region_id not in token_spans
        ]
        if missing:
            raise ValueError(
                f"regional LoRA {display_name!r} targets regions without active prompts: "
                + ", ".join(missing)
            )

        text_mask = [0.0] * text_token_count
        image_mask = [0.0] * geometry.image_lane_count
        names = []
        for region_id in region_ids:
            region = active_regions[region_id]
            span = token_spans[region_id]
            names.append(region.name)
            # Both regional modes gate text-fusion deltas to this clause. The
            # attention router prevents region-owned text from becoming a shared
            # scene-conditioning path; character mode additionally inserts its
            # explicit identity trigger anchor.
            for index in range(span.start, span.end):
                text_mask[index] = 1.0
            strict_box_mask = geometry.rasterize_box(region.box)
            image_mask = [
                max(current, candidate)
                for current, candidate in zip(image_mask, strict_box_mask, strict=True)
            ]
        routes.append(
            LoraDeltaRoute(
                lora_id=lora_id,
                display_name=display_name,
                strength=strength,
                global_scope=False,
                region_ids=region_ids,
                region_names=tuple(names),
                text_token_mask=tuple(text_mask),
                image_token_mask=tuple(image_mask),
                routing_mode=routing_mode,
                trigger_phrase=trigger_phrase,
            )
        )
    return tuple(routes)
