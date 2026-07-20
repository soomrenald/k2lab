from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from numbers import Integral
from typing import Callable

from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition


BACKEND = "krea-unified-spatial-attention-v6"
GLOBAL_EMPHASIS_SCOPE = "__global__"


@dataclass(frozen=True, slots=True)
class PromptEmphasis:
    """A user-selected phrase to reinforce in global or regional conditioning."""

    scope_id: str
    phrase: str
    strength: float = 0.5
    occurrence: int = 0

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise ValueError("prompt emphasis scope must not be empty")
        if not self.phrase.strip():
            raise ValueError("prompt emphasis phrase must not be empty")
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError("prompt emphasis strength must be between zero and two")
        if self.occurrence < 0:
            raise ValueError("prompt emphasis occurrence must not be negative")


@dataclass(frozen=True, slots=True)
class ResolvedPromptEmphasis:
    scope_id: str
    phrase: str
    strength: float
    character_span: tuple[int, int]
    image_token_field: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ResolvedCharacterIdentity:
    region_id: str
    trigger_phrase: str
    character_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ResolvedFaceIdentityPrompt:
    region_id: str
    prompt: str
    character_span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TextTokenEmphasis:
    scope_id: str
    phrase: str
    strength: float
    start: int
    end: int
    image_token_field: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CharacterIdentityTokenSpans:
    region_id: str
    trigger_phrase: str
    token_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FaceIdentityTokenSpan:
    region_id: str
    prompt: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class UnifiedPromptRegion:
    region_id: str
    name: str
    prompt: str
    face_identity_prompt: str
    negative_prompt: str
    box: PixelBox
    clause: str
    character_span: tuple[int, int]
    image_token_field: tuple[float, ...]
    image_token_mask: tuple[float, ...]
    spatial_role: str


@dataclass(frozen=True, slots=True)
class RegionalPromptPlan:
    width: int
    height: int
    image_token_width: int
    image_token_height: int
    prompt: str
    strength: float
    outside_penalty: float
    falloff_pixels: float
    subject_competition: bool
    subject_fill: bool
    late_step_scale: float
    regions: tuple[UnifiedPromptRegion, ...]
    emphases: tuple[ResolvedPromptEmphasis, ...] = ()
    character_identities: tuple[ResolvedCharacterIdentity, ...] = ()
    face_identities: tuple[ResolvedFaceIdentityPrompt, ...] = ()
    backend: str = BACKEND

    @property
    def image_token_count(self) -> int:
        return self.image_token_width * self.image_token_height

    def bind_tokens(
        self,
        prompt_prefix_token_count: Callable[[str], int],
        *,
        conditioning_text_token_count: int | None = None,
    ) -> "BoundRegionalPromptPlan":
        spans = tuple(
            RegionalTokenSpan(
                region_id=region.region_id,
                name=region.name,
                start=prompt_prefix_token_count(
                    self.prompt[: region.character_span[0]]
                ),
                end=prompt_prefix_token_count(
                    self.prompt[: region.character_span[1]]
                ),
                image_token_field=region.image_token_field,
                image_token_mask=region.image_token_mask,
                spatial_role=region.spatial_role,
            )
            for region in self.regions
        )
        if any(span.end <= span.start for span in spans):
            raise ValueError("each regional prompt must own at least one text token")
        text_token_count = (
            prompt_prefix_token_count(self.prompt)
            if conditioning_text_token_count is None
            else conditioning_text_token_count
        )
        if spans and max(span.end for span in spans) > text_token_count:
            raise ValueError("regional text span exceeds the conditioning sequence")
        emphases = tuple(
            TextTokenEmphasis(
                scope_id=emphasis.scope_id,
                phrase=emphasis.phrase,
                strength=emphasis.strength,
                start=prompt_prefix_token_count(
                    self.prompt[: emphasis.character_span[0]].rstrip()
                ),
                end=prompt_prefix_token_count(
                    self.prompt[: emphasis.character_span[1]]
                ),
                image_token_field=emphasis.image_token_field,
            )
            for emphasis in self.emphases
        )
        invalid_emphasis = next(
            (emphasis for emphasis in emphases if emphasis.end <= emphasis.start),
            None,
        )
        if invalid_emphasis is not None:
            raise ValueError(
                f"emphasized phrase {invalid_emphasis.phrase!r} does not align "
                "with a complete text token"
            )
        if emphases and max(emphasis.end for emphasis in emphases) > text_token_count:
            raise ValueError("emphasized text span exceeds the conditioning sequence")
        character_identities = tuple(
            CharacterIdentityTokenSpans(
                region_id=identity.region_id,
                trigger_phrase=identity.trigger_phrase,
                token_spans=tuple(
                    (
                        prompt_prefix_token_count(self.prompt[:start]),
                        prompt_prefix_token_count(self.prompt[:end]),
                    )
                    for start, end in identity.character_spans
                ),
            )
            for identity in self.character_identities
        )
        if any(
            end <= start
            for identity in character_identities
            for start, end in identity.token_spans
        ):
            raise ValueError("each character identity trigger must own at least one text token")
        if (
            character_identities
            and max(
                end
                for identity in character_identities
                for _start, end in identity.token_spans
            )
            > text_token_count
        ):
            raise ValueError("character identity trigger exceeds the conditioning sequence")
        face_identities = tuple(
            FaceIdentityTokenSpan(
                region_id=identity.region_id,
                prompt=identity.prompt,
                start=prompt_prefix_token_count(
                    self.prompt[: identity.character_span[0]]
                ),
                end=prompt_prefix_token_count(
                    self.prompt[: identity.character_span[1]]
                ),
            )
            for identity in self.face_identities
        )
        if any(identity.end <= identity.start for identity in face_identities):
            raise ValueError("each face identity prompt must own at least one text token")
        if face_identities and max(identity.end for identity in face_identities) > text_token_count:
            raise ValueError("face identity prompt exceeds the conditioning sequence")
        return BoundRegionalPromptPlan(
            prompt=self.prompt,
            text_token_count=text_token_count,
            image_token_count=self.image_token_count,
            strength=self.strength,
            outside_penalty=self.outside_penalty,
            falloff_pixels=self.falloff_pixels,
            late_step_scale=self.late_step_scale,
            spans=spans,
            emphases=emphases,
            character_identities=character_identities,
            face_identities=face_identities,
            backend=self.backend,
        )

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "compiled_prompt": self.prompt,
            "strength": self.strength,
            "outside_penalty": self.outside_penalty,
            "falloff_pixels": self.falloff_pixels,
            "subject_competition": self.subject_competition,
            "subject_fill": self.subject_fill,
            "late_step_scale": self.late_step_scale,
            "image_token_grid": [self.image_token_width, self.image_token_height],
            "region_count": len(self.regions),
            "emphases": [
                {
                    "scope_id": emphasis.scope_id,
                    "phrase": emphasis.phrase,
                    "strength": emphasis.strength,
                    "character_span": list(emphasis.character_span),
                }
                for emphasis in self.emphases
            ],
            "character_identities": [
                {
                    "region_id": identity.region_id,
                    "trigger_phrase": identity.trigger_phrase,
                    "character_spans": [list(span) for span in identity.character_spans],
                }
                for identity in self.character_identities
            ],
            "face_identities": [
                {
                    "region_id": identity.region_id,
                    "prompt": identity.prompt,
                    "character_span": list(identity.character_span),
                }
                for identity in self.face_identities
            ],
            "regions": [
                {
                    "id": region.region_id,
                    "name": region.name,
                    "box_pixels": [
                        region.box.x0,
                        region.box.y0,
                        region.box.x1,
                        region.box.y1,
                    ],
                    "character_span": list(region.character_span),
                    "spatial_role": region.spatial_role,
                    "peak_spatial_weight": max(region.image_token_field),
                }
                for region in self.regions
            ],
        }


@dataclass(frozen=True, slots=True)
class RegionalTokenSpan:
    region_id: str
    name: str
    start: int
    end: int
    image_token_field: tuple[float, ...]
    image_token_mask: tuple[float, ...]
    spatial_role: str


@dataclass(frozen=True, slots=True)
class BoundRegionalPromptPlan:
    prompt: str
    text_token_count: int
    image_token_count: int
    strength: float
    outside_penalty: float
    falloff_pixels: float
    late_step_scale: float
    spans: tuple[RegionalTokenSpan, ...]
    emphases: tuple[TextTokenEmphasis, ...] = ()
    character_identities: tuple[CharacterIdentityTokenSpans, ...] = ()
    face_identities: tuple[FaceIdentityTokenSpan, ...] = ()
    backend: str = BACKEND

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "compiled_prompt": self.prompt,
            "strength": self.strength,
            "outside_penalty": self.outside_penalty,
            "falloff_pixels": self.falloff_pixels,
            "late_step_scale": self.late_step_scale,
            "text_token_count": self.text_token_count,
            "image_token_count": self.image_token_count,
            "region_count": len(self.spans),
            "emphases": [
                {
                    "scope_id": emphasis.scope_id,
                    "phrase": emphasis.phrase,
                    "strength": emphasis.strength,
                    "text_token_span": [emphasis.start, emphasis.end],
                }
                for emphasis in self.emphases
            ],
            "character_identities": [
                {
                    "region_id": identity.region_id,
                    "trigger_phrase": identity.trigger_phrase,
                    "text_token_spans": [list(span) for span in identity.token_spans],
                }
                for identity in self.character_identities
            ],
            "face_identities": [
                {
                    "region_id": identity.region_id,
                    "prompt": identity.prompt,
                    "text_token_span": [identity.start, identity.end],
                }
                for identity in self.face_identities
            ],
            "regions": [
                {
                    "id": span.region_id,
                    "name": span.name,
                    "text_token_span": [span.start, span.end],
                    "spatial_role": span.spatial_role,
                }
                for span in self.spans
            ],
        }


def compile_regional_prompt_plan(
    width: int,
    height: int,
    global_prompt: str,
    regions: tuple[RegionDefinition, ...],
    *,
    strength: float = 1.0,
    outside_penalty: float = 1.0,
    falloff_pixels: float = 128.0,
    subject_competition: bool = True,
    subject_fill: bool = True,
    late_step_scale: float = 0.35,
    emphases: tuple[PromptEmphasis, ...] = (),
    character_identity_triggers: dict[str, tuple[str, ...]] | None = None,
) -> RegionalPromptPlan:
    if not 0.0 < strength <= 10.0:
        raise ValueError("spatial guidance strength must be in (0, 10]")
    if not 0.0 <= falloff_pixels <= 2048.0:
        raise ValueError("spatial falloff must be between 0 and 2048 pixels")
    if not 0.0 <= outside_penalty <= 10.0:
        raise ValueError("spatial outside penalty must be between 0 and 10")
    if not 0.0 <= late_step_scale <= 1.0:
        raise ValueError("late-step spatial scale must be between 0 and 1")

    geometry = CanvasGeometry.resolve(width, height)
    active = []
    for region in regions:
        if not region.enabled or not _regional_description(region):
            continue
        active.append((region, region.box.clipped(width, height)))
    # The project/list order is front-to-back. Priority keeps that ordering intact
    # when definitions arrive through the worker payload.
    active.sort(key=lambda item: -item[0].priority)

    roles = tuple(_effective_spatial_role(region, box, width) for region, box in active)
    raw_fields = tuple(
        _subject_target_field(
            geometry,
            box,
            float(falloff_pixels),
            edge_weight=0.85 if subject_fill else 0.5,
        )
        if role in {"subject", "edit"}
        else _soft_box_field(geometry, box, float(falloff_pixels))
        for (_, box), role in zip(active, roles, strict=True)
    )
    fields = (
        _apply_subject_competition(raw_fields, roles)
        if subject_competition
        else raw_fields
    )

    prompt = _sentence(global_prompt.strip())
    compiled: list[UnifiedPromptRegion] = []
    resolved_identities: list[ResolvedCharacterIdentity] = []
    resolved_face_identities: list[ResolvedFaceIdentityPrompt] = []
    identity_triggers = character_identity_triggers or {}
    for (region, box), role, image_token_field in zip(
        active, roles, fields, strict=True
    ):
        clause = _regional_clause(
            region,
            box,
            width,
            height,
            role=role,
            subject_fill=subject_fill,
        )
        if prompt:
            prompt += "\n"
        start = len(prompt)
        identity_description = _clean_description(region.face_identity_prompt)
        if identity_description:
            identity_start = clause.find(identity_description)
            if identity_start < 0:
                raise ValueError(
                    f"could not locate face identity prompt for region {region.name!r}"
                )
            resolved_face_identities.append(
                ResolvedFaceIdentityPrompt(
                    region_id=region.region_id,
                    prompt=region.face_identity_prompt.strip(),
                    character_span=(
                        start + identity_start,
                        start + identity_start + len(identity_description),
                    ),
                )
            )
        for trigger_phrase in dict.fromkeys(identity_triggers.get(region.region_id, ())):
            trigger_phrase = trigger_phrase.strip()
            if not trigger_phrase:
                raise ValueError("character identity trigger must not be empty")
            instruction = _character_identity_instruction(trigger_phrase)
            instruction_start = len(clause) + 1
            clause += f" {instruction}"
            resolved_identities.append(
                ResolvedCharacterIdentity(
                    region_id=region.region_id,
                    trigger_phrase=trigger_phrase,
                    character_spans=tuple(
                        (
                            start + instruction_start + offset,
                            start + instruction_start + offset + len(trigger_phrase),
                        )
                        for offset in _all_occurrences(instruction, trigger_phrase)
                    ),
                )
            )
        prompt += clause
        end = len(prompt)
        compiled.append(
            UnifiedPromptRegion(
                region_id=region.region_id,
                name=region.name,
                prompt=region.prompt.strip(),
                face_identity_prompt=region.face_identity_prompt.strip(),
                negative_prompt=region.negative_prompt.strip(),
                box=box,
                clause=clause,
                character_span=(start, end),
                image_token_field=image_token_field,
                image_token_mask=geometry.rasterize_box(box),
                spatial_role=role,
            )
        )

    relationship_clause = _relationship_clause(compiled, width, height)
    if relationship_clause:
        prompt += f"\n{relationship_clause}"
    resolved_emphases = _resolve_prompt_emphases(
        prompt,
        compiled,
        tuple(emphases),
        image_token_count=geometry.image_lane_count,
    )

    return RegionalPromptPlan(
        width=geometry.aligned_width,
        height=geometry.aligned_height,
        image_token_width=geometry.patch_width,
        image_token_height=geometry.patch_height,
        prompt=prompt,
        strength=float(strength),
        outside_penalty=float(outside_penalty),
        falloff_pixels=float(falloff_pixels),
        subject_competition=bool(subject_competition),
        subject_fill=bool(subject_fill),
        late_step_scale=float(late_step_scale),
        regions=tuple(compiled),
        emphases=resolved_emphases,
        character_identities=tuple(resolved_identities),
        face_identities=tuple(resolved_face_identities),
    )


def prompt_emphases_from_payload(
    payload: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> tuple[PromptEmphasis, ...]:
    return tuple(
        PromptEmphasis(
            scope_id=str(item.get("scope_id", GLOBAL_EMPHASIS_SCOPE)),
            phrase=str(item.get("phrase", "")),
            strength=float(item.get("strength", 0.5)),
            occurrence=int(item.get("occurrence", 0)),
        )
        for item in payload
    )


def _resolve_prompt_emphases(
    prompt: str,
    regions: list[UnifiedPromptRegion],
    emphases: tuple[PromptEmphasis, ...],
    *,
    image_token_count: int,
) -> tuple[ResolvedPromptEmphasis, ...]:
    if not emphases:
        return ()
    by_id = {region.region_id: region for region in regions}
    global_end = regions[0].character_span[0] if regions else len(prompt)
    resolved: list[ResolvedPromptEmphasis] = []
    for emphasis in emphases:
        if emphasis.scope_id == GLOBAL_EMPHASIS_SCOPE:
            start = _nth_occurrence(
                prompt[:global_end], emphasis.phrase, emphasis.occurrence
            )
            field = (1.0,) * image_token_count
        else:
            region = by_id.get(emphasis.scope_id)
            if region is None:
                raise ValueError(
                    "prompt emphasis references a region without an active prompt: "
                    f"{emphasis.scope_id}"
                )
            source_offset = _nth_occurrence(
                region.prompt, emphasis.phrase, emphasis.occurrence
            )
            description = region.prompt.strip().rstrip(".!? ")
            description_offset = region.clause.find(description)
            if description_offset < 0:
                raise ValueError(
                    f"could not locate emphasized phrase {emphasis.phrase!r} "
                    "in its compiled regional clause"
                )
            leading_whitespace = len(region.prompt) - len(region.prompt.lstrip())
            start = (
                region.character_span[0]
                + description_offset
                + source_offset
                - leading_whitespace
            )
            field = region.image_token_field
        resolved.append(
            ResolvedPromptEmphasis(
                scope_id=emphasis.scope_id,
                phrase=emphasis.phrase,
                strength=emphasis.strength,
                character_span=(start, start + len(emphasis.phrase)),
                image_token_field=field,
            )
        )
    return tuple(resolved)


def _nth_occurrence(text: str, phrase: str, occurrence: int) -> int:
    start = 0
    for _ in range(occurrence + 1):
        start = text.find(phrase, start)
        if start < 0:
            raise ValueError(
                f"emphasized phrase {phrase!r} no longer occurs in its prompt scope"
            )
        if _ < occurrence:
            start += len(phrase)
    return start


def _all_occurrences(text: str, phrase: str) -> tuple[int, ...]:
    offsets = []
    start = 0
    while (offset := text.find(phrase, start)) >= 0:
        offsets.append(offset)
        start = offset + len(phrase)
    return tuple(offsets)


def _character_identity_instruction(trigger_phrase: str) -> str:
    return (
        f"{trigger_phrase} identifies the person in this region. Generate this "
        f"person's face and facial identity from {trigger_phrase}, preserving one "
        "coherent person."
    )


def character_identity_prompt(prompt: str, triggers: tuple[str, ...]) -> str:
    """Append the same explicit identity anchors used by unified regional prompts."""
    result = _sentence(prompt.strip())
    for trigger_phrase in dict.fromkeys(triggers):
        trigger_phrase = trigger_phrase.strip()
        if not trigger_phrase:
            raise ValueError("character identity trigger must not be empty")
        instruction = _character_identity_instruction(trigger_phrase)
        result = f"{result} {instruction}" if result else instruction
    return result


def _sentence(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _regional_clause(
    region: RegionDefinition,
    box: PixelBox,
    width: int,
    height: int,
    *,
    role: str,
    subject_fill: bool,
) -> str:
    center_x = 100.0 * (box.x0 + box.x1) / (2.0 * width)
    center_y = 100.0 * (box.y0 + box.y1) / (2.0 * height)
    height_percent = 100.0 * box.height / height
    horizontal = _horizontal_position(center_x)
    vertical = _vertical_position(center_y)
    description = _regional_description(region)

    if role == "background":
        location = (
            f"Across the {vertical} of the image, occupying about "
            f"{height_percent:.0f}% of its height"
        )
        return f"{location}, there is {description}."

    if role == "edit":
        location = f"Inside the {vertical} {horizontal} edit area"
        return (
            f"{location}, the desired final appearance is {description}. Integrate the "
            "change naturally with the surrounding image without drawing a box, border, "
            "seam, label, or annotation."
        )

    location = f"In the {vertical} {horizontal}"
    framing = _subject_framing(height_percent)
    if not subject_fill:
        return f"{location}, render {description} as {framing}."

    return (
        f"{location}, render {description} as {framing}. The visible subject itself "
        "should fill most of its assigned image area with minimal empty margin. Keep "
        "the complete visible subject inside that area without drawing guides, borders, "
        "coordinates, labels, text, or annotations."
    )


def _clean_description(prompt: str) -> str:
    return prompt.strip().rstrip(".!? ")


def _regional_description(region: RegionDefinition) -> str:
    identity = _clean_description(region.face_identity_prompt)
    scene = _clean_description(region.prompt)
    return ". ".join(part for part in (identity, scene) if part)


def _subject_framing(height_percent: float) -> str:
    if height_percent >= 70.0:
        return "a large prominent near-frame-height foreground subject"
    if height_percent >= 45.0:
        return "a prominent medium-to-large subject"
    if height_percent >= 25.0:
        return "a medium-size subject"
    return "a small distant subject"


def _horizontal_position(percent: float) -> str:
    if percent < 20.0:
        return "far-left side"
    if percent < 40.0:
        return "left side"
    if percent < 60.0:
        return "center"
    if percent < 80.0:
        return "right side"
    return "far-right side"


def _vertical_position(percent: float) -> str:
    if percent < 20.0:
        return "top"
    if percent < 40.0:
        return "upper portion"
    if percent < 60.0:
        return "middle portion"
    if percent < 80.0:
        return "lower portion"
    return "bottom"


def _relationship_clause(
    regions: list[UnifiedPromptRegion], width: int, height: int
) -> str:
    subjects = [
        region
        for region in regions
        if region.spatial_role == "subject"
    ]
    if len(subjects) < 2:
        return ""
    left_to_right = sorted(
        subjects, key=lambda region: (region.box.x0 + region.box.x1) / 2.0
    )
    names = [region.name for region in left_to_right]
    if len(names) == 2:
        ordering = f"{names[0]} is to the left of {names[1]}"
    else:
        ordering = (
            "From left to right, the subjects are "
            + ", ".join(names[:-1])
            + f", and {names[-1]}"
        )

    lowest = max(
        subjects, key=lambda region: (region.box.y0 + region.box.y1) / 2.0
    )
    other_centers = [
        (region.box.y0 + region.box.y1) / 2.0
        for region in subjects
        if region.region_id != lowest.region_id
    ]
    lowest_center = (lowest.box.y0 + lowest.box.y1) / 2.0
    if other_centers and lowest_center - sum(other_centers) / len(other_centers) > 0.08 * height:
        ordering += f"; {lowest.name} is positioned below the other subjects"
    equally_scaled = []
    for index, first in enumerate(left_to_right):
        for second in left_to_right[index + 1 :]:
            height_ratio = first.box.height / second.box.height
            center_difference = abs(
                (first.box.y0 + first.box.y1)
                - (second.box.y0 + second.box.y1)
            ) / (2.0 * height)
            if 0.85 <= height_ratio <= 1.15 and center_difference <= 0.10:
                equally_scaled.append(
                    f"{first.name} and {second.name} are equally large, at the same "
                    "camera distance, with matching top and bottom levels"
                )
    if equally_scaled:
        ordering += "; " + "; ".join(equally_scaled)
    depth_relationships = []
    for index, front in enumerate(subjects):
        for behind in subjects[index + 1 :]:
            overlap_width = max(
                0.0, min(front.box.x1, behind.box.x1) - max(front.box.x0, behind.box.x0)
            )
            overlap_height = max(
                0.0, min(front.box.y1, behind.box.y1) - max(front.box.y0, behind.box.y0)
            )
            if overlap_width > 0.0 and overlap_height > 0.0:
                depth_relationships.append(
                    f"{front.name} appears in front of {behind.name} where their "
                    "target boxes overlap; both occupy the shared image area as "
                    f"distinct subjects, with {behind.name} naturally and partially "
                    f"occluded behind {front.name}"
                )
    relationships = [ordering, *depth_relationships]
    return ". ".join(relationships) + "."


def _soft_box_field(
    geometry: CanvasGeometry, box: PixelBox, falloff_pixels: float
) -> tuple[float, ...]:
    values: list[float] = []
    size = geometry.output_pixels_per_image_token
    for row in range(geometry.patch_height):
        center_y = (row + 0.5) * size
        for column in range(geometry.patch_width):
            center_x = (column + 0.5) * size
            dx = max(box.x0 - center_x, 0.0, center_x - box.x1)
            dy = max(box.y0 - center_y, 0.0, center_y - box.y1)
            distance = hypot(dx, dy)
            if distance == 0.0:
                value = 1.0
            elif falloff_pixels == 0.0 or distance >= falloff_pixels:
                value = 0.0
            else:
                u = 1.0 - distance / falloff_pixels
                value = u * u * (3.0 - 2.0 * u)
            values.append(value)
    return tuple(values)


def _effective_spatial_role(
    region: RegionDefinition, box: PixelBox, canvas_width: int
) -> str:
    if region.spatial_role != "auto":
        return region.spatial_role
    return "background" if box.width >= 0.70 * canvas_width else "subject"


def _subject_target_field(
    geometry: CanvasGeometry,
    box: PixelBox,
    falloff_pixels: float,
    *,
    edge_weight: float,
) -> tuple[float, ...]:
    """Create a box target with a center peak and configurable boundary strength."""
    values: list[float] = []
    size = geometry.output_pixels_per_image_token
    midpoint_x = (box.x0 + box.x1) / 2.0
    midpoint_y = (box.y0 + box.y1) / 2.0
    half_width = box.width / 2.0
    half_height = box.height / 2.0
    for row in range(geometry.patch_height):
        center_y = (row + 0.5) * size
        for column in range(geometry.patch_width):
            center_x = (column + 0.5) * size
            dx = max(box.x0 - center_x, 0.0, center_x - box.x1)
            dy = max(box.y0 - center_y, 0.0, center_y - box.y1)
            distance = hypot(dx, dy)
            if distance == 0.0:
                normalized = max(
                    abs(center_x - midpoint_x) / half_width,
                    abs(center_y - midpoint_y) / half_height,
                )
                u = min(1.0, normalized)
                smooth = u * u * (3.0 - 2.0 * u)
                value = 1.0 - (1.0 - edge_weight) * smooth
            elif falloff_pixels == 0.0 or distance >= falloff_pixels:
                value = 0.0
            else:
                u = 1.0 - distance / falloff_pixels
                smooth = u * u * (3.0 - 2.0 * u)
                value = edge_weight * smooth
            values.append(value)
    return tuple(values)


def _apply_subject_competition(
    fields: tuple[tuple[float, ...], ...], roles: tuple[str, ...]
) -> tuple[tuple[float, ...], ...]:
    """Give overlapping subject targets exclusive soft ownership per image token."""
    subject_indices = [index for index, role in enumerate(roles) if role == "subject"]
    if len(subject_indices) < 2:
        return fields
    competed = [list(field) for field in fields]
    for token_index in range(len(fields[0])):
        squared = {
            index: fields[index][token_index] ** 2 for index in subject_indices
        }
        denominator = sum(squared.values())
        if denominator == 0.0:
            continue
        for index in subject_indices:
            ownership = squared[index] / denominator
            competed[index][token_index] *= ownership
    return tuple(tuple(field) for field in competed)


def krea_prompt_token_count(tokenized: dict[str, list[list[tuple]]]) -> int:
    """Count prompt-owned lanes after Krea's fixed Qwen wrapper prefix is removed."""
    if not tokenized:
        raise ValueError("Krea tokenization returned no token groups")
    batches = next(iter(tokenized.values()))
    if len(batches) != 1:
        raise ValueError("Krea unified prompting requires one token batch")
    pairs = batches[0]
    second_im_start: int | None = None
    seen = 0
    for index, pair in enumerate(pairs):
        token = pair[0]
        if isinstance(token, Integral) and token == 151644:
            seen += 1
            if seen == 2:
                second_im_start = index
                break
    if second_im_start is None:
        raise ValueError("Krea Qwen wrapper is missing its second <|im_start|> token")

    prompt_start = second_im_start + 1
    if (
        len(pairs) > prompt_start + 1
        and pairs[prompt_start][0] == 872
        and pairs[prompt_start + 1][0] == 198
    ):
        prompt_start += 2
    for index in range(prompt_start, len(pairs)):
        if pairs[index][0] == 151645:
            return index - prompt_start
    raise ValueError("Krea Qwen wrapper is missing the user <|im_end|> token")


def region_definitions_from_payload(items: list[dict]) -> tuple[RegionDefinition, ...]:
    return tuple(
        RegionDefinition(
            region_id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            box=PixelBox(
                float(item["box"]["x0"]),
                float(item["box"]["y0"]),
                float(item["box"]["x1"]),
                float(item["box"]["y1"]),
            ),
            prompt=str(item.get("prompt", "")),
            negative_prompt=str(item.get("negative_prompt", "")),
            face_identity_prompt=str(item.get("face_identity_prompt", "")),
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in items
    )
