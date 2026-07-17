from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from numbers import Integral
from typing import Callable

from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition


BACKEND = "krea-unified-spatial-attention-v2"


@dataclass(frozen=True, slots=True)
class UnifiedPromptRegion:
    region_id: str
    name: str
    prompt: str
    negative_prompt: str
    box: PixelBox
    clause: str
    character_span: tuple[int, int]
    image_token_field: tuple[float, ...]
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
    late_step_scale: float
    regions: tuple[UnifiedPromptRegion, ...]
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
        return BoundRegionalPromptPlan(
            prompt=self.prompt,
            text_token_count=text_token_count,
            image_token_count=self.image_token_count,
            strength=self.strength,
            outside_penalty=self.outside_penalty,
            falloff_pixels=self.falloff_pixels,
            late_step_scale=self.late_step_scale,
            spans=spans,
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
            "late_step_scale": self.late_step_scale,
            "image_token_grid": [self.image_token_width, self.image_token_height],
            "region_count": len(self.regions),
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
    late_step_scale: float = 0.35,
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
        if not region.enabled or not region.prompt.strip():
            continue
        active.append((region, region.box.clipped(width, height)))
    active.sort(key=lambda item: _scene_order(item[0], item[1], width, height))

    roles = tuple(_effective_spatial_role(region, box, width) for region, box in active)
    raw_fields = tuple(
        _subject_target_field(geometry, box, float(falloff_pixels))
        if role == "subject"
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
    for (region, box), role, image_token_field in zip(
        active, roles, fields, strict=True
    ):
        clause = _regional_clause(region, box, width, height)
        if prompt:
            prompt += "\n"
        start = len(prompt)
        prompt += clause
        end = len(prompt)
        compiled.append(
            UnifiedPromptRegion(
                region_id=region.region_id,
                name=region.name,
                prompt=region.prompt.strip(),
                negative_prompt=region.negative_prompt.strip(),
                box=box,
                clause=clause,
                character_span=(start, end),
                image_token_field=image_token_field,
                spatial_role=role,
            )
        )

    relationship_clause = _relationship_clause(compiled, width, height)
    if relationship_clause:
        prompt += f"\n{relationship_clause}"

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
        late_step_scale=float(late_step_scale),
        regions=tuple(compiled),
    )


def _scene_order(
    region: RegionDefinition, box: PixelBox, width: int, height: int
) -> tuple[float, ...]:
    scene_layer = 0.0 if box.width / width >= 0.70 else 1.0
    center_y = (box.y0 + box.y1) / (2.0 * height)
    center_x = (box.x0 + box.x1) / (2.0 * width)
    area_fraction = box.width * box.height / (width * height)
    return (scene_layer, -float(region.priority), center_y, center_x, -area_fraction)


def _sentence(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _regional_clause(
    region: RegionDefinition, box: PixelBox, width: int, height: int
) -> str:
    center_x = 100.0 * (box.x0 + box.x1) / (2.0 * width)
    center_y = 100.0 * (box.y0 + box.y1) / (2.0 * height)
    width_percent = 100.0 * box.width / width
    height_percent = 100.0 * box.height / height
    horizontal = _horizontal_position(center_x)
    vertical = _vertical_position(center_y)
    description = region.prompt.strip().rstrip(".!? ")

    if width_percent >= 70.0:
        location = (
            f"Across the {vertical} of the image, occupying about "
            f"{height_percent:.0f}% of its height"
        )
    else:
        location = (
            f"In the {vertical} {horizontal}, centered about {center_x:.0f}% "
            f"across and {center_y:.0f}% down, occupying about "
            f"{width_percent:.0f}% of the image width and {height_percent:.0f}% "
            "of its height"
        )
    return f"{location}, there is {description}."


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
        if region.box.width < 0.70 * width
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
    return f"{ordering}."


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
    geometry: CanvasGeometry, box: PixelBox, falloff_pixels: float
) -> tuple[float, ...]:
    """Create a box target with a center peak and a half-strength boundary."""
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
                value = 1.0 - 0.5 * smooth
            elif falloff_pixels == 0.0 or distance >= falloff_pixels:
                value = 0.0
            else:
                u = 1.0 - distance / falloff_pixels
                smooth = u * u * (3.0 - 2.0 * u)
                value = 0.5 * smooth
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
            enabled=bool(item.get("enabled", True)),
            priority=int(item.get("priority", 0)),
            spatial_role=str(item.get("spatial_role", "auto")),
        )
        for item in items
    )
