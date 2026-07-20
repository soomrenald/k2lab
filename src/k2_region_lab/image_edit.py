from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

from k2_region_lab.regions import CanvasGeometry, RegionDefinition
from k2_region_lab.projector import DEFAULT_PROJECTOR_PRESET, PROJECTOR_PRESETS
from k2_region_lab.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis
from k2_region_lab.sampling import (
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    validate_sampler,
    validate_scheduler,
)


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True, slots=True)
class ImageEditState:
    source_image: Path | None = None
    associated_project: Path | None = None
    width: int = 0
    height: int = 0
    reference_global_prompt: str = ""
    reference_regions: tuple[RegionDefinition, ...] = ()
    reference_prompt_emphases: tuple[PromptEmphasis, ...] = ()
    reference_projector_enabled: bool = False
    reference_projector_preset: str = DEFAULT_PROJECTOR_PRESET
    reference_projector_values: tuple[float, ...] = PROJECTOR_PRESETS[
        DEFAULT_PROJECTOR_PRESET
    ]
    reference_projector_multiplier: float = 1.0
    reference_projector_identity_protection: float = 1.0
    global_prompt: str = ""
    steps: int = 8
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    seed: int = 0
    denoise: float = 0.15
    latent_feather_pixels: int = 64
    composite_feather_pixels: int = 48
    edit_entire_image: bool = False
    preserve_identity: bool = True
    reference_description_retention: float = 1.0
    regional_prompt_strength: float = 1.0
    regional_outside_penalty: float = 1.0
    regional_feather_pixels: int = 128
    regional_subject_competition: bool = True
    regional_subject_fill: bool = True
    regional_late_step_scale: float = 0.35
    regional_lora_delta_adaptation: bool = False
    regional_lora_delta_adaptation_gain: float = 0.35
    regions: tuple[RegionDefinition, ...] = ()

    def __post_init__(self) -> None:
        has_geometry = self.width != 0 or self.height != 0
        if has_geometry and not (
            256 <= self.width <= 4096 and 256 <= self.height <= 4096
        ):
            raise ValueError("image-edit dimensions must be between 256 and 4096 pixels")
        if not 1 <= self.steps <= 100:
            raise ValueError("image-edit steps must be between 1 and 100")
        validate_sampler(self.sampler)
        validate_scheduler(self.scheduler)
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("image-edit seed must be between 0 and 2147483647")
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError("image-edit denoise must be in (0, 1]")
        if not 0 <= self.latent_feather_pixels <= 256:
            raise ValueError("image-edit latent feather must be between 0 and 256 pixels")
        if not 0 <= self.composite_feather_pixels <= 256:
            raise ValueError("image-edit composite feather must be between 0 and 256 pixels")
        if not 0.0 < self.regional_prompt_strength <= 10.0:
            raise ValueError("image-edit regional prompt strength must be in (0, 10]")
        if not 0.0 <= self.regional_outside_penalty <= 10.0:
            raise ValueError("image-edit outside penalty must be between 0 and 10")
        if not 0 <= self.regional_feather_pixels <= 2048:
            raise ValueError("image-edit spatial falloff must be between 0 and 2048 pixels")
        if not 0.0 <= self.regional_late_step_scale <= 1.0:
            raise ValueError("image-edit late-step scale must be between zero and one")
        if not 0.0 <= self.regional_lora_delta_adaptation_gain <= 1.0:
            raise ValueError("image-edit LoRA delta adaptation gain must be between zero and one")
        if not 0.0 <= self.reference_description_retention <= 1.0:
            raise ValueError("reference description retention must be between zero and one")
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("image-edit region IDs must be unique")
        for region in self.regions:
            if region.box.width < 16 or region.box.height < 16:
                raise ValueError("image-edit region boxes must be at least 16x16 pixels")
            if has_geometry and (
                region.box.x0 < 0
                or region.box.y0 < 0
                or region.box.x1 > self.width
                or region.box.y1 > self.height
            ):
                raise ValueError("image-edit region boxes must stay inside the source image")
        reference_ids = [region.region_id for region in self.reference_regions]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("image-edit reference region IDs must be unique")
        for region in self.reference_regions:
            if region.box.width < 16 or region.box.height < 16:
                raise ValueError("image-edit reference boxes must be at least 16x16 pixels")
            if has_geometry and (
                region.box.x0 < 0
                or region.box.y0 < 0
                or region.box.x1 > self.width
                or region.box.y1 > self.height
            ):
                raise ValueError("image-edit reference boxes must stay inside the source image")


def load_source_image(path: Path) -> tuple[Image.Image, dict[str, str]]:
    source = path.expanduser().resolve()
    if source.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES or not source.is_file():
        raise ValueError(f"image editing requires a readable PNG, JPEG, or WebP: {source}")
    with Image.open(source) as opened:
        metadata = {
            str(key): str(value)
            for key, value in opened.info.items()
            if isinstance(value, (str, int, float, bool))
        }
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if not 256 <= image.width <= 4096 or not 256 <= image.height <= 4096:
        raise ValueError("image-edit dimensions must be between 256 and 4096 pixels")
    return image, metadata


def edge_pad_to_krea(image: Image.Image) -> tuple[Image.Image, CanvasGeometry]:
    geometry = CanvasGeometry.resolve(image.width, image.height)
    if (image.width, image.height) == (geometry.aligned_width, geometry.aligned_height):
        return image.copy(), geometry
    padded = Image.new("RGB", (geometry.aligned_width, geometry.aligned_height))
    padded.paste(image, (0, 0))
    if geometry.aligned_width > image.width:
        right = image.crop((image.width - 1, 0, image.width, image.height)).resize(
            (geometry.aligned_width - image.width, image.height)
        )
        padded.paste(right, (image.width, 0))
    if geometry.aligned_height > image.height:
        bottom = padded.crop((0, image.height - 1, geometry.aligned_width, image.height)).resize(
            (geometry.aligned_width, geometry.aligned_height - image.height)
        )
        padded.paste(bottom, (0, image.height))
    return padded, geometry


def regional_composite_mask(
    size: tuple[int, int],
    regions: tuple[RegionDefinition, ...],
    feather_pixels: int,
) -> Image.Image:
    union = Image.new("L", size, 0)
    feather = max(0, int(feather_pixels))
    for region in regions:
        if not region.enabled:
            continue
        region_mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(region_mask)
        if feather == 0:
            draw.rectangle(
                (
                    floor(region.box.x0),
                    floor(region.box.y0),
                    ceil(region.box.x1) - 1,
                    ceil(region.box.y1) - 1,
                ),
                fill=255,
            )
        else:
            for step in range(feather + 1):
                expansion = feather - step
                left = floor(region.box.x0 - expansion)
                top = floor(region.box.y0 - expansion)
                right = ceil(region.box.x1 + expansion) - 1
                bottom = ceil(region.box.y1 + expansion) - 1
                progress = step / feather
                smooth = progress * progress * (3.0 - 2.0 * progress)
                draw.rectangle((left, top, right, bottom), fill=round(255 * smooth))
        union = ImageChops.lighter(union, region_mask)
    return union


def regional_edit_conditioning(
    reference_regions: tuple[RegionDefinition, ...],
    edit_regions: tuple[RegionDefinition, ...],
    instruction: str,
    *,
    preserve_identity: bool = True,
) -> tuple[RegionDefinition, ...]:
    """Combine original layout clauses with non-owning edit clauses.

    Reference subject regions retain their exclusive identity ownership. Edit clauses use
    the ``edit`` role, so the same image tokens can read both the original subject identity
    and the desired local delta.
    """

    references = tuple(
        region
        if preserve_identity
        else replace(region, face_identity_prompt="")
        for region in reference_regions
    )
    edit_instruction = instruction.strip().rstrip(".!? ")
    edits: list[RegionDefinition] = []
    for region in edit_regions:
        if not region.enabled:
            continue
        local_prompt = region.prompt.strip().rstrip(".!? ")
        description = ". ".join(
            dict.fromkeys(
                part for part in (edit_instruction, local_prompt) if part
            )
        )
        if not description and not region.face_identity_prompt.strip():
            continue
        edits.append(
            replace(
                region,
                prompt=description,
                spatial_role="edit",
            )
        )
    return references + tuple(edits)


def edit_global_conditioning_prompt(
    reference_prompt: str,
    edit_instruction: str,
    *,
    edit_entire_image: bool,
) -> str:
    """Return global text that describes the requested delta, not the source scene.

    The source-global prompt commonly names content that a replacement or removal edit
    is trying to change. Repeating it during denoising makes the two instructions fight.
    Local edits already preserve the source outside their latent mask, while reference
    regions retain the useful subject/layout conditioning inside it. Whole-image edits
    therefore use only the new instruction and localized edits use regional clauses.
    """

    del reference_prompt
    return edit_instruction.strip() if edit_entire_image else ""


def regional_reference_emphases(
    emphases: tuple[PromptEmphasis, ...],
) -> tuple[PromptEmphasis, ...]:
    """Keep reference-region emphases after source-global text is omitted."""

    return tuple(
        emphasis
        for emphasis in emphases
        if emphasis.scope_id != GLOBAL_EMPHASIS_SCOPE
    )


def composite_regional_edit(
    source: Image.Image,
    candidate: Image.Image,
    regions: tuple[RegionDefinition, ...],
    feather_pixels: int,
) -> tuple[Image.Image, Image.Image]:
    if source.size != candidate.size:
        raise ValueError("source and edited candidate dimensions must match")
    mask = regional_composite_mask(source.size, regions, feather_pixels)
    return Image.composite(candidate.convert("RGB"), source.convert("RGB"), mask), mask
