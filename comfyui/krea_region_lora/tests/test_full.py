from __future__ import annotations

import json

import torch

from krea_region_lora.face_detailer import (
    coerce_face_boxes,
    expanded_square_box,
    face_blend_mask,
)
from krea_region_lora.masks import region_from_mask
from krea_region_lora.types import K2LoraReference
from krea_region_lora.upscaler import post_upscale
from krea_region_lora.workflow import (
    apply_prompt_emphases,
    bind_lora_reference,
    compile_spatial_conditioning,
    layout_from_document,
    parse_project_document,
)


class FakeClip:
    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        tensor = torch.ones((1, max(1, len(tokens.split())), 4), dtype=torch.float32)
        return [[tensor, {"prompt": tokens}]]


def sample_document():
    return {
        "schema": "k2-region-lab-comfy",
        "version": 1,
        "canvas": {"width": 64, "height": 64},
        "global_prompt": "two people in a room",
        "global_negative": "blur",
        "regions": [
            {
                "id": "left",
                "name": "Left person",
                "box": {"x0": 0, "y0": 0, "x1": 32, "y1": 64},
                "prompt": "red coat",
                "negative_prompt": "blue coat",
                "face_identity_prompt": "lface, a woman with brown hair",
                "enabled": True,
                "priority": 2,
                "spatial_role": "subject",
            },
            {
                "id": "right",
                "name": "Right person",
                "box": {"x0": 32, "y0": 0, "x1": 64, "y1": 64},
                "prompt": "blue coat",
                "negative_prompt": "red coat",
                "enabled": True,
                "priority": 1,
                "spatial_role": "subject",
            },
        ],
        "emphases": [{"scope_id": "left", "phrase": "red coat", "strength": 0.5, "occurrence": 0}],
        "regional": {"feather_pixels": 0, "inside_strength": 1.2, "outside_penalty": 0.8},
        "projector": {"enabled": False, "preset": "filter_bypass2", "values": [0] * 12},
        "lora_assignments": {"1": ["Left person"]},
    }


def test_desktop_project_payload_is_importable():
    desktop = {
        "schema": "k2-region-lab-project",
        "version": 14,
        "canvas": {"width": 1024, "height": 768},
        "generation": {"global_prompt": "portrait", "regional_prompt_strength": 1.5},
        "regions": [],
    }
    converted = parse_project_document(json.dumps(desktop))
    assert converted["canvas"] == {"width": 1024, "height": 768}
    assert converted["global_prompt"] == "portrait"
    assert converted["regional"]["inside_strength"] == 1.5


def test_editor_document_builds_named_encoded_regions():
    layout = layout_from_document(sample_document(), clip=FakeClip())
    assert (layout.width, layout.height) == (64, 64)
    assert [item.name for item in layout.regions] == ["Left person", "Right person"]
    assert layout.regions[0].region.pixel_bbox == (0, 0, 32, 64)
    encoded_prompt = layout.regions[0].positive[0][1]["prompt"]
    assert "lface, a woman with brown hair" in encoded_prompt
    assert "(red coat:1.500)" in encoded_prompt


def test_spatial_conditioning_contains_inside_and_outside_masks():
    layout = layout_from_document(sample_document(), clip=FakeClip())
    positive, negative = compile_spatial_conditioning(layout)
    assert len(positive) == 5
    assert len(negative) == 9
    left_mask = positive[1][1]["mask"]
    outside_mask = negative[3][1]["mask"]
    assert torch.all(left_mask[:, :, :32] == 1)
    assert torch.all(outside_mask[:, :, :32] == 0)
    assert positive[1][1]["mask_strength"] == 1.2
    assert positive[1][1]["end_percent"] == 0.5
    assert positive[2][1]["mask_strength"] == 1.2 * 0.35


def test_lora_reference_binds_by_region_name_and_requires_identity_prompt():
    layout = layout_from_document(sample_document(), clip=FakeClip())
    reference = K2LoraReference(
        "character.safetensors",
        strength=1.5,
        routing_mode="character_identity",
        trigger_phrase="lface",
    )
    bindings = bind_lora_reference(layout, reference, ["Left person"])
    assert len(bindings) == 1
    assert bindings[0].region_name == "Left person"
    assert bindings[0].lora_strength == 1.5
    try:
        bind_lora_reference(layout, reference, ["Right person"])
    except ValueError as error:
        assert "face identity prompt" in str(error)
    else:
        raise AssertionError("identity binding without identity prompt was accepted")


def test_native_mask_is_preserved_as_region():
    mask = torch.zeros((1, 32, 48))
    mask[:, 4:20, 8:30] = 1
    region = region_from_mask(mask, metadata={"region_id": "mask"})
    assert region.image_size == (48, 32)
    assert region.pixel_bbox == (8, 4, 30, 20)
    assert region.metadata["region_id"] == "mask"


def test_prompt_emphasis_only_changes_requested_occurrence():
    from krea_region_lora.types import K2PromptEmphasis

    result = apply_prompt_emphases(
        "red coat and red coat",
        (K2PromptEmphasis("global", "red coat", 0.4, 1),),
        "global",
    )
    assert result == "red coat and (red coat:1.400)"


def test_face_inputs_accept_xywh_and_segs_xyxy():
    boxes = coerce_face_boxes(bboxes=[(10, 12, 20, 24)], width=100, height=100)
    assert boxes == [(10, 12, 30, 36)]

    class Seg:
        bbox = (40, 42, 60, 68)
        crop_region = None

    boxes = coerce_face_boxes(segs=((100, 100), [Seg()]), width=100, height=100)
    assert boxes == [(40, 42, 60, 68)]
    assert expanded_square_box((40, 42, 60, 68), 100, 100, 2.0) == (24, 29, 76, 81)


def test_face_blend_mask_feathers_edges():
    mask = face_blend_mask(32, 32, 0.125, 0.5)
    assert mask.shape == (32, 32)
    assert mask[0, 0] == 0
    assert mask[16, 16] == 0.5


def test_lanczos_post_upscale_has_exact_requested_size():
    source = torch.full((2, 12, 20, 3), 0.5)
    output, report = post_upscale(source, scale=4, method="lanczos")
    assert output.shape == (2, 48, 80, 3)
    assert "Lanczos 4x" in report
