from __future__ import annotations

import torch

from krea_region_lora.engine import run_regional_velocity_sampler
from krea_region_lora.lora import (
    filter_lora_state_dict,
    is_krea_attention_out_lora_key,
    is_krea_mlp_lora_key,
    is_krea_writeback_lora_key,
)
from krea_region_lora.masks import region_from_bbox
from krea_region_lora.types import K2RegionalLora, K2RegionalLoraStack


def make_region(bbox, *, batch_size=1, bbox_format="xywh"):
    return region_from_bbox(
        [bbox],
        width=64,
        height=64,
        bbox_format=bbox_format,
        feather_px=0,
        snap_to_krea_token_grid=False,
        batch_size=batch_size,
        batch_mode="per_batch" if batch_size > 1 else "repeat",
    )


def make_lora(region, name, *, enabled=True):
    return K2RegionalLora(
        region=region,
        positive=[f"{name} positive"],
        negative=["negative"],
        lora_name=name,
        start_percent=0.0,
        end_percent=1.0,
        enabled=enabled,
    )


def run(stack, constants, *, batch_size=1):
    initial = torch.zeros((batch_size, 4, 8, 8), dtype=torch.float32)

    def predict(branch_name, x, sigma, cond, uncond):
        if branch_name == "base":
            return torch.zeros_like(x)
        return torch.full_like(x, constants[branch_name])

    return run_regional_velocity_sampler(
        initial=initial,
        stack=stack,
        base_positive=["base positive"],
        base_negative=["base negative"],
        cfg=4.0,
        schedule=[1.0, 0.0],
        predict=predict,
        pin_outside_regions=True,
    )


def test_xywh_to_mask_conversion():
    region = make_region((16, 16, 16, 16))
    assert region.pixel_bbox == (16, 16, 32, 32)
    assert region.pixel_mask.shape == (1, 64, 64)
    assert region.latent_mask.shape == (1, 1, 8, 8)
    assert region.token_mask.shape == (1, 16, 1)
    assert torch.all(region.latent_mask[:, :, 2:4, 2:4] == 1)
    assert torch.count_nonzero(region.latent_mask) == 4


def test_xyxy_to_mask_conversion():
    region = make_region((16, 16, 32, 32), bbox_format="xyxy")
    assert region.pixel_bbox == (16, 16, 32, 32)
    assert torch.all(region.latent_mask[:, :, 2:4, 2:4] == 1)


def test_empty_bbox_or_disabled_lora_returns_base_result():
    empty = region_from_bbox([], width=64, height=64, feather_px=0, snap_to_krea_token_grid=False)
    assert torch.count_nonzero(empty.pixel_mask) == 0

    region = make_region((16, 16, 16, 16))
    stack = K2RegionalLoraStack((make_lora(region, "char", enabled=False),))
    samples, base, _union, _debug = run(stack, {"char": 1.0})
    assert torch.equal(samples, base)


def test_fake_global_delta_changes_only_mask_region():
    region = make_region((16, 16, 16, 16))
    stack = K2RegionalLoraStack((make_lora(region, "char"),))
    samples, base, union, _debug = run(stack, {"char": 1.0})
    outside = union == 0
    inside = union == 1
    assert torch.equal(samples[outside.expand_as(samples)], base[outside.expand_as(base)])
    assert torch.all(samples[inside.expand_as(samples)] == -1)


def test_pin_outside_regions_exact_after_every_step():
    region = make_region((16, 16, 16, 16))
    stack = K2RegionalLoraStack((make_lora(region, "char"),))
    _samples, _base, _union, debug = run(stack, {"char": 9.0})
    assert debug.outside_equal_after_step == [True]


def test_three_disjoint_regions_produce_three_isolated_deltas():
    r1 = make_region((0, 0, 16, 16))
    r2 = make_region((16, 16, 16, 16))
    r3 = make_region((32, 32, 16, 16))
    stack = K2RegionalLoraStack(
        (make_lora(r1, "one"), make_lora(r2, "two"), make_lora(r3, "three"))
    )
    samples, _base, _union, _debug = run(stack, {"one": 1.0, "two": 2.0, "three": 3.0})
    assert torch.all(samples[:, :, 0:2, 0:2] == -1)
    assert torch.all(samples[:, :, 2:4, 2:4] == -2)
    assert torch.all(samples[:, :, 4:6, 4:6] == -3)
    assert torch.all(samples[:, :, 6:8, 6:8] == 0)


def test_overlapping_regions_follow_overlap_mode():
    r1 = make_region((16, 16, 16, 16))
    r2 = make_region((16, 16, 16, 16))
    for mode, expected in (("normalize", -2.0), ("priority_1", -1.0), ("priority_3", -3.0)):
        stack = K2RegionalLoraStack(
            (make_lora(r1, "one"), make_lora(r2, "three")), overlap_mode=mode
        )
        samples, _base, union, _debug = run(stack, {"one": 1.0, "three": 3.0})
        assert torch.all(samples[union.expand_as(samples) == 1] == expected)

    stack = K2RegionalLoraStack(
        (make_lora(r1, "a"), make_lora(r2, "b")), overlap_mode="add_clamped"
    )
    samples, _base, union, _debug = run(stack, {"a": 10.0, "b": 10.0})
    assert torch.all(samples[union.expand_as(samples) == 1] == -1.0)


def test_text_encoder_lora_keys_are_ignored_by_default():
    state = {
        "clip.transformer.text_model.encoder.layers.0.self_attn.q_proj.lora_up.weight": 1,
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": 2,
    }
    filtered = filter_lora_state_dict(
        state, attention_only_filter=False, ignore_text_encoder_lora=True
    )
    assert list(filtered) == ["diffusion_model.blocks.0.attn.wq.lora_up.weight"]


def test_attention_only_filter_keeps_krea_attention_targets():
    state = {
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": 1,
        "diffusion_model.blocks.0.attn.wk.lora_up.weight": 2,
        "diffusion_model.blocks.0.attn.wv.lora_up.weight": 3,
        "diffusion_model.blocks.0.attn.gate.lora_up.weight": 4,
        "diffusion_model.blocks.0.attn.wo.lora_up.weight": 5,
        "diffusion_model.blocks.0.mlp.fc1.lora_up.weight": 6,
    }
    filtered = filter_lora_state_dict(
        state, attention_only_filter=True, ignore_text_encoder_lora=True
    )
    assert set(filtered.values()) == {1, 2, 3, 4, 5}


def test_writeback_key_helpers_exclude_qkv_and_keep_attn_out_mlp():
    assert not is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.wq.lora_up.weight")
    assert not is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.wv.lora_up.weight")
    assert is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.wo.lora_up.weight")
    assert is_krea_attention_out_lora_key("diffusion_model.blocks.0.attn.out_proj.lora_up.weight")
    assert is_krea_mlp_lora_key("diffusion_model.blocks.0.mlp.fc1.lora_up.weight")
    assert is_krea_writeback_lora_key("diffusion_model.blocks.0.attn.wo.lora_up.weight")
    assert is_krea_writeback_lora_key("diffusion_model.blocks.0.mlp.fc2.lora_up.weight")


def test_output_shapes_support_batch_1_and_batch_gt_1():
    r1 = make_region((16, 16, 16, 16), batch_size=1)
    stack1 = K2RegionalLoraStack((make_lora(r1, "char"),))
    samples1, base1, union1, _debug1 = run(stack1, {"char": 1.0}, batch_size=1)
    assert samples1.shape == base1.shape == (1, 4, 8, 8)
    assert union1.shape == (1, 1, 8, 8)

    r2 = make_region((16, 16, 16, 16), batch_size=2)
    stack2 = K2RegionalLoraStack((make_lora(r2, "char"),))
    samples2, base2, union2, _debug2 = run(stack2, {"char": 1.0}, batch_size=2)
    assert samples2.shape == base2.shape == (2, 4, 8, 8)
    assert union2.shape == (2, 1, 8, 8)
