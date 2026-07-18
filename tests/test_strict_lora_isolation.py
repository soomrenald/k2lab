from __future__ import annotations

import pytest

from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.worker.runtime import (
    pin_regional_lora_latent_to_baseline,
    regional_lora_region_ids,
)


torch = pytest.importorskip("torch")


def test_two_subject_scene_restores_unassigned_subject_to_baseline() -> None:
    regions = (
        RegionDefinition(
            "left",
            "Left subject",
            PixelBox(0, 0, 128, 256),
            "first subject",
        ),
        RegionDefinition(
            "right",
            "Right subject",
            PixelBox(128, 0, 256, 256),
            "second subject",
        ),
    )
    loras = [
        {
            "id": "left-only-lora",
            "global": False,
            "region_ids": ["left"],
            "strength": 1.0,
        }
    ]
    baseline = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    regional = torch.ones_like(baseline)

    pinned, mask = pin_regional_lora_latent_to_baseline(
        regional,
        baseline,
        specifications=loras,
        regions=regions,
        width=256,
        height=256,
    )

    assert regional_lora_region_ids(loras) == frozenset({"left"})
    assert torch.all(mask[:, :, :, :16] == 1)
    assert torch.all(mask[:, :, :, 16:] == 0)
    assert torch.all(pinned[:, :, :, :16] == 1)
    assert torch.equal(pinned[:, :, :, 16:], baseline[:, :, :, 16:])


def test_global_and_disabled_loras_do_not_activate_strict_isolation() -> None:
    loras = [
        {"id": "global", "global": True, "region_ids": [], "strength": 1.0},
        {
            "id": "disabled-regional",
            "global": False,
            "region_ids": ["left"],
            "strength": 0.0,
        },
    ]

    assert regional_lora_region_ids(loras) == frozenset()
