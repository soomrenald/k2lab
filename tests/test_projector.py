from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from k2_region_lab.projector import (
    PROJECTOR_PRESETS,
    effective_projector_values,
    projector_token_delta_mask,
    validate_projector_values,
)
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


class ProjectorSettingsTests(unittest.TestCase):
    def test_reference_presets_have_twelve_values_and_multiplier_is_global(self) -> None:
        self.assertEqual(set(PROJECTOR_PRESETS), {
            "filter_bypass2",
            "filter_bypass3",
            "skc3vo",
            "z0jglf",
        })
        for values in PROJECTOR_PRESETS.values():
            self.assertEqual(len(values), 12)
        effective = effective_projector_values(
            PROJECTOR_PRESETS["filter_bypass2"], 3.0
        )
        self.assertAlmostEqual(effective[8], -1.5351)
        self.assertAlmostEqual(effective[9], -2.6718)

    def test_projector_vector_rejects_wrong_column_count(self) -> None:
        with self.assertRaises(ValueError):
            validate_projector_values((0.0,) * 11)

    def test_identity_protection_scales_only_selected_text_tokens(self) -> None:
        self.assertEqual(
            projector_token_delta_mask(8, ((2, 5),), 1.0),
            (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        )
        self.assertEqual(
            projector_token_delta_mask(5, ((1, 3),), 0.5),
            (1.0, 0.5, 0.5, 1.0, 1.0),
        )

    def test_runtime_applies_projector_as_one_global_diff_before_loras(self) -> None:
        class Weight:
            shape = (1, 12)

        class InnerModel:
            @staticmethod
            def state_dict():
                return {"diffusion_model.txtfusion.projector.weight": Weight()}

        class ModelPatcher:
            def __init__(self):
                self.model = InnerModel()
                self.patches = None
                self.attachments = {}

            def clone(self):
                return ModelPatcher()

            def add_patches(self, patches):
                self.patches = patches
                return list(patches)

            def set_attachments(self, key, value):
                self.attachments[key] = value

        torch = ModuleType("torch")
        torch.float32 = object()
        torch.tensor = lambda values, dtype: (values, dtype)
        runtime = ComfyBaselineRuntime(Path("/unused"))
        runtime.model = ModelPatcher()
        events = []

        with patch.dict(sys.modules, {"torch": torch}):
            patched, summary = runtime._apply_global_projector_vector(
                enabled=True,
                preset="filter_bypass2",
                values=PROJECTOR_PRESETS["filter_bypass2"],
                multiplier=2.0,
                event=lambda message, payload: events.append((message, payload)),
            )

        self.assertIsNot(patched, runtime.model)
        self.assertEqual(summary["scope"], "global")
        self.assertEqual(summary["status"], "applied_global_diff")
        delta, _dtype = patched.patches[
            "diffusion_model.txtfusion.projector.weight"
        ][1][0]
        self.assertAlmostEqual(delta[0][8], -1.0234)
        self.assertAlmostEqual(delta[0][9], -1.7812)
        self.assertEqual(patched.attachments["projector_settings"], summary)
        self.assertIn("Applied global projector vector", events[0][0])
