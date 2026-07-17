from __future__ import annotations

import sys
import unittest
from types import ModuleType
from types import MethodType
from unittest.mock import patch

from k2_region_lab.regional_lora import compile_lora_delta_routes
from k2_region_lab.regional_prompting import compile_regional_prompt_plan
from k2_region_lab.regions import PixelBox, RegionDefinition
from k2_region_lab.worker.runtime import ComfyBaselineRuntime


class RegionalLoraRoutingTests(unittest.TestCase):
    def _plans(self):
        regions = (
            RegionDefinition("left", "Left subject", PixelBox(0, 0, 16, 16), "red coat"),
            RegionDefinition("right", "Right subject", PixelBox(16, 0, 32, 16), "blue coat"),
        )
        plan = compile_regional_prompt_plan(32, 16, "portrait", regions)
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))
        return plan, bound

    def test_regional_route_enables_only_its_clause_and_pixel_box(self) -> None:
        plan, bound = self._plans()
        route = compile_lora_delta_routes(
            [
                {
                    "id": "face",
                    "name": "Face",
                    "strength": 0.8,
                    "global": False,
                    "region_ids": ["right"],
                }
            ],
            width=32,
            height=16,
            text_token_count=bound.text_token_count,
            regional_plan=plan,
            bound_plan=bound,
        )[0]
        right_span = next(span for span in bound.spans if span.region_id == "right")

        self.assertEqual(route.image_token_mask, (0.0, 1.0))
        self.assertTrue(
            all(
                route.text_token_mask[index] == 1.0
                for index in range(right_span.start, right_span.end)
            )
        )
        self.assertEqual(sum(route.text_token_mask), right_span.end - right_span.start)
        self.assertEqual(
            route.sequence_mask(bound.text_token_count, text_fusion=True),
            route.text_token_mask,
        )
        self.assertEqual(
            route.layerwise_text_batch_mask(bound.text_token_count * 2),
            route.text_token_mask * 2,
        )

    def test_multiple_regions_are_combined_as_a_union(self) -> None:
        plan, bound = self._plans()
        route = compile_lora_delta_routes(
            [
                {
                    "id": "style",
                    "name": "Style",
                    "global": False,
                    "region_ids": ["left", "right", "left"],
                }
            ],
            width=32,
            height=16,
            text_token_count=bound.text_token_count,
            regional_plan=plan,
            bound_plan=bound,
        )[0]

        self.assertEqual(route.region_ids, ("left", "right"))
        self.assertEqual(route.image_token_mask, (1.0, 1.0))
        self.assertEqual(route.region_names, ("Left subject", "Right subject"))

    def test_global_route_enables_every_lane_without_a_regional_plan(self) -> None:
        route = compile_lora_delta_routes(
            [{"id": "global", "name": "Global", "global": True}],
            width=32,
            height=16,
            text_token_count=5,
            regional_plan=None,
            bound_plan=None,
        )[0]

        self.assertEqual(route.text_token_mask, (1.0,) * 5)
        self.assertEqual(route.image_token_mask, (1.0, 1.0))

    def test_inactive_regional_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "without active prompts"):
            compile_lora_delta_routes(
                [
                    {
                        "id": "face",
                        "name": "Face",
                        "global": False,
                        "region_ids": ["missing"],
                    }
                ],
                width=32,
                height=16,
                text_token_count=5,
                regional_plan=None,
                bound_plan=None,
            )

    def test_global_and_regional_loras_share_one_composite_target(self) -> None:
        class FakeModel:
            def clone(self):
                return FakeModel()

            def set_attachments(self, key, value):
                del key, value

        plan, bound = self._plans()
        runtime = object.__new__(ComfyBaselineRuntime)
        runtime.model = FakeModel()
        installed = {}

        def fake_load(self, specification):
            del self
            lora_id = specification["id"]
            return (
                {"diffusion_model.blocks.0.attn.wq": object()},
                None,
                {
                    "id": lora_id,
                    "display_name": specification["name"],
                    "strength": specification["strength"],
                    "global": specification["global"],
                    "region_ids": specification.get("region_ids", []),
                    "compatible": True,
                    "adapter_count": 1,
                    "matched_model_targets": 1,
                },
            )

        def fake_install(self, generation_model, target_entries, statistics):
            del self, statistics
            installed.update(target_entries)
            return generation_model.clone(), len(target_entries)

        runtime._load_lora_patches = MethodType(fake_load, runtime)
        runtime._install_routed_lora_bypass = MethodType(fake_install, runtime)
        specifications = [
            {
                "id": "global",
                "name": "Global",
                "path": "/unused/global.safetensors",
                "strength": 0.5,
                "global": True,
            },
            {
                "id": "face",
                "name": "Face",
                "path": "/unused/face.safetensors",
                "strength": 1.0,
                "global": False,
                "region_ids": ["right"],
            },
        ]

        _model, reports, _statistics = runtime._apply_routed_loras(
            specifications,
            width=32,
            height=16,
            text_token_count=bound.text_token_count,
            regional_plan=plan,
            bound_plan=bound,
            event=None,
        )

        entries = installed["diffusion_model.blocks.0.attn.wq"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            [report["status"] for report in reports],
            ["applied_global", "applied_regional"],
        )

    def test_vae_handoff_unloads_model_before_discarding_adapter_hooks(self) -> None:
        calls = []
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        management = ModuleType("comfy.model_management")
        management.unload_all_models = lambda: calls.append("unload")
        management.soft_empty_cache = lambda force=False: calls.append(
            f"empty:{force}"
        )
        comfy.model_management = management

        class FakeGenerationModel:
            def remove_injections(self, key):
                self.assert_unloaded(key)

            @staticmethod
            def assert_unloaded(key):
                if calls != ["unload"]:
                    raise AssertionError("adapter hooks were removed before model unload")
                calls.append(f"remove:{key}")

        runtime = object.__new__(ComfyBaselineRuntime)
        with patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ):
            runtime._prepare_vae_handoff(FakeGenerationModel(), None)

        self.assertEqual(
            calls,
            ["unload", "remove:k2_routed_loras", "empty:True"],
        )


if __name__ == "__main__":
    unittest.main()
