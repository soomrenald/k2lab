from __future__ import annotations

import sys
import unittest
from types import ModuleType
from types import MethodType
from unittest.mock import patch

from k2_region_lab.lora import CHARACTER_IDENTITY_LORA_ROUTING
from k2_region_lab.regional_lora import (
    compile_lora_delta_routes,
    route_allows_adapter_target,
)
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

    def test_standard_regional_route_gates_its_clause_and_pixel_box(self) -> None:
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
        self.assertEqual(route.image_token_mask, (0.0, 1.0))
        right_span = next(span for span in bound.spans if span.region_id == "right")
        self.assertEqual(
            {index for index, value in enumerate(route.text_token_mask) if value},
            set(range(right_span.start, right_span.end)),
        )
        self.assertEqual(
            route.sequence_mask(bound.text_token_count, text_fusion=True),
            route.text_token_mask,
        )
        self.assertEqual(
            route.sequence_mask(len(route.image_token_mask), text_fusion=False),
            route.image_token_mask,
        )
        self.assertEqual(
            route.layerwise_text_batch_mask(bound.text_token_count * 2),
            route.text_token_mask * 2,
        )

    def test_standard_regional_route_excludes_broadcast_adapter_targets(self) -> None:
        plan, bound = self._plans()
        route = compile_lora_delta_routes(
            [
                {
                    "id": "style",
                    "name": "Style",
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

        self.assertTrue(
            route_allows_adapter_target(
                route, "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight"
            )
        )
        self.assertFalse(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.attn.wk.weight"
            )
        )
        self.assertFalse(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.attn.wv.weight"
            )
        )
        self.assertTrue(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.attn.wq.weight"
            )
        )
        self.assertTrue(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.attn.wo.weight"
            )
        )
        self.assertTrue(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.mlp.down.weight"
            )
        )
        self.assertFalse(
            route_allows_adapter_target(
                route, "diffusion_model.last.modulation.lin"
            )
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

    def test_character_identity_route_keeps_full_regional_text_coverage(self) -> None:
        regions = (
            RegionDefinition(
                "person",
                "Person",
                PixelBox(16, 0, 32, 16),
                "lface, an adult woman wearing a blue coat",
            ),
        )
        plan = compile_regional_prompt_plan(
            32,
            16,
            "portrait",
            regions,
            character_identity_triggers={"person": ("lface",)},
        )
        bound = plan.bind_tokens(len, conditioning_text_token_count=len(plan.prompt))

        route = compile_lora_delta_routes(
            [
                {
                    "id": "face",
                    "name": "Face",
                    "strength": 1.0,
                    "global": False,
                    "region_ids": ["person"],
                    "routing_mode": CHARACTER_IDENTITY_LORA_ROUTING,
                    "trigger_phrase": "lface",
                }
            ],
            width=32,
            height=16,
            text_token_count=bound.text_token_count,
            regional_plan=plan,
            bound_plan=bound,
        )[0]

        region_span = next(
            span for span in bound.spans if span.region_id == "person"
        )
        enabled_indices = set(range(region_span.start, region_span.end))
        self.assertEqual(route.image_token_mask, (0.0, 1.0))
        self.assertEqual(
            {index for index, value in enumerate(route.text_token_mask) if value},
            enabled_indices,
        )
        self.assertEqual(route.routing_mode, CHARACTER_IDENTITY_LORA_ROUTING)
        self.assertGreater(len(enabled_indices), 2)
        self.assertTrue(
            route_allows_adapter_target(
                route, "diffusion_model.blocks.0.attn.wv.weight"
            )
        )

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

            def get_attachment(self, key):
                del key
                return None

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
            base_model=runtime.model,
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

    def test_standard_runtime_installs_only_spatially_local_targets(self) -> None:
        class FakeModel:
            def clone(self):
                return FakeModel()

            def get_attachment(self, key):
                del key
                return None

            def set_attachments(self, key, value):
                del key, value

        plan, bound = self._plans()
        runtime = object.__new__(ComfyBaselineRuntime)
        runtime.model = FakeModel()
        installed = {}

        def fake_load(self, specification):
            del self
            patches = {
                "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight": object(),
                "diffusion_model.blocks.0.attn.wv.weight": object(),
                "diffusion_model.blocks.0.attn.wo.weight": object(),
            }
            return (
                patches,
                None,
                {
                    "id": specification["id"],
                    "display_name": specification["name"],
                    "strength": specification["strength"],
                    "global": False,
                    "region_ids": specification["region_ids"],
                    "compatible": True,
                    "adapter_count": len(patches),
                    "matched_model_targets": len(patches),
                },
            )

        def fake_install(self, generation_model, target_entries, statistics):
            del self, statistics
            installed.update(target_entries)
            return generation_model.clone(), len(target_entries)

        runtime._load_lora_patches = MethodType(fake_load, runtime)
        runtime._install_routed_lora_bypass = MethodType(fake_install, runtime)
        _model, reports, _statistics = runtime._apply_routed_loras(
            [
                {
                    "id": "style",
                    "name": "Style",
                    "path": "/unused/style.safetensors",
                    "strength": 1.0,
                    "global": False,
                    "region_ids": ["right"],
                }
            ],
            base_model=runtime.model,
            width=32,
            height=16,
            text_token_count=bound.text_token_count,
            regional_plan=plan,
            bound_plan=bound,
            event=None,
        )

        self.assertEqual(
            set(installed),
            {
                "diffusion_model.txtfusion.refiner_blocks.0.attn.wq.weight",
                "diffusion_model.blocks.0.attn.wo.weight",
            },
        )
        self.assertEqual(reports[0]["applied_model_targets"], 2)
        self.assertEqual(reports[0]["locality_skipped_targets"], 1)
        self.assertEqual(
            reports[0]["application_mode"],
            "unfused_region_text_image_delta_gate_v3",
        )

    def test_runtime_rejects_a_regional_lora_with_no_routable_targets(self) -> None:
        plan, bound = self._plans()
        runtime = object.__new__(ComfyBaselineRuntime)
        runtime.model = object()

        def fake_load(self, specification):
            del self
            return (
                {"diffusion_model.blocks.0.attn.wv.weight": object()},
                None,
                {
                    "id": specification["id"],
                    "display_name": specification["name"],
                    "strength": 1.0,
                    "global": False,
                    "region_ids": specification["region_ids"],
                    "compatible": True,
                    "adapter_count": 1,
                    "matched_model_targets": 1,
                },
            )

        runtime._load_lora_patches = MethodType(fake_load, runtime)
        with self.assertRaisesRegex(ValueError, "no targets that can be routed locally"):
            runtime._apply_routed_loras(
                [{
                    "id": "broadcast-only",
                    "name": "Broadcast only",
                    "path": "/unused/broadcast.safetensors",
                    "strength": 1.0,
                    "global": False,
                    "region_ids": ["right"],
                }],
                base_model=runtime.model,
                width=32,
                height=16,
                text_token_count=bound.text_token_count,
                regional_plan=plan,
                bound_plan=bound,
                event=None,
            )

    def test_global_distillation_lora_patches_bare_parameter_beside_bypass_hooks(
        self,
    ) -> None:
        torch = ModuleType("torch")
        torch.Tensor = type("Tensor", (), {})
        comfy = ModuleType("comfy")
        comfy.__path__ = []
        weight_adapter = ModuleType("comfy.weight_adapter")

        class FakeAdapterBase:
            pass

        class FakeManager:
            def __init__(self):
                self.adapters = []

            def add_adapter(self, key, adapter, strength):
                self.adapters.append((key, adapter, strength))

            def create_injections(self, model):
                del model
                return ["injection"]

            def get_hook_count(self):
                return len(self.adapters)

        weight_adapter.WeightAdapterBase = FakeAdapterBase
        weight_adapter.BypassInjectionManager = FakeManager
        comfy.weight_adapter = weight_adapter

        class FakePatcher:
            def __init__(self, calls):
                self.model = object()
                self.calls = calls

            def clone(self):
                return FakePatcher(self.calls)

            def add_patches(self, patches, strength_patch=1.0):
                self.calls.append(("patch", tuple(patches), strength_patch))
                return set(patches)

            def set_injections(self, name, injections):
                self.calls.append(("injections", name, tuple(injections)))

        route = compile_lora_delta_routes(
            [{"id": "distill", "name": "Turbo distill", "global": True}],
            width=32,
            height=16,
            text_token_count=5,
            regional_plan=None,
            bound_plan=None,
        )[0]
        calls = []
        target_entries = {
            "diffusion_model.blocks.0.attn.wq.weight": [(FakeAdapterBase(), route)],
            "diffusion_model.last.modulation.lin": [(FakeAdapterBase(), route)],
        }

        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                "comfy": comfy,
                "comfy.weight_adapter": weight_adapter,
            },
        ):
            _model, installed = ComfyBaselineRuntime._install_routed_lora_bypass(
                FakePatcher(calls), target_entries, object()
            )

        self.assertEqual(installed, 2)
        self.assertIn(
            ("patch", ("diffusion_model.last.modulation.lin",), 1.0),
            calls,
        )
        self.assertIn(("injections", "k2_routed_loras", ("injection",)), calls)

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
                if not calls or calls[0] != "unload":
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
            [
                "unload",
                "remove:k2_routed_loras",
                "remove:k2_projector_delta",
                "empty:True",
            ],
        )


if __name__ == "__main__":
    unittest.main()
