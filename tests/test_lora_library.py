from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    LoraLibrary,
    align_krea_lora_state_dict,
    inspect_lora_header,
    normalize_krea_lora_key,
    normalize_krea_lora_state_dict,
)


def write_lora(path: Path) -> None:
    header = {
        "blocks.0.attn.wq.lora_A.weight": {
            "dtype": "BF16",
            "shape": [4, 8],
            "data_offsets": [0, 64],
        },
        "blocks.0.attn.wq.lora_B.weight": {
            "dtype": "BF16",
            "shape": [8, 4],
            "data_offsets": [64, 128],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def write_lokr(path: Path) -> None:
    header = {
        "__metadata__": {
            "format": "pt",
            "name": "snofs",
            "ss_base_model_version": "krea2",
        },
        "diffusion_model.blocks.0.attn.wq.alpha": {
            "dtype": "BF16",
            "shape": [],
            "data_offsets": [0, 2],
        },
        "diffusion_model.blocks.0.attn.wq.lokr_w1": {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [2, 34],
        },
        "diffusion_model.blocks.0.attn.wq.lokr_w2": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [34, 42],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


class LoraLibraryTests(unittest.TestCase):
    def test_lora_defaults_global_and_can_target_multiple_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.safetensors"
            write_lora(path)
            library = LoraLibrary()
            entry = library.add(path)

            self.assertTrue(library.binding_for(entry.lora_id).global_scope)
            binding = library.assign_regions(entry.lora_id, ("region-1", "region-2"))
            self.assertFalse(binding.global_scope)
            self.assertEqual(binding.region_ids, ("region-1", "region-2"))

    def test_duplicate_path_creates_independent_lora_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.safetensors"
            write_lora(path)
            library = LoraLibrary()
            first = library.add(path)
            second = library.add(path)
            self.assertNotEqual(first.lora_id, second.lora_id)
            self.assertEqual(first.path, second.path)
            self.assertEqual(first.display_name, "style")
            self.assertEqual(second.display_name, "style #2")
            self.assertEqual(len(library.entries()), 2)

            library.set_strength(first.lora_id, 0.5)
            library.assign_regions(first.lora_id, ("left",))
            library.set_strength(second.lora_id, 1.5)
            library.assign_regions(second.lora_id, ("right",))

            first_binding = library.binding_for(first.lora_id)
            second_binding = library.binding_for(second.lora_id)
            self.assertEqual((first_binding.strength, first_binding.region_ids), (0.5, ("left",)))
            self.assertEqual(
                (second_binding.strength, second_binding.region_ids), (1.5, ("right",))
            )

    def test_character_identity_routing_keeps_an_editable_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.safetensors"
            write_lora(path)
            library = LoraLibrary()
            entry = library.add(path)

            library.set_trigger_phrase(entry.lora_id, "lface")
            binding = library.set_routing_mode(
                entry.lora_id, CHARACTER_IDENTITY_LORA_ROUTING
            )

            self.assertEqual(binding.routing_mode, CHARACTER_IDENTITY_LORA_ROUTING)
            self.assertEqual(binding.trigger_phrase, "lface")

    def test_deleted_last_region_falls_back_to_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.safetensors"
            write_lora(path)
            library = LoraLibrary()
            entry = library.add(path)
            library.assign_regions(entry.lora_id, ("region-1",))
            library.drop_region("region-1")
            self.assertTrue(library.binding_for(entry.lora_id).global_scope)

    def test_strength_is_stored_independently_from_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.safetensors"
            write_lora(path)
            library = LoraLibrary()
            entry = library.add(path)

            library.set_strength(entry.lora_id, 0.65)
            library.assign_regions(entry.lora_id, ("subject",))

            self.assertEqual(library.binding_for(entry.lora_id).strength, 0.65)
            self.assertEqual(library.binding_for(entry.lora_id).region_ids, ("subject",))

    def test_krea_ai_toolkit_namespace_is_normalized_without_touching_other_keys(self) -> None:
        self.assertEqual(
            normalize_krea_lora_key("diffusion_model.blocks.0.attn.wq.lora_A.weight"),
            "blocks.0.attn.wq.lora_A.weight",
        )
        self.assertEqual(
            normalize_krea_lora_key("transformer.transformer_blocks.0.attn.to_q.alpha"),
            "transformer.transformer_blocks.0.attn.to_q.alpha",
        )
        normalized = normalize_krea_lora_state_dict(
            {"diffusion_model.txtfusion.refiner_blocks.0.attn.wq.alpha": 32}
        )
        self.assertEqual(normalized, {"txtfusion.refiner_blocks.0.attn.wq.alpha": 32})
        original = {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": "a",
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": "b",
        }
        self.assertEqual(
            align_krea_lora_state_dict(original, {"diffusion_model.blocks.0.attn.wq"}),
            original,
        )
        self.assertEqual(
            align_krea_lora_state_dict(original, {"blocks.0.attn.wq"}),
            {
                "blocks.0.attn.wq.lora_A.weight": "a",
                "blocks.0.attn.wq.lora_B.weight": "b",
            },
        )
        lokr = {
            "diffusion_model.blocks.0.attn.wq.alpha": "alpha",
            "diffusion_model.blocks.0.attn.wq.lokr_w1": "w1",
            "diffusion_model.blocks.0.attn.wq.lokr_w2": "w2",
        }
        self.assertEqual(
            align_krea_lora_state_dict(lokr, {"blocks.0.attn.wq"}),
            {
                "blocks.0.attn.wq.alpha": "alpha",
                "blocks.0.attn.wq.lokr_w1": "w1",
                "blocks.0.attn.wq.lokr_w2": "w2",
            },
        )

    def test_header_inspection_reports_adapter_pairs_and_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.safetensors"
            write_lora(path)

            report = inspect_lora_header(path)

            self.assertEqual(report["tensor_count"], 2)
            self.assertEqual(report["adapter_count"], 1)
            self.assertEqual(report["complete_adapter_pairs"], 1)
            self.assertEqual(report["ranks"], {4: 1})
            self.assertEqual(report["adapter_types"], {"lora": 1})

    def test_header_inspection_reports_complete_lokr_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snofs.safetensors"
            write_lokr(path)

            report = inspect_lora_header(path)

            self.assertEqual(report["tensor_count"], 3)
            self.assertEqual(report["adapter_count"], 1)
            self.assertEqual(report["complete_adapter_pairs"], 1)
            self.assertEqual(report["adapter_types"], {"lokr": 1})
            self.assertEqual(report["namespaces"], {"blocks": 1})
            self.assertEqual(report["ranks"], {})
            self.assertEqual(report["base_model"], "krea2")


if __name__ == "__main__":
    unittest.main()
