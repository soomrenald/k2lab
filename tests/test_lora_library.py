from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.lora import (
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

    def test_duplicate_path_is_not_loaded_twice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "style.safetensors"
            write_lora(path)
            library = LoraLibrary()
            first = library.add(path)
            second = library.add(path)
            self.assertEqual(first.lora_id, second.lora_id)
            self.assertEqual(len(library.entries()), 1)

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
            self.assertEqual(
                library.binding_for(entry.lora_id).region_ids, ("subject",)
            )

    def test_krea_ai_toolkit_namespace_is_normalized_without_touching_other_keys(self) -> None:
        self.assertEqual(
            normalize_krea_lora_key(
                "diffusion_model.blocks.0.attn.wq.lora_A.weight"
            ),
            "blocks.0.attn.wq.lora_A.weight",
        )
        self.assertEqual(
            normalize_krea_lora_key("transformer.transformer_blocks.0.attn.to_q.alpha"),
            "transformer.transformer_blocks.0.attn.to_q.alpha",
        )
        normalized = normalize_krea_lora_state_dict(
            {"diffusion_model.txtfusion.refiner_blocks.0.attn.wq.alpha": 32}
        )
        self.assertEqual(
            normalized, {"txtfusion.refiner_blocks.0.attn.wq.alpha": 32}
        )
        original = {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": "a",
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": "b",
        }
        self.assertEqual(
            align_krea_lora_state_dict(
                original, {"diffusion_model.blocks.0.attn.wq"}
            ),
            original,
        )
        self.assertEqual(
            align_krea_lora_state_dict(original, {"blocks.0.attn.wq"}),
            {
                "blocks.0.attn.wq.lora_A.weight": "a",
                "blocks.0.attn.wq.lora_B.weight": "b",
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

if __name__ == "__main__":
    unittest.main()
