from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from k2_region_lab.lora import LoraLibrary


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


if __name__ == "__main__":
    unittest.main()
