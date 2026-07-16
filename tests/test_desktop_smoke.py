from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtWidgets import QApplication

    from k2_region_lab.config import AppSettings, ModelDirectories
    from k2_region_lab.desktop.main_window import GLOBAL_SCOPE_ID, MainWindow


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


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 desktop dependency is not installed")
class DesktopSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    @staticmethod
    def make_window(root: Path) -> "MainWindow":
        return MainWindow(
            AppSettings(
                model_directories=ModelDirectories(root, root, root),
                data_directory=root,
                auto_start_worker=False,
            )
        )

    def test_region_prompt_move_resize_and_delete_stay_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.canvas.region_created.emit(
                "region-one", 16.0, 32.0, 256.0, 512.0
            )
            window.region_prompt.setPlainText("a translucent blue sculpture")
            item = window.canvas.region_item("region-one")
            item.set_scene_geometry(QRectF(48.0, 64.0, 320.0, 400.0))
            self.application.processEvents()

            self.assertEqual(len(window.regions), 1)
            self.assertEqual(window.regions[0].prompt, "a translucent blue sculpture")
            self.assertEqual(window.regions[0].box.x0, 48.0)
            self.assertEqual(window.regions[0].box.y1, 464.0)

            item.setSelected(True)
            window.canvas.delete_selected_regions()
            self.application.processEvents()
            self.assertEqual(window.regions, [])
            self.assertEqual(window.region_list.count(), 0)
            window.close()

    def test_lora_browser_model_defaults_global_and_allows_multiple_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "character_or_style.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("region-one", 0.0, 0.0, 128.0, 128.0)
            window.canvas.region_created.emit(
                "region-two", 256.0, 256.0, 512.0, 512.0
            )

            self.assertTrue(window._add_lora_path(lora_path))
            lora_id = window.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertTrue(window.lora_library.binding_for(lora_id).global_scope)
            self.assertEqual(
                window.lora_scope_list.item(0).data(Qt.ItemDataRole.UserRole),
                GLOBAL_SCOPE_ID,
            )

            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            window.lora_scope_list.item(2).setCheckState(Qt.CheckState.Checked)
            self.application.processEvents()
            binding = window.lora_library.binding_for(lora_id)
            self.assertFalse(binding.global_scope)
            self.assertEqual(binding.region_ids, ("region-one", "region-two"))

            self.assertTrue(window._add_lora_path(lora_path))
            self.assertEqual(window.lora_list.count(), 1)
            window.close()

    def test_deleting_last_assigned_box_returns_lora_to_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "regional.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("region-one", 0.0, 0.0, 128.0, 128.0)
            window._add_lora_path(lora_path)
            lora_id = window.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            self.assertFalse(window.lora_library.binding_for(lora_id).global_scope)

            item = window.canvas.region_item("region-one")
            item.setSelected(True)
            window.canvas.delete_selected_regions()
            self.application.processEvents()
            self.assertTrue(window.lora_library.binding_for(lora_id).global_scope)
            window.close()


if __name__ == "__main__":
    unittest.main()
