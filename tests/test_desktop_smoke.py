from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPixmap
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

    def test_region_names_are_editable_unique_and_propagate_to_lora_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "named.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("region-one", 0.0, 0.0, 128.0, 128.0)
            window.canvas.region_created.emit("region-two", 256.0, 256.0, 512.0, 512.0)
            window._add_lora_path(lora_path)

            window.region_list.setCurrentRow(0)
            window.region_name.setText("Beach subject")
            window._region_name_edited()
            self.assertEqual(window.regions[0].name, "Beach subject")
            self.assertEqual(window.lora_scope_list.item(1).text(), "Beach subject")
            self.assertEqual(
                window.canvas.region_item("region-one")._label.text(),
                "Beach subject",
            )

            window.region_list.setCurrentRow(1)
            window.region_name.setText("Beach subject")
            window._region_name_edited()
            self.assertEqual(window.regions[1].name, "Region 2")
            self.assertEqual(window.region_name.text(), "Region 2")
            window.close()

    def test_project_save_and_load_restores_prompts_boxes_loras_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "project_lora.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.global_prompt.setPlainText("a sunny beach")
            window.width_input.setValue(768)
            window.height_input.setValue(512)
            window.canvas.set_canvas_size(768, 512)
            window.steps_input.setValue(6)
            window.seed_input.setValue(42)
            window.canvas.region_created.emit("subject", 32.0, 48.0, 320.0, 480.0)
            window.region_name.setText("Main subject")
            window._region_name_edited()
            window.region_prompt.setPlainText("a person by the water")
            window._add_lora_path(lora_path)
            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            project_path = root / "beach.k2lab.json"
            self.assertTrue(window._save_project_to(project_path))
            window.close()

            restored = self.make_window(root)
            self.assertTrue(restored._load_project_from(project_path))
            self.assertEqual(restored.global_prompt.toPlainText(), "a sunny beach")
            self.assertEqual(restored.width_input.value(), 768)
            self.assertEqual(restored.height_input.value(), 512)
            self.assertEqual(restored.steps_input.value(), 6)
            self.assertEqual(restored.seed_input.value(), 42)
            self.assertEqual(restored.regions[0].name, "Main subject")
            self.assertEqual(restored.regions[0].prompt, "a person by the water")
            self.assertEqual(restored.lora_list.count(), 1)
            lora_id = restored.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertEqual(
                restored.lora_library.binding_for(lora_id).region_ids,
                ("subject",),
            )
            restored.close()

    def test_unavailable_accelerator_exposes_diagnostic_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window._worker_event(
                {
                    "state": "unloaded",
                    "message": "Worker runtime probe complete",
                    "payload": {
                        "accelerator_available": False,
                        "python_executable": "/worker/python",
                        "torch_version": "2.9.1",
                        "hip_version": "6.4",
                        "devices": [],
                    },
                }
            )
            self.assertFalse(window.diagnostic_button.isHidden())
            self.assertIn("run diagnostic", window.accelerator_status.text())
            window.close()

    def test_baseline_request_uses_aligned_canvas_and_result_becomes_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = self.make_window(root)
            window.global_prompt.setPlainText("a red ceramic teapot")
            window.width_input.setValue(513)
            window.height_input.setValue(517)
            window.steps_input.setValue(8)
            window.seed_input.setValue(1234)
            window.worker_client.send = Mock()

            window._generate_baseline()
            command, payload = window.worker_client.send.call_args.args
            self.assertEqual(command.value, "generate_baseline")
            self.assertEqual(payload["prompt"], "a red ceramic teapot")
            self.assertEqual(payload["width"] % 16, 0)
            self.assertEqual(payload["height"] % 16, 0)
            self.assertEqual(payload["steps"], 8)
            self.assertEqual(payload["seed"], 1234)

            image_path = root / "baseline.png"
            pixmap = QPixmap(32, 32)
            pixmap.fill(QColor("#b32318"))
            self.assertTrue(pixmap.save(str(image_path)))
            window._worker_event(
                {
                    "state": "ready",
                    "message": "Baseline generation complete",
                    "payload": {"image_path": str(image_path)},
                }
            )
            self.assertIsNotNone(window.canvas._image_item)
            self.assertEqual(window.canvas._image_item.zValue(), -100.0)
            self.assertTrue(window.generate_button.isEnabled())
            window.close()


if __name__ == "__main__":
    unittest.main()
