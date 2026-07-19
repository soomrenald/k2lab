from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QModelIndex, QRectF, Qt
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDockWidget,
        QFileDialog,
        QListWidgetItem,
        QMessageBox,
        QTextEdit,
    )

    from k2_region_lab.config import AppSettings, ModelDirectories
    from k2_region_lab.desktop.main_window import GLOBAL_SCOPE_ID, MainWindow
    from k2_region_lab.lora import CHARACTER_IDENTITY_LORA_ROUTING
    from k2_region_lab.processes import WorkerProcess
    from k2_region_lab.project import ProjectState
    from k2_region_lab.regional_prompting import PromptEmphasis
    from k2_region_lab.worker.protocol import CommandKind


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
            window.canvas.region_created.emit("region-one", 16.0, 32.0, 256.0, 512.0)
            window.region_face_identity_prompt.setPlainText(
                "sculptureface, a specific blue glass face"
            )
            window.region_prompt.setPlainText("a translucent blue sculpture")
            item = window.canvas.region_item("region-one")
            item.set_scene_geometry(QRectF(48.0, 64.0, 320.0, 400.0))
            self.application.processEvents()

            self.assertEqual(len(window.regions), 1)
            self.assertEqual(window.regions[0].prompt, "a translucent blue sculpture")
            self.assertEqual(
                window.regions[0].face_identity_prompt,
                "sculptureface, a specific blue glass face",
            )
            self.assertEqual(window.regions[0].box.x0, 48.0)
            self.assertEqual(window.regions[0].box.y1, 464.0)

            item.setSelected(True)
            window.canvas.delete_selected_regions()
            self.application.processEvents()
            self.assertEqual(window.regions, [])
            self.assertEqual(window.region_list.count(), 0)
            window.close()

    def test_model_and_generation_settings_share_one_tabbed_pane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))

            self.assertEqual(window.settings_tabs.count(), 4)
            self.assertEqual(window.settings_tabs.tabText(0), "Model & memory")
            self.assertEqual(window.settings_tabs.tabText(1), "Generation & spatial")
            self.assertEqual(window.settings_tabs.tabText(2), "Token emphasis")
            self.assertEqual(window.settings_tabs.tabText(3), "Projector")
            runtime_page = window.settings_tabs.widget(0)
            generation_page = window.settings_tabs.widget(1)
            self.assertFalse(runtime_page.isHidden())
            self.assertTrue(generation_page.isHidden())

            window.settings_tabs.setCurrentIndex(1)
            self.assertTrue(runtime_page.isHidden())
            self.assertFalse(generation_page.isHidden())

    def test_view_menu_can_hide_show_float_and_restore_every_dock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.show()
            self.application.processEvents()

            self.assertEqual(window.file_menu.title(), "&File")
            self.assertEqual(window.view_menu.title(), "&View")
            self.assertLess(
                window.menuBar().actions().index(window.file_menu.menuAction()),
                window.menuBar().actions().index(window.view_menu.menuAction()),
            )
            expected = {
                "prompt_regions_dock": "Prompt and regions",
                "model_settings_dock": "Model and generation settings",
                "lora_library_dock": "LoRA library and scope",
                "events_dock": "Events",
            }
            required_features = (
                QDockWidget.DockWidgetFeature.DockWidgetClosable
                | QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            )
            for dock in window._dock_widgets:
                action = window._dock_toggle_actions[dock.objectName()]
                self.assertEqual(action.text(), expected[dock.objectName()])
                self.assertTrue(action.isCheckable())
                self.assertEqual(dock.features() & required_features, required_features)

            events_action = window._dock_toggle_actions["events_dock"]
            events_action.trigger()
            self.application.processEvents()
            self.assertTrue(window.event_dock.isHidden())
            events_action.trigger()
            self.application.processEvents()
            self.assertFalse(window.event_dock.isHidden())

            window.event_dock.setFloating(True)
            window.prompt_dock.hide()
            window._restore_default_dock_layout()
            self.application.processEvents()
            self.assertFalse(window.event_dock.isFloating())
            self.assertFalse(window.prompt_dock.isHidden())
            self.assertEqual(
                window.dockWidgetArea(window.event_dock),
                Qt.DockWidgetArea.BottomDockWidgetArea,
            )
            window.close()

    def test_clear_canvas_image_keeps_regions_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "loaded.png"
            pixmap = QPixmap(256, 256)
            pixmap.fill(QColor("#334455"))
            self.assertTrue(pixmap.save(str(image_path)))
            window = self.make_window(root)
            self.assertTrue(window.canvas.set_image(str(image_path)))
            window._background_image_path = image_path
            window.global_prompt.setPlainText("keep this prompt")
            window.canvas.region_created.emit("subject", 16.0, 16.0, 128.0, 128.0)

            window.clear_canvas_button.click()

            self.assertIsNone(window.canvas._image_item)
            self.assertIsNone(window._background_image_path)
            self.assertEqual(window.global_prompt.toPlainText(), "keep this prompt")
            self.assertEqual(len(window.regions), 1)
            window.close()

    def test_new_project_restores_defaults_and_clears_project_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "loaded.png"
            pixmap = QPixmap(256, 256)
            pixmap.fill(QColor("#556677"))
            self.assertTrue(pixmap.save(str(image_path)))
            lora_path = root / "character.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.global_prompt.setPlainText("old project prompt")
            window.steps_input.setValue(17)
            window.seed_input.setValue(999)
            window.projector_enabled_input.setChecked(True)
            window.face_detail_denoise_input.setValue(0.4)
            window.post_upscale_input.setChecked(True)
            window.canvas.region_created.emit("subject", 16.0, 16.0, 128.0, 128.0)
            window._add_lora_path(lora_path)
            self.assertTrue(window.canvas.set_image(str(image_path)))
            window._background_image_path = image_path
            self.assertTrue(window._set_face_refinement_source(image_path))
            window._face_result_path = image_path
            window.face_result_input.setText(str(image_path))
            window.face_result_preview.set_image(image_path)
            window._current_project_path = root / "old.k2lab.json"

            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window._new_project()

            self.assertEqual(window.windowTitle(), "K2 Region Lab")
            self.assertIsNone(window._current_project_path)
            self.assertEqual(window.global_prompt.toPlainText(), "")
            self.assertEqual(window.steps_input.value(), 8)
            self.assertEqual(window.seed_input.value(), 0)
            self.assertFalse(window.projector_enabled_input.isChecked())
            self.assertEqual(window.face_detail_denoise_input.value(), 0.15)
            self.assertFalse(window.post_upscale_input.isChecked())
            self.assertEqual(window.regions, [])
            self.assertEqual(window.region_list.count(), 0)
            self.assertEqual(window.lora_list.count(), 0)
            self.assertIsNone(window.canvas._image_item)
            self.assertIsNone(window._background_image_path)
            self.assertIsNone(window._face_source_path)
            self.assertIsNone(window._face_result_path)
            self.assertTrue(window.face_source_preview._original_pixmap.isNull())
            self.assertTrue(window.face_result_preview._original_pixmap.isNull())
            window.close()

    def test_highlighted_prompt_text_creates_editable_emphasis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.global_prompt.setPlainText("two distinct people on a beach")
            cursor = window.global_prompt.textCursor()
            cursor.setPosition(4)
            cursor.setPosition(19, cursor.MoveMode.KeepAnchor)
            window.global_prompt.setTextCursor(cursor)
            window.emphasis_strength_input.setValue(0.6)

            window._add_prompt_emphasis("__global__", window.global_prompt)

            self.assertEqual(len(window.prompt_emphases), 1)
            emphasis = window.prompt_emphases[0]
            self.assertEqual(emphasis.phrase, "distinct people")
            self.assertEqual(emphasis.strength, 0.6)
            window.close()
            window.close()

    def test_projector_preset_autofills_twelve_vectors_and_marks_custom_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.projector_preset_input.setCurrentIndex(
                window.projector_preset_input.findData("filter_bypass3")
            )

            self.assertEqual(len(window.projector_vector_inputs), 12)
            self.assertEqual(window.projector_vector_inputs[8].value(), -0.5117)
            self.assertEqual(window.projector_vector_inputs[9].value(), -0.8906)
            self.assertEqual(window.projector_vector_inputs[10].value(), -0.6094)
            self.assertEqual(window.projector_vector_inputs[11].value(), 0.0)

            window.projector_vector_inputs[0].setValue(1.25)
            self.assertEqual(window.projector_preset_input.currentData(), "custom")
            window.close()

    def test_unified_prompt_preview_is_resizable_and_contains_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.global_prompt.setPlainText("a detailed landscape")

            with patch.object(QDialog, "exec", return_value=0):
                window._preview_unified_prompt()

            preview = window._prompt_preview_dialog
            self.assertIsNotNone(preview)
            self.assertTrue(preview.isSizeGripEnabled())
            self.assertGreaterEqual(preview.minimumWidth(), 520)
            self.assertGreaterEqual(preview.minimumHeight(), 360)
            prompt_views = preview.findChildren(QTextEdit)
            self.assertEqual(len(prompt_views), 1)
            self.assertIn("a detailed landscape", prompt_views[0].toPlainText())
            window.close()

    def test_region_rows_drag_reorders_depth_priority_and_canvas_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.canvas.region_created.emit("back", 20.0, 20.0, 200.0, 240.0)
            window.canvas.region_created.emit("middle", 40.0, 40.0, 220.0, 260.0)
            window.canvas.region_created.emit("front", 60.0, 60.0, 240.0, 280.0)

            moved = window.region_list.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)
            self.application.processEvents()

            self.assertTrue(moved)
            self.assertEqual(
                [region.region_id for region in window.regions],
                ["front", "back", "middle"],
            )
            self.assertEqual([region.priority for region in window.regions], [3, 2, 1])
            self.assertGreater(
                window.canvas.region_item("front").zValue(),
                window.canvas.region_item("back").zValue(),
            )
            self.assertEqual(
                [region.region_id for region in window._project_state().regions],
                ["front", "back", "middle"],
            )
            window.close()

    def test_project_dialogs_start_in_repository_prompt_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            with patch.object(QFileDialog, "getOpenFileName", return_value=("", "")) as open_dialog:
                window._open_project()
            self.assertEqual(Path(open_dialog.call_args.args[2]), window._project_directory)

            with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")) as save_dialog:
                window._save_project_as()
            self.assertEqual(
                Path(save_dialog.call_args.args[2]).parent,
                window._project_directory,
            )
            window.close()

    def test_legacy_default_output_folder_migrates_to_repository_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = self.make_window(root)
            state = ProjectState(
                1024,
                1024,
                runtime={
                    "data_directory": str(root),
                    "output_directory": str(root / "baseline_outputs"),
                },
            )

            settings = window._settings_from_project(state)

            self.assertEqual(settings.output_directory, window._output_directory)
            self.assertEqual(settings.output_directory.name, "outputs")
            window.close()

    def test_lora_browser_model_defaults_global_and_allows_multiple_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "character_or_style.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("region-one", 0.0, 0.0, 128.0, 128.0)
            window.canvas.region_created.emit("region-two", 256.0, 256.0, 512.0, 512.0)

            self.assertTrue(window._add_lora_path(lora_path))
            lora_id = window.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertTrue(window.lora_library.binding_for(lora_id).global_scope)
            window.lora_strength_input.setValue(0.75)
            self.assertEqual(window.lora_library.binding_for(lora_id).strength, 0.75)
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
            second_lora_id = window.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertEqual(window.lora_list.count(), 2)
            self.assertNotEqual(second_lora_id, lora_id)
            self.assertEqual(
                window.lora_library.get(second_lora_id).display_name,
                "character_or_style #2",
            )
            second_binding = window.lora_library.binding_for(second_lora_id)
            self.assertTrue(second_binding.global_scope)
            self.assertEqual(second_binding.strength, 1.0)

            window.lora_strength_input.setValue(1.5)
            window.lora_scope_list.item(2).setCheckState(Qt.CheckState.Checked)
            self.application.processEvents()
            self.assertEqual(
                window.lora_library.binding_for(lora_id).region_ids,
                ("region-one", "region-two"),
            )
            self.assertEqual(window.lora_library.binding_for(lora_id).strength, 0.75)
            second_binding = window.lora_library.binding_for(second_lora_id)
            self.assertEqual(second_binding.region_ids, ("region-two",))
            self.assertEqual(second_binding.strength, 1.5)

            payload = window._lora_payload()
            self.assertEqual([item["path"] for item in payload], [str(lora_path)] * 2)
            self.assertNotEqual(payload[0]["id"], payload[1]["id"])
            self.assertEqual(payload[0]["region_ids"], ["region-one", "region-two"])
            self.assertEqual(payload[1]["region_ids"], ["region-two"])
            window.close()

    def test_duplicate_lora_instances_survive_project_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "identity.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("left", 0.0, 0.0, 128.0, 128.0)
            window.canvas.region_created.emit("right", 256.0, 0.0, 384.0, 128.0)

            window._add_lora_path(lora_path)
            window.lora_strength_input.setValue(0.5)
            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            window._add_lora_path(lora_path)
            window.lora_strength_input.setValue(1.5)
            window.lora_scope_list.item(2).setCheckState(Qt.CheckState.Checked)
            project_path = root / "duplicate-lora.k2lab.json"
            self.assertTrue(window._save_project_to(project_path))
            window.close()

            restored = self.make_window(root)
            self.assertTrue(restored._load_project_from(project_path))
            payload = restored._lora_payload()
            self.assertEqual(restored.lora_list.count(), 2)
            self.assertEqual([item["path"] for item in payload], [str(lora_path)] * 2)
            self.assertEqual(
                [(item["strength"], item["region_ids"]) for item in payload],
                [(0.5, ["left"]), (1.5, ["right"])],
            )
            self.assertEqual(
                [restored.lora_list.item(index).text().split("  ")[0] for index in range(2)],
                ["identity", "identity #2"],
            )
            restored.close()

    def test_character_identity_routing_controls_payload_and_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "identity_model.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.canvas.region_created.emit("person", 0.0, 0.0, 256.0, 512.0)
            window.region_face_identity_prompt.setPlainText(
                "lface, a specific adult woman with an oval face"
            )
            window.region_prompt.setPlainText("lface, an adult woman")
            window._add_lora_path(lora_path)

            window.lora_trigger_input.setText("lface")
            window._lora_trigger_edited()
            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            window.lora_routing_mode_input.setCurrentIndex(
                window.lora_routing_mode_input.findData(
                    CHARACTER_IDENTITY_LORA_ROUTING
                )
            )

            payload = window._lora_payload()[0]
            self.assertEqual(
                payload["routing_mode"], CHARACTER_IDENTITY_LORA_ROUTING
            )
            self.assertEqual(payload["trigger_phrase"], "lface")
            self.assertEqual(payload["region_ids"], ["person"])
            self.assertIn("[identity: lface]", window.lora_list.item(0).text())
            project_path = root / "identity-routing.k2lab.json"
            self.assertTrue(window._save_project_to(project_path))
            window.close()

            restored = self.make_window(root)
            self.assertTrue(restored._load_project_from(project_path))
            restored_payload = restored._lora_payload()[0]
            self.assertEqual(
                restored_payload["routing_mode"],
                CHARACTER_IDENTITY_LORA_ROUTING,
            )
            self.assertEqual(restored_payload["trigger_phrase"], "lface")
            self.assertEqual(
                restored.lora_routing_mode_input.currentData(),
                CHARACTER_IDENTITY_LORA_ROUTING,
            )
            self.assertEqual(restored.lora_trigger_input.text(), "lface")
            restored.close()

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
            window.sampler_input.setCurrentIndex(
                window.sampler_input.findData("dpmpp_2m")
            )
            window.scheduler_input.setCurrentIndex(
                window.scheduler_input.findData("karras")
            )
            window.seed_input.setValue(42)
            window.seed_mode_input.setCurrentIndex(window.seed_mode_input.findData("increment"))
            window.regional_prompting_input.setChecked(True)
            window.regional_prompt_strength_input.setValue(1.7)
            window.regional_outside_penalty_input.setValue(1.2)
            window.regional_feather_input.setValue(48)
            window.regional_subject_competition_input.setChecked(False)
            window.regional_subject_fill_input.setChecked(False)
            window.regional_relaxation_input.setChecked(False)
            window.regional_late_step_scale_input.setValue(0.8)
            window.regional_lora_delta_adaptation_input.setChecked(True)
            window.regional_lora_delta_adaptation_gain_input.setValue(0.6)
            window.projector_enabled_input.setChecked(True)
            window.projector_preset_input.setCurrentIndex(
                window.projector_preset_input.findData("filter_bypass3")
            )
            window.projector_multiplier_input.setValue(3.5)
            window.projector_identity_protection_input.setValue(0.7)
            window.face_detail_seed_input.setValue(123)
            window.face_detail_steps_input.setValue(10)
            window.face_detail_denoise_input.setValue(0.25)
            window.face_detail_crop_size_input.setCurrentIndex(
                window.face_detail_crop_size_input.findData(768)
            )
            window.face_detail_padding_input.setValue(1.8)
            window.face_detail_feather_input.setValue(0.16)
            window.face_detail_blend_input.setValue(0.4)
            window.face_detail_lora_scale_input.setValue(1.2)
            window.face_detail_detector_threshold_input.setValue(0.35)
            upscale_path = root / "4x-upscaler.pth"
            upscale_path.write_bytes(b"test upscaler placeholder")
            window.post_upscale_input.setChecked(True)
            window.upscale_scale_input.setCurrentIndex(window.upscale_scale_input.findData(4))
            window.upscale_method_input.setCurrentIndex(
                window.upscale_method_input.findData("model")
            )
            window._upscale_model_path = upscale_path
            window.upscale_model_input.setText(str(upscale_path))
            window.memory_policy_input.setCurrentIndex(
                window.memory_policy_input.findData("balanced")
            )
            window.reserve_vram_input.setValue(3.5)
            window.minimum_ram_input.setValue(13.0)
            window.cpu_vae_input.setChecked(True)
            output_directory = root / "custom renders"
            window._output_directory = output_directory
            window.output_directory_input.setText(str(output_directory))
            window.filename_prefix_input.setText("beach-study")
            window.canvas.region_created.emit("subject", 32.0, 48.0, 320.0, 480.0)
            window.region_name.setText("Main subject")
            window._region_name_edited()
            window.region_role.setCurrentIndex(window.region_role.findData("subject"))
            window.region_face_identity_prompt.setPlainText(
                "personface, a specific person with an oval face"
            )
            window.region_prompt.setPlainText("a person by the water")
            window.prompt_emphases = [PromptEmphasis("subject", "person", strength=0.4)]
            window._refresh_prompt_emphases()
            window._add_lora_path(lora_path)
            window.lora_strength_input.setValue(0.6)
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
            self.assertEqual(restored.sampler_input.currentData(), "dpmpp_2m")
            self.assertEqual(restored.scheduler_input.currentData(), "karras")
            self.assertEqual(restored.seed_input.value(), 42)
            self.assertEqual(restored.seed_mode_input.currentData(), "increment")
            self.assertTrue(restored.regional_prompting_input.isChecked())
            self.assertEqual(restored.regional_prompt_strength_input.value(), 1.7)
            self.assertEqual(restored.regional_outside_penalty_input.value(), 1.2)
            self.assertEqual(restored.regional_feather_input.value(), 48)
            self.assertFalse(restored.regional_subject_competition_input.isChecked())
            self.assertFalse(restored.regional_subject_fill_input.isChecked())
            self.assertFalse(restored.regional_relaxation_input.isChecked())
            self.assertEqual(restored.regional_late_step_scale_input.value(), 0.8)
            self.assertTrue(restored.regional_lora_delta_adaptation_input.isChecked())
            self.assertEqual(restored.regional_lora_delta_adaptation_gain_input.value(), 0.6)
            self.assertTrue(restored.projector_enabled_input.isChecked())
            self.assertEqual(restored.projector_preset_input.currentData(), "filter_bypass3")
            self.assertEqual(restored.projector_vector_inputs[10].value(), -0.6094)
            self.assertEqual(restored.projector_multiplier_input.value(), 3.5)
            self.assertEqual(
                restored.projector_identity_protection_input.value(), 0.7
            )
            self.assertEqual(restored.face_detail_seed_input.value(), 123)
            self.assertEqual(restored.face_detail_steps_input.value(), 10)
            self.assertEqual(restored.face_detail_denoise_input.value(), 0.25)
            self.assertEqual(restored.face_detail_crop_size_input.currentData(), 768)
            self.assertEqual(restored.face_detail_padding_input.value(), 1.8)
            self.assertEqual(restored.face_detail_feather_input.value(), 0.16)
            self.assertEqual(restored.face_detail_blend_input.value(), 0.4)
            self.assertEqual(restored.face_detail_lora_scale_input.value(), 1.2)
            self.assertEqual(restored.face_detail_detector_threshold_input.value(), 0.35)
            self.assertTrue(restored.post_upscale_input.isChecked())
            self.assertEqual(restored.upscale_scale_input.currentData(), 4)
            self.assertEqual(restored.upscale_method_input.currentData(), "model")
            self.assertEqual(restored._upscale_model_path, upscale_path)
            self.assertEqual(restored.memory_policy_input.currentData(), "balanced")
            self.assertEqual(restored.reserve_vram_input.value(), 3.5)
            self.assertEqual(restored.minimum_ram_input.value(), 13.0)
            self.assertTrue(restored.cpu_vae_input.isChecked())
            self.assertEqual(restored._output_directory, output_directory)
            self.assertEqual(restored.filename_prefix_input.text(), "beach-study")
            self.assertEqual(restored.regions[0].name, "Main subject")
            self.assertEqual(restored.regions[0].spatial_role, "subject")
            self.assertEqual(restored.regions[0].prompt, "a person by the water")
            self.assertEqual(
                restored.regions[0].face_identity_prompt,
                "personface, a specific person with an oval face",
            )
            self.assertFalse(hasattr(restored, "region_negative_prompt"))
            self.assertEqual(restored.prompt_emphases[0].phrase, "person")
            self.assertEqual(restored.prompt_emphases[0].strength, 0.4)
            self.assertEqual(restored.lora_list.count(), 1)
            lora_id = restored.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertEqual(
                restored.lora_library.binding_for(lora_id).region_ids,
                ("subject",),
            )
            self.assertEqual(restored.lora_library.binding_for(lora_id).strength, 0.6)
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

    def test_cuda_accelerator_is_identified_in_worker_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window._worker_event(
                {
                    "state": "unloaded",
                    "message": "Worker runtime probe complete",
                    "payload": {
                        "accelerator_available": True,
                        "accelerator_backend": "cuda",
                        "python_executable": "/cuda/python",
                        "torch_version": "2.9.0",
                        "cuda_version": "12.8",
                        "devices": [{"name": "NVIDIA GPU"}],
                    },
                }
            )
            messages = [
                window.events.item(row).text() for row in range(window.events.count())
            ]
            self.assertTrue(any("CUDA 12.8" in message for message in messages))
            self.assertEqual(window.accelerator_status.text(), "NVIDIA GPU")
            window.close()

    def test_worker_memory_event_updates_live_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window._worker_event(
                {
                    "state": "running",
                    "message": "Memory check: before VAE decode",
                    "payload": {
                        "memory": {
                            "gpu_free_bytes": 4 * 1024**3,
                            "gpu_total_bytes": 16 * 1024**3,
                            "ram_available_bytes": 32 * 1024**3,
                            "action": "offloaded_to_ram",
                        }
                    },
                }
            )
            self.assertIn("VRAM 4.0/16.0 GiB free", window.memory_status.text())
            self.assertIn("offloaded_to_ram", window.memory_status.text())
            window.close()

    def test_oom_event_surfaces_gpu_size_independent_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window._worker_event(
                {
                    "state": "error",
                    "message": "critical GPU memory pressure after denoising step 1/8",
                    "payload": {"exception_type": "CriticalGpuMemoryPressure"},
                }
            )
            messages = [window.events.item(row).text() for row in range(window.events.count())]
            self.assertTrue(any("GPU memory guidance" in message for message in messages))
            self.assertIn("exceeded available GPU memory", window.statusBar().currentMessage())
            window.close()

    def test_safe_worker_payload_enforces_memory_floors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.memory_policy_input.setCurrentIndex(
                window.memory_policy_input.findData("safe_16gb")
            )
            window.reserve_vram_input.setValue(2.0)
            window.minimum_ram_input.setValue(12.0)

            payload = window._worker_payload()

            self.assertEqual(payload["reserve_vram_gb"], 4.0)
            self.assertEqual(payload["minimum_system_ram_gb"], 14.0)
            window.close()

    def test_application_worker_python_overrides_saved_project_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "rocm7" / "bin" / "python"
            window = self.make_window(root)
            window.settings = AppSettings(
                model_directories=window.settings.model_directories,
                data_directory=root,
                worker_python=selected,
                auto_start_worker=False,
            )
            state = ProjectState(
                1024,
                1024,
                runtime={"worker_python": str(root / "rocm6" / "bin" / "python")},
            )

            with patch.dict(os.environ, {}, clear=True):
                restored = window._settings_from_project(state)

            self.assertEqual(restored.worker_python, selected)
            window.close()

    def test_release_gpu_memory_only_targets_discovered_k2_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            worker = WorkerProcess(
                8123,
                100,
                ("python", "-m", "k2_region_lab.worker.entrypoint"),
            )
            window.worker_client.kill_immediately = Mock(return_value=None)
            with (
                patch(
                    "k2_region_lab.desktop.main_window.find_owned_k2_workers",
                    side_effect=[(worker,), (worker,)],
                ),
                patch(
                    "k2_region_lab.desktop.main_window.terminate_workers",
                    return_value=((8123,), ()),
                ) as terminate,
                patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.Yes,
                ),
            ):
                window._release_k2_gpu_memory()

            terminate.assert_called_once_with((worker,))
            self.assertEqual(window.worker_status.text(), "Stopped")
            self.assertIn("allocations released", window.memory_status.text())
            window.close()

    def test_baseline_request_uses_aligned_canvas_and_result_becomes_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = self.make_window(root)
            window.artifacts = Mock(complete=True)
            window.global_prompt.setPlainText("a red ceramic teapot")
            window.width_input.setValue(513)
            window.height_input.setValue(517)
            window.steps_input.setValue(8)
            window.seed_input.setValue(1234)
            window._output_directory = root / "renders"
            window.filename_prefix_input.setText("teapot-test")
            lora_path = root / "global-style.safetensors"
            write_lora(lora_path)
            window._add_lora_path(lora_path)
            window.lora_strength_input.setValue(0.8)
            window.canvas.region_created.emit("teapot-region", 16.0, 16.0, 256.0, 256.0)
            window.region_prompt.setPlainText("a detailed red teapot")
            window.worker_client.send = Mock()
            window._accelerator_available = True
            window._models_compatible = True
            window._model_loaded = True

            with patch.object(
                type(window.worker_client),
                "running",
                new_callable=PropertyMock,
                return_value=True,
            ):
                window._generate_baseline()
            command, payload = window.worker_client.send.call_args.args
            self.assertEqual(command.value, "generate_baseline")
            self.assertEqual(payload["prompt"], "a red ceramic teapot")
            self.assertEqual(payload["width"] % 16, 0)
            self.assertEqual(payload["height"] % 16, 0)
            self.assertEqual(payload["steps"], 8)
            self.assertEqual(payload["sampler"], "euler")
            self.assertEqual(payload["scheduler"], "simple")
            self.assertEqual(payload["seed"], 1234)
            self.assertEqual(payload["output_directory"], str(root / "renders"))
            self.assertEqual(payload["filename_prefix"], "teapot-test")
            self.assertTrue(payload["regional_prompting"])
            self.assertEqual(payload["regional_feather_pixels"], 128)
            self.assertEqual(payload["regional_outside_penalty"], 1.0)
            self.assertTrue(payload["regional_subject_competition"])
            self.assertTrue(payload["regional_subject_fill"])
            self.assertEqual(payload["regional_late_step_scale"], 0.35)
            self.assertFalse(payload["regional_lora_delta_adaptation"])
            self.assertEqual(payload["regional_lora_delta_adaptation_gain"], 0.35)
            self.assertEqual(payload["prompt_emphases"], [])
            self.assertFalse(payload["projector_enabled"])
            self.assertEqual(payload["projector_preset"], "filter_bypass2")
            self.assertEqual(len(payload["projector_values"]), 12)
            self.assertEqual(payload["projector_multiplier"], 1.0)
            self.assertEqual(payload["projector_identity_protection"], 1.0)
            self.assertFalse(any(key.startswith("face_detail_") for key in payload))
            self.assertFalse(payload["post_upscale"])
            self.assertEqual(payload["upscale_scale"], 2)
            self.assertEqual(payload["upscale_method"], "lanczos")
            self.assertIsNone(payload["upscale_model_path"])
            self.assertEqual(payload["regions"][0]["id"], "teapot-region")
            self.assertEqual(payload["regions"][0]["spatial_role"], "auto")
            self.assertEqual(len(payload["loras"]), 1)
            self.assertTrue(payload["loras"][0]["global"])
            self.assertEqual(payload["loras"][0]["strength"], 0.8)
            self.assertEqual(payload["loras"][0]["path"], str(lora_path.resolve()))
            self.assertEqual(payload["regions"][0]["prompt"], "a detailed red teapot")
            self.assertNotIn("negative_prompt", payload["regions"][0])
            self.assertEqual(payload["regions"][0]["face_identity_prompt"], "")
            self.assertEqual(payload["project_json"]["schema"], "k2-region-lab-project")
            self.assertEqual(payload["project_json"]["generation"]["seed"], 1234)

            image_path = root / "baseline.png"
            pixmap = QPixmap(32, 32)
            pixmap.fill(QColor("#b32318"))
            self.assertTrue(pixmap.save(str(image_path)))
            window._worker_event(
                {
                    "state": "ready",
                    "message": "Generation complete",
                    "payload": {"image_path": str(image_path)},
                }
            )
            window._worker_process_status("stopped (exit 0, NormalExit)")
            self.assertIsNotNone(window.canvas._image_item)
            self.assertEqual(window.canvas._image_item.zValue(), -100.0)
            self.assertTrue(window.generate_button.isEnabled())
            window.close()

    def test_face_refinement_is_a_separate_png_worker_request_with_two_previews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "first-pass.png"
            source_pixmap = QPixmap(512, 512)
            source_pixmap.fill(QColor("#334455"))
            self.assertTrue(source_pixmap.save(str(source)))
            lora_path = root / "character.safetensors"
            write_lora(lora_path)
            window = self.make_window(root)
            window.artifacts = Mock(complete=True)
            window.canvas.region_created.emit("subject", 32.0, 64.0, 320.0, 448.0)
            window.region_prompt.setPlainText("a named person")
            window._add_lora_path(lora_path)
            window.lora_scope_list.item(1).setCheckState(Qt.CheckState.Checked)
            self.assertTrue(window._set_face_refinement_source(source))
            window._face_detections = [
                {
                    "index": 0,
                    "box": [40.0, 80.0, 100.0, 150.0],
                    "score": 0.9,
                    "region_id": window.regions[0].region_id,
                    "region_name": window.regions[0].name,
                }
            ]
            face_item = QListWidgetItem("Face 1")
            face_item.setData(Qt.ItemDataRole.UserRole, 0)
            face_item.setFlags(face_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            face_item.setCheckState(Qt.CheckState.Checked)
            window.face_selection_list.addItem(face_item)
            window.face_detail_seed_input.setValue(77)
            window.face_detail_blend_input.setValue(0.35)
            window.worker_client.send = Mock()
            window._accelerator_available = True
            window._models_compatible = True
            window._model_loaded = True

            with patch.object(
                type(window.worker_client),
                "running",
                new_callable=PropertyMock,
                return_value=True,
            ):
                window._run_face_refinement()

            command, payload = window.worker_client.send.call_args.args
            self.assertEqual(command, CommandKind.REFINE_FACES)
            self.assertEqual(payload["image_path"], str(source.resolve()))
            self.assertEqual(payload["output_directory"], str(root.resolve()))
            self.assertEqual(payload["seed"], 77)
            self.assertEqual(payload["blend"], 0.35)
            self.assertEqual(payload["crop_size"], 512)
            self.assertEqual(payload["selected_face_indices"], [0])
            self.assertEqual(payload["project_json"]["schema"], "k2-region-lab-project")
            self.assertEqual(payload["regions"][0]["box"]["x0"], 16.0)
            self.assertFalse(payload["loras"][0]["global"])

            result = root / "first-pass_face_refined_test.png"
            result_pixmap = QPixmap(512, 512)
            result_pixmap.fill(QColor("#556677"))
            self.assertTrue(result_pixmap.save(str(result)))
            window._worker_event(
                {
                    "state": "ready",
                    "message": "Face refinement complete",
                    "payload": {"image_path": str(result)},
                }
            )
            self.assertEqual(window._face_source_path, source.resolve())
            self.assertEqual(window._face_result_path, result)
            self.assertFalse(window.face_source_preview.pixmap().isNull())
            self.assertFalse(window.face_result_preview.pixmap().isNull())
            window.close()

    def test_generate_bootstraps_fresh_worker_and_release_restores_button(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.artifacts = Mock(complete=True)
            window.worker_client.send = Mock()

            with patch.object(
                type(window.worker_client),
                "running",
                new_callable=PropertyMock,
                return_value=True,
            ):
                window._generate_baseline()
                self.assertEqual(window.worker_client.send.call_args.args[0], CommandKind.PROBE)

                window._worker_event(
                    {
                        "state": "unloaded",
                        "message": "Worker runtime probe complete",
                        "payload": {"accelerator_available": True, "devices": []},
                    }
                )
                self.assertEqual(
                    window.worker_client.send.call_args.args[0],
                    CommandKind.VALIDATE_MODELS,
                )

                window._worker_event(
                    {
                        "state": "ready",
                        "message": "Model artifacts validated",
                        "payload": {"complete": True, "manifests": []},
                    }
                )
                self.assertEqual(
                    window.worker_client.send.call_args.args[0], CommandKind.LOAD_MODEL
                )

                window._worker_event(
                    {
                        "state": "ready",
                        "message": "Krea 2 baseline components loaded",
                        "payload": {},
                    }
                )
                command, payload = window.worker_client.send.call_args.args
                self.assertEqual(command, CommandKind.GENERATE_BASELINE)
                self.assertIn("prompt", payload)

                window._worker_event(
                    {
                        "state": "ready",
                        "message": "Generation complete",
                        "payload": {},
                    }
                )
                window._worker_process_status("stopped (exit 0, NormalExit)")

            self.assertFalse(window._model_loaded)
            self.assertIsNone(window._pending_generation_payload)
            self.assertEqual(window.memory_status.text(), "GPU and worker RAM released")
            self.assertTrue(window.generate_button.isEnabled())
            window.close()

    def test_random_and_increment_seed_modes_select_the_dispatched_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.worker_client.send = Mock()
            window._accelerator_available = True
            window._models_compatible = True
            window._model_loaded = True
            window.seed_mode_input.setCurrentIndex(window.seed_mode_input.findData("random"))
            with patch.object(
                type(window.worker_client),
                "running",
                new_callable=PropertyMock,
                return_value=True,
            ):
                with patch(
                    "k2_region_lab.desktop.main_window.secrets.randbelow",
                    return_value=9876,
                ):
                    window._generate_baseline()
                random_payload = window.worker_client.send.call_args.args[1]
                self.assertEqual(random_payload["seed"], 9876)
                self.assertEqual(random_payload["seed_mode"], "random")
                self.assertEqual(window.seed_input.value(), 9876)

                window.worker_client.send.reset_mock()
                window.seed_mode_input.setCurrentIndex(window.seed_mode_input.findData("increment"))
                window.seed_input.setValue(41)
                window._generate_baseline()
                increment_payload = window.worker_client.send.call_args.args[1]
                self.assertEqual(increment_payload["seed"], 41)
                self.assertEqual(increment_payload["seed_mode"], "increment")
                self.assertEqual(window.seed_input.value(), 42)
            window.close()

    def test_stop_generation_terminates_only_worker_and_resets_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.worker_client.cancel_generation = Mock(return_value=4321)
            window._generation_active = True

            window._stop_generation()

            window.worker_client.cancel_generation.assert_called_once()
            self.assertFalse(window._generation_active)
            self.assertFalse(window.generate_button.isEnabled())
            self.assertFalse(window.stop_generation_button.isEnabled())
            self.assertTrue(window.memory_policy_input.isEnabled())
            messages = [window.events.item(index).text() for index in range(window.events.count())]
            self.assertTrue(any("worker PID 4321" in message for message in messages))
            window.close()

    def test_event_view_follows_only_when_already_at_latest_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.show()
            window.events.setMinimumHeight(80)
            for index in range(100):
                window.events.addItem(f"Event {index}")
            self.application.processEvents()

            scrollbar = window.events.verticalScrollBar()
            window.events.scrollToBottom()
            self.application.processEvents()
            window.events.addItem("Followed event")
            self.application.processEvents()
            self.assertEqual(scrollbar.value(), scrollbar.maximum())

            scrollbar.setValue(max(scrollbar.minimum(), scrollbar.maximum() - 20))
            previous_position = scrollbar.value()
            window.events.addItem("Background event")
            self.application.processEvents()
            self.assertEqual(scrollbar.value(), previous_position)
            window.close()


if __name__ == "__main__":
    unittest.main()
