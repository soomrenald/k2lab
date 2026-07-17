from __future__ import annotations

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None

if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPixmap
    from PySide6.QtWidgets import QApplication, QMessageBox

    from k2_region_lab.config import AppSettings, ModelDirectories
    from k2_region_lab.desktop.main_window import GLOBAL_SCOPE_ID, MainWindow
    from k2_region_lab.processes import WorkerProcess
    from k2_region_lab.project import ProjectState


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
            window.seed_mode_input.setCurrentIndex(
                window.seed_mode_input.findData("increment")
            )
            window.regional_prompting_input.setChecked(True)
            window.regional_prompt_strength_input.setValue(1.7)
            window.regional_outside_penalty_input.setValue(1.2)
            window.regional_feather_input.setValue(48)
            window.regional_subject_competition_input.setChecked(False)
            window.regional_relaxation_input.setChecked(False)
            window.regional_refinement_input.setChecked(True)
            window.refinement_scale_input.setValue(1.7)
            window.refinement_steps_input.setValue(5)
            window.refinement_denoise_input.setValue(0.30)
            window.refinement_feather_input.setValue(56)
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
            window.region_role.setCurrentIndex(
                window.region_role.findData("subject")
            )
            window.region_prompt.setPlainText("a person by the water")
            window.region_negative_prompt.setPlainText("blurry face")
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
            self.assertEqual(restored.seed_input.value(), 42)
            self.assertEqual(restored.seed_mode_input.currentData(), "increment")
            self.assertTrue(restored.regional_prompting_input.isChecked())
            self.assertEqual(restored.regional_prompt_strength_input.value(), 1.7)
            self.assertEqual(restored.regional_outside_penalty_input.value(), 1.2)
            self.assertEqual(restored.regional_feather_input.value(), 48)
            self.assertFalse(
                restored.regional_subject_competition_input.isChecked()
            )
            self.assertFalse(restored.regional_relaxation_input.isChecked())
            self.assertTrue(restored.regional_refinement_input.isChecked())
            self.assertEqual(restored.refinement_scale_input.value(), 1.7)
            self.assertEqual(restored.refinement_steps_input.value(), 5)
            self.assertEqual(restored.refinement_denoise_input.value(), 0.30)
            self.assertEqual(restored.refinement_feather_input.value(), 56)
            self.assertEqual(restored.memory_policy_input.currentData(), "balanced")
            self.assertEqual(restored.reserve_vram_input.value(), 3.5)
            self.assertEqual(restored.minimum_ram_input.value(), 13.0)
            self.assertTrue(restored.cpu_vae_input.isChecked())
            self.assertEqual(restored._output_directory, output_directory)
            self.assertEqual(restored.filename_prefix_input.text(), "beach-study")
            self.assertEqual(restored.regions[0].name, "Main subject")
            self.assertEqual(restored.regions[0].spatial_role, "subject")
            self.assertEqual(restored.regions[0].prompt, "a person by the water")
            self.assertEqual(restored.regions[0].negative_prompt, "blurry face")
            self.assertEqual(restored.lora_list.count(), 1)
            lora_id = restored.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            self.assertEqual(
                restored.lora_library.binding_for(lora_id).region_ids,
                ("subject",),
            )
            self.assertEqual(
                restored.lora_library.binding_for(lora_id).strength, 0.6
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

    def test_oom_event_surfaces_in_app_16gb_guidance(self) -> None:
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
            self.assertTrue(any("16 GB guidance" in message for message in messages))
            self.assertIn("exceeded the 16 GB limit", window.statusBar().currentMessage())
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

    def test_explicit_worker_python_overrides_saved_project_runtime(self) -> None:
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

            with patch.dict(
                os.environ, {"K2LAB_WORKER_PYTHON": str(selected)}, clear=False
            ):
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
            window.canvas.region_created.emit(
                "teapot-region", 16.0, 16.0, 256.0, 256.0
            )
            window.region_prompt.setPlainText("a detailed red teapot")
            window.worker_client.send = Mock()

            window._generate_baseline()
            command, payload = window.worker_client.send.call_args.args
            self.assertEqual(command.value, "generate_baseline")
            self.assertEqual(payload["prompt"], "a red ceramic teapot")
            self.assertEqual(payload["width"] % 16, 0)
            self.assertEqual(payload["height"] % 16, 0)
            self.assertEqual(payload["steps"], 8)
            self.assertEqual(payload["seed"], 1234)
            self.assertEqual(payload["output_directory"], str(root / "renders"))
            self.assertEqual(payload["filename_prefix"], "teapot-test")
            self.assertTrue(payload["regional_prompting"])
            self.assertEqual(payload["regional_feather_pixels"], 128)
            self.assertEqual(payload["regional_outside_penalty"], 1.0)
            self.assertTrue(payload["regional_subject_competition"])
            self.assertEqual(payload["regional_late_step_scale"], 0.35)
            self.assertFalse(payload["regional_refinement"])
            self.assertEqual(payload["refinement_scale"], 1.5)
            self.assertEqual(payload["refinement_steps"], 4)
            self.assertEqual(payload["refinement_denoise"], 0.25)
            self.assertEqual(payload["refinement_feather_pixels"], 48)
            self.assertEqual(payload["regions"][0]["id"], "teapot-region")
            self.assertEqual(payload["regions"][0]["spatial_role"], "auto")
            self.assertEqual(len(payload["loras"]), 1)
            self.assertTrue(payload["loras"][0]["global"])
            self.assertEqual(payload["loras"][0]["strength"], 0.8)
            self.assertEqual(payload["loras"][0]["path"], str(lora_path.resolve()))
            self.assertEqual(
                payload["regions"][0]["prompt"], "a detailed red teapot"
            )

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
            self.assertIsNotNone(window.canvas._image_item)
            self.assertEqual(window.canvas._image_item.zValue(), -100.0)
            self.assertTrue(window.generate_button.isEnabled())
            window.close()

    def test_random_and_increment_seed_modes_select_the_dispatched_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(Path(directory))
            window.worker_client.send = Mock()
            window.seed_mode_input.setCurrentIndex(
                window.seed_mode_input.findData("random")
            )
            with patch(
                "k2_region_lab.desktop.main_window.secrets.randbelow", return_value=9876
            ):
                window._generate_baseline()
            random_payload = window.worker_client.send.call_args.args[1]
            self.assertEqual(random_payload["seed"], 9876)
            self.assertEqual(random_payload["seed_mode"], "random")
            self.assertEqual(window.seed_input.value(), 9876)

            window.worker_client.send.reset_mock()
            window.seed_mode_input.setCurrentIndex(
                window.seed_mode_input.findData("increment")
            )
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
            messages = [
                window.events.item(index).text()
                for index in range(window.events.count())
            ]
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
