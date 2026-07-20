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
    from PySide6.QtCore import QMetaObject, QObject, QUrl
    from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent, QQmlExpression
    from PySide6.QtWidgets import QApplication
    from PIL import Image

    from k2_region_lab.config import AppSettings, ModelDirectories
    from k2_region_lab.desktop.main_window import MainWindow
    from k2_region_lab.qml.controller import QmlWorkspaceController


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 desktop dependency is not installed")
class QmlWorkspaceTests(unittest.TestCase):
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

    def test_controller_keeps_reference_and_edit_regions_on_distinct_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = self.make_window(Path(directory))
            controller = QmlWorkspaceController(backend)

            generated_id = controller.createRegion(10, 20, 210, 320)
            controller.updateSelectedRegion("prompt", "a blue glass subject")
            self.assertEqual(backend.regions[0].region_id, generated_id)
            self.assertEqual(backend.regions[0].prompt, "a blue glass subject")

            controller.setMode("edit")
            edit_id = controller.createRegion(30, 40, 230, 280)
            controller.updateSelectedRegion("prompt", "remove this object")
            controller.setEditLayer("reference")
            reference_id = controller.createRegion(50, 60, 260, 300)
            controller.updateSelectedRegion("prompt", "original subject")

            self.assertEqual(backend.edit_regions[0].region_id, edit_id)
            self.assertEqual(backend.edit_regions[0].prompt, "remove this object")
            self.assertEqual(backend.edit_reference_regions[0].region_id, reference_id)
            self.assertEqual(backend.edit_reference_regions[0].prompt, "original subject")
            self.assertNotEqual(edit_id, reference_id)
            controller.deleteLater()
            backend.close()

    def test_qml_workspace_loads_with_the_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = self.make_window(Path(directory))
            engine = QQmlApplicationEngine()
            controller = QmlWorkspaceController(backend, engine)
            engine.rootContext().setContextProperty("controller", controller)
            qml_path = (
                Path(__file__).parents[1]
                / "src"
                / "k2_region_lab"
                / "qml"
                / "ui"
                / "Main.qml"
            )

            engine.load(QUrl.fromLocalFile(str(qml_path)))
            self.application.processEvents()

            self.assertEqual(len(engine.rootObjects()), 1)
            root_object = engine.rootObjects()[0]
            self.assertEqual(root_object.property("title"), "K2 Region Lab")

            comparison_mode = root_object.findChild(QObject, "comparisonMode")
            canvas_stage = root_object.findChild(QObject, "canvasStage")
            result_clip = root_object.findChild(QObject, "comparisonResultClip")
            self.assertIsNotNone(comparison_mode)
            self.assertIsNotNone(canvas_stage)
            self.assertIsNotNone(result_clip)
            comparison_mode.setProperty("currentIndex", 2)
            root_object.setProperty("compareValue", 0.25)
            self.application.processEvents()
            self.assertAlmostEqual(
                result_clip.property("width"),
                canvas_stage.property("width") * 0.25,
                delta=1.0,
            )
            root_object.setProperty("compareValue", 0.75)
            self.application.processEvents()
            self.assertAlmostEqual(
                result_clip.property("width"),
                canvas_stage.property("width") * 0.75,
                delta=1.0,
            )

            self.assertTrue(QMetaObject.invokeMethod(root_object, "openSetupWindow"))
            self.application.processEvents()
            setup_window = root_object.findChild(QObject, "setupWindow")
            self.assertIsNotNone(setup_window)
            self.assertTrue(setup_window.property("visible"))

            cpu_vae = setup_window.findChild(QObject, "cpuVaeCheckBox")
            apply_button = setup_window.findChild(QObject, "applySettingsButton")
            self.assertFalse(controller.setupController.dirty)
            self.assertFalse(apply_button.property("enabled"))
            self.assertTrue(QMetaObject.invokeMethod(cpu_vae, "click"))
            self.application.processEvents()
            self.assertTrue(controller.setupController.dirty)
            self.assertTrue(apply_button.property("enabled"))
            controller.setupController.reset()
            self.application.processEvents()

            reserve_slider = setup_window.findChild(QObject, "reserveVramSlider")
            slider_track = reserve_slider.findChild(QObject, "valueSliderTrack")
            value_input = reserve_slider.findChild(QObject, "valueSliderInput")
            slider_track.setProperty("value", 6.5)
            self.assertTrue(QMetaObject.invokeMethod(slider_track, "moved"))
            self.application.processEvents()
            self.assertEqual(reserve_slider.property("currentValue"), 6.5)
            self.assertEqual(value_input.property("text"), "6.5")
            self.assertEqual(controller.setupController.value("reserveVram"), 6.5)
            value_input.setProperty("text", "7.5")
            self.assertTrue(QMetaObject.invokeMethod(reserve_slider, "commitText"))
            self.application.processEvents()
            self.assertEqual(slider_track.property("value"), 7.5)
            self.assertEqual(controller.setupController.value("reserveVram"), 7.5)
            controller.setupController.reset()
            self.application.processEvents()

            self.assertTrue(QMetaObject.invokeMethod(setup_window, "requestClose"))
            self.application.processEvents()
            self.assertFalse(setup_window.property("visible"))
            self.assertTrue(root_object.property("visible"))
            root_object.close()
            backend.close()

    def test_lora_can_be_deactivated_reactivated_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "wardrobe.safetensors"
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
            lora_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            backend = self.make_window(root)
            self.assertTrue(backend._add_lora_path(lora_path))
            lora_id = backend.lora_library.entries()[0].lora_id
            controller = QmlWorkspaceController(backend)

            controller.setLoraStrength(lora_id, 1.25)
            controller.setLoraActive(lora_id, False)
            self.assertEqual(backend.lora_library.binding_for(lora_id).strength, 0.0)
            controller.setLoraActive(lora_id, True)
            self.assertEqual(backend.lora_library.binding_for(lora_id).strength, 1.25)

            controller.removeLora(lora_id)
            self.assertEqual(backend.lora_library.entries(), ())
            self.assertEqual(backend.lora_list.count(), 0)
            controller.deleteLater()
            backend.close()

    def test_region_boundary_preview_resizes_continuously_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = self.make_window(Path(directory))
            engine = QQmlApplicationEngine()
            controller = QmlWorkspaceController(backend, engine)
            region_id = controller.createRegion(100, 120, 360, 420)
            engine.rootContext().setContextProperty("controller", controller)
            qml_path = (
                Path(__file__).parents[1]
                / "src"
                / "k2_region_lab"
                / "qml"
                / "ui"
                / "Main.qml"
            )

            engine.load(QUrl.fromLocalFile(str(qml_path)))
            self.application.processEvents()
            root_object = engine.rootObjects()[0]
            canvas = root_object.findChild(QObject, "mainRegionCanvas")
            region_item = QQmlExpression(
                engine.rootContext(), canvas, "regionItemAt(0)"
            )
            region_box, is_undefined = region_item.evaluate()
            self.assertFalse(region_item.hasError(), region_item.error())
            self.assertFalse(is_undefined)
            self.assertIsNotNone(region_box)
            self.assertEqual(region_box.property("regionId"), region_id)
            initial_width = region_box.property("width")
            initial_height = region_box.property("height")

            preview = QQmlExpression(
                engine.rootContext(),
                region_box,
                "beginResize(); updateResize(1, 1, 40, 30)",
            )
            preview.evaluate()
            self.assertFalse(preview.hasError(), preview.error())
            self.application.processEvents()
            self.assertTrue(region_box.property("resizePreviewActive"))
            self.assertAlmostEqual(region_box.property("width"), initial_width + 40)
            self.assertAlmostEqual(region_box.property("height"), initial_height + 30)

            finish = QQmlExpression(engine.rootContext(), region_box, "finishResize()")
            finish.evaluate()
            self.assertFalse(finish.hasError(), finish.error())
            self.application.processEvents()
            self.assertGreater(backend.regions[0].box.x1, 360)
            self.assertGreater(backend.regions[0].box.y1, 420)
            root_object.close()
            backend.close()

    def test_non_live_value_slider_stays_draggable_and_commits_on_release(self) -> None:
        engine = QQmlApplicationEngine()
        component = QQmlComponent(engine)
        components = (
            Path(__file__).parents[1]
            / "src"
            / "k2_region_lab"
            / "qml"
            / "ui"
            / "components"
        )
        component.setData(
            b'import QtQuick\nimport "."\nValueSlider {'
            b' objectName: "testSlider"; from: -4; to: 4; value: 1.25; decimals: 2 }',
            QUrl.fromLocalFile(str(components / "SliderHarness.qml")),
        )
        slider = component.create()
        self.assertIsNotNone(slider, component.errorString())
        commits: list[float] = []
        slider.valueEdited.connect(commits.append)
        slider_track = slider.findChild(QObject, "valueSliderTrack")
        value_input = slider.findChild(QObject, "valueSliderInput")

        slider_track.setProperty("value", 1.5)
        self.assertTrue(QMetaObject.invokeMethod(slider_track, "moved"))
        self.application.processEvents()
        self.assertEqual(slider.property("currentValue"), 1.5)
        self.assertEqual(value_input.property("text"), "1.50")
        self.assertEqual(commits, [])

        self.assertTrue(QMetaObject.invokeMethod(slider_track, "pressedChanged"))
        self.application.processEvents()
        self.assertEqual(commits, [1.5])
        value_input.setProperty("text", "2.25")
        self.assertTrue(QMetaObject.invokeMethod(slider, "commitText"))
        self.application.processEvents()
        self.assertEqual(slider_track.property("value"), 2.25)
        self.assertEqual(commits, [1.5, 2.25])
        slider.deleteLater()

    def test_edit_canvas_source_remains_the_original_when_a_result_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.png"
            result = root / "edited.png"
            Image.new("RGB", (320, 256), "navy").save(source)
            Image.new("RGB", (320, 256), "orange").save(result)
            backend = self.make_window(root)
            self.assertTrue(backend._set_edit_source(source, confirm_reset=False))
            backend._edit_result_path = result
            controller = QmlWorkspaceController(backend)
            controller.setMode("edit")
            controller.createRegion(20, 20, 180, 180)

            self.assertEqual(Path(controller.imageSource.toLocalFile()), source.resolve())
            self.assertEqual(Path(controller.resultSource.toLocalFile()), result.resolve())
            self.assertEqual(controller.activeRegionCount, 1)
            controller.deleteLater()
            backend.close()

    def test_setup_settings_are_staged_until_apply_and_can_be_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "renders"
            backend = self.make_window(root)
            controller = QmlWorkspaceController(backend)
            setup = controller.setupController
            original_prefix = backend.filename_prefix_input.text()

            setup.setValue("filenamePrefix", "discard-me")
            self.assertTrue(setup.dirty)
            self.assertEqual(backend.filename_prefix_input.text(), original_prefix)
            setup.reset()
            self.assertFalse(setup.dirty)
            self.assertEqual(backend.filename_prefix_input.text(), original_prefix)

            setup.setValue("memoryPolicy", "custom")
            setup.setValue("reserveVram", 2.5)
            setup.setValue("minimumRam", 10.0)
            setup.setValue("outputDirectory", str(output))
            setup.setValue("filenamePrefix", "qml-setup")
            self.assertTrue(setup.apply())

            self.assertFalse(setup.dirty)
            self.assertEqual(backend.settings.memory_policy, "custom")
            self.assertEqual(backend.reserve_vram_input.value(), 2.5)
            self.assertEqual(backend.minimum_ram_input.value(), 10.0)
            self.assertEqual(backend._output_directory, output.resolve())
            self.assertEqual(backend.filename_prefix_input.text(), "qml-setup")
            controller.deleteLater()
            backend.close()


if __name__ == "__main__":
    unittest.main()
