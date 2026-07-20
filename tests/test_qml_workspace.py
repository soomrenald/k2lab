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
    from PySide6.QtCore import QUrl
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication

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
            self.assertEqual(engine.rootObjects()[0].property("title"), "K2 Region Lab")
            engine.rootObjects()[0].close()
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


if __name__ == "__main__":
    unittest.main()
