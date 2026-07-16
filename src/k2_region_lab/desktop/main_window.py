from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from k2_region_lab.config import AppSettings
from k2_region_lab.desktop.region_canvas import RegionCanvas
from k2_region_lab.desktop.worker_client import ExternalWorkerClient
from k2_region_lab.lora import LoraLibrary
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition
from k2_region_lab.worker.protocol import CommandKind


GLOBAL_SCOPE_ID = "__global__"


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.artifacts: ArtifactSet | None = None
        self.regions: list[RegionDefinition] = []
        self.lora_library = LoraLibrary()
        self._region_number = 0
        self._loading_region_form = False
        self._syncing_lora_scope = False
        self.setWindowTitle("K2 Region Lab")
        self.resize(1550, 950)

        self.canvas = RegionCanvas(settings.default_width, settings.default_height)
        self.setCentralWidget(self.canvas)
        self.canvas.region_created.connect(self._region_created)
        self.canvas.region_changed.connect(self._region_changed)
        self.canvas.region_deleted.connect(self._region_deleted)
        self.canvas.region_selected.connect(self._canvas_region_selected)

        self._build_prompt_dock()
        self._build_model_dock()
        self._build_lora_dock()
        self._build_event_dock()
        self.worker_client = ExternalWorkerClient(settings, self)
        self.worker_client.event_received.connect(self._worker_event)
        self.worker_client.stderr_received.connect(self._worker_stderr)
        self.worker_client.process_status.connect(self._worker_process_status)
        self._accelerator_available = False
        self.statusBar().showMessage("Foundation milestone — model not loaded")
        self.discover_models()
        if settings.auto_start_worker:
            self._start_worker()

    def _build_prompt_dock(self) -> None:
        dock = QDockWidget("Prompt and regions", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        body = QWidget(dock)
        layout = QVBoxLayout(body)
        layout.addWidget(QLabel("Global prompt"))
        self.global_prompt = QTextEdit()
        self.global_prompt.setPlaceholderText("Describe the complete image...")
        layout.addWidget(self.global_prompt)

        dimensions = QHBoxLayout()
        self.width_input = QSpinBox()
        self.height_input = QSpinBox()
        for control, value in (
            (self.width_input, self.settings.default_width),
            (self.height_input, self.settings.default_height),
        ):
            control.setRange(256, 4096)
            control.setSingleStep(16)
            control.setValue(value)
            control.editingFinished.connect(self._canvas_dimensions_changed)
        dimensions.addWidget(QLabel("W"))
        dimensions.addWidget(self.width_input)
        dimensions.addWidget(QLabel("H"))
        dimensions.addWidget(self.height_input)
        layout.addLayout(dimensions)

        region_buttons = QHBoxLayout()
        draw = QPushButton("Draw region")
        draw.clicked.connect(self.canvas.begin_region)
        delete = QPushButton("Delete selected")
        delete.clicked.connect(self.canvas.delete_selected_regions)
        region_buttons.addWidget(draw)
        region_buttons.addWidget(delete)
        layout.addLayout(region_buttons)

        layout.addWidget(QLabel("Regions (drag body; drag corner handles to resize)"))
        self.region_list = QListWidget()
        self.region_list.currentRowChanged.connect(self._selected_region_changed)
        layout.addWidget(self.region_list)
        layout.addWidget(QLabel("Selected region prompt"))
        self.region_prompt = QTextEdit()
        self.region_prompt.setPlaceholderText("Describe only the content controlled by this box...")
        self.region_prompt.setEnabled(False)
        self.region_prompt.textChanged.connect(self._region_form_edited)
        layout.addWidget(self.region_prompt)
        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_model_dock(self) -> None:
        dock = QDockWidget("Local model components", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        body = QWidget(dock)
        layout = QFormLayout(body)
        self.transformer_status = QLabel("Not discovered")
        self.text_status = QLabel("Not discovered")
        self.vae_status = QLabel("Not discovered")
        layout.addRow("Turbo", self.transformer_status)
        layout.addRow("Qwen", self.text_status)
        layout.addRow("VAE", self.vae_status)
        refresh = QPushButton("Discover models")
        refresh.clicked.connect(self.discover_models)
        layout.addRow(refresh)
        self.worker_status = QLabel("Stopped")
        self.accelerator_status = QLabel("Not probed")
        layout.addRow("GPU worker", self.worker_status)
        layout.addRow("Accelerator", self.accelerator_status)
        worker_buttons = QHBoxLayout()
        start_worker = QPushButton("Start worker")
        start_worker.clicked.connect(self._start_worker)
        validate = QPushButton("Validate tensors")
        validate.clicked.connect(self._validate_worker_models)
        worker_buttons.addWidget(start_worker)
        worker_buttons.addWidget(validate)
        layout.addRow(worker_buttons)
        self.load_model_button = QPushButton("Load Krea 2 baseline")
        self.load_model_button.clicked.connect(self._load_worker_model)
        self.load_model_button.setEnabled(False)
        layout.addRow(self.load_model_button)
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 100)
        self.steps_input.setValue(8)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(0)
        layout.addRow("Turbo steps", self.steps_input)
        layout.addRow("Seed", self.seed_input)
        self.generate_button = QPushButton("Generate baseline")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self._generate_baseline)
        layout.addRow(self.generate_button)
        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_lora_dock(self) -> None:
        dock = QDockWidget("LoRA library and scope", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        body = QWidget(dock)
        layout = QVBoxLayout(body)
        buttons = QHBoxLayout()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_lora)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected_lora)
        buttons.addWidget(browse)
        buttons.addWidget(remove)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Loaded LoRAs"))
        self.lora_list = QListWidget()
        self.lora_list.currentItemChanged.connect(self._selected_lora_changed)
        layout.addWidget(self.lora_list)
        layout.addWidget(QLabel("Apply selected LoRA to Global or one or more regions"))
        self.lora_scope_list = QListWidget()
        self.lora_scope_list.itemChanged.connect(self._lora_scope_changed)
        layout.addWidget(self.lora_scope_list)
        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_event_dock(self) -> None:
        dock = QDockWidget("Events", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.events = QListWidget(dock)
        dock.setWidget(self.events)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    @staticmethod
    def _artifact_label(artifact) -> str:
        if artifact is None:
            return "Missing"
        precision = ", ".join(f"{dtype}:{count}" for dtype, count in artifact.summary.dtypes)
        return f"{artifact.path.name}\n{artifact.summary.tensor_count} tensors; {precision}"

    @staticmethod
    def _region_label(region: RegionDefinition) -> str:
        box = region.box
        return (
            f"{region.name}: [{box.x0:.0f}, {box.y0:.0f}, "
            f"{box.x1:.0f}, {box.y1:.0f}] px"
        )

    def discover_models(self) -> None:
        try:
            self.artifacts = discover_model_artifacts(self.settings.model_directories)
        except (OSError, ValueError) as error:
            self.events.addItem(f"Model discovery failed: {error}")
            self.statusBar().showMessage("Model discovery failed")
            return
        self.transformer_status.setText(self._artifact_label(self.artifacts.transformer))
        self.text_status.setText(self._artifact_label(self.artifacts.text_encoder))
        self.vae_status.setText(self._artifact_label(self.artifacts.vae))
        state = "complete" if self.artifacts.complete else "incomplete"
        self.events.addItem(f"Model discovery {state}")
        self.statusBar().showMessage(f"Local model set: {state}; model loading not yet enabled")

    def _region_created(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        self._region_number += 1
        region = RegionDefinition(
            region_id=region_id,
            name=f"Region {self._region_number}",
            box=PixelBox(x0, y0, x1, y1),
        )
        self.regions.append(region)
        list_item = QListWidgetItem(self._region_label(region))
        list_item.setData(Qt.ItemDataRole.UserRole, region_id)
        self.region_list.addItem(list_item)
        self.canvas.add_region_box(region_id, QRectF(x0, y0, x1 - x0, y1 - y0))
        self.region_list.setCurrentItem(list_item)
        self._refresh_lora_scope()
        self.events.addItem(f"Created {region.name}")

    def _region_changed(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        index = self._region_index(region_id)
        self.regions[index] = replace(self.regions[index], box=PixelBox(x0, y0, x1, y1))
        item = self._region_list_item(region_id)
        if item is not None:
            item.setText(self._region_label(self.regions[index]))
        self.events.addItem(f"Updated {self.regions[index].name} geometry")

    def _region_deleted(self, region_id: str) -> None:
        try:
            index = self._region_index(region_id)
        except KeyError:
            return
        region = self.regions.pop(index)
        item = self._region_list_item(region_id)
        if item is not None:
            self.region_list.takeItem(self.region_list.row(item))
        self.lora_library.drop_region(region_id)
        self._refresh_lora_scope()
        self.events.addItem(f"Deleted {region.name}")

    def _canvas_region_selected(self, region_id: str) -> None:
        item = self._region_list_item(region_id)
        if item is not None and self.region_list.currentItem() is not item:
            self.region_list.setCurrentItem(item)

    def _region_index(self, region_id: str) -> int:
        for index, region in enumerate(self.regions):
            if region.region_id == region_id:
                return index
        raise KeyError(region_id)

    def _region_list_item(self, region_id: str) -> QListWidgetItem | None:
        for index in range(self.region_list.count()):
            item = self.region_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == region_id:
                return item
        return None

    def _selected_region_changed(self, row: int) -> None:
        selected = 0 <= row < len(self.regions)
        self.region_prompt.setEnabled(selected)
        self._loading_region_form = True
        try:
            if selected:
                region = self.regions[row]
                self.region_prompt.setPlainText(region.prompt)
                self.canvas.select_region(region.region_id)
            else:
                self.region_prompt.clear()
        finally:
            self._loading_region_form = False

    def _region_form_edited(self) -> None:
        if self._loading_region_form:
            return
        row = self.region_list.currentRow()
        if not 0 <= row < len(self.regions):
            return
        self.regions[row] = replace(
            self.regions[row], prompt=self.region_prompt.toPlainText()
        )

    def _canvas_dimensions_changed(self) -> None:
        geometry = CanvasGeometry.resolve(self.width_input.value(), self.height_input.value())
        self.canvas.set_canvas_size(geometry.aligned_width, geometry.aligned_height)
        self.events.addItem(
            "Canvas resolved to "
            f"{geometry.aligned_width}×{geometry.aligned_height} px "
            f"({geometry.patch_width}×{geometry.patch_height} image tokens)"
        )

    def _browse_lora(self) -> None:
        comfy_loras = Path("~/ComfyUI/models/loras").expanduser()
        start = comfy_loras if comfy_loras.is_dir() else Path.home()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load LoRA",
            str(start),
            "Safetensors LoRA (*.safetensors)",
        )
        if selected:
            self._add_lora_path(Path(selected), show_error_dialog=True)

    def _add_lora_path(self, path: Path, *, show_error_dialog: bool = False) -> bool:
        try:
            entry = self.lora_library.add(path)
        except (OSError, ValueError) as error:
            self.events.addItem(f"LoRA load failed: {error}")
            if show_error_dialog:
                QMessageBox.warning(self, "LoRA load failed", str(error))
            return False

        existing = self._lora_list_item(entry.lora_id)
        if existing is None:
            existing = QListWidgetItem(entry.display_name)
            existing.setData(Qt.ItemDataRole.UserRole, entry.lora_id)
            existing.setToolTip(str(entry.path))
            self.lora_list.addItem(existing)
            self.events.addItem(f"Loaded LoRA {entry.display_name}; scope defaults to Global")
        self.lora_list.setCurrentItem(existing)
        return True

    def _remove_selected_lora(self) -> None:
        item = self.lora_list.currentItem()
        if item is None:
            return
        lora_id = item.data(Qt.ItemDataRole.UserRole)
        entry = self.lora_library.get(lora_id)
        self.lora_library.remove(lora_id)
        self.lora_list.takeItem(self.lora_list.row(item))
        self._refresh_lora_scope()
        self.events.addItem(f"Removed LoRA {entry.display_name}")

    def _lora_list_item(self, lora_id: str) -> QListWidgetItem | None:
        for index in range(self.lora_list.count()):
            item = self.lora_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == lora_id:
                return item
        return None

    def _current_lora_id(self) -> str | None:
        item = self.lora_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _selected_lora_changed(self, current, previous) -> None:
        del current, previous
        self._refresh_lora_scope()

    def _refresh_lora_scope(self) -> None:
        self._syncing_lora_scope = True
        try:
            self.lora_scope_list.clear()
            lora_id = self._current_lora_id()
            if lora_id is None:
                self.lora_scope_list.setEnabled(False)
                return
            self.lora_scope_list.setEnabled(True)
            binding = self.lora_library.binding_for(lora_id)
            global_item = self._scope_item("Global", GLOBAL_SCOPE_ID, binding.global_scope)
            self.lora_scope_list.addItem(global_item)
            for region in self.regions:
                item = self._scope_item(
                    region.name, region.region_id, region.region_id in binding.region_ids
                )
                item.setToolTip(self._region_label(region))
                self.lora_scope_list.addItem(item)
        finally:
            self._syncing_lora_scope = False

    @staticmethod
    def _scope_item(label: str, scope_id: str, checked: bool) -> QListWidgetItem:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, scope_id)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        return item

    def _lora_scope_changed(self, changed_item: QListWidgetItem) -> None:
        if self._syncing_lora_scope:
            return
        lora_id = self._current_lora_id()
        if lora_id is None:
            return
        scope_id = changed_item.data(Qt.ItemDataRole.UserRole)
        if (
            scope_id == GLOBAL_SCOPE_ID
            and changed_item.checkState() == Qt.CheckState.Checked
        ):
            binding = self.lora_library.assign_global(lora_id)
        else:
            selected_regions = tuple(
                item.data(Qt.ItemDataRole.UserRole)
                for index in range(self.lora_scope_list.count())
                if (item := self.lora_scope_list.item(index)).data(Qt.ItemDataRole.UserRole)
                != GLOBAL_SCOPE_ID
                and item.checkState() == Qt.CheckState.Checked
            )
            binding = self.lora_library.assign_regions(lora_id, selected_regions)
        self._refresh_lora_scope()
        entry = self.lora_library.get(lora_id)
        scope = "Global" if binding.global_scope else ", ".join(binding.region_ids)
        self.events.addItem(f"Assigned {entry.display_name} to {scope}")

    def _worker_payload(self) -> dict[str, object]:
        directories = self.settings.model_directories
        return {
            "comfyui_root": str(self.settings.comfyui_root),
            "diffusion_models": str(directories.diffusion_models),
            "text_encoders": str(directories.text_encoders),
            "vae": str(directories.vae),
            "manifest_directory": str(self.settings.data_directory / "manifests"),
            "reserve_vram_gb": self.settings.reserve_vram_gb,
        }

    def _start_worker(self) -> None:
        if not self.worker_client.start():
            return
        if not self.worker_client.process.waitForStarted(3000):
            self.events.addItem("GPU worker did not start within three seconds")
            return
        self.worker_client.send(CommandKind.PROBE, self._worker_payload())

    def _validate_worker_models(self) -> None:
        if not self.worker_client.running:
            self._start_worker()
        if self.worker_client.running:
            self.worker_client.send(CommandKind.VALIDATE_MODELS, self._worker_payload())

    def _load_worker_model(self) -> None:
        if self.worker_client.running:
            self.load_model_button.setEnabled(False)
            self.worker_client.send(CommandKind.LOAD_MODEL, self._worker_payload())

    def _generate_baseline(self) -> None:
        geometry = CanvasGeometry.resolve(self.width_input.value(), self.height_input.value())
        payload = self._worker_payload()
        payload.update(
            {
                "prompt": self.global_prompt.toPlainText(),
                "width": geometry.aligned_width,
                "height": geometry.aligned_height,
                "steps": self.steps_input.value(),
                "seed": self.seed_input.value(),
                "output_directory": str(self.settings.data_directory / "baseline_outputs"),
            }
        )
        self.generate_button.setEnabled(False)
        self.worker_client.send(CommandKind.GENERATE_BASELINE, payload)

    def _worker_event(self, event: dict) -> None:
        state = event.get("state", "unknown")
        message = event.get("message", "")
        payload = event.get("payload", {})
        self.worker_status.setText(state)
        self.events.addItem(f"Worker [{state}]: {message}")
        if "accelerator_available" in payload:
            self._accelerator_available = bool(payload["accelerator_available"])
            devices = payload.get("devices", [])
            if devices:
                self.accelerator_status.setText(
                    ", ".join(device.get("name", "unknown") for device in devices)
                )
            else:
                self.accelerator_status.setText("Unavailable")
        if "manifests" in payload:
            compatible = payload.get("complete") and all(
                manifest.get("compatible") for manifest in payload["manifests"]
            )
            self.load_model_button.setEnabled(bool(compatible and self._accelerator_available))
            for manifest in payload["manifests"]:
                self.events.addItem(
                    f"{manifest['kind']} manifest: {manifest['manifest_path']}"
                )
        if message == "Krea 2 baseline components loaded":
            self.statusBar().showMessage("Krea 2 baseline loaded in GPU worker")
            self.generate_button.setEnabled(True)
        elif message == "Baseline generation complete":
            image_path = payload.get("image_path")
            if image_path and self.canvas.set_image(image_path):
                self.statusBar().showMessage(f"Baseline saved to {image_path}")
            self.generate_button.setEnabled(True)
        elif state == "error":
            self.load_model_button.setEnabled(self._accelerator_available)
            self.generate_button.setEnabled(False)

    def _worker_stderr(self, output: str) -> None:
        for line in output.splitlines():
            self.events.addItem(f"Worker stderr: {line}")

    def _worker_process_status(self, status: str) -> None:
        self.worker_status.setText(status)
        self.events.addItem(f"GPU worker process: {status}")

    def closeEvent(self, event) -> None:
        self.worker_client.stop()
        super().closeEvent(event)
