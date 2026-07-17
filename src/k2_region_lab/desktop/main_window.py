from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from k2_region_lab.config import AppSettings, ModelDirectories
from k2_region_lab.desktop.region_canvas import RegionCanvas
from k2_region_lab.desktop.worker_client import ExternalWorkerClient
from k2_region_lab.lora import LoraLibrary
from k2_region_lab.memory import (
    MEMORY_POLICIES,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
)
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.processes import find_owned_k2_workers, terminate_workers
from k2_region_lab.project import ProjectState, SavedLora, load_project, save_project
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
        self._models_compatible = False
        self._current_project_path: Path | None = None
        self._background_image_path: Path | None = None
        self.setWindowTitle("K2 Region Lab")
        self.resize(1550, 950)
        self._build_file_menu()

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
        self.statusBar().showMessage("Ready — model not loaded")
        if os.environ.get("DEBUG", "").strip() == "1":
            log_path = self.settings.data_directory / "logs" / "desktop-debug.log"
            self.events.addItem(f"DEBUG logging enabled: {log_path}")
        self.discover_models()
        if settings.auto_start_worker:
            self._start_worker()

    def _build_file_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open project…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)
        save_action = QAction("&Save project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        save_as_action = QAction("Save project &as…", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        menu.addAction(open_action)
        menu.addSeparator()
        menu.addAction(save_action)
        menu.addAction(save_as_action)
        menu.addSeparator()
        menu.addAction(exit_action)

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
        layout.addWidget(QLabel("Selected region name"))
        self.region_name = QLineEdit()
        self.region_name.setPlaceholderText("A unique region name…")
        self.region_name.setEnabled(False)
        self.region_name.editingFinished.connect(self._region_name_edited)
        layout.addWidget(self.region_name)
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
        self.diagnostic_button = QPushButton("Diagnose accelerator…")
        self.diagnostic_button.setVisible(False)
        self.diagnostic_button.clicked.connect(self._diagnose_accelerator)
        layout.addRow(self.diagnostic_button)
        worker_buttons = QHBoxLayout()
        start_worker = QPushButton("Start worker")
        start_worker.clicked.connect(self._start_worker)
        validate = QPushButton("Validate tensors")
        validate.clicked.connect(self._validate_worker_models)
        worker_buttons.addWidget(start_worker)
        worker_buttons.addWidget(validate)
        layout.addRow(worker_buttons)
        self.release_worker_button = QPushButton("Release K2 GPU memory…")
        self.release_worker_button.setToolTip(
            "Stop this user's K2 Region Lab GPU workers; other ROCm apps are untouched"
        )
        self.release_worker_button.clicked.connect(self._release_k2_gpu_memory)
        layout.addRow(self.release_worker_button)
        self.load_model_button = QPushButton("Load Krea 2 baseline")
        self.load_model_button.clicked.connect(self._load_worker_model)
        self.load_model_button.setEnabled(False)
        layout.addRow(self.load_model_button)
        self.memory_policy_input = QComboBox()
        for policy in MEMORY_POLICIES:
            self.memory_policy_input.addItem(policy.label, policy.key)
        policy_index = self.memory_policy_input.findData(self.settings.memory_policy)
        self.memory_policy_input.setCurrentIndex(max(0, policy_index))
        self.memory_policy_input.currentIndexChanged.connect(self._memory_policy_changed)
        layout.addRow("Memory policy", self.memory_policy_input)
        self.reserve_vram_input = QDoubleSpinBox()
        self.reserve_vram_input.setRange(0.5, 12.0)
        self.reserve_vram_input.setSingleStep(0.5)
        self.reserve_vram_input.setSuffix(" GiB")
        self.reserve_vram_input.setValue(self.settings.reserve_vram_gb)
        layout.addRow("Keep VRAM free", self.reserve_vram_input)
        self.minimum_ram_input = QDoubleSpinBox()
        self.minimum_ram_input.setRange(4.0, 48.0)
        self.minimum_ram_input.setSingleStep(1.0)
        self.minimum_ram_input.setSuffix(" GiB")
        self.minimum_ram_input.setValue(self.settings.minimum_system_ram_gb)
        layout.addRow("Minimum free RAM", self.minimum_ram_input)
        self.cpu_vae_input = QCheckBox("Decode with CPU VAE")
        self.cpu_vae_input.setChecked(self.settings.cpu_vae)
        layout.addRow(self.cpu_vae_input)
        self.oom_recovery_input = QCheckBox("Retry once after OOM")
        self.oom_recovery_input.setChecked(self.settings.oom_recovery)
        layout.addRow(self.oom_recovery_input)
        self.memory_status = QLabel("Not measured")
        self.memory_status.setWordWrap(True)
        layout.addRow("Memory", self.memory_status)
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

    def _memory_policy_changed(self) -> None:
        key = self.memory_policy_input.currentData()
        policy = memory_policy(key)
        self.reserve_vram_input.setValue(policy.reserve_vram_gb)
        self.minimum_ram_input.setValue(policy.minimum_system_ram_gb)
        self.cpu_vae_input.setChecked(policy.cpu_vae)
        self.oom_recovery_input.setChecked(policy.oom_recovery)
        self.events.addItem(f"Memory policy changed to {policy.label}")

    def _set_memory_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.memory_policy_input,
            self.reserve_vram_input,
            self.minimum_ram_input,
            self.cpu_vae_input,
            self.oom_recovery_input,
        ):
            control.setEnabled(enabled)

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
        self.statusBar().showMessage(f"Local model set: {state}")

    def _next_default_region_name(self) -> str:
        existing = {region.name.casefold() for region in self.regions}
        while True:
            self._region_number += 1
            candidate = f"Region {self._region_number}"
            if candidate.casefold() not in existing:
                return candidate

    def _region_created(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        region = RegionDefinition(
            region_id=region_id,
            name=self._next_default_region_name(),
            box=PixelBox(x0, y0, x1, y1),
        )
        self.regions.append(region)
        list_item = QListWidgetItem(self._region_label(region))
        list_item.setData(Qt.ItemDataRole.UserRole, region_id)
        self.region_list.addItem(list_item)
        self.canvas.add_region_box(
            region_id,
            QRectF(x0, y0, x1 - x0, y1 - y0),
            region.name,
        )
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
        self.region_name.setEnabled(selected)
        self.region_prompt.setEnabled(selected)
        self._loading_region_form = True
        try:
            if selected:
                region = self.regions[row]
                self.region_name.setText(region.name)
                self.region_prompt.setPlainText(region.prompt)
                self.canvas.select_region(region.region_id)
            else:
                self.region_name.clear()
                self.region_prompt.clear()
        finally:
            self._loading_region_form = False

    def _region_name_edited(self) -> None:
        if self._loading_region_form:
            return
        row = self.region_list.currentRow()
        if not 0 <= row < len(self.regions):
            return
        region = self.regions[row]
        name = self.region_name.text().strip()
        duplicate = any(
            index != row and candidate.name.casefold() == name.casefold()
            for index, candidate in enumerate(self.regions)
        )
        if not name or duplicate:
            self.region_name.setText(region.name)
            reason = "cannot be empty" if not name else "must be unique"
            self.statusBar().showMessage(f"Region name {reason}", 5000)
            return
        if name == region.name:
            return
        self.regions[row] = replace(region, name=name)
        item = self._region_list_item(region.region_id)
        if item is not None:
            item.setText(self._region_label(self.regions[row]))
        self.canvas.set_region_name(region.region_id, name)
        self._refresh_lora_scope()
        self.events.addItem(f"Renamed {region.name} to {name}")

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
        names = {
            region.region_id: region.name
            for region in self.regions
        }
        scope = (
            "Global"
            if binding.global_scope
            else ", ".join(names.get(region_id, region_id) for region_id in binding.region_ids)
        )
        self.events.addItem(f"Assigned {entry.display_name} to {scope}")

    def _project_state(self) -> ProjectState:
        directories = self.settings.model_directories
        runtime = {
            "diffusion_models": str(directories.diffusion_models),
            "text_encoders": str(directories.text_encoders),
            "vae": str(directories.vae),
            "worker_python": str(self.settings.worker_python),
            "comfyui_root": str(self.settings.comfyui_root),
            "data_directory": str(self.settings.data_directory),
            "memory_policy": self.memory_policy_input.currentData(),
            "reserve_vram_gb": self.reserve_vram_input.value(),
            "minimum_system_ram_gb": self.minimum_ram_input.value(),
            "cpu_vae": self.cpu_vae_input.isChecked(),
            "oom_recovery": self.oom_recovery_input.isChecked(),
        }
        saved_loras = tuple(
            SavedLora(
                path=entry.path,
                global_scope=(binding := self.lora_library.binding_for(entry.lora_id)).global_scope,
                region_ids=binding.region_ids,
            )
            for entry in self.lora_library.entries()
        )
        return ProjectState(
            canvas_width=self.width_input.value(),
            canvas_height=self.height_input.value(),
            global_prompt=self.global_prompt.toPlainText(),
            steps=self.steps_input.value(),
            seed=self.seed_input.value(),
            regions=tuple(self.regions),
            loras=saved_loras,
            runtime=runtime,
            background_image=self._background_image_path,
        )

    def _open_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open K2 Region Lab project",
            str(self._current_project_path.parent if self._current_project_path else Path.home()),
            "K2 Region Lab project (*.k2lab.json *.json)",
        )
        if selected:
            self._load_project_from(Path(selected), show_error_dialog=True)

    def _save_project(self) -> None:
        if self._current_project_path is None:
            self._save_project_as()
            return
        self._save_project_to(self._current_project_path, show_error_dialog=True)

    def _save_project_as(self) -> None:
        start = self._current_project_path or (Path.home() / "untitled.k2lab.json")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save K2 Region Lab project",
            str(start),
            "K2 Region Lab project (*.k2lab.json);;JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(path.suffix + ".json")
        self._save_project_to(path, show_error_dialog=True)

    def _save_project_to(self, path: Path, *, show_error_dialog: bool = False) -> bool:
        try:
            save_project(path, self._project_state())
        except (OSError, TypeError, ValueError) as error:
            logging.getLogger(__name__).exception("project save failed")
            self.events.addItem(f"Project save failed: {error}")
            if show_error_dialog:
                QMessageBox.warning(self, "Project save failed", str(error))
            return False
        self._current_project_path = path.expanduser().resolve()
        self.setWindowTitle(f"K2 Region Lab — {self._current_project_path.name}")
        self.statusBar().showMessage(f"Project saved to {self._current_project_path}", 5000)
        self.events.addItem(f"Saved project {self._current_project_path}")
        return True

    def _settings_from_project(self, state: ProjectState) -> AppSettings:
        runtime = state.runtime or {}
        current = self.settings
        current_directories = current.model_directories
        # A launch-time interpreter selection is an operator override. This lets
        # an old project run on a newer ROCm worker without first rewriting it.
        worker_python = (
            current.worker_python
            if "K2LAB_WORKER_PYTHON" in os.environ
            else Path(runtime.get("worker_python", current.worker_python)).expanduser()
        )
        return AppSettings(
            model_directories=ModelDirectories(
                Path(
                    runtime.get("diffusion_models", current_directories.diffusion_models)
                ).expanduser(),
                Path(runtime.get("text_encoders", current_directories.text_encoders)).expanduser(),
                Path(runtime.get("vae", current_directories.vae)).expanduser(),
            ),
            data_directory=Path(
                runtime.get("data_directory", current.data_directory)
            ).expanduser(),
            worker_python=worker_python,
            comfyui_root=Path(runtime.get("comfyui_root", current.comfyui_root)).expanduser(),
            auto_start_worker=current.auto_start_worker,
            memory_policy=str(runtime.get("memory_policy", current.memory_policy)),
            reserve_vram_gb=float(runtime.get("reserve_vram_gb", current.reserve_vram_gb)),
            minimum_system_ram_gb=float(
                runtime.get("minimum_system_ram_gb", current.minimum_system_ram_gb)
            ),
            cpu_vae=bool(runtime.get("cpu_vae", current.cpu_vae)),
            oom_recovery=bool(runtime.get("oom_recovery", current.oom_recovery)),
            default_width=state.canvas_width,
            default_height=state.canvas_height,
        )

    def _load_project_from(self, path: Path, *, show_error_dialog: bool = False) -> bool:
        try:
            state = load_project(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logging.getLogger(__name__).exception("project load failed")
            self.events.addItem(f"Project load failed: {error}")
            if show_error_dialog:
                QMessageBox.warning(self, "Project load failed", str(error))
            return False

        self.worker_client.stop()
        self.settings = self._settings_from_project(state)
        self.worker_client.settings = self.settings
        self.width_input.setValue(state.canvas_width)
        self.height_input.setValue(state.canvas_height)
        self.canvas.set_canvas_size(state.canvas_width, state.canvas_height)
        self.global_prompt.setPlainText(state.global_prompt)
        self.steps_input.setValue(state.steps)
        self.seed_input.setValue(state.seed)
        policy_index = self.memory_policy_input.findData(self.settings.memory_policy)
        self.memory_policy_input.setCurrentIndex(max(0, policy_index))
        self.reserve_vram_input.setValue(self.settings.reserve_vram_gb)
        self.minimum_ram_input.setValue(self.settings.minimum_system_ram_gb)
        self.cpu_vae_input.setChecked(self.settings.cpu_vae)
        self.oom_recovery_input.setChecked(self.settings.oom_recovery)

        self.canvas.clear_regions()
        self.region_list.clear()
        self.regions = list(state.regions)
        self._region_number = max(
            (
                int(match.group(1))
                for region in self.regions
                if (match := re.fullmatch(r"Region (\d+)", region.name))
            ),
            default=0,
        )
        for region in self.regions:
            item = QListWidgetItem(self._region_label(region))
            item.setData(Qt.ItemDataRole.UserRole, region.region_id)
            self.region_list.addItem(item)
            box = region.box
            self.canvas.add_region_box(
                region.region_id,
                QRectF(box.x0, box.y0, box.width, box.height),
                region.name,
            )

        self.lora_library = LoraLibrary()
        self.lora_list.clear()
        for saved_lora in state.loras:
            if not self._add_lora_path(saved_lora.path):
                continue
            lora_id = self.lora_list.currentItem().data(Qt.ItemDataRole.UserRole)
            if saved_lora.global_scope:
                self.lora_library.assign_global(lora_id)
            else:
                self.lora_library.assign_regions(lora_id, saved_lora.region_ids)
        self._refresh_lora_scope()

        self.canvas.clear_image()
        self._background_image_path = None
        if state.background_image and state.background_image.is_file():
            if self.canvas.set_image(str(state.background_image)):
                self._background_image_path = state.background_image
        if self.region_list.count():
            self.region_list.setCurrentRow(0)
        self._current_project_path = path.expanduser().resolve()
        self.setWindowTitle(f"K2 Region Lab — {self._current_project_path.name}")
        self.discover_models()
        self.events.addItem(f"Opened project {self._current_project_path}")
        self.statusBar().showMessage(f"Opened {self._current_project_path.name}", 5000)
        if self.settings.auto_start_worker:
            self._start_worker()
        return True

    def _worker_payload(self) -> dict[str, object]:
        directories = self.settings.model_directories
        policy_key = self.memory_policy_input.currentData()
        return {
            "comfyui_root": str(self.settings.comfyui_root),
            "diffusion_models": str(directories.diffusion_models),
            "text_encoders": str(directories.text_encoders),
            "vae": str(directories.vae),
            "manifest_directory": str(self.settings.data_directory / "manifests"),
            "memory_policy": policy_key,
            "reserve_vram_gb": effective_reserve_vram_gb(
                policy_key, self.reserve_vram_input.value()
            ),
            "minimum_system_ram_gb": effective_minimum_system_ram_gb(
                policy_key, self.minimum_ram_input.value()
            ),
            "cpu_vae": self.cpu_vae_input.isChecked(),
            "oom_recovery": self.oom_recovery_input.isChecked(),
        }

    def _start_worker(self) -> None:
        if not self.worker_client.start():
            return
        if not self.worker_client.process.waitForStarted(3000):
            self.events.addItem("GPU worker did not start within three seconds")
            return
        self.worker_client.send(CommandKind.PROBE, self._worker_payload())

    def _diagnose_accelerator(self) -> None:
        self.events.addItem("Restarting the GPU worker for a clean accelerator diagnostic")
        self.worker_client.stop()
        self._accelerator_available = False
        self.load_model_button.setEnabled(False)
        self.accelerator_status.setText("Diagnosing…")
        if not self.worker_client.start():
            return
        if not self.worker_client.process.waitForStarted(3000):
            self.events.addItem("Diagnostic worker did not start within three seconds")
            return
        self.worker_client.send(CommandKind.DIAGNOSE_ACCELERATOR, self._worker_payload())

    def _release_k2_gpu_memory(self) -> None:
        workers = find_owned_k2_workers()
        if not workers:
            QMessageBox.information(
                self,
                "No K2 GPU worker found",
                "No K2 Region Lab worker owned by your user is currently running. "
                "Other ROCm applications were not inspected or stopped.",
            )
            self.events.addItem("GPU memory release: no K2 workers found")
            return

        process_lines = "\n".join(
            f"PID {worker.pid}: {' '.join(worker.command[:4])}" for worker in workers
        )
        answer = QMessageBox.question(
            self,
            "Release K2 GPU memory?",
            "This will stop the following K2 Region Lab GPU worker processes:\n\n"
            f"{process_lines}\n\n"
            "Unsaved GUI configuration is unaffected. Other ROCm applications, "
            "including ComfyUI, will not be stopped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        current_pid = self.worker_client.kill_immediately()
        remaining = find_owned_k2_workers()
        terminated, failed = terminate_workers(remaining)
        stopped = set(terminated)
        if current_pid is not None:
            stopped.add(current_pid)

        self._accelerator_available = False
        self.worker_status.setText("Stopped")
        self.accelerator_status.setText("Not probed")
        self.memory_status.setText("K2 workers stopped; GPU allocations released")
        self.load_model_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self._set_memory_controls_enabled(True)
        if stopped:
            self.events.addItem(
                "Released K2 GPU workers: " + ", ".join(map(str, sorted(stopped)))
            )
        if failed:
            self.events.addItem(
                "Could not stop K2 GPU workers: " + ", ".join(map(str, failed))
            )
            QMessageBox.warning(
                self,
                "Some K2 workers remain",
                "These K2 worker PIDs could not be stopped: "
                + ", ".join(map(str, failed)),
            )
        else:
            self.statusBar().showMessage(
                "K2 GPU workers stopped; click Start worker when ready", 8000
            )

    def _validate_worker_models(self) -> None:
        if not self.worker_client.running:
            self._start_worker()
        if self.worker_client.running:
            self.worker_client.send(CommandKind.VALIDATE_MODELS, self._worker_payload())

    def _load_worker_model(self) -> None:
        if self.worker_client.running:
            self.load_model_button.setEnabled(False)
            self._set_memory_controls_enabled(False)
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
        if "memory" in payload:
            memory = payload["memory"]
            free_gib = memory.get("gpu_free_bytes", 0) / (1024**3)
            total_gib = memory.get("gpu_total_bytes", 0) / (1024**3)
            ram_gib = memory.get("ram_available_bytes", 0) / (1024**3)
            action = memory.get("action", "observed")
            self.memory_status.setText(
                f"VRAM {free_gib:.1f}/{total_gib:.1f} GiB free; "
                f"RAM {ram_gib:.1f} GiB available; {action}"
            )
        if "accelerator_available" in payload:
            self._accelerator_available = bool(payload["accelerator_available"])
            devices = payload.get("devices", [])
            if devices:
                self.accelerator_status.setText(
                    ", ".join(device.get("name", "unknown") for device in devices)
                )
            else:
                self.accelerator_status.setText("Unavailable — run diagnostic")
            self.diagnostic_button.setVisible(not self._accelerator_available)
            self.load_model_button.setEnabled(
                bool(self._models_compatible and self._accelerator_available)
            )
            self.events.addItem(
                "Worker runtime: "
                f"{payload.get('python_executable', 'unknown')}; "
                f"Torch {payload.get('torch_version', 'unavailable')}; "
                f"ROCm {payload.get('hip_version', 'unavailable')}"
            )
            if not self._accelerator_available:
                error = payload.get("initialization_error") or payload.get("error")
                if error:
                    self.events.addItem(f"Accelerator probe error: {error}")
        if "manifests" in payload:
            compatible = payload.get("complete") and all(
                manifest.get("compatible") for manifest in payload["manifests"]
            )
            self._models_compatible = bool(compatible)
            self.load_model_button.setEnabled(bool(compatible and self._accelerator_available))
            for manifest in payload["manifests"]:
                self.events.addItem(
                    f"{manifest['kind']} manifest: {manifest['manifest_path']}"
                )
        if message == "Accelerator diagnostics complete":
            recommendations = payload.get("recommendations", [])
            for recommendation in recommendations:
                self.events.addItem(f"Diagnostic: {recommendation}")
            report = QMessageBox(self)
            report.setWindowTitle("Accelerator diagnostic")
            report.setIcon(
                QMessageBox.Icon.Information
                if self._accelerator_available
                else QMessageBox.Icon.Warning
            )
            report.setText(
                "ROCm accelerator detected. Model loading is available."
                if self._accelerator_available
                else "The worker still cannot initialize a ROCm accelerator."
            )
            report.setInformativeText("\n".join(recommendations))
            report.setDetailedText(json.dumps(payload, indent=2, sort_keys=True))
            report.exec()
        elif message == "Krea 2 baseline components loaded":
            self.statusBar().showMessage("Krea 2 baseline loaded in GPU worker")
            if "reserve_vram_gb" in payload:
                self.reserve_vram_input.setValue(float(payload["reserve_vram_gb"]))
            self._set_memory_controls_enabled(False)
            self.generate_button.setEnabled(True)
        elif message == "Baseline generation complete":
            image_path = payload.get("image_path")
            if image_path and self.canvas.set_image(image_path):
                self._background_image_path = Path(image_path)
                self.statusBar().showMessage(f"Baseline saved to {image_path}")
            if payload.get("oom_recovered"):
                self.events.addItem(
                    "Generation recovered from GPU OOM with CPU VAE and a larger VRAM floor"
                )
            self.generate_button.setEnabled(True)
        elif state == "error":
            self.load_model_button.setEnabled(self._accelerator_available)
            self.generate_button.setEnabled(False)
            self._set_memory_controls_enabled(True)
            normalized_message = message.casefold()
            if "out of memory" in normalized_message or "gpu memory pressure" in normalized_message:
                self.events.addItem(
                    "16 GB guidance: use Release K2 GPU memory, restart the worker, "
                    "and reduce the canvas on ROCm 6.4; native scaled FP8 requires ROCm 6.5+"
                )
                self.statusBar().showMessage(
                    "Generation exceeded the 16 GB limit — release the K2 worker before retrying",
                    15000,
                )

    def _worker_stderr(self, output: str) -> None:
        logging.getLogger(__name__).debug("worker stderr received: %s", output)
        for line in output.splitlines():
            self.events.addItem(f"Worker stderr: {line}")

    def _worker_process_status(self, status: str) -> None:
        self.worker_status.setText(status)
        self.events.addItem(f"GPU worker process: {status}")
        if status.startswith("stopped") or "error" in status:
            self._set_memory_controls_enabled(True)

    def closeEvent(self, event) -> None:
        self.worker_client.stop()
        super().closeEvent(event)
