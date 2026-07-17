from __future__ import annotations

import json
import logging
import os
import re
import secrets
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from k2_region_lab.config import AppSettings, ModelDirectories
from k2_region_lab.desktop.region_canvas import RegionCanvas
from k2_region_lab.desktop.resource_monitor import ResourceMonitorWidget
from k2_region_lab.desktop.worker_client import ExternalWorkerClient
from k2_region_lab.lora import LoraLibrary
from k2_region_lab.memory import (
    MEMORY_POLICIES,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
)
from k2_region_lab.model import ArtifactSet, discover_model_artifacts
from k2_region_lab.output import (
    default_output_directory,
    default_prompt_directory,
    validate_filename_prefix,
)
from k2_region_lab.processes import find_owned_k2_workers, terminate_workers
from k2_region_lab.project import ProjectState, SavedLora, load_project, save_project
from k2_region_lab.projector import (
    CUSTOM_PROJECTOR_PRESET,
    DEFAULT_PROJECTOR_PRESET,
    PROJECTOR_PRESETS,
    PROJECTOR_PRESET_LABELS,
)
from k2_region_lab.regional_prompting import (
    GLOBAL_EMPHASIS_SCOPE,
    PromptEmphasis,
    compile_regional_prompt_plan,
)
from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition
from k2_region_lab.worker.protocol import CommandKind


GLOBAL_SCOPE_ID = "__global__"


class EventListWidget(QListWidget):
    """Follow new events only while the user is already viewing the end."""

    def addItem(self, item) -> None:
        scrollbar = self.verticalScrollBar()
        follow_latest = scrollbar.value() >= scrollbar.maximum()
        super().addItem(item)
        if follow_latest:
            self.scrollToBottom()


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.artifacts: ArtifactSet | None = None
        self.regions: list[RegionDefinition] = []
        self.prompt_emphases: list[PromptEmphasis] = []
        self.lora_library = LoraLibrary()
        self._region_number = 0
        self._loading_region_form = False
        self._syncing_lora_scope = False
        self._syncing_lora_strength = False
        self._syncing_projector_fields = False
        self._syncing_prompt_emphases = False
        self._models_compatible = False
        self._model_loaded = False
        self._current_project_path: Path | None = None
        self._background_image_path: Path | None = None
        self._upscale_model_path: Path | None = None
        self._generation_active = False
        self._pending_generation_payload: dict[str, object] | None = None
        self._worker_bootstrap_stage: str | None = None
        self._generation_completed = False
        self._prompt_preview_dialog: QDialog | None = None
        self._project_directory = default_prompt_directory()
        self._output_directory = settings.output_directory or default_output_directory(
            settings.data_directory
        )
        self.setWindowTitle("K2 Region Lab")
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
        self.splitDockWidget(
            self.model_dock, self.lora_dock, Qt.Orientation.Horizontal
        )
        self._build_event_dock()
        self._fit_initial_window_to_screen()
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

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _fit_initial_window_to_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1550, 900)
            return
        available = screen.availableGeometry()
        width = min(1800, max(1000, int(available.width() * 0.96)))
        height = min(900, max(700, int(available.height() * 0.92)))
        self.resize(width, height)

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
        self.global_prompt.setMinimumHeight(80)
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

        layout.addWidget(
            QLabel("Regions (front to back; drag rows to reorder)")
        )
        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(100)
        self.region_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.region_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.region_list.setToolTip(
            "Topmost overlapping subject regions appear in front of lower regions"
        )
        self.region_list.model().rowsMoved.connect(self._region_order_changed)
        self.region_list.currentRowChanged.connect(self._selected_region_changed)
        layout.addWidget(self.region_list)
        layout.addWidget(QLabel("Selected region name"))
        self.region_name = QLineEdit()
        self.region_name.setPlaceholderText("A unique region name…")
        self.region_name.setEnabled(False)
        self.region_name.editingFinished.connect(self._region_name_edited)
        layout.addWidget(self.region_name)
        layout.addWidget(QLabel("Selected region spatial role"))
        self.region_role = QComboBox()
        self.region_role.addItem("Auto (based on box width)", "auto")
        self.region_role.addItem("Subject target", "subject")
        self.region_role.addItem("Background band", "background")
        self.region_role.setEnabled(False)
        self.region_role.setToolTip(
            "Subject targets pull content toward the box center; background bands "
            "use softer full-box coverage"
        )
        self.region_role.currentIndexChanged.connect(self._region_role_changed)
        layout.addWidget(self.region_role)
        layout.addWidget(QLabel("Selected region prompt"))
        self.region_prompt = QTextEdit()
        self.region_prompt.setMinimumHeight(90)
        self.region_prompt.setPlaceholderText("Describe only the content controlled by this box...")
        self.region_prompt.setEnabled(False)
        self.region_prompt.textChanged.connect(self._region_form_edited)
        layout.addWidget(self.region_prompt)
        layout.addWidget(QLabel("Selected region negative prompt"))
        self.region_negative_prompt = QTextEdit()
        self.region_negative_prompt.setMinimumHeight(70)
        self.region_negative_prompt.setPlaceholderText(
            "Optional content to discourage inside this box..."
        )
        self.region_negative_prompt.setEnabled(False)
        self.region_negative_prompt.textChanged.connect(self._region_form_edited)
        layout.addWidget(self.region_negative_prompt)
        dock.setWidget(self._scrollable(body))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_model_dock(self) -> None:
        dock = QDockWidget("Model and generation settings", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        body = QWidget(dock)
        body_layout = QVBoxLayout(body)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)
        body_layout.addWidget(self.settings_tabs)
        runtime_page = QWidget()
        layout = QFormLayout(runtime_page)
        self.settings_tabs.addTab(
            self._scrollable(runtime_page), "Model & memory"
        )
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
        self.validate_models_button = QPushButton("Validate tensors")
        self.validate_models_button.clicked.connect(self._validate_worker_models)
        worker_buttons.addWidget(start_worker)
        worker_buttons.addWidget(self.validate_models_button)
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
        generation_page = QWidget()
        layout = QFormLayout(generation_page)
        self.settings_tabs.addTab(
            self._scrollable(generation_page), "Generation & spatial"
        )
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 100)
        self.steps_input.setValue(8)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(0)
        layout.addRow("Turbo steps", self.steps_input)
        layout.addRow("Seed", self.seed_input)
        self.seed_mode_input = QComboBox()
        self.seed_mode_input.addItem("Fixed", "fixed")
        self.seed_mode_input.addItem("Random", "random")
        self.seed_mode_input.addItem("Increment", "increment")
        layout.addRow("Seed behavior", self.seed_mode_input)
        self.regional_prompting_input = QCheckBox("Use unified spatial prompting")
        self.regional_prompting_input.setChecked(True)
        layout.addRow(self.regional_prompting_input)
        self.regional_prompt_strength_input = QDoubleSpinBox()
        self.regional_prompt_strength_input.setRange(0.1, 10.0)
        self.regional_prompt_strength_input.setSingleStep(0.1)
        self.regional_prompt_strength_input.setValue(1.0)
        self.regional_prompt_strength_input.setToolTip(
            "Additive attention guidance between each regional text span and its image area"
        )
        layout.addRow("Inside boost", self.regional_prompt_strength_input)
        self.regional_outside_penalty_input = QDoubleSpinBox()
        self.regional_outside_penalty_input.setRange(0.0, 10.0)
        self.regional_outside_penalty_input.setSingleStep(0.1)
        self.regional_outside_penalty_input.setValue(1.0)
        self.regional_outside_penalty_input.setToolTip(
            "Suppress subject prompt attention away from its target box; background "
            "bands automatically use one quarter of this penalty"
        )
        layout.addRow("Outside penalty", self.regional_outside_penalty_input)
        self.regional_feather_input = QSpinBox()
        self.regional_feather_input.setRange(0, 2048)
        self.regional_feather_input.setSingleStep(16)
        self.regional_feather_input.setSuffix(" px")
        self.regional_feather_input.setValue(128)
        self.regional_feather_input.setToolTip(
            "Distance outside each box over which its attention guidance smoothly fades"
        )
        layout.addRow("Spatial falloff", self.regional_feather_input)
        self.regional_subject_competition_input = QCheckBox(
            "Separate overlapping subject targets"
        )
        self.regional_subject_competition_input.setChecked(True)
        self.regional_subject_competition_input.setToolTip(
            "Give overlapping subject boxes exclusive soft ownership of image tokens"
        )
        layout.addRow(self.regional_subject_competition_input)
        self.regional_subject_fill_input = QCheckBox("Make subjects fill their boxes")
        self.regional_subject_fill_input.setChecked(True)
        self.regional_subject_fill_input.setToolTip(
            "Treat each subject box as its desired visible extent, not only its location"
        )
        layout.addRow(self.regional_subject_fill_input)
        self.regional_relaxation_input = QCheckBox(
            "Relax spatial guidance during late steps"
        )
        self.regional_relaxation_input.setChecked(True)
        self.regional_relaxation_input.setToolTip(
            "Keep placement guidance strong early, then reduce it to the selected "
            "scale for final detail"
        )
        layout.addRow(self.regional_relaxation_input)
        self.regional_late_step_scale_input = QDoubleSpinBox()
        self.regional_late_step_scale_input.setRange(0.0, 1.0)
        self.regional_late_step_scale_input.setDecimals(2)
        self.regional_late_step_scale_input.setSingleStep(0.05)
        self.regional_late_step_scale_input.setValue(0.35)
        self.regional_late_step_scale_input.setToolTip(
            "Spatial-guidance multiplier reached at the final denoising step. "
            "1.00 keeps full box guidance; lower values allow late refinement."
        )
        layout.addRow("Late-step spatial scale", self.regional_late_step_scale_input)
        self.regional_relaxation_input.toggled.connect(
            self._set_late_step_scale_enabled
        )
        self._set_late_step_scale_enabled(self.regional_relaxation_input.isChecked())
        self.regional_lora_delta_adaptation_input = QCheckBox(
            "Adapt spatial guidance from regional LoRA delta"
        )
        self.regional_lora_delta_adaptation_input.setChecked(False)
        self.regional_lora_delta_adaptation_input.setToolTip(
            "After each denoising step, compare each regional LoRA's routed delta "
            "to its first-step magnitude and gently tune that region's next-step "
            "spatial attention. The LoRA gate itself remains unchanged."
        )
        layout.addRow(self.regional_lora_delta_adaptation_input)
        self.regional_lora_delta_adaptation_gain_input = QDoubleSpinBox()
        self.regional_lora_delta_adaptation_gain_input.setRange(0.0, 1.0)
        self.regional_lora_delta_adaptation_gain_input.setDecimals(2)
        self.regional_lora_delta_adaptation_gain_input.setSingleStep(0.05)
        self.regional_lora_delta_adaptation_gain_input.setValue(0.35)
        self.regional_lora_delta_adaptation_gain_input.setToolTip(
            "How strongly measured LoRA delta changes can adjust spatial attention. "
            "The resulting per-region multiplier is always limited to 0.50–1.50."
        )
        layout.addRow(
            "LoRA delta response", self.regional_lora_delta_adaptation_gain_input
        )
        self.regional_lora_delta_adaptation_input.toggled.connect(
            self._set_lora_delta_adaptation_controls_enabled
        )
        self._set_lora_delta_adaptation_controls_enabled(False)
        self.post_upscale_input = QCheckBox("Post-upscale after releasing Krea VRAM")
        self.post_upscale_input.setToolTip(
            "Decode first, unload Krea/LoRAs/VAE from the GPU, then upscale the final image"
        )
        layout.addRow(self.post_upscale_input)
        self.upscale_scale_input = QComboBox()
        self.upscale_scale_input.addItem("2×", 2)
        self.upscale_scale_input.addItem("4×", 4)
        layout.addRow("Output scale", self.upscale_scale_input)
        self.upscale_method_input = QComboBox()
        self.upscale_method_input.addItem("CPU Lanczos", "lanczos")
        self.upscale_method_input.addItem("Neural model (tiled GPU)", "model")
        layout.addRow("Upscaler", self.upscale_method_input)
        upscale_model_row = QWidget()
        upscale_model_layout = QHBoxLayout(upscale_model_row)
        upscale_model_layout.setContentsMargins(0, 0, 0, 0)
        self.upscale_model_input = QLineEdit()
        self.upscale_model_input.setReadOnly(True)
        self.upscale_model_input.setPlaceholderText("Select ESRGAN-compatible model…")
        self.upscale_model_browse = QPushButton("Browse…")
        self.upscale_model_browse.clicked.connect(self._browse_upscale_model)
        self.upscale_model_clear = QPushButton("Clear")
        self.upscale_model_clear.clicked.connect(self._clear_upscale_model)
        upscale_model_layout.addWidget(self.upscale_model_input)
        upscale_model_layout.addWidget(self.upscale_model_browse)
        upscale_model_layout.addWidget(self.upscale_model_clear)
        layout.addRow("Model", upscale_model_row)
        self.post_upscale_input.toggled.connect(self._set_upscale_controls_enabled)
        self.upscale_method_input.currentIndexChanged.connect(
            self._set_upscale_controls_enabled
        )
        self._set_upscale_controls_enabled()
        preview_prompt = QPushButton("Preview unified prompt…")
        preview_prompt.clicked.connect(self._preview_unified_prompt)
        layout.addRow(preview_prompt)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_directory_input = QLineEdit(str(self._output_directory))
        self.output_directory_input.setReadOnly(True)
        output_browse = QPushButton("Browse…")
        output_browse.clicked.connect(self._browse_output_directory)
        output_layout.addWidget(self.output_directory_input)
        output_layout.addWidget(output_browse)
        layout.addRow("Output folder", output_row)
        self.filename_prefix_input = QLineEdit(self.settings.filename_prefix)
        self.filename_prefix_input.setPlaceholderText("baseline")
        self.filename_prefix_input.editingFinished.connect(self._filename_prefix_edited)
        layout.addRow("Filename prefix", self.filename_prefix_input)
        generation_buttons = QWidget()
        generation_button_layout = QHBoxLayout(generation_buttons)
        generation_button_layout.setContentsMargins(0, 0, 0, 0)
        self.generate_button = QPushButton("Generate image")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self._generate_baseline)
        self.stop_generation_button = QPushButton("Stop generation")
        self.stop_generation_button.setEnabled(False)
        self.stop_generation_button.setToolTip(
            "Stop only the K2 GPU worker and release its GPU/system memory"
        )
        self.stop_generation_button.clicked.connect(self._stop_generation)
        generation_button_layout.addWidget(self.generate_button)
        generation_button_layout.addWidget(self.stop_generation_button)
        layout.addRow(generation_buttons)
        self._build_token_emphasis_tab()
        self._build_projector_tab()
        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.model_dock = dock

    def _build_token_emphasis_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        explanation = QLabel(
            "Highlight a complete word or phrase in the Global prompt or Selected "
            "region prompt, then add it here. The boost applies to its resolved Qwen "
            "tokens; regional phrases are boosted only through their box's soft field."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.emphasis_strength_input = QDoubleSpinBox()
        self.emphasis_strength_input.setRange(0.0, 2.0)
        self.emphasis_strength_input.setDecimals(2)
        self.emphasis_strength_input.setSingleStep(0.1)
        self.emphasis_strength_input.setValue(0.5)
        self.emphasis_strength_input.setToolTip(
            "Additive attention-logit boost. Start around 0.30–0.60; very high "
            "values can make a concept dominate the image."
        )
        form = QFormLayout()
        form.addRow("Selected phrase boost", self.emphasis_strength_input)
        layout.addLayout(form)
        actions = QHBoxLayout()
        add_global = QPushButton("Emphasize global selection")
        add_global.clicked.connect(
            lambda: self._add_prompt_emphasis(
                GLOBAL_EMPHASIS_SCOPE, self.global_prompt
            )
        )
        add_region = QPushButton("Emphasize region selection")
        add_region.clicked.connect(self._add_selected_region_emphasis)
        actions.addWidget(add_global)
        actions.addWidget(add_region)
        layout.addLayout(actions)
        layout.addWidget(QLabel("Emphasized phrases"))
        self.prompt_emphasis_list = QListWidget()
        self.prompt_emphasis_list.setMinimumHeight(150)
        self.prompt_emphasis_list.currentRowChanged.connect(
            self._selected_prompt_emphasis_changed
        )
        layout.addWidget(self.prompt_emphasis_list, 1)
        remove = QPushButton("Remove selected emphasis")
        remove.clicked.connect(self._remove_selected_prompt_emphasis)
        layout.addWidget(remove)
        self.emphasis_strength_input.valueChanged.connect(
            self._prompt_emphasis_strength_changed
        )
        self.settings_tabs.addTab(self._scrollable(page), "Token emphasis")

    def _build_projector_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        explanation = QLabel(
            "Applies one global txtfusion.projector vector before regional LoRA "
            "routing. It cannot be safely localized to a pixel box."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.projector_enabled_input = QCheckBox("Apply global projector vector")
        self.projector_enabled_input.setChecked(False)
        self.projector_enabled_input.setToolTip(
            "Affects all text and image tokens; regional LoRA routing remains unchanged"
        )
        layout.addWidget(self.projector_enabled_input)
        form = QFormLayout()
        self.projector_preset_input = QComboBox()
        for preset in PROJECTOR_PRESETS:
            self.projector_preset_input.addItem(PROJECTOR_PRESET_LABELS[preset], preset)
        self.projector_preset_input.addItem(
            PROJECTOR_PRESET_LABELS[CUSTOM_PROJECTOR_PRESET],
            CUSTOM_PROJECTOR_PRESET,
        )
        form.addRow("Preset", self.projector_preset_input)
        vector_grid = QWidget()
        grid = QGridLayout(vector_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        self.projector_vector_inputs: list[QDoubleSpinBox] = []
        default_values = PROJECTOR_PRESETS[DEFAULT_PROJECTOR_PRESET]
        for index, value in enumerate(default_values):
            row = (index % 4) * 2
            column = (index // 4) * 2
            label = QLabel(f"Vector {index + 1}")
            field = QDoubleSpinBox()
            field.setRange(-1000.0, 1000.0)
            field.setDecimals(4)
            field.setSingleStep(0.0001)
            field.setValue(value)
            field.valueChanged.connect(self._projector_value_edited)
            self.projector_vector_inputs.append(field)
            grid.addWidget(label, row, column)
            grid.addWidget(field, row + 1, column)
        form.addRow("Vector values", vector_grid)
        self.projector_multiplier_input = QDoubleSpinBox()
        self.projector_multiplier_input.setRange(-20.0, 20.0)
        self.projector_multiplier_input.setDecimals(4)
        self.projector_multiplier_input.setSingleStep(0.1)
        self.projector_multiplier_input.setValue(1.0)
        self.projector_multiplier_input.setToolTip(
            "Multiplies every vector value before it is added to the projector weight"
        )
        form.addRow("Global multiplier", self.projector_multiplier_input)
        layout.addLayout(form)
        layout.addStretch(1)
        self.projector_preset_input.currentIndexChanged.connect(
            self._projector_preset_changed
        )
        self.settings_tabs.addTab(self._scrollable(page), "Projector")

    def _projector_values(self) -> tuple[float, ...]:
        return tuple(field.value() for field in self.projector_vector_inputs)

    def _projector_preset_changed(self, *_args) -> None:
        if self._syncing_projector_fields:
            return
        preset = str(self.projector_preset_input.currentData())
        if preset == CUSTOM_PROJECTOR_PRESET:
            return
        self._syncing_projector_fields = True
        try:
            for field, value in zip(
                self.projector_vector_inputs,
                PROJECTOR_PRESETS[preset],
                strict=True,
            ):
                field.setValue(value)
        finally:
            self._syncing_projector_fields = False

    def _projector_value_edited(self, *_args) -> None:
        if self._syncing_projector_fields:
            return
        custom_index = self.projector_preset_input.findData(CUSTOM_PROJECTOR_PRESET)
        if custom_index >= 0:
            self._syncing_projector_fields = True
            try:
                self.projector_preset_input.setCurrentIndex(custom_index)
            finally:
                self._syncing_projector_fields = False

    def _set_projector_controls(
        self,
        *,
        enabled: bool,
        preset: str,
        values: tuple[float, ...],
        multiplier: float,
    ) -> None:
        self._syncing_projector_fields = True
        try:
            self.projector_enabled_input.setChecked(enabled)
            for field, value in zip(self.projector_vector_inputs, values, strict=True):
                field.setValue(value)
            preset_index = self.projector_preset_input.findData(preset)
            if preset_index < 0:
                preset_index = self.projector_preset_input.findData(
                    CUSTOM_PROJECTOR_PRESET
                )
            self.projector_preset_input.setCurrentIndex(preset_index)
            self.projector_multiplier_input.setValue(multiplier)
        finally:
            self._syncing_projector_fields = False

    def _set_upscale_controls_enabled(self, _value=None) -> None:
        enabled = self.post_upscale_input.isChecked()
        self.upscale_scale_input.setEnabled(enabled)
        self.upscale_method_input.setEnabled(enabled)
        model_enabled = enabled and self.upscale_method_input.currentData() == "model"
        self.upscale_model_input.setEnabled(model_enabled)
        self.upscale_model_browse.setEnabled(model_enabled)
        self.upscale_model_clear.setEnabled(model_enabled and self._upscale_model_path is not None)

    def _set_late_step_scale_enabled(self, enabled: bool) -> None:
        self.regional_late_step_scale_input.setEnabled(enabled)

    def _set_lora_delta_adaptation_controls_enabled(self, enabled: bool) -> None:
        self.regional_lora_delta_adaptation_gain_input.setEnabled(enabled)

    def _add_selected_region_emphasis(self) -> None:
        row = self.region_list.currentRow()
        if not 0 <= row < len(self.regions):
            self.statusBar().showMessage("Select a region prompt first", 5000)
            return
        self._add_prompt_emphasis(self.regions[row].region_id, self.region_prompt)

    def _add_prompt_emphasis(self, scope_id: str, editor: QTextEdit) -> None:
        cursor = editor.textCursor()
        phrase = cursor.selectedText().replace("\u2029", "\n")
        if scope_id != GLOBAL_EMPHASIS_SCOPE:
            phrase = phrase.rstrip(".!? ")
        if not phrase.strip():
            self.statusBar().showMessage(
                "Highlight a complete word or phrase before adding emphasis", 5000
            )
            return
        source = editor.toPlainText()
        selection_start = cursor.selectionStart()
        occurrence = 0
        offset = source.find(phrase)
        while 0 <= offset < selection_start:
            occurrence += 1
            offset = source.find(phrase, offset + len(phrase))
        self.prompt_emphases.append(
            PromptEmphasis(
                scope_id=scope_id,
                phrase=phrase,
                strength=self.emphasis_strength_input.value(),
                occurrence=occurrence,
            )
        )
        self._refresh_prompt_emphases(select_row=len(self.prompt_emphases) - 1)
        scope = self._prompt_emphasis_scope_label(scope_id)
        self.events.addItem(f"Added {scope} token emphasis: {phrase!r}")

    def _prompt_emphasis_scope_label(self, scope_id: str) -> str:
        if scope_id == GLOBAL_EMPHASIS_SCOPE:
            return "Global"
        return next(
            (region.name for region in self.regions if region.region_id == scope_id),
            "Missing region",
        )

    def _refresh_prompt_emphases(self, *, select_row: int | None = None) -> None:
        self._syncing_prompt_emphases = True
        try:
            self.prompt_emphasis_list.clear()
            for index, emphasis in enumerate(self.prompt_emphases):
                scope = self._prompt_emphasis_scope_label(emphasis.scope_id)
                item = QListWidgetItem(
                    f"{scope}: {emphasis.phrase!r}  ({emphasis.strength:.2f})"
                )
                item.setData(Qt.ItemDataRole.UserRole, index)
                self.prompt_emphasis_list.addItem(item)
            if select_row is not None and 0 <= select_row < len(self.prompt_emphases):
                self.prompt_emphasis_list.setCurrentRow(select_row)
        finally:
            self._syncing_prompt_emphases = False

    def _selected_prompt_emphasis_changed(self, row: int) -> None:
        if self._syncing_prompt_emphases or not 0 <= row < len(self.prompt_emphases):
            return
        self._syncing_prompt_emphases = True
        try:
            self.emphasis_strength_input.setValue(self.prompt_emphases[row].strength)
        finally:
            self._syncing_prompt_emphases = False

    def _prompt_emphasis_strength_changed(self, strength: float) -> None:
        if self._syncing_prompt_emphases:
            return
        row = self.prompt_emphasis_list.currentRow()
        if not 0 <= row < len(self.prompt_emphases):
            return
        self.prompt_emphases[row] = replace(
            self.prompt_emphases[row], strength=float(strength)
        )
        self._refresh_prompt_emphases(select_row=row)

    def _remove_selected_prompt_emphasis(self) -> None:
        row = self.prompt_emphasis_list.currentRow()
        if not 0 <= row < len(self.prompt_emphases):
            return
        removed = self.prompt_emphases.pop(row)
        self._refresh_prompt_emphases(select_row=min(row, len(self.prompt_emphases) - 1))
        self.events.addItem(f"Removed token emphasis: {removed.phrase!r}")

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
        columns = QHBoxLayout(body)
        library_group = QGroupBox("Library")
        layout = QVBoxLayout(library_group)
        columns.addWidget(library_group, 1)
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
        strength_row = QFormLayout()
        self.lora_strength_input = QDoubleSpinBox()
        self.lora_strength_input.setRange(-4.0, 4.0)
        self.lora_strength_input.setSingleStep(0.05)
        self.lora_strength_input.setDecimals(2)
        self.lora_strength_input.setValue(1.0)
        self.lora_strength_input.setEnabled(False)
        self.lora_strength_input.setToolTip(
            "Model-only LoRA multiplier; set to zero to disable without removing"
        )
        self.lora_strength_input.valueChanged.connect(self._lora_strength_changed)
        strength_row.addRow("Selected strength", self.lora_strength_input)
        layout.addLayout(strength_row)
        self.lora_diagnostic_button = QPushButton("Inspect selected LoRA…")
        self.lora_diagnostic_button.setEnabled(False)
        self.lora_diagnostic_button.clicked.connect(self._diagnose_selected_lora)
        layout.addWidget(self.lora_diagnostic_button)
        self.lora_status = QLabel("Select a LoRA to inspect its Krea compatibility")
        self.lora_status.setWordWrap(True)
        layout.addWidget(self.lora_status)
        scope_group = QGroupBox("Apply selected LoRA")
        scope_layout = QVBoxLayout(scope_group)
        scope_layout.addWidget(QLabel("Choose Global or one or more named regions"))
        self.lora_scope_list = QListWidget()
        self.lora_scope_list.itemChanged.connect(self._lora_scope_changed)
        scope_layout.addWidget(self.lora_scope_list)
        columns.addWidget(scope_group, 1)
        dock.setWidget(self._scrollable(body))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.lora_dock = dock

    def _build_event_dock(self) -> None:
        dock = QDockWidget("Events", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        splitter = QSplitter(Qt.Orientation.Horizontal, dock)
        self.events = EventListWidget(splitter)
        self.resource_monitor = ResourceMonitorWidget(splitter)
        splitter.addWidget(self.events)
        splitter.addWidget(self.resource_monitor)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1100, 320])
        dock.setWidget(splitter)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def _browse_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select generation output folder",
            str(self._output_directory),
        )
        if selected:
            self._output_directory = Path(selected).expanduser().resolve()
            self.output_directory_input.setText(str(self._output_directory))
            self.events.addItem(f"Output folder set to {self._output_directory}")

    def _browse_upscale_model(self) -> None:
        start = (
            self._upscale_model_path.parent
            if self._upscale_model_path is not None
            else self.settings.comfyui_root / "models" / "upscale_models"
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select neural upscaler model",
            str(start),
            "Upscaler models (*.safetensors *.pth *.pt);;All files (*)",
        )
        if selected:
            self._upscale_model_path = Path(selected).expanduser().resolve()
            self.upscale_model_input.setText(str(self._upscale_model_path))
            self.events.addItem(f"Upscaler model set to {self._upscale_model_path.name}")
            self._set_upscale_controls_enabled()

    def _clear_upscale_model(self) -> None:
        self._upscale_model_path = None
        self.upscale_model_input.clear()
        self._set_upscale_controls_enabled()

    def _filename_prefix_edited(self) -> None:
        try:
            prefix = validate_filename_prefix(self.filename_prefix_input.text())
        except ValueError as error:
            self.filename_prefix_input.setText("baseline")
            self.events.addItem(f"Invalid filename prefix: {error}; reset to baseline")
            return
        self.filename_prefix_input.setText(prefix)

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
            f"{region.name} [{region.spatial_role.title()}]: "
            f"[{box.x0:.0f}, {box.y0:.0f}, "
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
        self.generate_button.setEnabled(
            bool(self.artifacts.complete and not self._generation_active)
        )
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
        self._normalize_region_priorities()
        list_item = QListWidgetItem(self._region_label(region))
        list_item.setData(Qt.ItemDataRole.UserRole, region_id)
        self.region_list.addItem(list_item)
        self.canvas.add_region_box(
            region_id,
            QRectF(x0, y0, x1 - x0, y1 - y0),
            region.name,
        )
        self._sync_canvas_region_stack()
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
        removed_emphases = sum(
            emphasis.scope_id == region_id for emphasis in self.prompt_emphases
        )
        self.prompt_emphases = [
            emphasis
            for emphasis in self.prompt_emphases
            if emphasis.scope_id != region_id
        ]
        if removed_emphases:
            self._refresh_prompt_emphases()
        self._normalize_region_priorities()
        self._sync_canvas_region_stack()
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

    def _normalize_region_priorities(self) -> None:
        count = len(self.regions)
        self.regions = [
            replace(region, priority=count - index)
            for index, region in enumerate(self.regions)
        ]

    def _sync_canvas_region_stack(self) -> None:
        self.canvas.set_region_stack_order(
            tuple(region.region_id for region in self.regions)
        )

    def _region_order_changed(self, *_args) -> None:
        by_id = {region.region_id: region for region in self.regions}
        ordered_ids = [
            str(self.region_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.region_list.count())
        ]
        if set(ordered_ids) != set(by_id):
            return
        self.regions = [by_id[region_id] for region_id in ordered_ids]
        self._normalize_region_priorities()
        self._sync_canvas_region_stack()
        self._refresh_lora_scope()
        self._selected_region_changed(self.region_list.currentRow())
        self.events.addItem(
            "Region depth order updated (top row is frontmost)"
        )

    def _selected_region_changed(self, row: int) -> None:
        selected = 0 <= row < len(self.regions)
        self.region_name.setEnabled(selected)
        self.region_role.setEnabled(selected)
        self.region_prompt.setEnabled(selected)
        self.region_negative_prompt.setEnabled(selected)
        self._loading_region_form = True
        try:
            if selected:
                region = self.regions[row]
                self.region_name.setText(region.name)
                role_index = self.region_role.findData(region.spatial_role)
                self.region_role.setCurrentIndex(max(0, role_index))
                self.region_prompt.setPlainText(region.prompt)
                self.region_negative_prompt.setPlainText(region.negative_prompt)
                self.canvas.select_region(region.region_id)
            else:
                self.region_name.clear()
                self.region_role.setCurrentIndex(0)
                self.region_prompt.clear()
                self.region_negative_prompt.clear()
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
        self._refresh_prompt_emphases()
        self.events.addItem(f"Renamed {region.name} to {name}")

    def _region_role_changed(self, *_args) -> None:
        if self._loading_region_form:
            return
        row = self.region_list.currentRow()
        if not 0 <= row < len(self.regions):
            return
        region = self.regions[row]
        spatial_role = str(self.region_role.currentData())
        if spatial_role == region.spatial_role:
            return
        self.regions[row] = replace(region, spatial_role=spatial_role)
        item = self._region_list_item(region.region_id)
        if item is not None:
            item.setText(self._region_label(self.regions[row]))
        self.events.addItem(
            f"Set {region.name} spatial role to {self.region_role.currentText()}"
        )

    def _region_form_edited(self) -> None:
        if self._loading_region_form:
            return
        row = self.region_list.currentRow()
        if not 0 <= row < len(self.regions):
            return
        self.regions[row] = replace(
            self.regions[row],
            prompt=self.region_prompt.toPlainText(),
            negative_prompt=self.region_negative_prompt.toPlainText(),
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
            existing = QListWidgetItem(self._lora_label(entry.lora_id))
            existing.setData(Qt.ItemDataRole.UserRole, entry.lora_id)
            existing.setToolTip(str(entry.path))
            self.lora_list.addItem(existing)
            self.events.addItem(f"Loaded LoRA {entry.display_name}; scope defaults to Global")
        self.lora_list.setCurrentItem(existing)
        return True

    def _lora_label(self, lora_id: str) -> str:
        entry = self.lora_library.get(lora_id)
        binding = self.lora_library.binding_for(lora_id)
        return f"{entry.display_name}  ×{binding.strength:.2f}"

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
        lora_id = self._current_lora_id()
        self._syncing_lora_strength = True
        try:
            self.lora_strength_input.setEnabled(lora_id is not None)
            self.lora_diagnostic_button.setEnabled(lora_id is not None)
            if lora_id is None:
                self.lora_strength_input.setValue(1.0)
                self.lora_status.setText("Select a LoRA to inspect its Krea compatibility")
            else:
                binding = self.lora_library.binding_for(lora_id)
                self.lora_strength_input.setValue(binding.strength)
                entry = self.lora_library.get(lora_id)
                self.lora_status.setText(
                    f"{entry.summary.tensor_count} tensors; compatibility not yet checked"
                )
        finally:
            self._syncing_lora_strength = False

    def _lora_strength_changed(self, strength: float) -> None:
        if self._syncing_lora_strength:
            return
        lora_id = self._current_lora_id()
        if lora_id is None:
            return
        self.lora_library.set_strength(lora_id, strength)
        item = self._lora_list_item(lora_id)
        if item is not None:
            item.setText(self._lora_label(lora_id))
        entry = self.lora_library.get(lora_id)
        self.events.addItem(f"Set {entry.display_name} strength to {strength:.2f}")

    def _lora_payload(self, lora_ids: set[str] | None = None) -> list[dict[str, object]]:
        payload = []
        for entry in self.lora_library.entries():
            if lora_ids is not None and entry.lora_id not in lora_ids:
                continue
            binding = self.lora_library.binding_for(entry.lora_id)
            payload.append(
                {
                    "id": entry.lora_id,
                    "name": entry.display_name,
                    "path": str(entry.path),
                    "strength": binding.strength,
                    "global": binding.global_scope,
                    "region_ids": list(binding.region_ids),
                }
            )
        return payload

    def _diagnose_selected_lora(self) -> None:
        lora_id = self._current_lora_id()
        if lora_id is None:
            return
        if not self.worker_client.running or not self._model_loaded:
            self.events.addItem("Load the Krea 2 baseline before inspecting LoRA compatibility")
            self.lora_status.setText("Baseline must be loaded before compatibility testing")
            return
        payload = self._worker_payload()
        payload["loras"] = self._lora_payload({lora_id})
        self.lora_status.setText("Inspecting model-key compatibility…")
        self.worker_client.send(CommandKind.VALIDATE_LORAS, payload)

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
            "output_directory": str(self._output_directory),
            "filename_prefix": validate_filename_prefix(
                self.filename_prefix_input.text()
            ),
        }
        saved_loras = tuple(
            SavedLora(
                path=entry.path,
                global_scope=(binding := self.lora_library.binding_for(entry.lora_id)).global_scope,
                region_ids=binding.region_ids,
                strength=binding.strength,
            )
            for entry in self.lora_library.entries()
        )
        return ProjectState(
            canvas_width=self.width_input.value(),
            canvas_height=self.height_input.value(),
            global_prompt=self.global_prompt.toPlainText(),
            steps=self.steps_input.value(),
            seed=self.seed_input.value(),
            seed_mode=str(self.seed_mode_input.currentData()),
            regional_prompting=self.regional_prompting_input.isChecked(),
            regional_prompt_strength=self.regional_prompt_strength_input.value(),
            regional_outside_penalty=self.regional_outside_penalty_input.value(),
            regional_feather_pixels=self.regional_feather_input.value(),
            regional_subject_competition=(
                self.regional_subject_competition_input.isChecked()
            ),
            regional_subject_fill=self.regional_subject_fill_input.isChecked(),
            regional_relaxation=self.regional_relaxation_input.isChecked(),
            regional_late_step_scale=self.regional_late_step_scale_input.value(),
            regional_lora_delta_adaptation=(
                self.regional_lora_delta_adaptation_input.isChecked()
            ),
            regional_lora_delta_adaptation_gain=(
                self.regional_lora_delta_adaptation_gain_input.value()
            ),
            prompt_emphases=tuple(self.prompt_emphases),
            projector_enabled=self.projector_enabled_input.isChecked(),
            projector_preset=str(self.projector_preset_input.currentData()),
            projector_values=self._projector_values(),
            projector_multiplier=self.projector_multiplier_input.value(),
            post_upscale=self.post_upscale_input.isChecked(),
            upscale_scale=int(self.upscale_scale_input.currentData()),
            upscale_method=str(self.upscale_method_input.currentData()),
            upscale_model=self._upscale_model_path,
            regions=tuple(self.regions),
            loras=saved_loras,
            runtime=runtime,
            background_image=self._background_image_path,
        )

    def _open_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open K2 Region Lab project",
            str(self._project_directory),
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
        filename = (
            self._current_project_path.name
            if self._current_project_path
            else "untitled.k2lab.json"
        )
        start = self._project_directory / filename
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
        data_directory = Path(
            runtime.get("data_directory", current.data_directory)
        ).expanduser()
        current_output = current.output_directory or default_output_directory(
            current.data_directory
        )
        saved_output = runtime.get("output_directory")
        legacy_output = data_directory / "baseline_outputs"
        output_directory = (
            current_output
            if saved_output is None
            or Path(saved_output).expanduser() == legacy_output
            else Path(saved_output).expanduser()
        )
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
            data_directory=data_directory,
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
            output_directory=output_directory,
            filename_prefix=validate_filename_prefix(
                runtime.get("filename_prefix", current.filename_prefix)
            ),
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
        seed_mode_index = self.seed_mode_input.findData(state.seed_mode)
        self.seed_mode_input.setCurrentIndex(max(0, seed_mode_index))
        self.regional_prompting_input.setChecked(state.regional_prompting)
        self.regional_prompt_strength_input.setValue(state.regional_prompt_strength)
        self.regional_outside_penalty_input.setValue(state.regional_outside_penalty)
        self.regional_feather_input.setValue(state.regional_feather_pixels)
        self.regional_subject_competition_input.setChecked(
            state.regional_subject_competition
        )
        self.regional_subject_fill_input.setChecked(state.regional_subject_fill)
        self.regional_relaxation_input.setChecked(state.regional_relaxation)
        self.regional_late_step_scale_input.setValue(state.regional_late_step_scale)
        self.regional_lora_delta_adaptation_input.setChecked(
            state.regional_lora_delta_adaptation
        )
        self.regional_lora_delta_adaptation_gain_input.setValue(
            state.regional_lora_delta_adaptation_gain
        )
        self.prompt_emphases = list(state.prompt_emphases)
        self._refresh_prompt_emphases()
        self._set_projector_controls(
            enabled=state.projector_enabled,
            preset=state.projector_preset,
            values=state.projector_values,
            multiplier=state.projector_multiplier,
        )
        self.post_upscale_input.setChecked(state.post_upscale)
        scale_index = self.upscale_scale_input.findData(state.upscale_scale)
        self.upscale_scale_input.setCurrentIndex(max(0, scale_index))
        method_index = self.upscale_method_input.findData(state.upscale_method)
        self.upscale_method_input.setCurrentIndex(max(0, method_index))
        self._upscale_model_path = state.upscale_model
        self.upscale_model_input.setText(
            str(self._upscale_model_path) if self._upscale_model_path else ""
        )
        self._set_upscale_controls_enabled()
        policy_index = self.memory_policy_input.findData(self.settings.memory_policy)
        self.memory_policy_input.setCurrentIndex(max(0, policy_index))
        self.reserve_vram_input.setValue(self.settings.reserve_vram_gb)
        self.minimum_ram_input.setValue(self.settings.minimum_system_ram_gb)
        self.cpu_vae_input.setChecked(self.settings.cpu_vae)
        self.oom_recovery_input.setChecked(self.settings.oom_recovery)
        self._output_directory = (
            self.settings.output_directory
            or default_output_directory(self.settings.data_directory)
        )
        self.output_directory_input.setText(str(self._output_directory))
        self.filename_prefix_input.setText(self.settings.filename_prefix)

        self.canvas.clear_regions()
        self.region_list.clear()
        self.regions = list(state.regions)
        self._normalize_region_priorities()
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
        self._sync_canvas_region_stack()

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
            self.lora_library.set_strength(lora_id, saved_lora.strength)
            item = self._lora_list_item(lora_id)
            if item is not None:
                item.setText(self._lora_label(lora_id))
        self._selected_lora_changed(self.lora_list.currentItem(), None)

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
        if self.worker_client.running:
            self._advance_pending_generation()
            return
        self._model_loaded = False
        self._accelerator_available = False
        if not self.worker_client.start():
            return
        if not self.worker_client.process.waitForStarted(3000):
            self.events.addItem("GPU worker did not start within three seconds")
            return
        self._worker_bootstrap_stage = "probe"
        self.worker_client.send(CommandKind.PROBE, self._worker_payload())

    def _diagnose_accelerator(self) -> None:
        self.events.addItem("Restarting the GPU worker for a clean accelerator diagnostic")
        self.worker_client.stop()
        self._model_loaded = False
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
        self._model_loaded = False
        self.worker_status.setText("Stopped")
        self.accelerator_status.setText("Not probed")
        self.memory_status.setText("K2 workers stopped; GPU allocations released")
        self.load_model_button.setEnabled(False)
        self.generate_button.setEnabled(bool(self.artifacts and self.artifacts.complete))
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
                "K2 GPU workers stopped; click Generate when ready", 8000
            )

    def _validate_worker_models(self) -> None:
        if self._model_loaded:
            self.events.addItem("Krea baseline is already loaded; validation skipped")
            return
        if not self.worker_client.running:
            self._start_worker()
        if self.worker_client.running:
            self._worker_bootstrap_stage = "validate"
            self.worker_client.send(CommandKind.VALIDATE_MODELS, self._worker_payload())

    def _load_worker_model(self) -> None:
        if self._model_loaded:
            self.events.addItem("Krea baseline is already loaded; duplicate load skipped")
            return
        if self.worker_client.running and self._worker_bootstrap_stage is None:
            self._worker_bootstrap_stage = "load"
            self.load_model_button.setEnabled(False)
            self._set_memory_controls_enabled(False)
            self.worker_client.send(CommandKind.LOAD_MODEL, self._worker_payload())

    def _advance_pending_generation(self) -> None:
        if self._pending_generation_payload is None or self._worker_bootstrap_stage:
            return
        if not self.worker_client.running:
            self.events.addItem("Starting a fresh disposable generation worker")
            self._start_worker()
            return
        if not self._accelerator_available:
            self._worker_bootstrap_stage = "probe"
            self.worker_client.send(CommandKind.PROBE, self._worker_payload())
            return
        if not self._models_compatible:
            self._worker_bootstrap_stage = "validate"
            self.worker_client.send(CommandKind.VALIDATE_MODELS, self._worker_payload())
            return
        if not self._model_loaded:
            self._load_worker_model()
            return
        payload = self._pending_generation_payload
        try:
            self.worker_client.send(CommandKind.GENERATE_BASELINE, payload)
        except RuntimeError as error:
            self._pending_generation_payload = None
            self._set_generation_active(False)
            self.events.addItem(f"Could not start generation: {error}")
            return
        self._pending_generation_payload = None
        self.events.addItem("Fresh worker ready; generation dispatched")

    def _generate_baseline(self) -> None:
        self._filename_prefix_edited()
        if (
            self.post_upscale_input.isChecked()
            and self.upscale_method_input.currentData() == "model"
            and (
                self._upscale_model_path is None
                or not self._upscale_model_path.is_file()
            )
        ):
            message = "Select a readable neural upscaler model before generation."
            self.events.addItem(message)
            QMessageBox.warning(self, "Upscaler model required", message)
            return
        seed_mode = str(self.seed_mode_input.currentData())
        seed = self.seed_input.value()
        if seed_mode == "random":
            seed = secrets.randbelow(2_147_483_648)
            self.seed_input.setValue(seed)
        elif seed_mode == "increment":
            self.seed_input.setValue((seed + 1) % 2_147_483_648)
        geometry = CanvasGeometry.resolve(self.width_input.value(), self.height_input.value())
        payload = self._worker_payload()
        payload.update(
            {
                "prompt": self.global_prompt.toPlainText(),
                "width": geometry.aligned_width,
                "height": geometry.aligned_height,
                "steps": self.steps_input.value(),
                "seed": seed,
                "seed_mode": seed_mode,
                "output_directory": str(self._output_directory),
                "filename_prefix": validate_filename_prefix(
                    self.filename_prefix_input.text()
                ),
                "regional_prompting": self.regional_prompting_input.isChecked(),
                "regional_prompt_strength": self.regional_prompt_strength_input.value(),
                "regional_outside_penalty": (
                    self.regional_outside_penalty_input.value()
                ),
                "regional_feather_pixels": self.regional_feather_input.value(),
                "regional_subject_competition": (
                    self.regional_subject_competition_input.isChecked()
                ),
                "regional_subject_fill": self.regional_subject_fill_input.isChecked(),
                "regional_late_step_scale": (
                    self.regional_late_step_scale_input.value()
                    if self.regional_relaxation_input.isChecked()
                    else 1.0
                ),
                "regional_lora_delta_adaptation": (
                    self.regional_lora_delta_adaptation_input.isChecked()
                ),
                "regional_lora_delta_adaptation_gain": (
                    self.regional_lora_delta_adaptation_gain_input.value()
                ),
                "prompt_emphases": [
                    {
                        "scope_id": emphasis.scope_id,
                        "phrase": emphasis.phrase,
                        "strength": emphasis.strength,
                        "occurrence": emphasis.occurrence,
                    }
                    for emphasis in self.prompt_emphases
                ],
                "projector_enabled": self.projector_enabled_input.isChecked(),
                "projector_preset": str(self.projector_preset_input.currentData()),
                "projector_values": list(self._projector_values()),
                "projector_multiplier": self.projector_multiplier_input.value(),
                "post_upscale": self.post_upscale_input.isChecked(),
                "upscale_scale": int(self.upscale_scale_input.currentData()),
                "upscale_method": str(self.upscale_method_input.currentData()),
                "upscale_model_path": (
                    str(self._upscale_model_path)
                    if self._upscale_model_path is not None
                    else None
                ),
                "regions": [
                    {
                        "id": region.region_id,
                        "name": region.name,
                        "box": {
                            "x0": region.box.x0,
                            "y0": region.box.y0,
                            "x1": region.box.x1,
                            "y1": region.box.y1,
                        },
                        "prompt": region.prompt,
                        "negative_prompt": region.negative_prompt,
                        "enabled": region.enabled,
                        "priority": region.priority,
                        "spatial_role": region.spatial_role,
                    }
                    for region in self.regions
                ],
                "loras": self._lora_payload(),
            }
        )
        self._pending_generation_payload = payload
        self._generation_completed = False
        self._set_generation_active(True)
        self._advance_pending_generation()

    def _preview_unified_prompt(self) -> None:
        plan = compile_regional_prompt_plan(
            self.width_input.value(),
            self.height_input.value(),
            self.global_prompt.toPlainText(),
            tuple(self.regions),
            strength=self.regional_prompt_strength_input.value(),
            outside_penalty=self.regional_outside_penalty_input.value(),
            falloff_pixels=self.regional_feather_input.value(),
            subject_competition=(
                self.regional_subject_competition_input.isChecked()
            ),
            subject_fill=self.regional_subject_fill_input.isChecked(),
            late_step_scale=(
                self.regional_late_step_scale_input.value()
                if self.regional_relaxation_input.isChecked()
                else 1.0
            ),
            emphases=tuple(self.prompt_emphases),
        )
        preview = QDialog(self)
        preview.setWindowTitle("Unified spatial prompt")
        preview.setMinimumSize(520, 360)
        preview.resize(900, 650)
        preview.setSizeGripEnabled(True)
        preview_layout = QVBoxLayout(preview)
        summary = QLabel(
            f"{len(plan.regions)} regional clauses will be encoded in one prompt."
        )
        summary.setWordWrap(True)
        preview_layout.addWidget(summary)
        explanation = QLabel(
            "Pixel boxes are applied separately as a hidden soft attention grid."
        )
        explanation.setWordWrap(True)
        preview_layout.addWidget(explanation)
        prompt_text = QTextEdit()
        prompt_text.setReadOnly(True)
        prompt_text.setPlainText(plan.prompt)
        preview_layout.addWidget(prompt_text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(preview.reject)
        preview_layout.addWidget(buttons)
        self._prompt_preview_dialog = preview
        preview.exec()

    def _set_generation_active(self, active: bool) -> None:
        self._generation_active = active
        self.generate_button.setEnabled(
            bool(not active and self.artifacts is not None and self.artifacts.complete)
        )
        self.stop_generation_button.setEnabled(active and self.worker_client.running)

    def _stop_generation(self) -> None:
        if not self._generation_active:
            return
        pid = self.worker_client.cancel_generation()
        self._pending_generation_payload = None
        self._worker_bootstrap_stage = None
        self._set_generation_active(False)
        self._accelerator_available = False
        self._model_loaded = False
        self.worker_status.setText("Stopped")
        self.accelerator_status.setText("Not probed")
        self.memory_status.setText("Generation stopped; worker memory released")
        self.load_model_button.setEnabled(False)
        self.generate_button.setEnabled(bool(self.artifacts and self.artifacts.complete))
        self._set_memory_controls_enabled(True)
        detail = f" (worker PID {pid})" if pid is not None else ""
        self.events.addItem(f"Generation stopped by user{detail}; GPU/RAM released")
        self.statusBar().showMessage(
            "Generation stopped — make changes, then click Generate when ready", 10000
        )

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
                bool(
                    self._models_compatible
                    and self._accelerator_available
                    and not self._model_loaded
                )
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
            if message == "Worker runtime probe complete":
                self._worker_bootstrap_stage = None
                if (
                    self._pending_generation_payload is not None
                    and not self._accelerator_available
                ):
                    self._pending_generation_payload = None
                    self._set_generation_active(False)
                    self.events.addItem(
                        "Automatic generation stopped: accelerator probe failed"
                    )
        if "manifests" in payload:
            compatible = payload.get("complete") and all(
                manifest.get("compatible") for manifest in payload["manifests"]
            )
            self._models_compatible = bool(compatible)
            self._worker_bootstrap_stage = None
            if self._pending_generation_payload is not None and not compatible:
                self._pending_generation_payload = None
                self._set_generation_active(False)
                self.events.addItem(
                    "Automatic generation stopped: model validation failed"
                )
            self.load_model_button.setEnabled(
                bool(compatible and self._accelerator_available and not self._model_loaded)
            )
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
        elif message in {
            "Krea 2 baseline components loaded",
            "Krea 2 baseline already loaded",
        }:
            self._worker_bootstrap_stage = None
            self._model_loaded = True
            self.statusBar().showMessage("Krea 2 baseline loaded in GPU worker")
            if "reserve_vram_gb" in payload:
                self.reserve_vram_input.setValue(float(payload["reserve_vram_gb"]))
            self._set_memory_controls_enabled(False)
            self.validate_models_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
            self._advance_pending_generation()
        elif message == "LoRA diagnostics complete":
            reports = payload.get("loras", [])
            if reports:
                report_data = reports[0]
                matched = int(report_data.get("matched_model_targets", 0))
                adapters = int(report_data.get("adapter_count", 0))
                compatible = bool(report_data.get("compatible"))
                self.lora_status.setText(
                    f"{'Compatible' if compatible else 'Incompatible'}: "
                    f"{matched}/{adapters} Krea model targets matched"
                )
                report = QMessageBox(self)
                report.setWindowTitle("LoRA compatibility")
                report.setIcon(
                    QMessageBox.Icon.Information
                    if compatible
                    else QMessageBox.Icon.Warning
                )
                report.setText(self.lora_status.text())
                report.setInformativeText(
                    "Compatible LoRAs can be routed globally or to one or more named "
                    "regions. Regional routes gate both their prompt-token spans and "
                    "their exact pixel-box image tokens."
                )
                report.setDetailedText(json.dumps(report_data, indent=2, sort_keys=True))
                report.exec()
        elif message in {"Generation complete", "Baseline generation complete"}:
            self._generation_completed = True
            self._set_generation_active(False)
            image_path = payload.get("image_path")
            if image_path and self.canvas.set_image(image_path):
                self._background_image_path = Path(image_path)
                self.statusBar().showMessage(f"Image saved to {image_path}")
            if payload.get("oom_recovered"):
                self.reserve_vram_input.setValue(
                    max(5.0, self.reserve_vram_input.value())
                )
                self.cpu_vae_input.setChecked(True)
                self.events.addItem(
                    "Generation recovered from GPU OOM; future fresh workers will start "
                    "with CPU VAE and at least 5 GiB reserved"
                )
            # The worker exits immediately after this event. Keep Generate disabled
            # until QProcess confirms that all GPU/system allocations are gone.
            self.generate_button.setEnabled(False)
        elif message == "Generation worker releasing GPU and system RAM":
            self.memory_status.setText("Releasing generation worker memory…")
        elif state == "error":
            self._pending_generation_payload = None
            self._worker_bootstrap_stage = None
            self._set_generation_active(False)
            if message != "LoRA diagnostics complete":
                self._model_loaded = False
            self.load_model_button.setEnabled(self._accelerator_available)
            self.generate_button.setEnabled(False)
            self._set_memory_controls_enabled(True)
            normalized_message = message.casefold()
            if any(
                marker in normalized_message
                for marker in ("out of memory", "gpu memory pressure", "gpu oom")
            ):
                self.events.addItem(
                    "16 GB guidance: use Release K2 GPU memory, restart the worker, "
                    "close RAM-heavy applications, or reduce the canvas when using "
                    "multiple LoRAs"
                )
                self.statusBar().showMessage(
                    "Generation exceeded the 16 GB limit — release the K2 worker before retrying",
                    15000,
                )

        if self._pending_generation_payload is not None and state != "error":
            self._advance_pending_generation()

    def _worker_stderr(self, output: str) -> None:
        logging.getLogger(__name__).debug("worker stderr received: %s", output)
        for line in output.splitlines():
            self.events.addItem(f"Worker stderr: {line}")

    def _worker_process_status(self, status: str) -> None:
        self.worker_status.setText(status)
        self.events.addItem(f"GPU worker process: {status}")
        if status.startswith("stopped") or "error" in status:
            completed = self._generation_completed
            self._generation_completed = False
            if not completed:
                self._pending_generation_payload = None
            self._model_loaded = False
            self._accelerator_available = False
            self._models_compatible = False
            self._worker_bootstrap_stage = None
            self._set_generation_active(False)
            self.generate_button.setEnabled(bool(self.artifacts and self.artifacts.complete))
            self.load_model_button.setEnabled(False)
            self.validate_models_button.setEnabled(True)
            self._set_memory_controls_enabled(True)
            if completed:
                self.memory_status.setText("GPU and generation RAM released")
                self.events.addItem(
                    "Disposable generation worker exited; GPU/system RAM released"
                )

    def closeEvent(self, event) -> None:
        self.worker_client.stop()
        super().closeEvent(event)
