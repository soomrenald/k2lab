from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
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

from k2_region_lab.config import AppSettings, ModelDirectories, discover_worker_python
from k2_region_lab.desktop.region_canvas import RegionCanvas
from k2_region_lab.desktop.resource_monitor import ResourceMonitorWidget
from k2_region_lab.desktop.worker_client import ExternalWorkerClient
from k2_region_lab.face_detail import (
    DetectedFace,
    assign_faces_to_regional_loras,
    discover_face_detector,
    expanded_square_crop,
)
from k2_region_lab.image_edit import ImageEditState, load_source_image
from k2_region_lab.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    STANDARD_LORA_ROUTING,
    LoraBinding,
    LoraLibrary,
)
from k2_region_lab.memory import (
    MEMORY_POLICIES,
    effective_minimum_system_ram_gb,
    effective_reserve_vram_gb,
    memory_policy,
)
from k2_region_lab.model import (
    ArtifactSet,
    discover_krea_transformers,
    discover_model_artifacts,
)
from k2_region_lab.output import (
    default_output_directory,
    default_prompt_directory,
    validate_filename_prefix,
)
from k2_region_lab.processes import find_owned_k2_workers, terminate_workers
from k2_region_lab.project import (
    ProjectState,
    SavedLora,
    load_project,
    load_project_image,
    project_document,
    save_project,
)
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
from k2_region_lab.regional_lora import character_identity_triggers
from k2_region_lab.regions import CanvasGeometry, PixelBox, RegionDefinition
from k2_region_lab.sampling import COMFYUI_SAMPLERS, COMFYUI_SCHEDULERS
from k2_region_lab.worker.protocol import CommandKind


GLOBAL_SCOPE_ID = "__global__"
DISABLED_SCOPE_ID = "__disabled__"


class EventListWidget(QListWidget):
    """Follow new events only while the user is already viewing the end."""

    def addItem(self, item) -> None:
        message = item.text() if isinstance(item, QListWidgetItem) else str(item)
        logging.getLogger("k2_region_lab.events").info(message)
        scrollbar = self.verticalScrollBar()
        follow_latest = scrollbar.value() >= scrollbar.maximum()
        super().addItem(item)
        if follow_latest:
            self.scrollToBottom()


class ScaledImagePreview(QLabel):
    """A simple aspect-fit PNG preview that keeps the original pixmap in memory."""

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._original_pixmap = QPixmap()
        self._overlays: list[tuple[float, float, float, float, str, bool]] = []
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Keep the comparison usable when the bottom Events dock is dragged up.
        # The preview expands normally, but must not impose a tall minimum on the
        # complete upper dock row.
        self.setMinimumSize(128, 128)
        self.setWordWrap(True)
        self.setStyleSheet(
            "QLabel { background: #191919; border: 1px solid #555; color: #aaa; }"
        )

    def set_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self._original_pixmap = pixmap
        self._fit_pixmap()
        return True

    def clear_image(self, placeholder: str) -> None:
        self._original_pixmap = QPixmap()
        self._overlays = []
        self.clear()
        self.setText(placeholder)

    def set_overlays(
        self,
        overlays: list[tuple[float, float, float, float, str, bool]],
    ) -> None:
        self._overlays = list(overlays)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._original_pixmap.isNull():
            return
        fitted = self._original_pixmap.scaled(
            self.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._overlays:
            annotated = fitted.copy()
            painter = QPainter(annotated)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            scale_x = fitted.width() / max(1, self._original_pixmap.width())
            scale_y = fitted.height() / max(1, self._original_pixmap.height())
            for x0, y0, x1, y1, label, selected in self._overlays:
                color = QColor("#39ff88" if selected else "#ffb340")
                painter.setPen(QPen(color, 3))
                rectangle = QRectF(
                    x0 * scale_x,
                    y0 * scale_y,
                    max(1.0, (x1 - x0) * scale_x),
                    max(1.0, (y1 - y0) * scale_y),
                )
                painter.drawRect(rectangle)
                painter.fillRect(
                    QRectF(rectangle.left(), rectangle.top(), max(28, len(label) * 8), 22),
                    color,
                )
                painter.setPen(QColor("#101010"))
                painter.drawText(
                    QRectF(rectangle.left() + 4, rectangle.top(), max(24, len(label) * 8), 22),
                    Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
            painter.end()
            fitted = annotated
        self.setPixmap(fitted)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()


class LassoImagePreview(ScaledImagePreview):
    lassos_changed = Signal(object)

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._drawing_enabled = False
        self._lassos: list[tuple[tuple[float, float], ...]] = []
        self._active_lasso: list[tuple[float, float]] = []
        self._selected_lassos: set[int] = set()
        self._lassos_visible = False

    def lasso_paths(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        return tuple(self._lassos)

    def set_lasso_paths(self, paths, *, emit: bool = True) -> None:
        self._lassos = [tuple(tuple(point) for point in path) for path in paths]
        self._active_lasso = []
        self._selected_lassos = set(range(len(self._lassos)))
        self._fit_pixmap()
        if emit:
            self.lassos_changed.emit(list(self._lassos))

    def set_lassos_visible(self, visible: bool) -> None:
        self._lassos_visible = bool(visible)
        self._fit_pixmap()

    def set_drawing_enabled(self, enabled: bool) -> None:
        self._drawing_enabled = bool(enabled)
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
            self._active_lasso = []
            self._fit_pixmap()

    def clear_lassos(self, *, emit: bool = True) -> None:
        self._lassos = []
        self._active_lasso = []
        self._selected_lassos = set()
        self._fit_pixmap()
        if emit:
            self.lassos_changed.emit([])

    def undo_lasso(self) -> None:
        if not self._lassos:
            return
        self._lassos.pop()
        self._selected_lassos = set(range(len(self._lassos)))
        self._fit_pixmap()
        self.lassos_changed.emit(list(self._lassos))

    def set_selected_lassos(self, selected: set[int]) -> None:
        self._selected_lassos = set(selected)
        self._fit_pixmap()

    def _image_point(self, widget_point: QPointF) -> tuple[float, float] | None:
        displayed = self.pixmap()
        if displayed is None or displayed.isNull() or self._original_pixmap.isNull():
            return None
        area = self.contentsRect()
        left = area.left() + (area.width() - displayed.width()) / 2.0
        top = area.top() + (area.height() - displayed.height()) / 2.0
        local_x = widget_point.x() - left
        local_y = widget_point.y() - top
        if not (0.0 <= local_x <= displayed.width() and 0.0 <= local_y <= displayed.height()):
            return None
        return (
            local_x * self._original_pixmap.width() / max(1, displayed.width()),
            local_y * self._original_pixmap.height() / max(1, displayed.height()),
        )

    def _fit_pixmap(self) -> None:
        super()._fit_pixmap()
        displayed = self.pixmap()
        if (
            displayed is None
            or displayed.isNull()
            or self._original_pixmap.isNull()
            or not ((self._lassos_visible and self._lassos) or self._active_lasso)
        ):
            return
        annotated = displayed.copy()
        painter = QPainter(annotated)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale_x = annotated.width() / max(1, self._original_pixmap.width())
        scale_y = annotated.height() / max(1, self._original_pixmap.height())
        if self._lassos_visible:
            for index, path in enumerate(self._lassos):
                color = QColor("#39ff88" if index in self._selected_lassos else "#ffb340")
                polygon = QPolygonF(
                    [QPointF(x * scale_x, y * scale_y) for x, y in path]
                )
                painter.setPen(QPen(color, 3))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 45))
                painter.drawPolygon(polygon)
                if polygon:
                    painter.setPen(color)
                    painter.drawText(
                        polygon.boundingRect().topLeft() + QPointF(4, 18),
                        str(index + 1),
                    )
        if self._active_lasso:
            painter.setPen(QPen(QColor("#42c7f5"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(
                QPolygonF(
                    [
                        QPointF(x * scale_x, y * scale_y)
                        for x, y in self._active_lasso
                    ]
                )
            )
        painter.end()
        self.setPixmap(annotated)

    def mousePressEvent(self, event) -> None:
        if self._drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self._image_point(event.position())
            if point is not None:
                self._active_lasso = [point]
                self._fit_pixmap()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing_enabled and self._active_lasso:
            point = self._image_point(event.position())
            if point is not None:
                previous = self._active_lasso[-1]
                if (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2 >= 4.0:
                    self._active_lasso.append(point)
                    self._fit_pixmap()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            if len(self._active_lasso) >= 3:
                self._lassos.append(tuple(self._active_lasso))
                self._selected_lassos = set(range(len(self._lassos)))
            self._active_lasso = []
            self._fit_pixmap()
            self.lassos_changed.emit(list(self._lassos))
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self._default_settings = settings
        self.settings = settings
        self.artifacts: ArtifactSet | None = None
        self.regions: list[RegionDefinition] = []
        self.prompt_emphases: list[PromptEmphasis] = []
        self.lora_library = LoraLibrary()
        self._region_number = 0
        self._loading_region_form = False
        self._syncing_lora_scope = False
        self._syncing_lora_strength = False
        self._syncing_lora_routing = False
        self._syncing_projector_fields = False
        self._syncing_prompt_emphases = False
        self._models_compatible = False
        self._model_loaded = False
        self._current_project_path: Path | None = None
        self._background_image_path: Path | None = None
        self.edit_regions: list[RegionDefinition] = []
        self._edit_region_number = 0
        self._edit_loading_region_form = False
        self._edit_source_path: Path | None = None
        self._edit_result_path: Path | None = None
        self._edit_lora_bindings: dict[str, LoraBinding] = {}
        self._face_source_path: Path | None = None
        self._face_result_path: Path | None = None
        self._face_detections: list[dict[str, object]] = []
        self._manual_face_paths: list[tuple[tuple[float, float], ...]] = []
        self._lasso_dialog_active = False
        self._syncing_face_selection = False
        self._upscale_model_path = settings.default_upscale_model
        self._generation_active = False
        self._pending_generation_payload: dict[str, object] | None = None
        self._pending_image_edit_payload: dict[str, object] | None = None
        self._pending_face_refinement_payload: dict[str, object] | None = None
        self._batch_base_payload: dict[str, object] | None = None
        self._batch_runs_remaining = 0
        self._batch_completed = 0
        self._batch_total = 0
        self._active_task: str | None = None
        self._worker_bootstrap_stage: str | None = None
        self._generation_completed = False
        self._prompt_preview_dialog: QDialog | None = None
        self._project_directory = default_prompt_directory()
        self._output_directory = settings.output_directory or default_output_directory(
            settings.data_directory
        )
        self.setWindowTitle("K2 Region Lab")
        self.setDockNestingEnabled(True)
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._build_file_menu()

        self.canvas = RegionCanvas(settings.default_width, settings.default_height)
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setDocumentMode(True)
        generation_page = QWidget()
        generation_layout = QVBoxLayout(generation_page)
        generation_layout.setContentsMargins(0, 0, 0, 0)
        canvas_actions = QHBoxLayout()
        canvas_actions.addStretch(1)
        self.clear_canvas_button = QPushButton("Clear canvas image")
        self.clear_canvas_button.setToolTip(
            "Remove the loaded/generated image without changing regions or prompts"
        )
        self.clear_canvas_button.clicked.connect(self._clear_generation_canvas)
        canvas_actions.addWidget(self.clear_canvas_button)
        generation_layout.addLayout(canvas_actions)
        generation_layout.addWidget(self.canvas, 1)
        self.workspace_tabs.addTab(generation_page, "Generation canvas")
        self._build_image_edit_tab()
        self._build_face_refinement_tab()
        self.workspace_tabs.currentChanged.connect(self._workspace_tab_changed)
        self.setCentralWidget(self.workspace_tabs)
        self.canvas.region_created.connect(self._region_created)
        self.canvas.region_changed.connect(self._region_changed)
        self.canvas.region_deleted.connect(self._region_deleted)
        self.canvas.region_selected.connect(self._canvas_region_selected)

        self._build_prompt_dock()
        self._build_model_dock()
        self._build_event_dock()
        self._build_view_menu()
        self._fit_initial_window_to_screen()
        self.worker_client = ExternalWorkerClient(settings, self)
        self.worker_client.event_received.connect(self._worker_event)
        self.worker_client.stderr_received.connect(self._worker_stderr)
        self.worker_client.process_status.connect(self._worker_process_status)
        self._use_latest_face_source(show_message=False)
        self._accelerator_available = False
        self.statusBar().showMessage("Ready — model not loaded")
        log_path = self.settings.data_directory / "logs" / "desktop-debug.log"
        self.events.addItem(f"Event logging enabled: {log_path}")
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
        self.file_menu = self.menuBar().addMenu("&File")
        menu = self.file_menu
        new_action = QAction("&New project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_project)
        open_action = QAction("&Open project…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)
        self.import_image_action = QAction("&Import image…", self)
        self.import_image_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.import_image_action.triggered.connect(self._import_image)
        save_action = QAction("&Save project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)
        save_as_action = QAction("Save project &as…", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        menu.addAction(new_action)
        menu.addAction(open_action)
        menu.addAction(self.import_image_action)
        menu.addSeparator()
        menu.addAction(save_action)
        menu.addAction(save_as_action)
        menu.addSeparator()
        menu.addAction(exit_action)

    @staticmethod
    def _configure_dock(dock: QDockWidget, object_name: str) -> None:
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

    def _build_view_menu(self) -> None:
        self.view_menu = self.menuBar().addMenu("&View")
        self._dock_widgets = (
            self.prompt_dock,
            self.model_dock,
            self.event_dock,
        )
        self._dock_toggle_actions: dict[str, QAction] = {}
        for dock in self._dock_widgets:
            action = dock.toggleViewAction()
            action.setText(dock.windowTitle())
            self.view_menu.addAction(action)
            self._dock_toggle_actions[dock.objectName()] = action
        self.view_menu.addSeparator()
        restore_action = QAction("Restore default pane layout", self)
        restore_action.triggered.connect(self._restore_default_dock_layout)
        self.view_menu.addAction(restore_action)

    def _restore_default_dock_layout(self) -> None:
        for dock in self._dock_widgets:
            dock.setFloating(False)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.prompt_dock
        )
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.model_dock
        )
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.event_dock
        )
        for dock in self._dock_widgets:
            dock.show()
        self.resizeDocks(
            [self.prompt_dock, self.model_dock],
            [360, 520],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.event_dock], [240], Qt.Orientation.Vertical
        )
        self.events.addItem("Restored default pane layout")
        self.statusBar().showMessage("Default pane layout restored", 5000)

    def _build_prompt_dock(self) -> None:
        dock = QDockWidget("Prompt and regions", self)
        self._configure_dock(dock, "prompt_regions_dock")
        dock.setMinimumWidth(260)
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
        layout.addWidget(QLabel("Selected region face identity prompt"))
        self.region_face_identity_prompt = QTextEdit()
        self.region_face_identity_prompt.setMinimumHeight(75)
        self.region_face_identity_prompt.setPlaceholderText(
            "Stable facial identity only: trigger, person class, face, and hair…"
        )
        self.region_face_identity_prompt.setToolTip(
            "These tokens can retain the baseline projector mixture and are reused "
            "as the face-refinement identity prompt"
        )
        self.region_face_identity_prompt.setEnabled(False)
        self.region_face_identity_prompt.textChanged.connect(self._region_form_edited)
        layout.addWidget(self.region_face_identity_prompt)
        layout.addWidget(QLabel("Selected region prompt"))
        self.region_prompt = QTextEdit()
        self.region_prompt.setMinimumHeight(90)
        self.region_prompt.setPlaceholderText("Describe only the content controlled by this box...")
        self.region_prompt.setEnabled(False)
        self.region_prompt.textChanged.connect(self._region_form_edited)
        layout.addWidget(self.region_prompt)
        dock.setWidget(self._scrollable(body))
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.prompt_dock = dock

    def _build_image_edit_tab(self) -> None:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        source_row = QHBoxLayout()
        self.edit_source_input = QLineEdit()
        self.edit_source_input.setReadOnly(True)
        self.edit_source_input.setPlaceholderText("Select a PNG, JPEG, or WebP image…")
        browse = QPushButton("Load image…")
        browse.clicked.connect(self._browse_edit_source)
        source_row.addWidget(QLabel("Source"))
        source_row.addWidget(self.edit_source_input, 1)
        source_row.addWidget(browse)
        page_layout.addLayout(source_row)

        controls_body = QWidget()
        controls = QGridLayout(controls_body)
        self.edit_seed_input = QSpinBox()
        self.edit_seed_input.setRange(0, 2_147_483_647)
        self.edit_steps_input = QSpinBox()
        self.edit_steps_input.setRange(1, 100)
        self.edit_steps_input.setValue(8)
        self.edit_sampler_input = QComboBox()
        for sampler in COMFYUI_SAMPLERS:
            self.edit_sampler_input.addItem(sampler, sampler)
        self.edit_scheduler_input = QComboBox()
        for scheduler in COMFYUI_SCHEDULERS:
            self.edit_scheduler_input.addItem(scheduler, scheduler)
        self.edit_denoise_input = QDoubleSpinBox()
        self.edit_denoise_input.setRange(0.05, 1.0)
        self.edit_denoise_input.setDecimals(2)
        self.edit_denoise_input.setSingleStep(0.05)
        self.edit_denoise_input.setValue(0.35)
        self.edit_composite_feather_input = QSpinBox()
        self.edit_composite_feather_input.setRange(0, 256)
        self.edit_composite_feather_input.setSuffix(" px")
        self.edit_composite_feather_input.setValue(32)
        self.edit_regional_strength_input = QDoubleSpinBox()
        self.edit_regional_strength_input.setRange(0.1, 10.0)
        self.edit_regional_strength_input.setValue(1.0)
        self.edit_outside_penalty_input = QDoubleSpinBox()
        self.edit_outside_penalty_input.setRange(0.0, 10.0)
        self.edit_outside_penalty_input.setValue(1.0)
        self.edit_spatial_falloff_input = QSpinBox()
        self.edit_spatial_falloff_input.setRange(0, 2048)
        self.edit_spatial_falloff_input.setSuffix(" px")
        self.edit_spatial_falloff_input.setValue(128)
        self.edit_subject_competition_input = QCheckBox("Separate overlapping subjects")
        self.edit_subject_competition_input.setChecked(True)
        self.edit_subject_fill_input = QCheckBox("Make subjects fill boxes")
        self.edit_subject_fill_input.setChecked(True)
        self.edit_late_step_scale_input = QDoubleSpinBox()
        self.edit_late_step_scale_input.setRange(0.0, 1.0)
        self.edit_late_step_scale_input.setDecimals(2)
        self.edit_late_step_scale_input.setValue(0.35)
        self.edit_lora_adaptation_input = QCheckBox("Adapt from regional LoRA delta")
        self.edit_lora_adaptation_gain_input = QDoubleSpinBox()
        self.edit_lora_adaptation_gain_input.setRange(0.0, 1.0)
        self.edit_lora_adaptation_gain_input.setDecimals(2)
        self.edit_lora_adaptation_gain_input.setValue(0.35)
        edit_controls = (
            ("Seed", self.edit_seed_input),
            ("Steps", self.edit_steps_input),
            ("Sampler", self.edit_sampler_input),
            ("Scheduler", self.edit_scheduler_input),
            ("Denoise", self.edit_denoise_input),
            ("Composite feather", self.edit_composite_feather_input),
            ("Inside boost", self.edit_regional_strength_input),
            ("Outside penalty", self.edit_outside_penalty_input),
            ("Spatial falloff", self.edit_spatial_falloff_input),
            ("Late-step scale", self.edit_late_step_scale_input),
            ("LoRA response", self.edit_lora_adaptation_gain_input),
        )
        for index, (label, control) in enumerate(edit_controls):
            row = index // 4
            column = (index % 4) * 2
            controls.addWidget(QLabel(label), row, column)
            controls.addWidget(control, row, column + 1)
        option_row = len(edit_controls) // 4 + 1
        controls.addWidget(self.edit_subject_competition_input, option_row, 0, 1, 2)
        controls.addWidget(self.edit_subject_fill_input, option_row, 2, 1, 2)
        controls.addWidget(self.edit_lora_adaptation_input, option_row, 4, 1, 2)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_body)
        controls_scroll.setMaximumHeight(155)
        controls_scroll.setMinimumWidth(0)
        page_layout.addWidget(controls_scroll)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        prompt_panel = QWidget()
        prompt_layout = QVBoxLayout(prompt_panel)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.addWidget(QLabel("Global edit prompt"))
        self.edit_global_prompt = QTextEdit()
        self.edit_global_prompt.setMinimumHeight(75)
        self.edit_global_prompt.setPlaceholderText(
            "Optional: when non-empty the entire image may change…"
        )
        prompt_layout.addWidget(self.edit_global_prompt)
        region_actions = QHBoxLayout()
        draw = QPushButton("Draw edit region")
        delete = QPushButton("Delete selected")
        region_actions.addWidget(draw)
        region_actions.addWidget(delete)
        prompt_layout.addLayout(region_actions)
        prompt_layout.addWidget(QLabel("Edit regions (front to back)"))
        self.edit_region_list = QListWidget()
        self.edit_region_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.edit_region_list.model().rowsMoved.connect(self._edit_region_order_changed)
        self.edit_region_list.currentRowChanged.connect(self._selected_edit_region_changed)
        prompt_layout.addWidget(self.edit_region_list, 1)
        self.edit_region_name = QLineEdit()
        self.edit_region_name.setPlaceholderText("Selected region name")
        self.edit_region_name.setEnabled(False)
        self.edit_region_name.editingFinished.connect(self._edit_region_name_edited)
        prompt_layout.addWidget(self.edit_region_name)
        self.edit_region_role = QComboBox()
        self.edit_region_role.addItem("Auto (based on box width)", "auto")
        self.edit_region_role.addItem("Subject target", "subject")
        self.edit_region_role.addItem("Background band", "background")
        self.edit_region_role.setEnabled(False)
        self.edit_region_role.currentIndexChanged.connect(self._edit_region_form_edited)
        prompt_layout.addWidget(self.edit_region_role)
        self.edit_region_face_prompt = QTextEdit()
        self.edit_region_face_prompt.setPlaceholderText("Optional face identity prompt…")
        self.edit_region_face_prompt.setMaximumHeight(70)
        self.edit_region_face_prompt.setEnabled(False)
        self.edit_region_face_prompt.textChanged.connect(self._edit_region_form_edited)
        prompt_layout.addWidget(self.edit_region_face_prompt)
        self.edit_region_prompt = QTextEdit()
        self.edit_region_prompt.setPlaceholderText("Describe the edit inside this box…")
        self.edit_region_prompt.setEnabled(False)
        self.edit_region_prompt.textChanged.connect(self._edit_region_form_edited)
        prompt_layout.addWidget(self.edit_region_prompt)

        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("Source with editable regions"))
        self.edit_canvas = RegionCanvas(
            self.settings.default_width, self.settings.default_height
        )
        source_layout.addWidget(self.edit_canvas, 1)
        draw.clicked.connect(self.edit_canvas.begin_region)
        delete.clicked.connect(self.edit_canvas.delete_selected_regions)
        self.edit_canvas.region_created.connect(self._edit_region_created)
        self.edit_canvas.region_changed.connect(self._edit_region_changed)
        self.edit_canvas.region_deleted.connect(self._edit_region_deleted)
        self.edit_canvas.region_selected.connect(self._edit_canvas_region_selected)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(QLabel("Edited result"))
        self.edit_result_preview = ScaledImagePreview("Run an edit to compare the result")
        result_layout.addWidget(self.edit_result_preview, 1)
        self.edit_result_input = QLineEdit()
        self.edit_result_input.setReadOnly(True)
        self.edit_result_input.setPlaceholderText("Result path")
        result_layout.addWidget(self.edit_result_input)
        self.edit_run_button = QPushButton("Run image edit")
        self.edit_run_button.setEnabled(False)
        self.edit_run_button.clicked.connect(self._run_image_edit)
        result_layout.addWidget(self.edit_run_button)
        note = QLabel(
            "Blank global prompt preserves source pixels outside the feathered box union. "
            "A global prompt permits whole-image changes."
        )
        note.setWordWrap(True)
        result_layout.addWidget(note)

        splitter.addWidget(prompt_panel)
        splitter.addWidget(source_panel)
        splitter.addWidget(result_panel)
        splitter.setSizes([330, 700, 500])
        page_layout.addWidget(splitter, 1)
        self.workspace_tabs.addTab(page, "Image editing")

    def _build_face_refinement_tab(self) -> None:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        explanation = QLabel(
            "Load a completed first-pass PNG, then refine detected faces using only "
            "the LoRAs assigned to their current subject regions. Global LoRAs are "
            "excluded. Crop size is the square Krea working resolution; it does not "
            "change the detected area, which is controlled by crop padding."
        )
        explanation.setWordWrap(True)
        page_layout.addWidget(explanation)

        source_row = QHBoxLayout()
        self.face_source_input = QLineEdit()
        self.face_source_input.setReadOnly(True)
        self.face_source_input.setPlaceholderText("Select a first-pass PNG…")
        browse = QPushButton("Load PNG…")
        browse.clicked.connect(self._browse_face_source)
        latest = QPushButton("Use latest first pass")
        latest.clicked.connect(self._use_latest_face_source)
        source_row.addWidget(QLabel("Source"))
        source_row.addWidget(self.face_source_input, 1)
        source_row.addWidget(browse)
        source_row.addWidget(latest)
        page_layout.addLayout(source_row)

        controls = QGridLayout()
        self.face_detail_seed_input = QSpinBox()
        self.face_detail_seed_input.setRange(0, 2_147_483_647)
        self.face_detail_seed_input.setValue(0)
        self.face_detail_steps_input = QSpinBox()
        self.face_detail_steps_input.setRange(1, 100)
        self.face_detail_steps_input.setValue(8)
        self.face_detail_denoise_input = QDoubleSpinBox()
        self.face_detail_denoise_input.setRange(0.05, 1.0)
        self.face_detail_denoise_input.setDecimals(2)
        self.face_detail_denoise_input.setSingleStep(0.05)
        self.face_detail_denoise_input.setValue(0.15)
        self.face_detail_denoise_input.setToolTip(
            "How much Krea may redraw the crop. Lower this first when faces deform."
        )
        self.face_detail_crop_size_input = QComboBox()
        for crop_size in (256, 512, 768, 1024):
            self.face_detail_crop_size_input.addItem(f"{crop_size} px", crop_size)
        self.face_detail_crop_size_input.setCurrentIndex(
            self.face_detail_crop_size_input.findData(512)
        )
        self.face_detail_crop_size_input.setToolTip(
            "Square working resolution sent through Krea. 512 is the safe default; "
            "256 often loses facial structure, while 768/1024 use more memory."
        )
        self.face_detail_padding_input = QDoubleSpinBox()
        self.face_detail_padding_input.setRange(1.0, 4.0)
        self.face_detail_padding_input.setDecimals(2)
        self.face_detail_padding_input.setSingleStep(0.1)
        self.face_detail_padding_input.setValue(2.0)
        self.face_detail_padding_input.setSuffix("× face")
        self.face_detail_padding_input.setToolTip(
            "Physical crop area relative to the detected face. Around 1.6–2.0 keeps "
            "hair and head context without redrawing too much background."
        )
        self.face_detail_padding_input.valueChanged.connect(
            self._face_selection_changed
        )
        self.face_detail_feather_input = QDoubleSpinBox()
        self.face_detail_feather_input.setRange(0.0, 0.5)
        self.face_detail_feather_input.setDecimals(2)
        self.face_detail_feather_input.setSingleStep(0.02)
        self.face_detail_feather_input.setValue(0.12)
        self.face_detail_blend_input = QDoubleSpinBox()
        self.face_detail_blend_input.setRange(0.0, 1.0)
        self.face_detail_blend_input.setDecimals(2)
        self.face_detail_blend_input.setSingleStep(0.05)
        self.face_detail_blend_input.setValue(0.5)
        self.face_detail_blend_input.setToolTip(
            "Pixel-delta amount blended over the original face. Lower this when the "
            "refined crop has seams or invented features."
        )
        self.face_detail_lora_scale_input = QDoubleSpinBox()
        self.face_detail_lora_scale_input.setRange(0.0, 4.0)
        self.face_detail_lora_scale_input.setDecimals(2)
        self.face_detail_lora_scale_input.setSingleStep(0.05)
        self.face_detail_lora_scale_input.setValue(0.5)
        self.face_detail_lora_scale_input.setSuffix("×")
        self.face_detail_lora_scale_input.setToolTip(
            "Multiplier over each regional LoRA's saved strength. If the saved "
            "strength is 2.0, the default applies 1.0 during face refinement."
        )
        self.face_detail_detector_threshold_input = QDoubleSpinBox()
        self.face_detail_detector_threshold_input.setRange(0.05, 0.95)
        self.face_detail_detector_threshold_input.setDecimals(2)
        self.face_detail_detector_threshold_input.setSingleStep(0.05)
        self.face_detail_detector_threshold_input.setValue(0.15)
        self.face_detail_detector_threshold_input.setToolTip(
            "Minimum detector confidence. Lower values find subtler or smaller faces "
            "but may add false positives. Detection is only a proposal: review the "
            "numbered boxes and select exactly which faces should be refined."
        )
        self.face_detail_detector_threshold_input.valueChanged.connect(
            self._face_detection_settings_changed
        )
        self.face_detail_detector_provider_input = QComboBox()
        self.face_detail_detector_provider_input.addItem(
            "Auto (CUDA when available)", "auto"
        )
        self.face_detail_detector_provider_input.addItem("CPU", "cpu")
        self.face_detail_detector_provider_input.addItem("NVIDIA CUDA", "cuda")
        self.face_detail_detector_provider_input.setToolTip(
            "Auto prefers ONNX Runtime's CUDA provider on NVIDIA and otherwise "
            "uses CPU. AMD/ROCm users should use Auto or CPU."
        )
        self.face_detail_detector_provider_input.currentIndexChanged.connect(
            self._face_detection_settings_changed
        )

        control_items = (
            ("Seed", self.face_detail_seed_input),
            ("Steps", self.face_detail_steps_input),
            ("Denoise", self.face_detail_denoise_input),
            ("Crop working resolution", self.face_detail_crop_size_input),
            ("Crop padding", self.face_detail_padding_input),
            ("Edge feather", self.face_detail_feather_input),
            ("Refined-pixel blend", self.face_detail_blend_input),
            ("Regional LoRA scale", self.face_detail_lora_scale_input),
            ("Detector threshold", self.face_detail_detector_threshold_input),
            ("Detector device", self.face_detail_detector_provider_input),
        )
        for index, (label, control) in enumerate(control_items):
            row = index // 3
            column = (index % 3) * 2
            controls.addWidget(QLabel(label), row, column)
            controls.addWidget(control, row, column + 1)
        page_layout.addLayout(controls)

        action_row = QHBoxLayout()
        assignment_note = QLabel(
            "1. Detect faces. 2. Review the numbered boxes and choose one or more. "
            "3. Refine only the selected faces using their current regional LoRAs."
        )
        assignment_note.setWordWrap(True)
        self.face_detect_button = QPushButton("1. Detect faces")
        self.face_detect_button.setEnabled(False)
        self.face_detect_button.setToolTip(
            "Runs the CPU face detector and draws a numbered box around every candidate."
        )
        self.face_detect_button.clicked.connect(self._detect_faces_for_refinement)
        self.face_lasso_button = QPushButton("Draw face lasso")
        self.face_lasso_button.setEnabled(False)
        self.face_lasso_undo_button = QPushButton("Undo lasso")
        self.face_lasso_clear_button = QPushButton("Clear lassos")
        self.face_refine_button = QPushButton("Run face refinement")
        self.face_refine_button.setEnabled(False)
        self.face_refine_button.clicked.connect(self._run_face_refinement)
        action_row.addWidget(assignment_note, 1)
        action_row.addWidget(self.face_detect_button)
        action_row.addWidget(self.face_lasso_button)
        action_row.addWidget(self.face_lasso_undo_button)
        action_row.addWidget(self.face_lasso_clear_button)
        action_row.addWidget(self.face_refine_button)
        page_layout.addLayout(action_row)

        selection_row = QHBoxLayout()
        self.face_selection_list = QListWidget()
        self.face_selection_list.setMaximumHeight(110)
        self.face_selection_list.setToolTip(
            "Checked faces will be refined. Orange boxes are detected but excluded; "
            "green boxes are selected."
        )
        self.face_selection_list.itemChanged.connect(self._face_selection_changed)
        selection_buttons = QVBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_face_selections(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_face_selections(False))
        selection_buttons.addWidget(select_all)
        selection_buttons.addWidget(select_none)
        selection_buttons.addStretch(1)
        selection_row.addWidget(QLabel("Detected faces"))
        selection_row.addWidget(self.face_selection_list, 1)
        selection_row.addLayout(selection_buttons)
        page_layout.addLayout(selection_row)

        previews = QSplitter(Qt.Orientation.Horizontal)
        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        self.face_source_panel = source_panel
        self.face_source_preview_layout = source_layout
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("First pass with detected-face selection"))
        self.face_source_preview = LassoImagePreview("No source PNG loaded")
        self.face_lasso_button.clicked.connect(self._open_face_lasso_dialog)
        self.face_lasso_undo_button.clicked.connect(
            self.face_source_preview.undo_lasso
        )
        self.face_lasso_clear_button.clicked.connect(
            self.face_source_preview.clear_lassos
        )
        self.face_source_preview.lassos_changed.connect(self._manual_lassos_changed)
        source_layout.addWidget(self.face_source_preview, 1)
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(QLabel("Face-refined result"))
        self.face_result_preview = ScaledImagePreview(
            "Run face refinement to compare the result"
        )
        result_layout.addWidget(self.face_result_preview, 1)
        self.face_result_input = QLineEdit()
        self.face_result_input.setReadOnly(True)
        self.face_result_input.setPlaceholderText("Result path")
        result_layout.addWidget(self.face_result_input)
        previews.addWidget(source_panel)
        previews.addWidget(result_panel)
        previews.setSizes([700, 700])
        page_layout.addWidget(previews, 1)
        self.workspace_tabs.addTab(page, "Face refinement")

    def _build_model_dock(self) -> None:
        dock = QDockWidget("Model and generation settings", self)
        self._configure_dock(dock, "model_settings_dock")
        dock.setMinimumWidth(340)
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
        self.comfyui_root_input = QLineEdit()
        self.comfyui_root_input.setReadOnly(True)
        self.comfyui_root_browse = QPushButton("Choose…")
        self.comfyui_root_browse.setToolTip(
            "Select the ComfyUI checkout used by the isolated GPU worker"
        )
        self.comfyui_root_browse.clicked.connect(self._browse_comfyui_root)
        comfyui_row = QWidget()
        comfyui_row_layout = QHBoxLayout(comfyui_row)
        comfyui_row_layout.setContentsMargins(0, 0, 0, 0)
        comfyui_row_layout.addWidget(self.comfyui_root_input, 1)
        comfyui_row_layout.addWidget(self.comfyui_root_browse)
        layout.addRow("ComfyUI checkout", comfyui_row)
        self.worker_python_input = QLineEdit()
        self.worker_python_input.setReadOnly(True)
        self.worker_python_browse = QPushButton("Choose…")
        self.worker_python_browse.setToolTip(
            "Select Python from a CUDA- or ROCm-enabled ComfyUI environment"
        )
        self.worker_python_browse.clicked.connect(self._browse_worker_python)
        self.worker_python_auto = QPushButton("Auto")
        self.worker_python_auto.setToolTip(
            "Search the selected ComfyUI checkout for a common virtual environment"
        )
        self.worker_python_auto.clicked.connect(self._auto_select_worker_python)
        worker_row = QWidget()
        worker_row_layout = QHBoxLayout(worker_row)
        worker_row_layout.setContentsMargins(0, 0, 0, 0)
        worker_row_layout.addWidget(self.worker_python_input, 1)
        worker_row_layout.addWidget(self.worker_python_browse)
        worker_row_layout.addWidget(self.worker_python_auto)
        layout.addRow("GPU worker Python", worker_row)
        self.transformer_status = QLabel("Not discovered")
        self.text_status = QLabel("Not discovered")
        self.vae_status = QLabel("Not discovered")
        self.diffusion_model_input = QComboBox()
        self.diffusion_model_input.setToolTip(
            "Select a Krea 2 transformer found in the configured ComfyUI model directory"
        )
        self.diffusion_model_input.currentIndexChanged.connect(
            self._diffusion_model_changed
        )
        layout.addRow("Krea checkpoint", self.diffusion_model_input)
        self.transformer_browse = QPushButton("Choose…")
        self.transformer_browse.clicked.connect(
            lambda: self._browse_primary_model("diffusion_model_file")
        )
        self.transformer_auto = QPushButton("Auto")
        self.transformer_auto.clicked.connect(
            lambda: self._clear_primary_model("diffusion_model_file")
        )
        self.text_encoder_browse = QPushButton("Choose…")
        self.text_encoder_browse.clicked.connect(
            lambda: self._browse_primary_model("text_encoder_file")
        )
        self.text_encoder_auto = QPushButton("Auto")
        self.text_encoder_auto.clicked.connect(
            lambda: self._clear_primary_model("text_encoder_file")
        )
        self.vae_browse = QPushButton("Choose…")
        self.vae_browse.clicked.connect(
            lambda: self._browse_primary_model("vae_file")
        )
        self.vae_auto = QPushButton("Auto")
        self.vae_auto.clicked.connect(
            lambda: self._clear_primary_model("vae_file")
        )
        for label, status, browse, automatic in (
            ("Transformer", self.transformer_status, self.transformer_browse, self.transformer_auto),
            ("Text encoder", self.text_status, self.text_encoder_browse, self.text_encoder_auto),
            ("VAE", self.vae_status, self.vae_browse, self.vae_auto),
        ):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(status, 1)
            row_layout.addWidget(browse)
            row_layout.addWidget(automatic)
            automatic.setToolTip("Clear the exact file and use compatible name-based discovery")
            layout.addRow(label, row)
        self.face_detector_input = QLineEdit()
        self.face_detector_input.setReadOnly(True)
        self.face_detector_browse = QPushButton("Choose…")
        self.face_detector_browse.clicked.connect(self._browse_face_detector_model)
        self.face_detector_auto = QPushButton("Auto")
        self.face_detector_auto.clicked.connect(self._clear_face_detector_model)
        self.face_detector_auto.setToolTip(
            "Use the compatible face detector found under the selected ComfyUI checkout"
        )
        detector_row = QWidget()
        detector_row_layout = QHBoxLayout(detector_row)
        detector_row_layout.setContentsMargins(0, 0, 0, 0)
        detector_row_layout.addWidget(self.face_detector_input, 1)
        detector_row_layout.addWidget(self.face_detector_browse)
        detector_row_layout.addWidget(self.face_detector_auto)
        layout.addRow("Face detector", detector_row)
        self._refresh_runtime_path_controls()
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
            "Stop this user's K2 Region Lab GPU workers; other GPU apps are untouched"
        )
        self.release_worker_button.clicked.connect(self._release_k2_gpu_memory)
        layout.addRow(self.release_worker_button)
        self.load_model_button = QPushButton("Load selected Krea 2 model")
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
        self.reserve_vram_input.setRange(0.5, 128.0)
        self.reserve_vram_input.setSingleStep(0.5)
        self.reserve_vram_input.setSuffix(" GiB")
        self.reserve_vram_input.setValue(self.settings.reserve_vram_gb)
        layout.addRow("Keep VRAM free", self.reserve_vram_input)
        self.minimum_ram_input = QDoubleSpinBox()
        self.minimum_ram_input.setRange(4.0, 256.0)
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
        self.steps_input.setValue(self.settings.default_steps)
        self.sampler_input = QComboBox()
        for sampler in COMFYUI_SAMPLERS:
            self.sampler_input.addItem(sampler, sampler)
        sampler_index = self.sampler_input.findData(self.settings.default_sampler)
        self.sampler_input.setCurrentIndex(max(0, sampler_index))
        self.sampler_input.setToolTip(
            "Sampling algorithm passed directly to ComfyUI KSampler. Euler is the "
            "Krea 2 Turbo default."
        )
        self.scheduler_input = QComboBox()
        for scheduler in COMFYUI_SCHEDULERS:
            self.scheduler_input.addItem(scheduler, scheduler)
        scheduler_index = self.scheduler_input.findData(self.settings.default_scheduler)
        self.scheduler_input.setCurrentIndex(max(0, scheduler_index))
        self.scheduler_input.setToolTip(
            "Noise schedule passed directly to ComfyUI KSampler. Simple is the "
            "Krea 2 Turbo default."
        )
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_147_483_647)
        self.seed_input.setValue(self.settings.default_seed)
        layout.addRow("Turbo steps", self.steps_input)
        layout.addRow("Sampler", self.sampler_input)
        layout.addRow("Scheduler", self.scheduler_input)
        layout.addRow("Seed", self.seed_input)
        self.seed_mode_input = QComboBox()
        self.seed_mode_input.addItem("Fixed", "fixed")
        self.seed_mode_input.addItem("Random", "random")
        self.seed_mode_input.addItem("Increment", "increment")
        seed_mode_index = self.seed_mode_input.findData(self.settings.default_seed_mode)
        self.seed_mode_input.setCurrentIndex(max(0, seed_mode_index))
        layout.addRow("Seed behavior", self.seed_mode_input)
        self.batch_mode_input = QCheckBox("Run generation in batch mode")
        self.batch_mode_input.setChecked(False)
        self.batch_mode_input.setToolTip(
            "Run each image in a fresh worker so all GPU models are unloaded between runs"
        )
        layout.addRow(self.batch_mode_input)
        self.batch_count_input = QSpinBox()
        self.batch_count_input.setRange(1, 100)
        self.batch_count_input.setValue(2)
        self.batch_count_input.setEnabled(False)
        layout.addRow("Batch runs", self.batch_count_input)
        self.batch_mode_input.toggled.connect(self._batch_mode_changed)
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
            "Increase center-to-edge contrast inside subject boxes; subject text is "
            "hard-blocked outside its box. Background bands use one quarter of this "
            "penalty and may feather beyond their boxes."
        )
        layout.addRow("Outside penalty", self.regional_outside_penalty_input)
        self.regional_feather_input = QSpinBox()
        self.regional_feather_input.setRange(0, 2048)
        self.regional_feather_input.setSingleStep(16)
        self.regional_feather_input.setSuffix(" px")
        self.regional_feather_input.setValue(128)
        self.regional_feather_input.setToolTip(
            "Distance over which background-band guidance fades beyond its box. "
            "Subject text remains hard-confined to image tokens intersecting its box."
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
        self.upscale_model_input.setText(
            str(self._upscale_model_path) if self._upscale_model_path else ""
        )
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
        self._build_lora_tab()
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
            "Applies a txtfusion.projector vector to prompt tokens before regional "
            "LoRA routing. Face identity prompt tokens can retain the baseline "
            "projector mixture without creating an image-space mask."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.projector_enabled_input = QCheckBox("Apply global projector vector")
        self.projector_enabled_input.setChecked(False)
        self.projector_enabled_input.setToolTip(
            "Applies to all prompt tokens except the protected portion of each "
            "Face identity prompt; regional LoRA routing remains unchanged"
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
        self.projector_identity_protection_input = QDoubleSpinBox()
        self.projector_identity_protection_input.setRange(0.0, 1.0)
        self.projector_identity_protection_input.setDecimals(2)
        self.projector_identity_protection_input.setSingleStep(0.05)
        self.projector_identity_protection_input.setValue(1.0)
        self.projector_identity_protection_input.setToolTip(
            "0 applies the complete projector delta to identity tokens; 1 keeps "
            "their original projector mixture"
        )
        form.addRow(
            "Face identity protection",
            self.projector_identity_protection_input,
        )
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
        identity_protection: float,
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
            self.projector_identity_protection_input.setValue(identity_protection)
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

    def _build_lora_tab(self) -> None:
        body = QWidget()
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
        self.lora_routing_mode_input = QComboBox()
        self.lora_routing_mode_input.addItem("Standard regional", STANDARD_LORA_ROUTING)
        self.lora_routing_mode_input.addItem(
            "Character identity (face)", CHARACTER_IDENTITY_LORA_ROUTING
        )
        self.lora_routing_mode_input.setEnabled(False)
        self.lora_routing_mode_input.setToolTip(
            "Standard regional routing gates text-fusion deltas to each assigned "
            "prompt clause and lets those tokens condition only image tokens inside "
            "the assigned box. Subject-owned image keys also remain private so the "
            "delta cannot relay through the shared image stream. Main-stream "
            "attention key/value targets are omitted. "
            "Character identity additionally inserts an explicit face anchor."
        )
        self.lora_routing_mode_input.currentIndexChanged.connect(
            self._lora_routing_mode_changed
        )
        strength_row.addRow("Routing mode", self.lora_routing_mode_input)
        self.lora_trigger_input = QLineEdit()
        self.lora_trigger_input.setEnabled(False)
        self.lora_trigger_input.setPlaceholderText("Training trigger, for example lface")
        self.lora_trigger_input.setToolTip(
            "Exact trigger learned during LoRA training. This is inserted into the "
            "character identity anchor."
        )
        self.lora_trigger_input.editingFinished.connect(self._lora_trigger_edited)
        strength_row.addRow("Identity trigger", self.lora_trigger_input)
        layout.addLayout(strength_row)
        self.lora_routing_note = QLabel(
            "Standard routing partitions each assigned prompt clause and confines "
            "its LoRA-conditioned image attention to the assigned boxes."
        )
        self.lora_routing_note.setWordWrap(True)
        layout.addWidget(self.lora_routing_note)
        self.lora_diagnostic_button = QPushButton("Inspect selected LoRA…")
        self.lora_diagnostic_button.setEnabled(False)
        self.lora_diagnostic_button.clicked.connect(self._diagnose_selected_lora)
        layout.addWidget(self.lora_diagnostic_button)
        self.lora_status = QLabel("Select a LoRA to inspect its Krea compatibility")
        self.lora_status.setWordWrap(True)
        layout.addWidget(self.lora_status)
        scope_group = QGroupBox("Apply selected LoRA")
        scope_layout = QVBoxLayout(scope_group)
        self.lora_scope_context_label = QLabel(
            "Generation scope: choose Global or one or more named regions"
        )
        self.lora_scope_context_label.setWordWrap(True)
        scope_layout.addWidget(self.lora_scope_context_label)
        self.lora_scope_list = QListWidget()
        self.lora_scope_list.itemChanged.connect(self._lora_scope_changed)
        scope_layout.addWidget(self.lora_scope_list)
        columns.addWidget(scope_group, 1)
        self.settings_tabs.addTab(self._scrollable(body), "LoRA library & scope")

    def _build_event_dock(self) -> None:
        dock = QDockWidget("Events", self)
        self._configure_dock(dock, "events_dock")
        dock.setMinimumHeight(100)
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
        self.event_dock = dock

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

    def _latest_first_pass_output(self) -> Path | None:
        if not self._output_directory.is_dir():
            return None
        candidates = tuple(
            path
            for path in self._output_directory.glob("*.png")
            if "face_refined" not in path.stem.casefold()
            and not path.stem.casefold().startswith("face-detail-verify")
        )
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

    def _set_face_refinement_source(self, path: Path) -> bool:
        source_path = path.expanduser().resolve()
        if source_path.suffix.casefold() != ".png" or not source_path.is_file():
            return False
        if not self.face_source_preview.set_image(source_path):
            return False
        self._face_source_path = source_path
        self.face_source_input.setText(str(source_path))
        self._clear_face_detections()
        self._face_result_path = None
        self.face_result_input.clear()
        self.face_result_preview.clear_image(
            "Run face refinement to compare the result"
        )
        self.face_detect_button.setEnabled(not self._generation_active)
        self.face_lasso_button.setEnabled(not self._generation_active)
        self.face_refine_button.setEnabled(False)
        return True

    def _clear_face_detections(self) -> None:
        self._face_detections = []
        self._manual_face_paths = []
        self._syncing_face_selection = True
        self.face_selection_list.clear()
        self._syncing_face_selection = False
        self.face_source_preview.set_overlays([])
        self.face_source_preview.clear_lassos(emit=False)
        self.face_refine_button.setEnabled(False)

    def _manual_lassos_changed(self, paths) -> None:
        self._manual_face_paths = [tuple(tuple(point) for point in path) for path in paths]
        detections = tuple(
            DetectedFace(
                PixelBox(
                    min(point[0] for point in path),
                    min(point[1] for point in path),
                    max(point[0] for point in path),
                    max(point[1] for point in path),
                ),
                1.0,
                path,
            )
            for path in self._manual_face_paths
            if len(path) >= 3
        )
        if self.face_source_preview._original_pixmap.isNull():
            return
        targets = assign_faces_to_regional_loras(
            detections,
            self._scaled_face_regions(
                self.face_source_preview._original_pixmap.width(),
                self.face_source_preview._original_pixmap.height(),
            ),
            self._lora_payload(),
        )
        target_by_box = {
            (target.face.box.x0, target.face.box.y0, target.face.box.x1, target.face.box.y1): target
            for target in targets
        }
        self._face_detections = []
        self._syncing_face_selection = True
        self.face_selection_list.clear()
        for index, face in enumerate(detections):
            key = (face.box.x0, face.box.y0, face.box.x1, face.box.y1)
            target = target_by_box.get(key)
            matched = target is not None
            self._face_detections.append(
                {
                    "index": index,
                    "box": list(key),
                    "score": 1.0,
                    "region_id": target.region_id if target else None,
                    "region_name": target.region_name if target else None,
                    "manual": True,
                }
            )
            assignment = target.region_name if target else "no regional LoRA match"
            item = QListWidgetItem(f"Lasso {index + 1} - {assignment}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if matched else Qt.CheckState.Unchecked)
            self.face_selection_list.addItem(item)
        self._syncing_face_selection = False
        self._face_selection_changed()
        self.events.addItem(
            f"Prepared {len(detections)} manual face lasso(s); "
            f"{len(targets)} matched regional LoRAs"
        )

    def _open_face_lasso_dialog(self) -> None:
        if self._face_source_path is None or self.face_source_preview._original_pixmap.isNull():
            QMessageBox.warning(
                self, "Source PNG required", "Load a first-pass PNG before drawing lassos."
            )
            return
        preview = self.face_source_preview
        original_paths = preview.lasso_paths()
        dialog = QDialog(self)
        dialog.setWindowTitle("Draw face lassos")
        dialog.setMinimumSize(640, 480)
        dialog.resize(1200, 850)
        layout = QVBoxLayout(dialog)
        instructions = QLabel(
            "Drag freehand around each face. Draw additional lassos for additional "
            "faces. The main preview will show each padded refinement crop after OK."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.face_source_preview_layout.removeWidget(preview)
        preview.setParent(dialog)
        layout.addWidget(preview, 1)
        controls = QHBoxLayout()
        undo = QPushButton("Undo lasso")
        undo.clicked.connect(preview.undo_lasso)
        clear = QPushButton("Clear lassos")
        clear.clicked.connect(preview.clear_lassos)
        controls.addWidget(undo)
        controls.addWidget(clear)
        controls.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        controls.addWidget(buttons)
        layout.addLayout(controls)
        self._lasso_dialog_active = True
        preview.set_lassos_visible(True)
        preview.set_drawing_enabled(True)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        preview.set_drawing_enabled(False)
        if not accepted:
            preview.set_lasso_paths(original_paths)
        self._lasso_dialog_active = False
        preview.set_lassos_visible(False)
        layout.removeWidget(preview)
        preview.setParent(self.face_source_panel)
        self.face_source_preview_layout.insertWidget(1, preview, 1)
        preview.show()
        self._face_selection_changed()

    def _face_detection_settings_changed(self, _value=None) -> None:
        if not self._face_detections:
            return
        self._clear_face_detections()
        self.events.addItem(
            "Face detector threshold changed; run detection again before refinement"
        )

    def _scaled_face_regions(self, image_width: int, image_height: int):
        scale_x = image_width / max(1, self.width_input.value())
        scale_y = image_height / max(1, self.height_input.value())
        return tuple(
            replace(
                region,
                box=PixelBox(
                    region.box.x0 * scale_x,
                    region.box.y0 * scale_y,
                    region.box.x1 * scale_x,
                    region.box.y1 * scale_y,
                ),
            )
            for region in self.regions
        )

    def _detect_faces_for_refinement(self) -> None:
        source_path = self._face_source_path
        if source_path is None or not source_path.is_file():
            QMessageBox.warning(self, "Source PNG required", "Load a first-pass PNG first.")
            return
        detector_path = self.settings.face_detector_path or discover_face_detector(
            self.settings.comfyui_root
        )
        if detector_path is None:
            QMessageBox.warning(
                self,
                "Face detector unavailable",
                "The bundled face_det.onnx model was not found under the configured "
                "ComfyUI root.",
            )
            return
        # Do not resolve a virtualenv Python symlink: Python uses the symlink's
        # location to find pyvenv.cfg and select the intended environment.
        worker_python = self.settings.worker_python.expanduser().absolute()
        if not worker_python.is_file():
            QMessageBox.warning(
                self,
                "Face detector unavailable",
                f"The configured worker interpreter does not exist: {worker_python}",
            )
            return
        project_root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment["VIRTUAL_ENV"] = str(worker_python.parent.parent)
        environment["PATH"] = os.pathsep.join(
            (str(worker_python.parent), environment.get("PATH", ""))
        )
        python_paths = [str(project_root / "src"), str(self.settings.comfyui_root)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        try:
            completed = subprocess.run(
                (
                    str(worker_python),
                    "-m",
                    "k2_region_lab.worker.detect_faces",
                    "--image",
                    str(source_path),
                    "--comfyui-root",
                    str(self.settings.comfyui_root),
                    "--detector-path",
                    str(detector_path),
                    "--threshold",
                    str(self.face_detail_detector_threshold_input.value()),
                    "--provider",
                    str(self.face_detail_detector_provider_input.currentData()),
                ),
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            detection_report = json.loads(completed.stdout)
            image_width = int(detection_report["width"])
            image_height = int(detection_report["height"])
            execution_provider = str(
                detection_report.get("execution_provider", "unknown provider")
            )
            detections = tuple(
                DetectedFace(
                    PixelBox(*map(float, item["box"])),
                    float(item["score"]),
                )
                for item in detection_report["faces"]
            )
        except (
            json.JSONDecodeError,
            KeyError,
            OSError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as error:
            logging.getLogger(__name__).exception("face detection failed")
            error_stderr = getattr(error, "stderr", None)
            details = (
                error_stderr.strip()
                if isinstance(error_stderr, str) and error_stderr.strip()
                else str(error)
            )
            QMessageBox.warning(self, "Face detection failed", details)
            return

        targets = assign_faces_to_regional_loras(
            detections,
            self._scaled_face_regions(image_width, image_height),
            self._lora_payload(),
        )
        target_by_box = {
            (
                target.face.box.x0,
                target.face.box.y0,
                target.face.box.x1,
                target.face.box.y1,
            ): target
            for target in targets
        }
        self._face_detections = []
        self._manual_face_paths = []
        self.face_source_preview.clear_lassos(emit=False)
        self._syncing_face_selection = True
        self.face_selection_list.clear()
        for index, face in enumerate(detections):
            box_key = (face.box.x0, face.box.y0, face.box.x1, face.box.y1)
            target = target_by_box.get(box_key)
            matched = target is not None
            record: dict[str, object] = {
                "index": index,
                "box": [face.box.x0, face.box.y0, face.box.x1, face.box.y1],
                "score": face.score,
                "region_id": target.region_id if target else None,
                "region_name": target.region_name if target else None,
            }
            self._face_detections.append(record)
            assignment = (
                f"region {target.region_name}"
                if target is not None
                else "no regional LoRA match"
            )
            item = QListWidgetItem(
                f"Face {index + 1} — confidence {face.score:.3f} — {assignment}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if matched else Qt.CheckState.Unchecked
            )
            self.face_selection_list.addItem(item)
        self._syncing_face_selection = False
        self._face_selection_changed()
        if detections:
            self.events.addItem(
                f"Detected {len(detections)} face(s) at threshold "
                f"{self.face_detail_detector_threshold_input.value():.2f}; "
                f"{len(targets)} matched regional LoRAs; {execution_provider}"
            )
        else:
            self.events.addItem(
                "No faces detected. Lower the detector threshold and try again."
            )
            QMessageBox.information(
                self,
                "No faces detected",
                "Lower the detector threshold and run detection again.",
            )

    def _selected_face_indices(self) -> list[int]:
        return [
            int(self.face_selection_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.face_selection_list.count())
            if self.face_selection_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def _face_selection_changed(self, _item=None) -> None:
        if self._syncing_face_selection:
            return
        selected = set(self._selected_face_indices())
        overlays = []
        for record in self._face_detections:
            box = list(record["box"])
            index = int(record["index"])
            region_name = record.get("region_name")
            label = f"{index + 1}" + (f" {region_name}" if region_name else "")
            if record.get("manual") and not self._lasso_dialog_active:
                padded_box = expanded_square_crop(
                    PixelBox(*map(float, box)),
                    self.face_source_preview._original_pixmap.width(),
                    self.face_source_preview._original_pixmap.height(),
                    self.face_detail_padding_input.value(),
                )
                overlays.append((*map(float, padded_box), f"{label} padded", index in selected))
            elif not record.get("manual"):
                overlays.append((*map(float, box), label, index in selected))
        self.face_source_preview.set_overlays(overlays)
        self.face_source_preview.set_selected_lassos(selected)
        self.face_refine_button.setEnabled(
            bool(
                selected
                and not self._generation_active
                and self.artifacts is not None
                and self.artifacts.complete
            )
        )

    def _set_all_face_selections(self, checked: bool) -> None:
        self._syncing_face_selection = True
        for row in range(self.face_selection_list.count()):
            self.face_selection_list.item(row).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self._syncing_face_selection = False
        self._face_selection_changed()

    def _workspace_tab_changed(self, _index: int) -> None:
        self._refresh_lora_scope()
        self._selected_lora_changed(self.lora_list.currentItem(), None)

    def _editing_scope_active(self) -> bool:
        return self.workspace_tabs.currentIndex() == 1

    def _browse_edit_source(self) -> None:
        start = (
            self._edit_source_path.parent
            if self._edit_source_path is not None
            else self._output_directory
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select image to edit",
            str(start),
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if selected and not self._set_edit_source(Path(selected), confirm_reset=True):
            QMessageBox.warning(self, "Could not load image", selected)

    def _set_edit_source(self, path: Path, *, confirm_reset: bool) -> bool:
        try:
            image, _metadata = load_source_image(path)
        except (OSError, ValueError) as error:
            self.events.addItem(f"Could not load image-edit source: {error}")
            return False
        dimensions_changed = (
            bool(self.edit_regions or self._edit_source_path is not None)
            and (image.width, image.height)
            != (self.edit_canvas.canvas_width, self.edit_canvas.canvas_height)
        )
        if dimensions_changed and self.edit_regions and confirm_reset:
            answer = QMessageBox.question(
                self,
                "Reset edit regions?",
                "The replacement image has different dimensions. Loading it will clear "
                "the current edit boxes and their LoRA assignments.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        if dimensions_changed:
            self._clear_edit_regions()
        self.edit_canvas.set_canvas_size(image.width, image.height)
        if not self.edit_canvas.set_image(str(path)):
            return False
        self._edit_source_path = path.expanduser().resolve()
        self.edit_source_input.setText(str(self._edit_source_path))
        self._edit_result_path = None
        self.edit_result_input.clear()
        self.edit_result_preview.clear_image("Run an edit to compare the result")
        self.edit_run_button.setEnabled(
            bool(not self._generation_active and self.artifacts and self.artifacts.complete)
        )
        self.events.addItem(
            f"Image-edit source loaded: {self._edit_source_path.name} "
            f"({image.width}x{image.height})"
        )
        return True

    def _next_edit_region_name(self) -> str:
        existing = {region.name.casefold() for region in self.edit_regions}
        while True:
            self._edit_region_number += 1
            candidate = f"Edit region {self._edit_region_number}"
            if candidate.casefold() not in existing:
                return candidate

    def _edit_region_created(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        region = RegionDefinition(
            region_id=region_id,
            name=self._next_edit_region_name(),
            box=PixelBox(x0, y0, x1, y1),
        )
        self.edit_regions.append(region)
        self._normalize_edit_region_priorities()
        item = QListWidgetItem(self._region_label(region))
        item.setData(Qt.ItemDataRole.UserRole, region_id)
        self.edit_region_list.addItem(item)
        self.edit_canvas.add_region_box(
            region_id, QRectF(x0, y0, x1 - x0, y1 - y0), region.name
        )
        self._sync_edit_canvas_stack()
        self.edit_region_list.setCurrentItem(item)
        self._refresh_lora_scope()

    def _edit_region_changed(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        index = self._edit_region_index(region_id)
        self.edit_regions[index] = replace(
            self.edit_regions[index], box=PixelBox(x0, y0, x1, y1)
        )
        item = self._edit_region_list_item(region_id)
        if item is not None:
            item.setText(self._region_label(self.edit_regions[index]))

    def _edit_region_deleted(self, region_id: str) -> None:
        try:
            index = self._edit_region_index(region_id)
        except KeyError:
            return
        self.edit_regions.pop(index)
        item = self._edit_region_list_item(region_id)
        if item is not None:
            self.edit_region_list.takeItem(self.edit_region_list.row(item))
        for lora_id, binding in tuple(self._edit_lora_bindings.items()):
            remaining = tuple(item for item in binding.region_ids if item != region_id)
            if remaining != binding.region_ids:
                self._edit_lora_bindings[lora_id] = replace(
                    binding, global_scope=False, region_ids=remaining
                )
        self._normalize_edit_region_priorities()
        self._sync_edit_canvas_stack()
        self._refresh_lora_scope()

    def _clear_edit_regions(self) -> None:
        self.edit_canvas.clear_regions()
        self.edit_region_list.clear()
        self.edit_regions = []
        self._edit_lora_bindings = {
            lora_id: replace(binding, global_scope=False, region_ids=())
            for lora_id, binding in self._edit_lora_bindings.items()
        }
        self._selected_edit_region_changed(-1)

    def _edit_region_index(self, region_id: str) -> int:
        for index, region in enumerate(self.edit_regions):
            if region.region_id == region_id:
                return index
        raise KeyError(region_id)

    def _edit_region_list_item(self, region_id: str) -> QListWidgetItem | None:
        for row in range(self.edit_region_list.count()):
            item = self.edit_region_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == region_id:
                return item
        return None

    def _normalize_edit_region_priorities(self) -> None:
        count = len(self.edit_regions)
        self.edit_regions = [
            replace(region, priority=count - index)
            for index, region in enumerate(self.edit_regions)
        ]

    def _sync_edit_canvas_stack(self) -> None:
        self.edit_canvas.set_region_stack_order(
            tuple(region.region_id for region in self.edit_regions)
        )

    def _edit_region_order_changed(self, *_args) -> None:
        by_id = {region.region_id: region for region in self.edit_regions}
        ordered_ids = [
            str(self.edit_region_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.edit_region_list.count())
        ]
        if set(ordered_ids) != set(by_id):
            return
        self.edit_regions = [by_id[region_id] for region_id in ordered_ids]
        self._normalize_edit_region_priorities()
        self._sync_edit_canvas_stack()
        self._selected_edit_region_changed(self.edit_region_list.currentRow())

    def _edit_canvas_region_selected(self, region_id: str) -> None:
        item = self._edit_region_list_item(region_id)
        if item is not None and self.edit_region_list.currentItem() is not item:
            self.edit_region_list.setCurrentItem(item)

    def _selected_edit_region_changed(self, row: int) -> None:
        selected = 0 <= row < len(self.edit_regions)
        for control in (
            self.edit_region_name,
            self.edit_region_role,
            self.edit_region_face_prompt,
            self.edit_region_prompt,
        ):
            control.setEnabled(selected)
        self._edit_loading_region_form = True
        try:
            if selected:
                region = self.edit_regions[row]
                self.edit_region_name.setText(region.name)
                self.edit_region_role.setCurrentIndex(
                    max(0, self.edit_region_role.findData(region.spatial_role))
                )
                self.edit_region_face_prompt.setPlainText(region.face_identity_prompt)
                self.edit_region_prompt.setPlainText(region.prompt)
                self.edit_canvas.select_region(region.region_id)
            else:
                self.edit_region_name.clear()
                self.edit_region_role.setCurrentIndex(0)
                self.edit_region_face_prompt.clear()
                self.edit_region_prompt.clear()
        finally:
            self._edit_loading_region_form = False

    def _edit_region_name_edited(self) -> None:
        if self._edit_loading_region_form:
            return
        row = self.edit_region_list.currentRow()
        if not 0 <= row < len(self.edit_regions):
            return
        region = self.edit_regions[row]
        name = self.edit_region_name.text().strip()
        duplicate = any(
            index != row and candidate.name.casefold() == name.casefold()
            for index, candidate in enumerate(self.edit_regions)
        )
        if not name or duplicate:
            self.edit_region_name.setText(region.name)
            return
        self.edit_regions[row] = replace(region, name=name)
        item = self._edit_region_list_item(region.region_id)
        if item is not None:
            item.setText(self._region_label(self.edit_regions[row]))
        self.edit_canvas.set_region_name(region.region_id, name)
        self._refresh_lora_scope()

    def _edit_region_form_edited(self, *_args) -> None:
        if self._edit_loading_region_form:
            return
        row = self.edit_region_list.currentRow()
        if not 0 <= row < len(self.edit_regions):
            return
        self.edit_regions[row] = replace(
            self.edit_regions[row],
            spatial_role=str(self.edit_region_role.currentData()),
            face_identity_prompt=self.edit_region_face_prompt.toPlainText(),
            prompt=self.edit_region_prompt.toPlainText(),
        )
        item = self._edit_region_list_item(self.edit_regions[row].region_id)
        if item is not None:
            item.setText(self._region_label(self.edit_regions[row]))

    def _browse_face_source(self) -> None:
        start = (
            self._face_source_path.parent
            if self._face_source_path is not None
            else self._output_directory
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select first-pass PNG",
            str(start),
            "PNG images (*.png)",
        )
        if not selected:
            return
        if not self._set_face_refinement_source(Path(selected)):
            QMessageBox.warning(self, "Could not load PNG", selected)
            return
        self.events.addItem(f"Face-refinement source set to {selected}")

    def _use_latest_face_source(self, _checked=False, *, show_message: bool = True) -> None:
        latest = self._latest_first_pass_output()
        if latest is None or not self._set_face_refinement_source(latest):
            if show_message:
                self.statusBar().showMessage("No first-pass PNG found", 5000)
            return
        if show_message:
            self.events.addItem(f"Loaded latest first-pass PNG: {latest.name}")

    def _run_image_edit(self) -> None:
        source_path = self._edit_source_path
        if source_path is None or not source_path.is_file():
            QMessageBox.warning(self, "Source image required", "Load an image before editing.")
            return
        active_regions = [
            region
            for region in self.edit_regions
            if region.enabled
            and (region.prompt.strip() or region.face_identity_prompt.strip())
        ]
        if not self.edit_global_prompt.toPlainText().strip() and not active_regions:
            QMessageBox.warning(
                self,
                "Edit prompt required",
                "With a blank global prompt, add at least one box with a regional prompt.",
            )
            return
        active_ids = {region.region_id for region in active_regions}
        invalid_lora = next(
            (
                lora["name"]
                for lora in self._edit_lora_payload()
                if not lora["global"]
                and not set(lora["region_ids"]).issubset(active_ids)
            ),
            None,
        )
        if invalid_lora is not None:
            QMessageBox.warning(
                self,
                "Regional prompt required",
                f"{invalid_lora} targets an edit region without an active prompt.",
            )
            return
        payload = self._worker_payload()
        payload.update(
            {
                "image_path": str(source_path),
                "output_directory": str(self._output_directory),
                "prompt": self.edit_global_prompt.toPlainText(),
                "seed": self.edit_seed_input.value(),
                "steps": self.edit_steps_input.value(),
                "sampler": str(self.edit_sampler_input.currentData()),
                "scheduler": str(self.edit_scheduler_input.currentData()),
                "denoise": self.edit_denoise_input.value(),
                "composite_feather_pixels": (
                    self.edit_composite_feather_input.value()
                ),
                "regional_prompt_strength": self.edit_regional_strength_input.value(),
                "regional_outside_penalty": self.edit_outside_penalty_input.value(),
                "regional_feather_pixels": self.edit_spatial_falloff_input.value(),
                "regional_subject_competition": (
                    self.edit_subject_competition_input.isChecked()
                ),
                "regional_subject_fill": self.edit_subject_fill_input.isChecked(),
                "regional_late_step_scale": self.edit_late_step_scale_input.value(),
                "regional_lora_delta_adaptation": (
                    self.edit_lora_adaptation_input.isChecked()
                ),
                "regional_lora_delta_adaptation_gain": (
                    self.edit_lora_adaptation_gain_input.value()
                ),
                "regions": [self._region_payload(region) for region in self.edit_regions],
                "loras": self._edit_lora_payload(),
                "project_json": project_document(self._project_state()),
            }
        )
        self._pending_image_edit_payload = payload
        self._generation_completed = False
        self._active_task = "image_edit"
        self._set_generation_active(True)
        self._advance_pending_generation()

    @staticmethod
    def _region_payload(region: RegionDefinition) -> dict[str, object]:
        return {
            "id": region.region_id,
            "name": region.name,
            "box": {
                "x0": region.box.x0,
                "y0": region.box.y0,
                "x1": region.box.x1,
                "y1": region.box.y1,
            },
            "prompt": region.prompt,
            "face_identity_prompt": region.face_identity_prompt,
            "enabled": region.enabled,
            "priority": region.priority,
            "spatial_role": region.spatial_role,
        }

    def _run_face_refinement(self) -> None:
        source_path = self._face_source_path
        if source_path is None or not source_path.is_file():
            QMessageBox.warning(
                self, "Source PNG required", "Load a first-pass PNG before refining."
            )
            return
        selected_face_indices = self._selected_face_indices()
        if not self._face_detections or not selected_face_indices:
            QMessageBox.warning(
                self,
                "Select faces first",
                "Run face detection, then check one or more numbered faces to refine.",
            )
            return
        loras = self._lora_payload()
        regional_loras = [
            lora
            for lora in loras
            if not lora["global"]
            and lora["region_ids"]
            and float(lora["strength"]) != 0.0
        ]
        if not regional_loras:
            message = "Assign at least one enabled LoRA to a subject region first."
            self.events.addItem(message)
            QMessageBox.warning(self, "Regional LoRA required", message)
            return
        pixmap = QPixmap(str(source_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Could not load PNG", str(source_path))
            return
        base_width = max(1, self.width_input.value())
        base_height = max(1, self.height_input.value())
        scale_x = pixmap.width() / base_width
        scale_y = pixmap.height() / base_height
        regions = [
            {
                "id": region.region_id,
                "name": region.name,
                "box": {
                    "x0": region.box.x0 * scale_x,
                    "y0": region.box.y0 * scale_y,
                    "x1": region.box.x1 * scale_x,
                    "y1": region.box.y1 * scale_y,
                },
                "prompt": region.prompt,
                "face_identity_prompt": region.face_identity_prompt,
                "enabled": region.enabled,
                "priority": region.priority,
                "spatial_role": region.spatial_role,
            }
            for region in self.regions
        ]
        payload = self._worker_payload()
        payload.update(
            {
                "image_path": str(source_path),
                "output_directory": str(source_path.parent),
                "seed": self.face_detail_seed_input.value(),
                "steps": self.face_detail_steps_input.value(),
                "denoise": self.face_detail_denoise_input.value(),
                "crop_size": int(self.face_detail_crop_size_input.currentData()),
                "padding": self.face_detail_padding_input.value(),
                "feather": self.face_detail_feather_input.value(),
                "blend": self.face_detail_blend_input.value(),
                "lora_scale": self.face_detail_lora_scale_input.value(),
                "detector_threshold": (
                    self.face_detail_detector_threshold_input.value()
                ),
                "detector_provider": str(
                    self.face_detail_detector_provider_input.currentData()
                ),
                "selected_face_indices": selected_face_indices,
                "manual_face_paths": [
                    [list(point) for point in path]
                    for path in self._manual_face_paths
                ],
                "regions": regions,
                "loras": loras,
                "project_json": project_document(self._project_state()),
            }
        )
        self._pending_face_refinement_payload = payload
        self._generation_completed = False
        self._active_task = "face_refinement"
        self._set_generation_active(True)
        self._advance_pending_generation()

    def _browse_upscale_model(self) -> None:
        start = (
            self._upscale_model_path.parent
            if self._upscale_model_path is not None
            else self.settings.model_directories.upscale_models
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

    def _refresh_runtime_path_controls(self) -> None:
        if not hasattr(self, "worker_python_input"):
            return
        directories = self.settings.model_directories
        self.comfyui_root_input.setText(str(self.settings.comfyui_root))
        self.comfyui_root_input.setToolTip(str(self.settings.comfyui_root))
        self.worker_python_input.setText(str(self.settings.worker_python))
        self.worker_python_input.setToolTip(str(self.settings.worker_python))
        detector = self.settings.face_detector_path
        self.face_detector_input.setText(str(detector) if detector else "Auto-discover")
        self.face_detector_input.setToolTip(
            str(detector) if detector else "Search the selected ComfyUI checkout"
        )
        selections = (
            (self.transformer_status, directories.diffusion_model_file),
            (self.text_status, directories.text_encoder_file),
            (self.vae_status, directories.vae_file),
        )
        for status, path in selections:
            status.setToolTip(str(path) if path else "Automatic compatible-model discovery")

    def _apply_runtime_settings(self, settings: AppSettings, *, rediscover: bool) -> bool:
        if self._generation_active:
            QMessageBox.warning(
                self,
                "Generation is running",
                "Stop the current generation before changing GPU runtime or model files.",
            )
            return False
        if hasattr(self, "worker_client") and self.worker_client.running:
            self.worker_client.stop()
        self.settings = settings
        if hasattr(self, "worker_client"):
            self.worker_client.settings = settings
        self._model_loaded = False
        self._models_compatible = False
        self._accelerator_available = False
        self.load_model_button.setEnabled(False)
        self.worker_status.setText("Stopped")
        self.accelerator_status.setText("Not probed")
        self._refresh_runtime_path_controls()
        if rediscover:
            self.artifacts = None
            self.transformer_status.setText("Not discovered")
            self.text_status.setText("Not discovered")
            self.vae_status.setText("Not discovered")
            self.discover_models()
        return True

    def _browse_comfyui_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select ComfyUI checkout",
            str(self.settings.comfyui_root),
        )
        if not selected:
            return
        root = Path(selected).expanduser().resolve()
        directories = ModelDirectories(
            diffusion_models=root / "models" / "diffusion_models",
            text_encoders=root / "models" / "text_encoders",
            vae=root / "models" / "vae",
            loras=root / "models" / "loras",
            upscale_models=root / "models" / "upscale_models",
        )
        settings = replace(
            self.settings,
            comfyui_root=root,
            worker_python=discover_worker_python(root),
            model_directories=directories,
            face_detector_path=None,
            default_upscale_model=None,
        )
        if self._apply_runtime_settings(settings, rediscover=True):
            self._upscale_model_path = None
            self.upscale_model_input.clear()
            self._set_upscale_controls_enabled()
            self.events.addItem(f"ComfyUI checkout set to {root}")

    def _browse_worker_python(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select CUDA- or ROCm-enabled ComfyUI Python",
            str(self.settings.worker_python.parent),
            "Python interpreter (python python3 python.exe);;All files (*)",
        )
        if not selected:
            return
        path = Path(selected).expanduser().absolute()
        if self._apply_runtime_settings(
            replace(self.settings, worker_python=path), rediscover=False
        ):
            self.events.addItem(f"GPU worker Python set to {path}")

    def _auto_select_worker_python(self) -> None:
        path = discover_worker_python(self.settings.comfyui_root)
        if self._apply_runtime_settings(
            replace(self.settings, worker_python=path), rediscover=False
        ):
            self.events.addItem(f"Auto-selected GPU worker Python: {path}")

    def _browse_primary_model(self, field_name: str) -> None:
        directories = self.settings.model_directories
        labels = {
            "diffusion_model_file": ("Krea transformer", directories.diffusion_models),
            "text_encoder_file": ("Qwen text encoder", directories.text_encoders),
            "vae_file": ("Krea VAE", directories.vae),
        }
        label, directory = labels[field_name]
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {label}",
            str(directory),
            "Safetensors model (*.safetensors);;All files (*)",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        updated_directories = replace(directories, **{field_name: path})
        if self._apply_runtime_settings(
            replace(self.settings, model_directories=updated_directories),
            rediscover=True,
        ):
            self.events.addItem(f"Selected {label}: {path.name}")

    def _clear_primary_model(self, field_name: str) -> None:
        directories = replace(
            self.settings.model_directories, **{field_name: None}
        )
        if self._apply_runtime_settings(
            replace(self.settings, model_directories=directories), rediscover=True
        ):
            self.events.addItem("Returned model selection to automatic discovery")

    def _browse_face_detector_model(self) -> None:
        start = (
            self.settings.face_detector_path.parent
            if self.settings.face_detector_path
            else self.settings.comfyui_root
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select ONNX face detector",
            str(start),
            "ONNX model (*.onnx);;All files (*)",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if self._apply_runtime_settings(
            replace(self.settings, face_detector_path=path), rediscover=False
        ):
            self.events.addItem(f"Face detector set to {path.name}")

    def _clear_face_detector_model(self) -> None:
        if self._apply_runtime_settings(
            replace(self.settings, face_detector_path=None), rediscover=False
        ):
            self.events.addItem("Face detector returned to automatic discovery")

    def _diffusion_model_changed(self, index: int) -> None:
        selected = self.diffusion_model_input.itemData(index)
        if not selected:
            return
        selected_path = Path(str(selected)).expanduser().resolve()
        current_path = self.settings.model_directories.diffusion_model_file
        if current_path is not None and current_path.expanduser().resolve() == selected_path:
            return
        directories = replace(
            self.settings.model_directories,
            diffusion_model_file=selected_path,
        )
        changed = self._apply_runtime_settings(
            replace(self.settings, model_directories=directories),
            rediscover=True,
        )
        if not changed:
            self.discover_models()
            return
        self.events.addItem(f"Selected Krea transformer: {selected_path.name}")
        if "raw" in selected_path.name.casefold():
            self.events.addItem(
                "Raw checkpoint selected: regional hooks are architecture-compatible, "
                "and generation still uses the CFG-free Turbo sampling path; apply "
                "a Raw-to-Turbo distillation LoRA globally for that path"
            )

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
            candidates = discover_krea_transformers(
                self.settings.model_directories.diffusion_models
            )
        except (OSError, ValueError) as error:
            self.events.addItem(f"Model discovery failed: {error}")
            self.statusBar().showMessage("Model discovery failed")
            return
        selected_transformer = (
            self.artifacts.transformer.path if self.artifacts.transformer else None
        )
        candidate_paths = [artifact.path for artifact in candidates]
        if selected_transformer is not None and selected_transformer not in candidate_paths:
            candidate_paths.append(selected_transformer)
        self.diffusion_model_input.blockSignals(True)
        self.diffusion_model_input.clear()
        for path in sorted(candidate_paths, key=lambda item: item.name.casefold()):
            self.diffusion_model_input.addItem(path.name, str(path))
            self.diffusion_model_input.setItemData(
                self.diffusion_model_input.count() - 1,
                str(path),
                Qt.ItemDataRole.ToolTipRole,
            )
        if selected_transformer is not None:
            selected_index = self.diffusion_model_input.findData(
                str(selected_transformer)
            )
            self.diffusion_model_input.setCurrentIndex(selected_index)
        self.diffusion_model_input.blockSignals(False)
        self.transformer_status.setText(self._artifact_label(self.artifacts.transformer))
        self.text_status.setText(self._artifact_label(self.artifacts.text_encoder))
        self.vae_status.setText(self._artifact_label(self.artifacts.vae))
        self._refresh_runtime_path_controls()
        state = "complete" if self.artifacts.complete else "incomplete"
        self.generate_button.setEnabled(
            bool(self.artifacts.complete and not self._generation_active)
        )
        self.face_refine_button.setEnabled(
            bool(
                self.artifacts.complete
                and not self._generation_active
                and self._face_source_path is not None
                and self._face_source_path.is_file()
                and self._selected_face_indices()
            )
        )
        self.face_detect_button.setEnabled(
            bool(
                not self._generation_active
                and self._face_source_path is not None
                and self._face_source_path.is_file()
            )
        )
        self.edit_run_button.setEnabled(
            bool(
                self.artifacts.complete
                and not self._generation_active
                and self._edit_source_path is not None
                and self._edit_source_path.is_file()
            )
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
        self.region_face_identity_prompt.setEnabled(selected)
        self._loading_region_form = True
        try:
            if selected:
                region = self.regions[row]
                self.region_name.setText(region.name)
                role_index = self.region_role.findData(region.spatial_role)
                self.region_role.setCurrentIndex(max(0, role_index))
                self.region_prompt.setPlainText(region.prompt)
                self.region_face_identity_prompt.setPlainText(
                    region.face_identity_prompt
                )
                self.canvas.select_region(region.region_id)
            else:
                self.region_name.clear()
                self.region_role.setCurrentIndex(0)
                self.region_prompt.clear()
                self.region_face_identity_prompt.clear()
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
            face_identity_prompt=self.region_face_identity_prompt.toPlainText(),
        )

    def _canvas_dimensions_changed(self) -> None:
        previous_width = max(1, self.canvas.canvas_width)
        previous_height = max(1, self.canvas.canvas_height)
        geometry = CanvasGeometry.resolve(self.width_input.value(), self.height_input.value())
        scale_x = geometry.aligned_width / previous_width
        scale_y = geometry.aligned_height / previous_height
        original_regions = tuple(self.regions)
        self.canvas.set_canvas_size(geometry.aligned_width, geometry.aligned_height)
        scaled_regions = []
        for region in original_regions:
            box = PixelBox(
                region.box.x0 * scale_x,
                region.box.y0 * scale_y,
                region.box.x1 * scale_x,
                region.box.y1 * scale_y,
            )
            scaled_region = replace(region, box=box)
            scaled_regions.append(scaled_region)
            self.canvas.region_item(region.region_id).set_scene_geometry(
                QRectF(box.x0, box.y0, box.width, box.height),
                notify=False,
                enforce_minimum=False,
            )
            item = self._region_list_item(region.region_id)
            if item is not None:
                item.setText(self._region_label(scaled_region))
        self.regions = scaled_regions
        self.events.addItem(
            "Canvas resolved to "
            f"{geometry.aligned_width}×{geometry.aligned_height} px "
            f"({geometry.patch_width}×{geometry.patch_height} image tokens); "
            f"scaled {len(scaled_regions)} region box(es) proportionally"
        )

    def _browse_lora(self) -> None:
        comfy_loras = self.settings.model_directories.loras
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
        self._edit_lora_bindings[entry.lora_id] = LoraBinding(
            lora_id=entry.lora_id,
            global_scope=False,
            region_ids=(),
            strength=self.lora_library.binding_for(entry.lora_id).strength,
            trigger_phrase=entry.path.stem,
        )
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
        identity = (
            f"  [identity: {binding.trigger_phrase}]"
            if binding.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING
            else ""
        )
        return f"{entry.display_name}  ×{binding.strength:.2f}{identity}"

    def _remove_selected_lora(self) -> None:
        item = self.lora_list.currentItem()
        if item is None:
            return
        lora_id = item.data(Qt.ItemDataRole.UserRole)
        entry = self.lora_library.get(lora_id)
        self.lora_library.remove(lora_id)
        self._edit_lora_bindings.pop(lora_id, None)
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
        self._syncing_lora_routing = True
        try:
            self.lora_routing_mode_input.setEnabled(lora_id is not None)
            if lora_id is None:
                self.lora_routing_mode_input.setCurrentIndex(
                    self.lora_routing_mode_input.findData(STANDARD_LORA_ROUTING)
                )
                self.lora_trigger_input.clear()
            else:
                binding = self._active_lora_binding(lora_id)
                self.lora_routing_mode_input.setCurrentIndex(
                    self.lora_routing_mode_input.findData(binding.routing_mode)
                )
                self.lora_trigger_input.setText(binding.trigger_phrase)
        finally:
            self._syncing_lora_routing = False
        self._refresh_lora_routing_controls()

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

    def _active_lora_binding(self, lora_id: str) -> LoraBinding:
        if self._editing_scope_active():
            return self._edit_lora_bindings[lora_id]
        return self.lora_library.binding_for(lora_id)

    def _refresh_lora_routing_controls(self) -> None:
        lora_id = self._current_lora_id()
        character_identity = (
            lora_id is not None
            and self.lora_routing_mode_input.currentData()
            == CHARACTER_IDENTITY_LORA_ROUTING
        )
        self.lora_trigger_input.setEnabled(character_identity)
        self.lora_routing_note.setText(
            "Character identity routing adds a person/face identity instruction to "
            "each assigned region. The LoRA keeps full regional text coverage and "
            "its image delta remains confined to the region box."
            if character_identity
            else "Standard routing gates text-fusion deltas to assigned regional "
            "clauses. Image tokens outside the assigned box cannot attend those clauses, "
            "and global or other-region text cannot read the box's image keys. "
            "Image-to-image attention stays continuous to avoid a rectangular seam. "
            "Main-stream attention key/value targets remain omitted because their "
            "outputs broadcast."
        )

    def _lora_routing_mode_changed(self) -> None:
        if self._syncing_lora_routing:
            return
        lora_id = self._current_lora_id()
        if lora_id is None:
            return
        routing_mode = str(self.lora_routing_mode_input.currentData())
        binding = self._active_lora_binding(lora_id)
        if (
            routing_mode == CHARACTER_IDENTITY_LORA_ROUTING
            and binding.global_scope
        ):
            self._syncing_lora_routing = True
            try:
                self.lora_routing_mode_input.setCurrentIndex(
                    self.lora_routing_mode_input.findData(binding.routing_mode)
                )
            finally:
                self._syncing_lora_routing = False
            self.events.addItem(
                "Assign the LoRA to one or more regions before enabling Character identity routing"
            )
            self._refresh_lora_routing_controls()
            return
        if self._editing_scope_active():
            self._edit_lora_bindings[lora_id] = replace(
                binding, routing_mode=routing_mode
            )
        else:
            self.lora_library.set_routing_mode(lora_id, routing_mode)
        self._refresh_lora_routing_controls()
        item = self._lora_list_item(lora_id)
        if item is not None:
            item.setText(self._lora_label(lora_id))
        entry = self.lora_library.get(lora_id)
        self.events.addItem(
            f"Set {entry.display_name} routing to "
            f"{'Character identity' if routing_mode == CHARACTER_IDENTITY_LORA_ROUTING else 'Standard'}"
        )

    def _lora_trigger_edited(self) -> None:
        if self._syncing_lora_routing:
            return
        lora_id = self._current_lora_id()
        if lora_id is None:
            return
        binding = self._active_lora_binding(lora_id)
        previous = binding.trigger_phrase
        phrase = self.lora_trigger_input.text().strip()
        if not phrase:
            self.lora_trigger_input.setText(previous)
            self.events.addItem("Character identity trigger cannot be empty")
            return
        if self._editing_scope_active():
            self._edit_lora_bindings[lora_id] = replace(
                binding, trigger_phrase=phrase
            )
        else:
            self.lora_library.set_trigger_phrase(lora_id, phrase)
        self.lora_trigger_input.setText(phrase)
        item = self._lora_list_item(lora_id)
        if item is not None:
            item.setText(self._lora_label(lora_id))
        entry = self.lora_library.get(lora_id)
        self.events.addItem(f"Set {entry.display_name} identity trigger to {phrase!r}")

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
                    "routing_mode": binding.routing_mode,
                    "trigger_phrase": binding.trigger_phrase,
                }
            )
        return payload

    def _edit_lora_payload(self) -> list[dict[str, object]]:
        payload = []
        for entry in self.lora_library.entries():
            binding = self._edit_lora_bindings.get(entry.lora_id)
            if binding is None or (not binding.global_scope and not binding.region_ids):
                continue
            payload.append(
                {
                    "id": entry.lora_id,
                    "name": entry.display_name,
                    "path": str(entry.path),
                    "strength": self.lora_library.binding_for(entry.lora_id).strength,
                    "global": binding.global_scope,
                    "region_ids": list(binding.region_ids),
                    "routing_mode": binding.routing_mode,
                    "trigger_phrase": binding.trigger_phrase,
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
            editing = self._editing_scope_active()
            self.lora_scope_context_label.setText(
                "Image-edit scope: Disabled, Global, or one or more edit regions"
                if editing
                else "Generation scope: choose Global or one or more named regions"
            )
            lora_id = self._current_lora_id()
            if lora_id is None:
                self.lora_scope_list.setEnabled(False)
                return
            self.lora_scope_list.setEnabled(True)
            binding = self._active_lora_binding(lora_id)
            if editing:
                disabled = not binding.global_scope and not binding.region_ids
                self.lora_scope_list.addItem(
                    self._scope_item("Disabled for image editing", DISABLED_SCOPE_ID, disabled)
                )
            global_item = self._scope_item("Global", GLOBAL_SCOPE_ID, binding.global_scope)
            self.lora_scope_list.addItem(global_item)
            regions = self.edit_regions if editing else self.regions
            for region in regions:
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
        editing = self._editing_scope_active()
        current = self._active_lora_binding(lora_id)
        if editing and (
            scope_id == DISABLED_SCOPE_ID
            and changed_item.checkState() == Qt.CheckState.Checked
        ):
            binding = replace(current, global_scope=False, region_ids=())
            self._edit_lora_bindings[lora_id] = binding
        elif scope_id == GLOBAL_SCOPE_ID and changed_item.checkState() == Qt.CheckState.Checked:
            binding = replace(current, global_scope=True, region_ids=())
            if editing:
                self._edit_lora_bindings[lora_id] = binding
            else:
                binding = self.lora_library.assign_global(lora_id)
        else:
            selected_regions = tuple(
                item.data(Qt.ItemDataRole.UserRole)
                for index in range(self.lora_scope_list.count())
                if (item := self.lora_scope_list.item(index)).data(Qt.ItemDataRole.UserRole)
                not in {GLOBAL_SCOPE_ID, DISABLED_SCOPE_ID}
                and item.checkState() == Qt.CheckState.Checked
            )
            if editing:
                binding = replace(
                    current, global_scope=False, region_ids=selected_regions
                )
                self._edit_lora_bindings[lora_id] = binding
            else:
                binding = self.lora_library.assign_regions(lora_id, selected_regions)
        if (
            binding.global_scope
            and binding.routing_mode == CHARACTER_IDENTITY_LORA_ROUTING
        ):
            if editing:
                binding = replace(binding, routing_mode=STANDARD_LORA_ROUTING)
                self._edit_lora_bindings[lora_id] = binding
            else:
                binding = self.lora_library.set_routing_mode(
                    lora_id, STANDARD_LORA_ROUTING
                )
            self.events.addItem(
                "Character identity routing returned to Standard because the LoRA is Global"
            )
        self._refresh_lora_scope()
        self._selected_lora_changed(self.lora_list.currentItem(), None)
        entry = self.lora_library.get(lora_id)
        regions = self.edit_regions if editing else self.regions
        names = {
            region.region_id: region.name
            for region in regions
        }
        scope = (
            "Global"
            if binding.global_scope
            else (
                ", ".join(names.get(region_id, region_id) for region_id in binding.region_ids)
                or "Disabled"
            )
        )
        self.events.addItem(f"Assigned {entry.display_name} to {scope}")

    def _project_state(self) -> ProjectState:
        directories = self.settings.model_directories
        runtime = {
            "diffusion_models": str(directories.diffusion_models),
            "text_encoders": str(directories.text_encoders),
            "vae": str(directories.vae),
            "loras": str(directories.loras),
            "upscale_models": str(directories.upscale_models),
            "diffusion_model_file": (
                str(directories.diffusion_model_file)
                if directories.diffusion_model_file else None
            ),
            "text_encoder_file": (
                str(directories.text_encoder_file)
                if directories.text_encoder_file else None
            ),
            "vae_file": str(directories.vae_file) if directories.vae_file else None,
            "face_detector_path": (
                str(self.settings.face_detector_path)
                if self.settings.face_detector_path else None
            ),
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
                routing_mode=binding.routing_mode,
                trigger_phrase=binding.trigger_phrase,
                edit_enabled=bool(
                    (edit_binding := self._edit_lora_bindings[entry.lora_id]).global_scope
                    or edit_binding.region_ids
                ),
                edit_global_scope=edit_binding.global_scope,
                edit_region_ids=edit_binding.region_ids,
                edit_routing_mode=edit_binding.routing_mode,
                edit_trigger_phrase=edit_binding.trigger_phrase,
            )
            for entry in self.lora_library.entries()
        )
        return ProjectState(
            canvas_width=self.width_input.value(),
            canvas_height=self.height_input.value(),
            global_prompt=self.global_prompt.toPlainText(),
            steps=self.steps_input.value(),
            sampler=str(self.sampler_input.currentData()),
            scheduler=str(self.scheduler_input.currentData()),
            seed=self.seed_input.value(),
            seed_mode=str(self.seed_mode_input.currentData()),
            batch_mode=self.batch_mode_input.isChecked(),
            batch_count=self.batch_count_input.value(),
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
            projector_identity_protection=(
                self.projector_identity_protection_input.value()
            ),
            face_detail_seed=self.face_detail_seed_input.value(),
            face_detail_steps=self.face_detail_steps_input.value(),
            face_detail_denoise=self.face_detail_denoise_input.value(),
            face_detail_crop_size=int(self.face_detail_crop_size_input.currentData()),
            face_detail_padding=self.face_detail_padding_input.value(),
            face_detail_feather=self.face_detail_feather_input.value(),
            face_detail_blend=self.face_detail_blend_input.value(),
            face_detail_lora_scale=self.face_detail_lora_scale_input.value(),
            face_detail_detector_threshold=(
                self.face_detail_detector_threshold_input.value()
            ),
            face_detail_detector_provider=str(
                self.face_detail_detector_provider_input.currentData()
            ),
            post_upscale=self.post_upscale_input.isChecked(),
            upscale_scale=int(self.upscale_scale_input.currentData()),
            upscale_method=str(self.upscale_method_input.currentData()),
            upscale_model=self._upscale_model_path,
            regions=tuple(self.regions),
            loras=saved_loras,
            runtime=runtime,
            background_image=self._background_image_path,
            image_edit=ImageEditState(
                source_image=self._edit_source_path,
                width=(self.edit_canvas.canvas_width if self._edit_source_path else 0),
                height=(self.edit_canvas.canvas_height if self._edit_source_path else 0),
                global_prompt=self.edit_global_prompt.toPlainText(),
                steps=self.edit_steps_input.value(),
                sampler=str(self.edit_sampler_input.currentData()),
                scheduler=str(self.edit_scheduler_input.currentData()),
                seed=self.edit_seed_input.value(),
                denoise=self.edit_denoise_input.value(),
                composite_feather_pixels=self.edit_composite_feather_input.value(),
                regional_prompt_strength=self.edit_regional_strength_input.value(),
                regional_outside_penalty=self.edit_outside_penalty_input.value(),
                regional_feather_pixels=self.edit_spatial_falloff_input.value(),
                regional_subject_competition=(
                    self.edit_subject_competition_input.isChecked()
                ),
                regional_subject_fill=self.edit_subject_fill_input.isChecked(),
                regional_late_step_scale=self.edit_late_step_scale_input.value(),
                regional_lora_delta_adaptation=(
                    self.edit_lora_adaptation_input.isChecked()
                ),
                regional_lora_delta_adaptation_gain=(
                    self.edit_lora_adaptation_gain_input.value()
                ),
                regions=tuple(self.edit_regions),
            ),
        )

    def _clear_generation_canvas(self) -> None:
        self.canvas.clear_image()
        self._background_image_path = None
        self.events.addItem("Cleared generation canvas image")
        self.statusBar().showMessage(
            "Canvas image cleared; regions and prompts were kept", 5000
        )

    def _new_project(self) -> None:
        answer = QMessageBox.question(
            self,
            "Start a new project?",
            "This clears the current prompts, regions, LoRAs, project settings, "
            "and image previews. Saved project and PNG files are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.settings = self._default_settings
        state = ProjectState(
            canvas_width=self._default_settings.default_width,
            canvas_height=self._default_settings.default_height,
            steps=self._default_settings.default_steps,
            sampler=self._default_settings.default_sampler,
            scheduler=self._default_settings.default_scheduler,
            seed=self._default_settings.default_seed,
            seed_mode=self._default_settings.default_seed_mode,
            upscale_model=self._default_settings.default_upscale_model,
        )
        self._apply_project_state(state, load_latest_face_source=False)
        self._current_project_path = None
        self.setWindowTitle("K2 Region Lab")
        self.events.addItem("Started a new project with default settings")
        self.statusBar().showMessage("New project ready", 5000)
        if self.settings.auto_start_worker:
            self._start_worker()

    def _open_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open K2 Region Lab project",
            str(self._project_directory),
            "K2 Region Lab project (*.k2lab.json *.json)",
        )
        if selected:
            self._load_project_from(Path(selected), show_error_dialog=True)

    def _import_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import K2 Region Lab image",
            str(self._output_directory),
            "K2 Region Lab image (*.png);;PNG image (*.png)",
        )
        if selected:
            self._load_project_image_from(Path(selected), show_error_dialog=True)

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
        # The worker interpreter is an application/runtime choice, not project
        # content. In particular, opening an older project must not silently
        # downgrade a launch configured for a different CUDA/ROCm environment.
        worker_python = current.worker_python
        return AppSettings(
            model_directories=ModelDirectories(
                diffusion_models=Path(
                    runtime.get("diffusion_models", current_directories.diffusion_models)
                ).expanduser(),
                text_encoders=Path(
                    runtime.get("text_encoders", current_directories.text_encoders)
                ).expanduser(),
                vae=Path(runtime.get("vae", current_directories.vae)).expanduser(),
                loras=Path(runtime.get("loras", current_directories.loras)).expanduser(),
                upscale_models=Path(
                    runtime.get("upscale_models", current_directories.upscale_models)
                ).expanduser(),
                diffusion_model_file=(
                    Path(value).expanduser()
                    if (value := runtime.get(
                        "diffusion_model_file", current_directories.diffusion_model_file
                    ))
                    else None
                ),
                text_encoder_file=(
                    Path(value).expanduser()
                    if (value := runtime.get(
                        "text_encoder_file", current_directories.text_encoder_file
                    ))
                    else None
                ),
                vae_file=(
                    Path(value).expanduser()
                    if (value := runtime.get("vae_file", current_directories.vae_file))
                    else None
                ),
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
            face_detector_path=(
                Path(value).expanduser()
                if (value := runtime.get(
                    "face_detector_path", current.face_detector_path
                ))
                else None
            ),
            default_upscale_model=current.default_upscale_model,
            default_width=state.canvas_width,
            default_height=state.canvas_height,
            default_steps=current.default_steps,
            default_sampler=current.default_sampler,
            default_scheduler=current.default_scheduler,
            default_seed=current.default_seed,
            default_seed_mode=current.default_seed_mode,
            config_file=current.config_file,
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

        self._apply_project_state(state, load_latest_face_source=True)
        self._current_project_path = path.expanduser().resolve()
        self.setWindowTitle(f"K2 Region Lab — {self._current_project_path.name}")
        self.events.addItem(f"Opened project {self._current_project_path}")
        self.statusBar().showMessage(f"Opened {self._current_project_path.name}", 5000)
        if self.settings.auto_start_worker:
            self._start_worker()
        return True

    def _load_project_image_from(
        self, path: Path, *, show_error_dialog: bool = False
    ) -> bool:
        try:
            state = load_project_image(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logging.getLogger(__name__).exception("project image import failed")
            self.events.addItem(f"Project image import failed: {error}")
            if show_error_dialog:
                QMessageBox.warning(self, "Project image import failed", str(error))
            return False

        self._apply_project_state(state, load_latest_face_source=False)
        imported = path.expanduser().resolve()
        self._current_project_path = None
        self.setWindowTitle(f"K2 Region Lab — {imported.name} (imported)")
        self.events.addItem(f"Imported project metadata from {imported}")
        self.statusBar().showMessage(f"Imported {imported.name}", 5000)
        if self.settings.auto_start_worker:
            self._start_worker()
        return True

    def _apply_project_state(
        self, state: ProjectState, *, load_latest_face_source: bool
    ) -> None:
        self.worker_client.stop()
        self._pending_generation_payload = None
        self._pending_image_edit_payload = None
        self._pending_face_refinement_payload = None
        self._clear_batch_state()
        self._active_task = None
        self._worker_bootstrap_stage = None
        self._generation_completed = False
        self._set_generation_active(False)
        self.settings = self._settings_from_project(state)
        self.worker_client.settings = self.settings
        self._refresh_runtime_path_controls()
        self.width_input.setValue(state.canvas_width)
        self.height_input.setValue(state.canvas_height)
        self.canvas.set_canvas_size(state.canvas_width, state.canvas_height)
        self.global_prompt.setPlainText(state.global_prompt)
        self.steps_input.setValue(state.steps)
        sampler_index = self.sampler_input.findData(state.sampler)
        self.sampler_input.setCurrentIndex(max(0, sampler_index))
        scheduler_index = self.scheduler_input.findData(state.scheduler)
        self.scheduler_input.setCurrentIndex(max(0, scheduler_index))
        self.seed_input.setValue(state.seed)
        seed_mode_index = self.seed_mode_input.findData(state.seed_mode)
        self.seed_mode_input.setCurrentIndex(max(0, seed_mode_index))
        self.batch_count_input.setValue(state.batch_count)
        self.batch_mode_input.setChecked(state.batch_mode)
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
        self._set_projector_controls(
            enabled=state.projector_enabled,
            preset=state.projector_preset,
            values=state.projector_values,
            multiplier=state.projector_multiplier,
            identity_protection=state.projector_identity_protection,
        )
        self.face_detail_seed_input.setValue(state.face_detail_seed)
        self.face_detail_steps_input.setValue(state.face_detail_steps)
        self.face_detail_denoise_input.setValue(state.face_detail_denoise)
        crop_size_index = self.face_detail_crop_size_input.findData(
            state.face_detail_crop_size
        )
        self.face_detail_crop_size_input.setCurrentIndex(max(0, crop_size_index))
        self.face_detail_padding_input.setValue(state.face_detail_padding)
        self.face_detail_feather_input.setValue(state.face_detail_feather)
        self.face_detail_blend_input.setValue(state.face_detail_blend)
        self.face_detail_lora_scale_input.setValue(state.face_detail_lora_scale)
        self.face_detail_detector_threshold_input.setValue(
            state.face_detail_detector_threshold
        )
        detector_provider_index = self.face_detail_detector_provider_input.findData(
            state.face_detail_detector_provider
        )
        self.face_detail_detector_provider_input.setCurrentIndex(
            max(0, detector_provider_index)
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
        self._refresh_prompt_emphases()

        edit_state = state.image_edit
        self.edit_global_prompt.setPlainText(edit_state.global_prompt)
        self.edit_steps_input.setValue(edit_state.steps)
        self.edit_sampler_input.setCurrentIndex(
            max(0, self.edit_sampler_input.findData(edit_state.sampler))
        )
        self.edit_scheduler_input.setCurrentIndex(
            max(0, self.edit_scheduler_input.findData(edit_state.scheduler))
        )
        self.edit_seed_input.setValue(edit_state.seed)
        self.edit_denoise_input.setValue(edit_state.denoise)
        self.edit_composite_feather_input.setValue(
            edit_state.composite_feather_pixels
        )
        self.edit_regional_strength_input.setValue(
            edit_state.regional_prompt_strength
        )
        self.edit_outside_penalty_input.setValue(
            edit_state.regional_outside_penalty
        )
        self.edit_spatial_falloff_input.setValue(edit_state.regional_feather_pixels)
        self.edit_subject_competition_input.setChecked(
            edit_state.regional_subject_competition
        )
        self.edit_subject_fill_input.setChecked(edit_state.regional_subject_fill)
        self.edit_late_step_scale_input.setValue(edit_state.regional_late_step_scale)
        self.edit_lora_adaptation_input.setChecked(
            edit_state.regional_lora_delta_adaptation
        )
        self.edit_lora_adaptation_gain_input.setValue(
            edit_state.regional_lora_delta_adaptation_gain
        )
        self.edit_canvas.clear_regions()
        self.edit_canvas.clear_image()
        self.edit_region_list.clear()
        self.edit_regions = list(edit_state.regions)
        if edit_state.width and edit_state.height:
            self.edit_canvas.set_canvas_size(edit_state.width, edit_state.height)
        self._normalize_edit_region_priorities()
        self._edit_region_number = max(
            (
                int(match.group(1))
                for region in self.edit_regions
                if (match := re.fullmatch(r"Edit region (\d+)", region.name))
            ),
            default=0,
        )
        for region in self.edit_regions:
            item = QListWidgetItem(self._region_label(region))
            item.setData(Qt.ItemDataRole.UserRole, region.region_id)
            self.edit_region_list.addItem(item)
            box = region.box
            self.edit_canvas.add_region_box(
                region.region_id,
                QRectF(box.x0, box.y0, box.width, box.height),
                region.name,
            )
        self._sync_edit_canvas_stack()
        self._edit_source_path = None
        self._edit_result_path = None
        self.edit_source_input.clear()
        self.edit_result_input.clear()
        self.edit_result_preview.clear_image("Run an edit to compare the result")
        if edit_state.source_image:
            self._edit_source_path = edit_state.source_image.expanduser().resolve()
            self.edit_source_input.setText(
                str(self._edit_source_path)
                if self._edit_source_path.is_file()
                else f"{self._edit_source_path} (missing)"
            )
            if self._edit_source_path.is_file():
                self.edit_canvas.set_image(str(self._edit_source_path))

        self.lora_library = LoraLibrary()
        self._edit_lora_bindings = {}
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
            if saved_lora.trigger_phrase:
                self.lora_library.set_trigger_phrase(lora_id, saved_lora.trigger_phrase)
            self.lora_library.set_routing_mode(lora_id, saved_lora.routing_mode)
            self._edit_lora_bindings[lora_id] = LoraBinding(
                lora_id=lora_id,
                global_scope=(
                    saved_lora.edit_global_scope if saved_lora.edit_enabled else False
                ),
                region_ids=(
                    saved_lora.edit_region_ids if saved_lora.edit_enabled else ()
                ),
                strength=saved_lora.strength,
                routing_mode=saved_lora.edit_routing_mode,
                trigger_phrase=(
                    saved_lora.edit_trigger_phrase or saved_lora.path.stem
                ),
            )
            item = self._lora_list_item(lora_id)
            if item is not None:
                item.setText(self._lora_label(lora_id))
        self._selected_lora_changed(self.lora_list.currentItem(), None)

        self.canvas.clear_image()
        self._background_image_path = None
        self._face_source_path = None
        self._face_result_path = None
        self.face_source_input.clear()
        self.face_result_input.clear()
        self.face_source_preview.clear_image("No source PNG loaded")
        self._face_detections = []
        self.face_selection_list.clear()
        self.face_result_preview.clear_image(
            "Run face refinement to compare the result"
        )
        self.face_detect_button.setEnabled(False)
        self.face_refine_button.setEnabled(False)
        self.edit_run_button.setEnabled(
            bool(
                self.artifacts
                and self.artifacts.complete
                and self._edit_source_path is not None
                and self._edit_source_path.is_file()
            )
        )
        if state.background_image and state.background_image.is_file():
            if self.canvas.set_image(str(state.background_image)):
                self._background_image_path = state.background_image
                self._set_face_refinement_source(state.background_image)
        elif load_latest_face_source:
            self._use_latest_face_source(show_message=False)
        if self.region_list.count():
            self.region_list.setCurrentRow(0)
        else:
            self._selected_region_changed(-1)
        if self.edit_region_list.count():
            self.edit_region_list.setCurrentRow(0)
        else:
            self._selected_edit_region_changed(-1)
        self.discover_models()

    def _worker_payload(self) -> dict[str, object]:
        directories = self.settings.model_directories
        policy_key = self.memory_policy_input.currentData()
        return {
            "comfyui_root": str(self.settings.comfyui_root),
            "diffusion_models": str(directories.diffusion_models),
            "text_encoders": str(directories.text_encoders),
            "vae": str(directories.vae),
            "loras": str(directories.loras),
            "upscale_models": str(directories.upscale_models),
            "diffusion_model_file": (
                str(directories.diffusion_model_file)
                if directories.diffusion_model_file else None
            ),
            "text_encoder_file": (
                str(directories.text_encoder_file)
                if directories.text_encoder_file else None
            ),
            "vae_file": str(directories.vae_file) if directories.vae_file else None,
            "face_detector_path": (
                str(self.settings.face_detector_path)
                if self.settings.face_detector_path else None
            ),
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
                "Other GPU applications were not inspected or stopped.",
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
            "Unsaved GUI configuration is unaffected. Other GPU applications, "
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
        self.face_refine_button.setEnabled(
            bool(
                self.artifacts
                and self.artifacts.complete
                and self._face_source_path is not None
                and self._face_source_path.is_file()
                and self._selected_face_indices()
            )
        )
        self.face_detect_button.setEnabled(
            bool(self._face_source_path is not None and self._face_source_path.is_file())
        )
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
        pending_payload = (
            self._pending_face_refinement_payload
            if self._pending_face_refinement_payload is not None
            else (
                self._pending_image_edit_payload
                if self._pending_image_edit_payload is not None
                else self._pending_generation_payload
            )
        )
        if pending_payload is None or self._worker_bootstrap_stage:
            return
        if not self.worker_client.running:
            task_name = (
                "face-refinement"
                if self._pending_face_refinement_payload is not None
                else (
                    "image-edit"
                    if self._pending_image_edit_payload is not None
                    else "generation"
                )
            )
            self.events.addItem(f"Starting a fresh disposable {task_name} worker")
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
        is_refinement = self._pending_face_refinement_payload is not None
        is_image_edit = self._pending_image_edit_payload is not None
        command = CommandKind.GENERATE_BASELINE
        if is_refinement:
            command = CommandKind.REFINE_FACES
        elif is_image_edit:
            command = CommandKind.EDIT_IMAGE
        try:
            self.worker_client.send(command, pending_payload)
        except RuntimeError as error:
            self._pending_generation_payload = None
            self._pending_image_edit_payload = None
            self._pending_face_refinement_payload = None
            self._active_task = None
            self._set_generation_active(False)
            self.events.addItem(f"Could not start worker task: {error}")
            return
        if is_refinement:
            self._pending_face_refinement_payload = None
            self.events.addItem("Fresh worker ready; face refinement dispatched")
        elif is_image_edit:
            self._pending_image_edit_payload = None
            self.events.addItem("Fresh worker ready; image edit dispatched")
        else:
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
                "sampler": str(self.sampler_input.currentData()),
                "scheduler": str(self.scheduler_input.currentData()),
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
                "projector_identity_protection": (
                    self.projector_identity_protection_input.value()
                ),
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
                        "face_identity_prompt": region.face_identity_prompt,
                        "enabled": region.enabled,
                        "priority": region.priority,
                        "spatial_role": region.spatial_role,
                    }
                    for region in self.regions
                ],
                "loras": self._lora_payload(),
                "project_json": project_document(
                    replace(self._project_state(), seed=seed)
                ),
            }
        )
        self._pending_generation_payload = payload
        batch_count = (
            self.batch_count_input.value()
            if self.batch_mode_input.isChecked()
            else 1
        )
        self._batch_base_payload = dict(payload) if batch_count > 1 else None
        self._batch_runs_remaining = batch_count - 1
        self._batch_completed = 0
        self._batch_total = batch_count
        if batch_count > 1:
            self.events.addItem(
                f"Batch generation started: run 1/{batch_count}, seed {seed}"
            )
        self._generation_completed = False
        self._active_task = "generation"
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
            character_identity_triggers=character_identity_triggers(
                self._lora_payload()
            ),
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

    def _batch_mode_changed(self, checked: bool) -> None:
        self.batch_count_input.setEnabled(checked)
        fixed_index = self.seed_mode_input.findData("fixed")
        fixed_item = self.seed_mode_input.model().item(fixed_index)
        if fixed_item is not None:
            fixed_item.setEnabled(not checked)
        if checked and self.seed_mode_input.currentData() == "fixed":
            random_index = self.seed_mode_input.findData("random")
            self.seed_mode_input.setCurrentIndex(random_index)

    def _clear_batch_state(self) -> None:
        self._batch_base_payload = None
        self._batch_runs_remaining = 0
        self._batch_completed = 0
        self._batch_total = 0

    def _queue_next_batch_generation(self) -> None:
        if self._batch_base_payload is None or self._batch_runs_remaining <= 0:
            return
        payload = dict(self._batch_base_payload)
        seed_mode = str(payload["seed_mode"])
        seed = self.seed_input.value()
        if seed_mode == "random":
            seed = secrets.randbelow(2_147_483_648)
            self.seed_input.setValue(seed)
        elif seed_mode == "increment":
            self.seed_input.setValue((seed + 1) % 2_147_483_648)
        else:
            raise RuntimeError("batch generation requires random or increment seeds")
        payload["seed"] = seed
        payload["project_json"] = project_document(
            replace(self._project_state(), seed=seed)
        )
        self._batch_runs_remaining -= 1
        run_number = self._batch_completed + 1
        self._pending_generation_payload = payload
        self._generation_completed = False
        self._active_task = "generation"
        self._set_generation_active(True)
        self.events.addItem(
            f"Batch generation queued: run {run_number}/{self._batch_total}, seed {seed}"
        )
        self._advance_pending_generation()

    def _set_generation_active(self, active: bool) -> None:
        self._generation_active = active
        self.diffusion_model_input.setEnabled(not active)
        self.generate_button.setEnabled(
            bool(not active and self.artifacts is not None and self.artifacts.complete)
        )
        self.face_refine_button.setEnabled(
            bool(
                not active
                and self.artifacts is not None
                and self.artifacts.complete
                and self._face_source_path is not None
                and self._face_source_path.is_file()
                and self._selected_face_indices()
            )
        )
        self.face_detect_button.setEnabled(
            bool(
                not active
                and self._face_source_path is not None
                and self._face_source_path.is_file()
            )
        )
        self.edit_run_button.setEnabled(
            bool(
                not active
                and self.artifacts is not None
                and self.artifacts.complete
                and self._edit_source_path is not None
                and self._edit_source_path.is_file()
            )
        )
        self.stop_generation_button.setEnabled(active)

    def _stop_generation(self) -> None:
        if not self._generation_active:
            return
        pid = self.worker_client.cancel_generation()
        self._pending_generation_payload = None
        self._pending_image_edit_payload = None
        self._pending_face_refinement_payload = None
        self._clear_batch_state()
        self._worker_bootstrap_stage = None
        task = self._active_task or "worker task"
        self._active_task = None
        self._set_generation_active(False)
        self._accelerator_available = False
        self._model_loaded = False
        self.worker_status.setText("Stopped")
        self.accelerator_status.setText("Not probed")
        self.memory_status.setText("Worker task stopped; memory released")
        self.load_model_button.setEnabled(False)
        self.generate_button.setEnabled(bool(self.artifacts and self.artifacts.complete))
        self._set_memory_controls_enabled(True)
        detail = f" (worker PID {pid})" if pid is not None else ""
        self.events.addItem(f"{task.replace('_', ' ').title()} stopped{detail}; GPU/RAM released")
        self.statusBar().showMessage(
            "Worker task stopped — settings and loaded images were preserved", 10000
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
            backend = str(payload.get("accelerator_backend", "unknown"))
            runtime_version = (
                payload.get("hip_version")
                if backend == "rocm"
                else payload.get("cuda_version")
            )
            runtime_label = (
                f"{backend.upper()} {runtime_version}"
                if backend in {"rocm", "cuda"}
                else "GPU runtime unavailable"
            )
            self.events.addItem(
                "Worker runtime: "
                f"{payload.get('python_executable', 'unknown')}; "
                f"Torch {payload.get('torch_version', 'unavailable')}; "
                f"{runtime_label}"
            )
            if not self._accelerator_available:
                error = payload.get("initialization_error") or payload.get("error")
                if error:
                    self.events.addItem(f"Accelerator probe error: {error}")
            if message == "Worker runtime probe complete":
                self._worker_bootstrap_stage = None
                if (
                    (
                        self._pending_generation_payload is not None
                        or self._pending_image_edit_payload is not None
                        or self._pending_face_refinement_payload is not None
                    )
                    and not self._accelerator_available
                ):
                    self._pending_generation_payload = None
                    self._pending_image_edit_payload = None
                    self._pending_face_refinement_payload = None
                    self._active_task = None
                    self._set_generation_active(False)
                    self.events.addItem(
                        "Automatic worker task stopped: accelerator probe failed"
                    )
        if "manifests" in payload:
            compatible = payload.get("complete") and all(
                manifest.get("compatible") for manifest in payload["manifests"]
            )
            self._models_compatible = bool(compatible)
            self._worker_bootstrap_stage = None
            if (
                self._pending_generation_payload is not None
                or self._pending_image_edit_payload is not None
                or self._pending_face_refinement_payload is not None
            ) and not compatible:
                self._pending_generation_payload = None
                self._pending_image_edit_payload = None
                self._pending_face_refinement_payload = None
                self._active_task = None
                self._set_generation_active(False)
                self.events.addItem(
                    "Automatic worker task stopped: model validation failed"
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
            backend = str(payload.get("accelerator_backend", "GPU")).upper()
            report.setText(
                f"{backend} accelerator detected. Model loading is available."
                if self._accelerator_available
                else "The worker still cannot initialize a GPU accelerator."
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
            if self._active_task == "generation":
                self._batch_completed += 1
                if self._batch_total > 1:
                    self.events.addItem(
                        f"Batch generation completed run "
                        f"{self._batch_completed}/{self._batch_total}"
                    )
            self._set_generation_active(self._batch_runs_remaining > 0)
            image_path = payload.get("image_path")
            if image_path and self.canvas.set_image(image_path):
                self._background_image_path = Path(image_path)
                self._set_face_refinement_source(Path(image_path))
                self.statusBar().showMessage(f"Image saved to {image_path}")
            if payload.get("oom_recovered"):
                self.reserve_vram_input.setValue(
                    max(
                        float(payload.get("reserve_vram_gb", 0.5)),
                        self.reserve_vram_input.value(),
                    )
                )
                self.cpu_vae_input.setChecked(True)
                self.events.addItem(
                    "Generation recovered from GPU OOM; future fresh workers will start "
                    f"with CPU VAE and {self.reserve_vram_input.value():.1f} GiB reserved"
                )
            # The worker exits immediately after this event. Keep Generate disabled
            # until QProcess confirms that all GPU/system allocations are gone.
            self.generate_button.setEnabled(False)
            self.edit_run_button.setEnabled(False)
            self.face_refine_button.setEnabled(False)
        elif message == "Image editing complete":
            self._generation_completed = True
            self._set_generation_active(False)
            image_path = payload.get("image_path")
            if image_path:
                result_path = Path(image_path)
                if self.edit_result_preview.set_image(result_path):
                    self._edit_result_path = result_path
                    self.edit_result_input.setText(str(result_path))
                    self._set_face_refinement_source(result_path)
                    self.statusBar().showMessage(
                        f"Edited image saved to {result_path}"
                    )
            self.generate_button.setEnabled(False)
            self.edit_run_button.setEnabled(False)
            self.face_refine_button.setEnabled(False)
        elif message == "Face refinement complete":
            self._generation_completed = True
            self._set_generation_active(False)
            image_path = payload.get("image_path")
            if image_path:
                result_path = Path(image_path)
                if self.face_result_preview.set_image(result_path):
                    self._face_result_path = result_path
                    self.face_result_input.setText(str(result_path))
                    self.statusBar().showMessage(
                        f"Face-refined image saved to {result_path}"
                    )
            self.generate_button.setEnabled(False)
            self.face_refine_button.setEnabled(False)
        elif message in {
            "Generation worker releasing GPU and system RAM",
            "Image-edit worker releasing GPU and system RAM",
            "Face refinement worker releasing GPU and system RAM",
        }:
            self.memory_status.setText("Releasing disposable worker memory…")
        elif state == "error":
            self._pending_generation_payload = None
            self._pending_image_edit_payload = None
            self._pending_face_refinement_payload = None
            self._clear_batch_state()
            self._worker_bootstrap_stage = None
            self._active_task = None
            self._set_generation_active(False)
            if message != "LoRA diagnostics complete":
                self._model_loaded = False
            self.load_model_button.setEnabled(self._accelerator_available)
            self.generate_button.setEnabled(False)
            self.edit_run_button.setEnabled(False)
            self.face_refine_button.setEnabled(False)
            self._set_memory_controls_enabled(True)
            normalized_message = message.casefold()
            if any(
                marker in normalized_message
                for marker in ("out of memory", "gpu memory pressure", "gpu oom")
            ):
                self.events.addItem(
                    "GPU memory guidance: choose a lower-VRAM or Custom policy, use "
                    "Release K2 GPU memory, close RAM-heavy applications, or reduce "
                    "the canvas when using multiple LoRAs"
                )
                self.statusBar().showMessage(
                    "Generation exceeded available GPU memory — tune the memory policy before retrying",
                    15000,
                )

        if (
            self._pending_generation_payload is not None
            or self._pending_image_edit_payload is not None
            or self._pending_face_refinement_payload is not None
        ) and state != "error":
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
                self._pending_image_edit_payload = None
                self._pending_face_refinement_payload = None
                self._clear_batch_state()
            completed_task = self._active_task
            self._active_task = None
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
                self.memory_status.setText("GPU and worker RAM released")
                self.events.addItem(
                    "Disposable "
                    f"{(completed_task or 'task').replace('_', ' ')} worker exited; "
                    "GPU/system RAM released"
                )
                if completed_task == "generation" and self._batch_runs_remaining > 0:
                    self._queue_next_batch_generation()
                elif completed_task == "generation":
                    if self._batch_total > 1:
                        self.events.addItem(
                            f"Batch generation finished: {self._batch_completed} "
                            "runs completed"
                        )
                    self._clear_batch_state()

    def closeEvent(self, event) -> None:
        self.worker_client.stop()
        super().closeEvent(event)
