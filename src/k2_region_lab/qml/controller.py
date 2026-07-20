from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    Property,
    QTimer,
    QUrl,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import QFileDialog

from k2_region_lab.lora import LoraBinding
from k2_region_lab.regions import PixelBox, RegionDefinition


class RegionListModel(QAbstractListModel):
    """Read-only QML view of one semantic region layer."""

    RegionIdRole = Qt.ItemDataRole.UserRole + 1
    NameRole = RegionIdRole + 1
    X0Role = NameRole + 1
    Y0Role = X0Role + 1
    X1Role = Y0Role + 1
    Y1Role = X1Role + 1
    PromptRole = Y1Role + 1
    FacePromptRole = PromptRole + 1
    SpatialRole = FacePromptRole + 1
    EnabledRole = SpatialRole + 1
    PriorityRole = EnabledRole + 1

    _ROLES = {
        RegionIdRole: b"regionId",
        NameRole: b"name",
        X0Role: b"x0",
        Y0Role: b"y0",
        X1Role: b"x1",
        Y1Role: b"y1",
        PromptRole: b"prompt",
        FacePromptRole: b"facePrompt",
        SpatialRole: b"spatialRole",
        EnabledRole: b"regionEnabled",
        PriorityRole: b"priority",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._regions: tuple[RegionDefinition, ...] = ()

    def roleNames(self) -> dict[int, bytes]:
        return self._ROLES

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._regions)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._regions):
            return None
        region = self._regions[index.row()]
        values = {
            self.RegionIdRole: region.region_id,
            self.NameRole: region.name,
            self.X0Role: region.box.x0,
            self.Y0Role: region.box.y0,
            self.X1Role: region.box.x1,
            self.Y1Role: region.box.y1,
            self.PromptRole: region.prompt,
            self.FacePromptRole: region.face_identity_prompt,
            self.SpatialRole: region.spatial_role,
            self.EnabledRole: region.enabled,
            self.PriorityRole: region.priority,
            Qt.ItemDataRole.DisplayRole: region.name,
        }
        return values.get(role)

    def set_regions(self, regions: tuple[RegionDefinition, ...] | list[RegionDefinition]) -> None:
        normalized = tuple(regions)
        if normalized == self._regions:
            return
        self.beginResetModel()
        self._regions = normalized
        self.endResetModel()

    @Slot(int, result="QVariantMap")
    def get(self, row: int) -> dict[str, Any]:
        if not 0 <= row < len(self._regions):
            return {}
        return self.as_mapping(self._regions[row])

    @staticmethod
    def as_mapping(region: RegionDefinition) -> dict[str, Any]:
        return {
            "regionId": region.region_id,
            "name": region.name,
            "x0": region.box.x0,
            "y0": region.box.y0,
            "x1": region.box.x1,
            "y1": region.box.y1,
            "prompt": region.prompt,
            "facePrompt": region.face_identity_prompt,
            "spatialRole": region.spatial_role,
            "enabled": region.enabled,
            "priority": region.priority,
        }


class LoraListModel(QAbstractListModel):
    LoraIdRole = Qt.ItemDataRole.UserRole + 1
    NameRole = LoraIdRole + 1
    PathRole = NameRole + 1
    StrengthRole = PathRole + 1
    ScopeRole = StrengthRole + 1
    RoutingRole = ScopeRole + 1
    TriggerRole = RoutingRole + 1
    ActiveRole = TriggerRole + 1

    _ROLES = {
        LoraIdRole: b"loraId",
        NameRole: b"name",
        PathRole: b"path",
        StrengthRole: b"strength",
        ScopeRole: b"scope",
        RoutingRole: b"routingMode",
        TriggerRole: b"triggerPhrase",
        ActiveRole: b"active",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: tuple[dict[str, Any], ...] = ()

    def roleNames(self) -> dict[int, bytes]:
        return self._ROLES

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._items):
            return None
        item = self._items[index.row()]
        values = {
            self.LoraIdRole: item["loraId"],
            self.NameRole: item["name"],
            self.PathRole: item["path"],
            self.StrengthRole: item["strength"],
            self.ScopeRole: item["scope"],
            self.RoutingRole: item["routingMode"],
            self.TriggerRole: item["triggerPhrase"],
            self.ActiveRole: item["active"],
            Qt.ItemDataRole.DisplayRole: item["name"],
        }
        return values.get(role)

    def set_items(self, items: list[dict[str, Any]]) -> None:
        normalized = tuple(items)
        if normalized == self._items:
            return
        self.beginResetModel()
        self._items = normalized
        self.endResetModel()


class QmlWorkspaceController(QObject):
    """Presentation adapter over the stable Widgets-era application controller.

    The visible UI is Qt Quick. Until every legacy controller method has been extracted
    into standalone QObjects, an unshown MainWindow remains the compatibility owner of
    project serialization, model discovery, validation, and worker orchestration.
    """

    stateChanged = Signal()
    activeRegionModelChanged = Signal()
    selectionChanged = Signal()
    notification = Signal(str)

    GENERATION = "generation"
    IMAGE_EDIT = "edit"
    FACE_REFINEMENT = "face"

    def __init__(self, backend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self._mode = self.GENERATION
        self._edit_layer = "targets"
        self._selected_region_id = ""
        self._draw_mode = False
        self._progress = 0.0
        self._last_worker_message = ""
        self._lora_last_strength: dict[tuple[str, str, str], float] = {}
        self._state_snapshot: tuple[Any, ...] | None = None
        self._state_revision = 0
        self._generation_regions = RegionListModel(self)
        self._edit_regions = RegionListModel(self)
        self._reference_regions = RegionListModel(self)
        self._loras = LoraListModel(self)
        self.backend.worker_client.event_received.connect(self._worker_event)
        self.backend.worker_client.process_status.connect(self._worker_status)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(350)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start()
        self.refresh()

    @Property(str, notify=stateChanged)
    def mode(self) -> str:
        return self._mode

    @Property(str, notify=stateChanged)
    def editLayer(self) -> str:
        return self._edit_layer

    @Property(QObject, notify=activeRegionModelChanged)
    def activeRegionModel(self) -> QObject:
        if self._mode == self.GENERATION:
            return self._generation_regions
        if self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            return self._reference_regions
        if self._mode == self.IMAGE_EDIT:
            return self._edit_regions
        return self._generation_regions

    @Property(QObject, constant=True)
    def loraModel(self) -> QObject:
        return self._loras

    @Property("QVariantMap", notify=selectionChanged)
    def selectedRegion(self) -> dict[str, Any]:
        region = self._find_active_region(self._selected_region_id)
        return RegionListModel.as_mapping(region) if region is not None else {}

    @Property(str, notify=selectionChanged)
    def selectedRegionId(self) -> str:
        return self._selected_region_id

    @Property(bool, notify=stateChanged)
    def drawMode(self) -> bool:
        return self._draw_mode

    @Property(int, notify=stateChanged)
    def stateRevision(self) -> int:
        return self._state_revision

    @Property(str, notify=stateChanged)
    def modeTitle(self) -> str:
        return {
            self.GENERATION: "Generation",
            self.IMAGE_EDIT: "Image editing",
            self.FACE_REFINEMENT: "Face refinement",
        }[self._mode]

    @Property(str, notify=stateChanged)
    def canvasCaption(self) -> str:
        if self._mode == self.IMAGE_EDIT:
            return "Original reference layout" if self._edit_layer == "reference" else "Edit targets"
        if self._mode == self.FACE_REFINEMENT:
            return "Face refinement source"
        return "Generation canvas"

    @Property(str, notify=stateChanged)
    def runLabel(self) -> str:
        return {
            self.GENERATION: "Generate",
            self.IMAGE_EDIT: "Run image edit",
            self.FACE_REFINEMENT: "Refine selected faces",
        }[self._mode]

    @Property(bool, notify=stateChanged)
    def runEnabled(self) -> bool:
        if self.backend._generation_active:
            return False
        complete = bool(self.backend.artifacts and self.backend.artifacts.complete)
        if self._mode == self.IMAGE_EDIT:
            return complete and self.backend._edit_source_path is not None
        if self._mode == self.FACE_REFINEMENT:
            return complete and bool(self.backend._selected_face_indices())
        return complete

    @Property(bool, notify=stateChanged)
    def busy(self) -> bool:
        return bool(self.backend._generation_active)

    @Property(float, notify=stateChanged)
    def progress(self) -> float:
        return self._progress

    @Property(str, notify=stateChanged)
    def statusText(self) -> str:
        return self._last_worker_message or self.backend.statusBar().currentMessage() or "Ready"

    @Property(str, notify=stateChanged)
    def memoryText(self) -> str:
        return self.backend.memory_status.text()

    @Property(str, notify=stateChanged)
    def globalPrompt(self) -> str:
        if self._mode == self.IMAGE_EDIT:
            if self._edit_layer == "reference":
                return self.backend.edit_reference_global_prompt.toPlainText()
            return self.backend.edit_global_prompt.toPlainText()
        return self.backend.global_prompt.toPlainText()

    @Property(str, notify=stateChanged)
    def promptLabel(self) -> str:
        if self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            return "Original global prompt (reference only)"
        if self._mode == self.IMAGE_EDIT:
            return "Edit instruction"
        if self._mode == self.FACE_REFINEMENT:
            return "Generation prompt (reference)"
        return "Global prompt"

    @Property(QUrl, notify=stateChanged)
    def imageSource(self) -> QUrl:
        path: Path | None
        if self._mode == self.IMAGE_EDIT:
            path = self.backend._edit_source_path
        elif self._mode == self.FACE_REFINEMENT:
            path = self.backend._face_source_path
        else:
            path = self.backend._background_image_path
        return QUrl.fromLocalFile(str(path)) if path else QUrl()

    @Property(QUrl, notify=stateChanged)
    def resultSource(self) -> QUrl:
        path = None
        if self._mode == self.IMAGE_EDIT:
            path = self.backend._edit_result_path
        elif self._mode == self.FACE_REFINEMENT:
            path = self.backend._face_result_path
        return QUrl.fromLocalFile(str(path)) if path else QUrl()

    @Property(int, notify=stateChanged)
    def canvasWidth(self) -> int:
        if self._mode == self.IMAGE_EDIT:
            return self.backend.edit_canvas.canvas_width
        if (
            self._mode == self.FACE_REFINEMENT
            and not self.backend.face_source_preview._original_pixmap.isNull()
        ):
            return self.backend.face_source_preview._original_pixmap.width()
        return self.backend.canvas.canvas_width

    @Property(int, notify=stateChanged)
    def canvasHeight(self) -> int:
        if self._mode == self.IMAGE_EDIT:
            return self.backend.edit_canvas.canvas_height
        if (
            self._mode == self.FACE_REFINEMENT
            and not self.backend.face_source_preview._original_pixmap.isNull()
        ):
            return self.backend.face_source_preview._original_pixmap.height()
        return self.backend.canvas.canvas_height

    @Property(bool, notify=stateChanged)
    def canDrawRegions(self) -> bool:
        return self._mode != self.FACE_REFINEMENT

    @Property(int, notify=stateChanged)
    def faceCount(self) -> int:
        return len(self.backend._face_detections)

    @Property("QVariantList", notify=stateChanged)
    def faces(self) -> list[dict[str, Any]]:
        selected = set(self.backend._selected_face_indices())
        return [
            {
                "index": int(record["index"]),
                "label": f"Face {int(record['index']) + 1}",
                "score": float(record.get("score", 0.0)),
                "regionName": str(record.get("region_name") or "Unassigned"),
                "selected": int(record["index"]) in selected,
                "x0": float(record["box"][0]),
                "y0": float(record["box"][1]),
                "x1": float(record["box"][2]),
                "y1": float(record["box"][3]),
            }
            for record in self.backend._face_detections
        ]

    @Property("QStringList", constant=True)
    def samplerOptions(self) -> list[str]:
        return [self.backend.sampler_input.itemText(index) for index in range(self.backend.sampler_input.count())]

    @Property("QStringList", constant=True)
    def schedulerOptions(self) -> list[str]:
        return [
            self.backend.scheduler_input.itemText(index)
            for index in range(self.backend.scheduler_input.count())
        ]

    @Slot(str)
    def setMode(self, mode: str) -> None:
        if mode not in {self.GENERATION, self.IMAGE_EDIT, self.FACE_REFINEMENT}:
            return
        if mode == self._mode:
            return
        self._mode = mode
        self._draw_mode = False
        self._selected_region_id = ""
        self.backend.workspace_tabs.setCurrentIndex(
            {self.GENERATION: 0, self.IMAGE_EDIT: 1, self.FACE_REFINEMENT: 2}[mode]
        )
        self.refresh()
        self.activeRegionModelChanged.emit()
        self.selectionChanged.emit()

    @Slot(str)
    def setEditLayer(self, layer: str) -> None:
        if layer not in {"reference", "targets"} or layer == self._edit_layer:
            return
        self._edit_layer = layer
        index = 0 if layer == "reference" else 1
        self.backend.edit_layer_prompt_tabs.setCurrentIndex(index)
        self.backend.edit_layer_canvas_tabs.setCurrentIndex(index)
        self._selected_region_id = ""
        self.refresh()
        self.activeRegionModelChanged.emit()
        self.selectionChanged.emit()

    @Slot(bool)
    def setDrawMode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled and self.canDrawRegions)
        self.stateChanged.emit()

    @Slot(str)
    def setGlobalPrompt(self, prompt: str) -> None:
        target = self.backend.global_prompt
        if self._mode == self.IMAGE_EDIT:
            target = (
                self.backend.edit_reference_global_prompt
                if self._edit_layer == "reference"
                else self.backend.edit_global_prompt
            )
        if target.toPlainText() != prompt:
            target.setPlainText(prompt)

    @Slot(str)
    def selectRegion(self, region_id: str) -> None:
        if self._find_active_region(region_id) is None:
            region_id = ""
        if region_id == self._selected_region_id:
            return
        self._selected_region_id = region_id
        self.selectionChanged.emit()

    @Slot(float, float, float, float, result=str)
    def createRegion(self, x0: float, y0: float, x1: float, y1: float) -> str:
        if not self.canDrawRegions:
            return ""
        try:
            box = PixelBox(x0, y0, x1, y1).clipped(
                self.canvasWidth, self.canvasHeight
            )
        except ValueError:
            return ""
        if box.width < 16 or box.height < 16:
            self.notification.emit("Regions must be at least 16 × 16 pixels")
            return ""
        region_id = uuid4().hex
        if self._mode == self.GENERATION:
            self.backend._region_created(
                region_id, box.x0, box.y0, box.x1, box.y1
            )
        elif self._edit_layer == "reference":
            self.backend._edit_reference_region_created(
                region_id, box.x0, box.y0, box.x1, box.y1
            )
        else:
            self.backend._edit_region_created(
                region_id, box.x0, box.y0, box.x1, box.y1
            )
        self._draw_mode = False
        self._selected_region_id = region_id
        self.refresh()
        self.selectionChanged.emit()
        return region_id

    @Slot(str, float, float, float, float)
    def updateRegionGeometry(
        self, region_id: str, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        try:
            box = PixelBox(x0, y0, x1, y1).clipped(
                self.canvasWidth, self.canvasHeight
            )
            if box.width < 16 or box.height < 16:
                return
            if self._mode == self.GENERATION:
                self.backend._region_changed(
                    region_id, box.x0, box.y0, box.x1, box.y1
                )
            elif self._edit_layer == "reference":
                self.backend._edit_reference_region_changed(
                    region_id, box.x0, box.y0, box.x1, box.y1
                )
            else:
                self.backend._edit_region_changed(
                    region_id, box.x0, box.y0, box.x1, box.y1
                )
        except (KeyError, ValueError):
            return
        self.refresh()

    @Slot(str)
    def deleteRegion(self, region_id: str) -> None:
        if not region_id:
            return
        if self._mode == self.GENERATION:
            self.backend._region_deleted(region_id)
        elif self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            self.backend._edit_reference_region_deleted(region_id)
        elif self._mode == self.IMAGE_EDIT:
            self.backend._edit_region_deleted(region_id)
        self._selected_region_id = ""
        self.refresh()
        self.selectionChanged.emit()

    @Slot(str, "QVariant")
    def updateSelectedRegion(self, field: str, value) -> None:
        collection = self._active_regions()
        try:
            index = next(
                index
                for index, region in enumerate(collection)
                if region.region_id == self._selected_region_id
            )
        except StopIteration:
            return
        region = collection[index]
        if field == "name":
            name = str(value).strip()
            if not name or any(
                candidate.region_id != region.region_id
                and candidate.name.casefold() == name.casefold()
                for candidate in collection
            ):
                self.notification.emit("Region names must be non-empty and unique")
                return
            updated = replace(region, name=name)
        elif field == "prompt":
            updated = replace(region, prompt=str(value))
        elif field == "facePrompt":
            updated = replace(region, face_identity_prompt=str(value))
        elif field == "spatialRole" and str(value) in {"auto", "subject", "background", "edit"}:
            updated = replace(region, spatial_role=str(value))
        elif field == "enabled":
            updated = replace(region, enabled=bool(value))
        else:
            return
        collection[index] = updated
        self._sync_legacy_region_display(updated)
        self.refresh()
        self.selectionChanged.emit()

    @Slot(str, result="QVariant")
    def setting(self, name: str):
        control = self._setting_control(name)
        if control is None:
            return None
        if hasattr(control, "isChecked"):
            return control.isChecked()
        if hasattr(control, "currentData"):
            return control.currentData()
        return control.value()

    @Slot(str, "QVariant")
    def setSetting(self, name: str, value) -> None:
        control = self._setting_control(name)
        if control is None:
            return
        if hasattr(control, "setChecked"):
            control.setChecked(bool(value))
        elif hasattr(control, "findData"):
            index = control.findData(value)
            if index >= 0:
                control.setCurrentIndex(index)
        else:
            control.setValue(value)
        if name in {"width", "height"}:
            self.backend._canvas_dimensions_changed()
        self.refresh()

    @Slot()
    def addLora(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Add Krea LoRA",
            str(self.backend.settings.model_directories.loras),
            "Safetensors (*.safetensors)",
        )
        if selected and self.backend._add_lora_path(Path(selected), show_error_dialog=True):
            self.refresh()

    @Slot(str, float)
    def setLoraStrength(self, lora_id: str, strength: float) -> None:
        try:
            binding = replace(
                self._active_lora_binding(lora_id), strength=float(strength)
            )
        except (KeyError, ValueError):
            return
        if self._mode == self.IMAGE_EDIT:
            self._set_active_lora_binding(lora_id, binding)
        else:
            self.backend.lora_library.set_strength(lora_id, float(strength))
        self.refresh()

    @Slot(str, bool)
    def setLoraActive(self, lora_id: str, active: bool) -> None:
        try:
            binding = self._active_lora_binding(lora_id)
        except KeyError:
            return
        key = (self._mode, self._edit_layer, lora_id)
        if active:
            strength = self._lora_last_strength.get(key, 1.0)
            if strength == 0:
                strength = 1.0
        else:
            if binding.strength != 0:
                self._lora_last_strength[key] = binding.strength
            strength = 0.0
        self.setLoraStrength(lora_id, strength)

    @Slot(str)
    def removeLora(self, lora_id: str) -> None:
        try:
            entry = self.backend.lora_library.get(lora_id)
        except KeyError:
            return
        self.backend.lora_library.remove(lora_id)
        self.backend._edit_lora_bindings.pop(lora_id, None)
        self.backend._edit_reference_lora_bindings.pop(lora_id, None)
        item = self.backend._lora_list_item(lora_id)
        if item is not None:
            self.backend.lora_list.takeItem(self.backend.lora_list.row(item))
        self.backend._refresh_lora_scope()
        self.backend.events.addItem(f"Removed LoRA {entry.display_name}")
        for key in tuple(self._lora_last_strength):
            if key[2] == lora_id:
                self._lora_last_strength.pop(key)
        self.refresh()

    @Slot(int, bool)
    def setFaceSelected(self, face_index: int, selected: bool) -> None:
        for row in range(self.backend.face_selection_list.count()):
            item = self.backend.face_selection_list.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole)) == face_index:
                item.setCheckState(
                    Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
                )
                break
        self.backend._face_selection_changed()
        self.refresh()

    @Slot(bool)
    def setAllFacesSelected(self, selected: bool) -> None:
        self.backend._set_all_face_selections(selected)
        self.refresh()

    @Slot(str)
    def assignLoraGlobal(self, lora_id: str) -> None:
        try:
            binding = replace(
                self._active_lora_binding(lora_id), global_scope=True, region_ids=()
            )
        except KeyError:
            return
        self._set_active_lora_binding(lora_id, binding)
        self.refresh()

    @Slot(str)
    def assignLoraToSelectedRegion(self, lora_id: str) -> None:
        if not self._selected_region_id:
            self.notification.emit("Select a region before assigning a regional LoRA")
            return
        try:
            current = self._active_lora_binding(lora_id)
            binding = replace(
                current,
                global_scope=False,
                region_ids=tuple(
                    dict.fromkeys((*current.region_ids, self._selected_region_id))
                ),
            )
        except KeyError:
            return
        self._set_active_lora_binding(lora_id, binding)
        self.refresh()

    @Slot()
    def loadCanvasImage(self) -> None:
        title = "Load canvas image"
        image_filter = "Images (*.png *.jpg *.jpeg *.webp)"
        if self._mode == self.FACE_REFINEMENT:
            title = "Load face-refinement source"
            image_filter = "PNG images (*.png)"
        selected, _ = QFileDialog.getOpenFileName(
            None,
            title,
            str(self.backend._output_directory),
            image_filter,
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if self._mode == self.IMAGE_EDIT:
            self.backend._set_edit_source(path, confirm_reset=True)
        elif self._mode == self.FACE_REFINEMENT:
            self.backend._set_face_refinement_source(path)
        elif self.backend.canvas.set_image(path):
            self.backend._background_image_path = path
        self.refresh()

    @Slot()
    def clearCanvasImage(self) -> None:
        if self._mode == self.GENERATION:
            self.backend._clear_generation_canvas()
        self.refresh()

    @Slot()
    def detectFaces(self) -> None:
        self.backend._detect_faces_for_refinement()
        self.refresh()

    @Slot()
    def runActive(self) -> None:
        if self._mode == self.IMAGE_EDIT:
            self.backend._run_image_edit()
        elif self._mode == self.FACE_REFINEMENT:
            self.backend._run_face_refinement()
        else:
            self.backend._generate_baseline()
        self._progress = 0.0
        self.refresh()

    @Slot()
    def stopActive(self) -> None:
        self.backend._stop_generation()
        self._progress = 0.0
        self.refresh()

    @Slot()
    def newProject(self) -> None:
        self.backend._new_project()
        self.refresh()

    @Slot()
    def openProject(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Open K2 Region Lab project",
            str(self.backend._project_directory),
            "K2 Region Lab project (*.k2lab.json *.json)",
        )
        if selected:
            self.backend._load_project_from(Path(selected), show_error_dialog=True)
            self.refresh()

    @Slot()
    def importProjectImage(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Import K2 Region Lab image",
            str(self.backend._output_directory),
            "K2 Region Lab image (*.png);;PNG image (*.png)",
        )
        if selected:
            self.backend._load_project_image_from(Path(selected), show_error_dialog=True)
            self.refresh()

    @Slot()
    def saveProject(self) -> None:
        if self.backend._current_project_path is not None:
            self.backend._save_project_to(
                self.backend._current_project_path, show_error_dialog=True
            )
            return
        self.saveProjectAs()

    @Slot()
    def saveProjectAs(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            None,
            "Save K2 Region Lab project",
            str(self.backend._project_directory / "untitled.k2lab.json"),
            "K2 Region Lab project (*.k2lab.json);;JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.casefold() != ".json":
            path = path.with_suffix(path.suffix + ".json")
        self.backend._save_project_to(path, show_error_dialog=True)
        self.refresh()

    @Slot()
    def showLegacyWindow(self) -> None:
        self.backend.show()
        self.backend.raise_()
        self.backend.activateWindow()

    @Slot()
    def refresh(self) -> None:
        self._generation_regions.set_regions(self.backend.regions)
        self._edit_regions.set_regions(self.backend.edit_regions)
        self._reference_regions.set_regions(self.backend.edit_reference_regions)
        if self._selected_region_id and self._find_active_region(self._selected_region_id) is None:
            self._selected_region_id = ""
            self.selectionChanged.emit()
        self._refresh_loras()
        snapshot = self._current_state_snapshot()
        if snapshot != self._state_snapshot:
            self._state_snapshot = snapshot
            self._state_revision += 1
            self.stateChanged.emit()

    def _current_state_snapshot(self) -> tuple[Any, ...]:
        setting_names = {
            self.GENERATION: (
                "width",
                "height",
                "steps",
                "seed",
                "sampler",
                "scheduler",
                "insideBoost",
                "outsidePenalty",
                "spatialFalloff",
                "lateStepScale",
            ),
            self.IMAGE_EDIT: (
                "steps",
                "seed",
                "sampler",
                "scheduler",
                "denoise",
                "latentFeather",
                "compositeFeather",
                "referenceRetention",
                "insideBoost",
                "outsidePenalty",
                "spatialFalloff",
                "lateStepScale",
                "preserveIdentity",
                "editEntireImage",
                "loraAdaptation",
                "loraResponse",
            ),
            self.FACE_REFINEMENT: (
                "steps",
                "seed",
                "denoise",
                "padding",
                "feather",
                "blend",
                "loraScale",
                "detectorThreshold",
            ),
        }[self._mode]
        settings = tuple((name, self.setting(name)) for name in setting_names)
        face_state = tuple(
            (
                int(record["index"]),
                tuple(record.get("box", ())),
                record.get("score"),
                record.get("region_id"),
                int(record["index"]) in self.backend._selected_face_indices(),
            )
            for record in self.backend._face_detections
        )
        return (
            self._mode,
            self._edit_layer,
            self._draw_mode,
            self._progress,
            self._last_worker_message,
            bool(self.backend._generation_active),
            bool(self.backend.artifacts and self.backend.artifacts.complete),
            str(self.imageSource.toString()),
            str(self.resultSource.toString()),
            self.canvasWidth,
            self.canvasHeight,
            self.globalPrompt,
            self.backend.statusBar().currentMessage(),
            self.backend.memory_status.text(),
            face_state,
            settings,
        )

    def _active_regions(self) -> list[RegionDefinition]:
        if self._mode == self.GENERATION or self._mode == self.FACE_REFINEMENT:
            return self.backend.regions
        if self._edit_layer == "reference":
            return self.backend.edit_reference_regions
        return self.backend.edit_regions

    def _find_active_region(self, region_id: str) -> RegionDefinition | None:
        return next(
            (region for region in self._active_regions() if region.region_id == region_id),
            None,
        )

    def _sync_legacy_region_display(self, region: RegionDefinition) -> None:
        if self._mode == self.GENERATION or self._mode == self.FACE_REFINEMENT:
            item = self.backend._region_list_item(region.region_id)
            canvas = self.backend.canvas
        elif self._edit_layer == "reference":
            item = self.backend._edit_reference_region_list_item(region.region_id)
            canvas = self.backend.edit_reference_canvas
        else:
            item = self.backend._edit_region_list_item(region.region_id)
            canvas = self.backend.edit_canvas
        if item is not None:
            item.setText(self.backend._region_label(region))
        canvas.set_region_name(region.region_id, region.name)
        self.backend._refresh_lora_scope()

    def _setting_control(self, name: str):
        generation = {
            "width": self.backend.width_input,
            "height": self.backend.height_input,
            "steps": self.backend.steps_input,
            "seed": self.backend.seed_input,
            "sampler": self.backend.sampler_input,
            "scheduler": self.backend.scheduler_input,
            "insideBoost": self.backend.regional_prompt_strength_input,
            "outsidePenalty": self.backend.regional_outside_penalty_input,
            "spatialFalloff": self.backend.regional_feather_input,
            "lateStepScale": self.backend.regional_late_step_scale_input,
        }
        editing = {
            "steps": self.backend.edit_steps_input,
            "seed": self.backend.edit_seed_input,
            "sampler": self.backend.edit_sampler_input,
            "scheduler": self.backend.edit_scheduler_input,
            "denoise": self.backend.edit_denoise_input,
            "latentFeather": self.backend.edit_latent_feather_input,
            "compositeFeather": self.backend.edit_composite_feather_input,
            "referenceRetention": self.backend.edit_reference_retention_input,
            "insideBoost": self.backend.edit_regional_strength_input,
            "outsidePenalty": self.backend.edit_outside_penalty_input,
            "spatialFalloff": self.backend.edit_spatial_falloff_input,
            "lateStepScale": self.backend.edit_late_step_scale_input,
            "preserveIdentity": self.backend.edit_preserve_identity_input,
            "editEntireImage": self.backend.edit_entire_image_input,
            "loraAdaptation": self.backend.edit_lora_adaptation_input,
            "loraResponse": self.backend.edit_lora_adaptation_gain_input,
        }
        face = {
            "steps": self.backend.face_detail_steps_input,
            "seed": self.backend.face_detail_seed_input,
            "denoise": self.backend.face_detail_denoise_input,
            "padding": self.backend.face_detail_padding_input,
            "feather": self.backend.face_detail_feather_input,
            "blend": self.backend.face_detail_blend_input,
            "loraScale": self.backend.face_detail_lora_scale_input,
            "detectorThreshold": self.backend.face_detail_detector_threshold_input,
        }
        if self._mode == self.IMAGE_EDIT:
            return editing.get(name)
        if self._mode == self.FACE_REFINEMENT:
            return face.get(name)
        return generation.get(name)

    def _active_lora_binding(self, lora_id: str) -> LoraBinding:
        if self._mode != self.IMAGE_EDIT:
            return self.backend.lora_library.binding_for(lora_id)
        source = (
            self.backend._edit_reference_lora_bindings
            if self._edit_layer == "reference"
            else self.backend._edit_lora_bindings
        )
        return source.get(lora_id, self.backend.lora_library.binding_for(lora_id))

    def _set_active_lora_binding(self, lora_id: str, binding: LoraBinding) -> None:
        if self._mode != self.IMAGE_EDIT:
            if binding.global_scope:
                self.backend.lora_library.assign_global(lora_id)
            else:
                self.backend.lora_library.assign_regions(lora_id, binding.region_ids)
            return
        target = (
            self.backend._edit_reference_lora_bindings
            if self._edit_layer == "reference"
            else self.backend._edit_lora_bindings
        )
        target[lora_id] = binding

    def _refresh_loras(self) -> None:
        items = []
        for entry in self.backend.lora_library.entries():
            binding = self._active_lora_binding(entry.lora_id)
            if binding.global_scope:
                scope = "Global"
            else:
                names = [
                    region.name
                    for region in self._active_regions()
                    if region.region_id in binding.region_ids
                ]
                scope = ", ".join(names) if names else "Unassigned"
            items.append(
                {
                    "loraId": entry.lora_id,
                    "name": entry.display_name,
                    "path": str(entry.path),
                    "strength": binding.strength,
                    "scope": scope,
                    "routingMode": binding.routing_mode,
                    "triggerPhrase": binding.trigger_phrase,
                    "active": binding.strength != 0.0,
                }
            )
        self._loras.set_items(items)

    def _worker_event(self, event: dict[str, Any]) -> None:
        self._last_worker_message = str(event.get("message", ""))
        payload = event.get("payload", {})
        total = int(payload.get("total_steps", 0) or 0)
        step = int(payload.get("step", 0) or 0)
        if total > 0:
            self._progress = min(1.0, max(0.0, step / total))
        if self._last_worker_message in {
            "Generation complete",
            "Baseline generation complete",
            "Image editing complete",
            "Face refinement complete",
        }:
            self._progress = 1.0
        self.refresh()

    def _worker_status(self, status: str) -> None:
        self._last_worker_message = status
        self.refresh()
