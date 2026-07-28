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

from k2_region_lab.config import ModelDirectories, discover_worker_python
from k2core.lora import (
    CHARACTER_IDENTITY_LORA_ROUTING,
    STANDARD_LORA_ROUTING,
    LoraBinding,
)
from k2core.memory import MEMORY_POLICIES, memory_policy
from k2_region_lab.output import validate_filename_prefix
from k2core.regional_prompting import GLOBAL_EMPHASIS_SCOPE, PromptEmphasis
from k2core.regions import PixelBox, RegionDefinition


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


class SetupController(QObject):
    """Staged runtime/model settings for the dedicated Qt Quick setup window."""

    changed = Signal()
    applied = Signal()
    notification = Signal(str)

    def __init__(self, backend, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.backend = backend
        self._values: dict[str, Any] = {}
        self._baseline: dict[str, Any] = {}
        self._revision = 0
        self._status_snapshot: tuple[str, ...] | None = None
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self.refreshStatus)
        self._status_timer.start()
        self.reset()

    @Property(bool, notify=changed)
    def dirty(self) -> bool:
        return self._values != self._baseline

    @Property(int, notify=changed)
    def revision(self) -> int:
        return self._revision

    @Property("QVariantList", constant=True)
    def memoryPolicyOptions(self) -> list[dict[str, str]]:
        return [{"label": policy.label, "value": policy.key} for policy in MEMORY_POLICIES]

    @Property("QVariantList", notify=changed)
    def checkpointOptions(self) -> list[dict[str, str]]:
        control = self.backend.diffusion_model_input
        return [
            {"label": control.itemText(index), "value": str(control.itemData(index))}
            for index in range(control.count())
            if control.itemData(index)
        ]

    @Property(str, notify=changed)
    def workerStatus(self) -> str:
        return self.backend.worker_status.text()

    @Property(str, notify=changed)
    def acceleratorStatus(self) -> str:
        return self.backend.accelerator_status.text()

    @Property(str, notify=changed)
    def modelStatus(self) -> str:
        if self.backend.artifacts is None:
            return "Models have not been discovered"
        return "Model set complete" if self.backend.artifacts.complete else "Model set incomplete"

    @Property(str, notify=changed)
    def transformerStatus(self) -> str:
        return self.backend.transformer_status.text()

    @Property(str, notify=changed)
    def textEncoderStatus(self) -> str:
        return self.backend.text_status.text()

    @Property(str, notify=changed)
    def vaeStatus(self) -> str:
        return self.backend.vae_status.text()

    @Property(str, notify=changed)
    def memoryStatus(self) -> str:
        return self.backend.memory_status.text()

    @Slot(str, result="QVariant")
    def value(self, name: str):
        return self._values.get(name)

    @Slot(str, "QVariant")
    def setValue(self, name: str, value) -> None:
        if name not in self._values:
            return
        if name in {"reserveVram", "minimumRam"}:
            normalized: Any = float(value)
        elif name in {"cpuVae", "oomRecovery"}:
            normalized = bool(value)
        else:
            normalized = str(value).strip()
        if self._values[name] == normalized:
            return
        self._values[name] = normalized
        if name == "memoryPolicy":
            try:
                policy = memory_policy(normalized)
            except ValueError:
                pass
            else:
                self._values.update(
                    {
                        "reserveVram": policy.reserve_vram_gb,
                        "minimumRam": policy.minimum_system_ram_gb,
                        "cpuVae": policy.cpu_vae,
                        "oomRecovery": policy.oom_recovery,
                    }
                )
        self._revision += 1
        self.changed.emit()

    @Slot()
    def reset(self) -> None:
        settings = self.backend.settings
        directories = settings.model_directories
        self._values = {
            "comfyuiRoot": str(settings.comfyui_root),
            "workerPython": str(settings.worker_python),
            "transformer": str(directories.diffusion_model_file or ""),
            "textEncoder": str(directories.text_encoder_file or ""),
            "vae": str(directories.vae_file or ""),
            "faceDetector": str(settings.face_detector_path or ""),
            "memoryPolicy": settings.memory_policy,
            "reserveVram": float(self.backend.reserve_vram_input.value()),
            "minimumRam": float(self.backend.minimum_ram_input.value()),
            "cpuVae": bool(self.backend.cpu_vae_input.isChecked()),
            "oomRecovery": bool(self.backend.oom_recovery_input.isChecked()),
            "outputDirectory": str(self.backend._output_directory),
            "filenamePrefix": self.backend.filename_prefix_input.text(),
        }
        self._baseline = dict(self._values)
        self._revision += 1
        self.changed.emit()

    @Slot(str)
    def browseDirectory(self, name: str) -> None:
        if name not in {"comfyuiRoot", "outputDirectory"}:
            return
        selected = QFileDialog.getExistingDirectory(
            None,
            "Select ComfyUI checkout" if name == "comfyuiRoot" else "Select output folder",
            str(self._values[name]),
        )
        if selected:
            self.setValue(name, str(Path(selected).expanduser().resolve()))

    @Slot(str)
    def browseFile(self, name: str) -> None:
        definitions = {
            "workerPython": ("Select GPU worker Python", "All files (*)"),
            "transformer": ("Select Krea transformer", "Safetensors (*.safetensors)"),
            "textEncoder": ("Select text encoder", "Safetensors (*.safetensors)"),
            "vae": ("Select VAE", "Safetensors (*.safetensors)"),
            "faceDetector": ("Select face detector", "ONNX model (*.onnx)"),
        }
        if name not in definitions:
            return
        title, file_filter = definitions[name]
        current = Path(str(self._values[name] or self._values["comfyuiRoot"]))
        start = current if current.is_dir() else current.parent
        selected, _ = QFileDialog.getOpenFileName(None, title, str(start), file_filter)
        if selected:
            path = Path(selected).expanduser()
            self.setValue(name, str(path.absolute() if name == "workerPython" else path.resolve()))

    @Slot(str)
    def useAutomatic(self, name: str) -> None:
        if name == "workerPython":
            self.setValue(
                name,
                str(discover_worker_python(Path(self._values["comfyuiRoot"]))),
            )
        elif name in {"transformer", "textEncoder", "vae", "faceDetector"}:
            self.setValue(name, "")

    @Slot(result=bool)
    def apply(self) -> bool:
        if self.backend._generation_active:
            self.notification.emit("Stop the current generation before applying setup changes")
            return False
        try:
            policy = memory_policy(str(self._values["memoryPolicy"]))
            prefix = validate_filename_prefix(str(self._values["filenamePrefix"]))
            comfyui_root = Path(self._values["comfyuiRoot"]).expanduser().resolve()
            worker_python = Path(self._values["workerPython"]).expanduser().absolute()
            output_directory = Path(self._values["outputDirectory"]).expanduser().resolve()
            reserve_vram = max(policy.reserve_vram_gb, float(self._values["reserveVram"]))
            minimum_ram = max(policy.minimum_system_ram_gb, float(self._values["minimumRam"]))
        except (OSError, TypeError, ValueError) as error:
            self.notification.emit(f"Could not apply settings: {error}")
            return False

        current = self.backend.settings
        current_directories = current.model_directories
        if comfyui_root != current.comfyui_root:
            current_directories = ModelDirectories(
                diffusion_models=comfyui_root / "models" / "diffusion_models",
                text_encoders=comfyui_root / "models" / "text_encoders",
                vae=comfyui_root / "models" / "vae",
                loras=comfyui_root / "models" / "loras",
                upscale_models=comfyui_root / "models" / "upscale_models",
            )

        def optional_path(name: str) -> Path | None:
            text = str(self._values[name]).strip()
            return Path(text).expanduser().resolve() if text else None

        directories = replace(
            current_directories,
            diffusion_model_file=optional_path("transformer"),
            text_encoder_file=optional_path("textEncoder"),
            vae_file=optional_path("vae"),
        )
        updated = replace(
            current,
            comfyui_root=comfyui_root,
            worker_python=worker_python,
            model_directories=directories,
            face_detector_path=optional_path("faceDetector"),
            memory_policy=policy.key,
            reserve_vram_gb=reserve_vram,
            minimum_system_ram_gb=minimum_ram,
            cpu_vae=bool(self._values["cpuVae"]),
            oom_recovery=bool(self._values["oomRecovery"]),
            output_directory=output_directory,
            filename_prefix=prefix,
        )
        if not self.backend._apply_runtime_settings(updated, rediscover=True):
            return False

        policy_index = self.backend.memory_policy_input.findData(policy.key)
        self.backend.memory_policy_input.blockSignals(True)
        self.backend.memory_policy_input.setCurrentIndex(policy_index)
        self.backend.memory_policy_input.blockSignals(False)
        self.backend.reserve_vram_input.setValue(reserve_vram)
        self.backend.minimum_ram_input.setValue(minimum_ram)
        self.backend.cpu_vae_input.setChecked(bool(self._values["cpuVae"]))
        self.backend.oom_recovery_input.setChecked(bool(self._values["oomRecovery"]))
        self.backend._output_directory = output_directory
        self.backend.output_directory_input.setText(str(output_directory))
        self.backend.filename_prefix_input.setText(prefix)
        self.backend.events.addItem("Applied runtime and model setup changes")
        self.reset()
        self.applied.emit()
        self.notification.emit("Settings applied")
        return True

    @Slot()
    def discoverModels(self) -> None:
        if not self._require_applied():
            return
        self.backend.discover_models()
        self._revision += 1
        self.changed.emit()
        self.refreshStatus()

    @Slot()
    def startWorker(self) -> None:
        if self._require_applied():
            self.backend._start_worker()

    @Slot()
    def validateModels(self) -> None:
        if self._require_applied():
            self.backend._validate_worker_models()

    @Slot()
    def loadModel(self) -> None:
        if self._require_applied():
            self.backend._load_worker_model()

    @Slot()
    def releaseGpuMemory(self) -> None:
        if self._require_applied():
            self.backend._release_k2_gpu_memory()

    @Slot()
    def diagnoseAccelerator(self) -> None:
        if self._require_applied():
            self.backend._diagnose_accelerator()

    @Slot()
    def refreshStatus(self) -> None:
        snapshot = (
            self.workerStatus,
            self.acceleratorStatus,
            self.modelStatus,
            self.memoryStatus,
            self.transformerStatus,
            self.textEncoderStatus,
            self.vaeStatus,
            tuple((item["label"], item["value"]) for item in self.checkpointOptions),
        )
        if snapshot != self._status_snapshot:
            self._status_snapshot = snapshot
            self._revision += 1
            self.changed.emit()

    def _require_applied(self) -> bool:
        if self.dirty:
            self.notification.emit("Apply the staged settings before running this action")
            return False
        return True


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
        self._setup_controller = SetupController(self.backend, self)
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

    @Property(QObject, constant=True)
    def setupController(self) -> QObject:
        return self._setup_controller

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
            return (
                "Original reference layout" if self._edit_layer == "reference" else "Edit targets"
            )
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
    def resourceGpuText(self) -> str:
        return self.backend.resource_monitor.gpu_label.text()

    @Property(str, notify=stateChanged)
    def resourceRamText(self) -> str:
        return self.backend.resource_monitor.ram_label.text()

    @Property(str, notify=stateChanged)
    def resourceActivityText(self) -> str:
        return self.backend.resource_monitor.busy_label.text()

    @Property("QStringList", notify=stateChanged)
    def eventMessages(self) -> list[str]:
        return [
            self.backend.events.item(index).text() for index in range(self.backend.events.count())
        ]

    @Property(int, constant=True)
    def eventLimit(self) -> int:
        return self.backend.events.maximum_items

    @Slot()
    def clearEvents(self) -> None:
        self.backend.events.clear()
        self.refresh()

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

    @Property(str, constant=True)
    def globalEmphasisScope(self) -> str:
        return GLOBAL_EMPHASIS_SCOPE

    @Property("QVariantList", notify=stateChanged)
    def promptEmphases(self) -> list[dict[str, Any]]:
        emphases = self._active_prompt_emphases()
        return [
            {
                "index": index,
                "scopeId": emphasis.scope_id,
                "scopeLabel": self._emphasis_scope_label(emphasis.scope_id),
                "phrase": emphasis.phrase,
                "strength": emphasis.strength,
                "occurrence": emphasis.occurrence,
                "matches": self._emphasis_matches(emphasis),
            }
            for index, emphasis in enumerate(emphases)
        ]

    @Property(bool, notify=stateChanged)
    def promptEmphasisAvailable(self) -> bool:
        return self._mode == self.GENERATION or (
            self._mode == self.IMAGE_EDIT and self._edit_layer == "reference"
        )

    @Property("QVariantList", notify=stateChanged)
    def projectorVector(self) -> list[float]:
        return [float(control.value()) for control in self.backend.projector_vector_inputs]

    @Property(str, notify=stateChanged)
    def upscaleModelPath(self) -> str:
        return str(self.backend._upscale_model_path or "")

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
        return self._mode == self.GENERATION or (
            self._mode == self.IMAGE_EDIT and self._edit_layer == "targets"
        )

    @Property(int, notify=stateChanged)
    def faceCount(self) -> int:
        return len(self.backend._face_detections)

    @Property(int, notify=stateChanged)
    def activeRegionCount(self) -> int:
        if self._mode == self.FACE_REFINEMENT:
            return len(self.backend._face_detections)
        return len(self._active_regions())

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
        return [
            self.backend.sampler_input.itemText(index)
            for index in range(self.backend.sampler_input.count())
        ]

    @Property("QStringList", constant=True)
    def schedulerOptions(self) -> list[str]:
        return [
            self.backend.scheduler_input.itemText(index)
            for index in range(self.backend.scheduler_input.count())
        ]

    @Property("QVariantList", notify=stateChanged)
    def spatialRoleOptions(self) -> list[dict[str, str]]:
        # These are the three choices exposed by every Widgets-era region editor.
        # "edit" is an internal compiled role, not a selectable region definition role.
        return [
            {"label": "Auto (based on box width)", "value": "auto"},
            {"label": "Subject target", "value": "subject"},
            {"label": "Background band", "value": "background"},
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
        if self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            return
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
            box = PixelBox(x0, y0, x1, y1).clipped(self.canvasWidth, self.canvasHeight)
        except ValueError:
            return ""
        if box.width < 16 or box.height < 16:
            self.notification.emit("Regions must be at least 16 × 16 pixels")
            return ""
        region_id = uuid4().hex
        if self._mode == self.GENERATION:
            self.backend._region_created(region_id, box.x0, box.y0, box.x1, box.y1)
        elif self._edit_layer == "reference":
            self.backend._edit_reference_region_created(region_id, box.x0, box.y0, box.x1, box.y1)
        else:
            self.backend._edit_region_created(region_id, box.x0, box.y0, box.x1, box.y1)
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
            box = PixelBox(x0, y0, x1, y1).clipped(self.canvasWidth, self.canvasHeight)
            if box.width < 16 or box.height < 16:
                return
            if self._mode == self.GENERATION:
                self.backend._region_changed(region_id, box.x0, box.y0, box.x1, box.y1)
            elif self._edit_layer == "reference":
                self.backend._edit_reference_region_changed(
                    region_id, box.x0, box.y0, box.x1, box.y1
                )
            else:
                self.backend._edit_region_changed(region_id, box.x0, box.y0, box.x1, box.y1)
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
        if self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            return
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
        elif field == "spatialRole" and str(value) in {"auto", "subject", "background"}:
            updated = replace(region, spatial_role=str(value))
        elif field == "enabled":
            updated = replace(region, enabled=bool(value))
        else:
            return
        collection[index] = updated
        self._sync_legacy_region_display(updated)
        self.refresh()
        self.selectionChanged.emit()

    @Slot(str, int)
    def moveRegion(self, region_id: str, offset: int) -> None:
        collection = self._active_regions()
        try:
            source = next(
                index for index, region in enumerate(collection) if region.region_id == region_id
            )
        except StopIteration:
            return
        target = source + int(offset)
        if target < 0 or target >= len(collection):
            return
        if self._mode == self.GENERATION:
            widget = self.backend.region_list
        elif self._edit_layer == "reference":
            widget = self.backend.edit_reference_region_list
        else:
            widget = self.backend.edit_region_list
        destination = target + 1 if target > source else target
        widget.model().moveRow(QModelIndex(), source, QModelIndex(), destination)
        self._selected_region_id = region_id
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

    @Slot(str, result="QVariantMap")
    def settingSpec(self, name: str) -> dict[str, Any]:
        """Return the visible contract of the corresponding legacy control.

        Qt Quick consumes this instead of copying numeric limits and ComboBox choices.
        That keeps the replacement workspace byte-for-byte aligned with the Widgets-era
        settings surface as controls evolve.
        """
        control = self._setting_control(name)
        if control is None:
            return {}
        spec: dict[str, Any] = {"enabled": bool(control.isEnabled())}
        if hasattr(control, "minimum"):
            spec.update(
                {
                    "minimum": control.minimum(),
                    "maximum": control.maximum(),
                    "step": control.singleStep(),
                    "decimals": control.decimals() if hasattr(control, "decimals") else 0,
                    "suffix": control.suffix().strip() if hasattr(control, "suffix") else "",
                }
            )
        if hasattr(control, "count") and hasattr(control, "itemData"):
            spec["options"] = [
                {
                    "label": control.itemText(index),
                    "value": control.itemData(index),
                    "enabled": bool(
                        control.model().item(index) is None
                        or control.model().item(index).isEnabled()
                    ),
                }
                for index in range(control.count())
            ]
        return spec

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

    @Slot(str, str, int, float)
    def addPromptEmphasis(
        self, scope_id: str, phrase: str, selection_start: int, strength: float
    ) -> None:
        if not self.promptEmphasisAvailable:
            return
        source = self.globalPrompt
        if scope_id != GLOBAL_EMPHASIS_SCOPE:
            region = self._find_active_region(scope_id)
            if region is None:
                self.notification.emit("Select a region before adding regional emphasis")
                return
            source = region.prompt
            phrase = phrase.rstrip(".!? ")
        phrase = str(phrase).replace("\u2029", "\n")
        selection_start = int(selection_start)
        if not phrase.strip():
            self.notification.emit("Highlight a complete prompt word or phrase first")
            return
        offsets: list[int] = []
        offset = source.find(phrase)
        while offset >= 0:
            offsets.append(offset)
            offset = source.find(phrase, offset + len(phrase))
        selected_offset = next(
            (
                offset
                for offset in offsets
                if len(source[:offset].encode("utf-16-le")) // 2 == selection_start
            ),
            None,
        )
        if selected_offset is None:
            self.notification.emit("Highlight a complete prompt word or phrase first")
            return
        occurrence = offsets.index(selected_offset)
        emphasis = PromptEmphasis(
            scope_id=scope_id,
            phrase=phrase,
            strength=float(strength),
            occurrence=occurrence,
        )
        emphases = self._active_prompt_emphases()
        emphases.append(emphasis)
        self._set_active_prompt_emphases(emphases)
        self.notification.emit(f"Added token emphasis for {phrase!r}")
        self.refresh()

    @Slot(int, float)
    def setPromptEmphasisStrength(self, index: int, strength: float) -> None:
        emphases = self._active_prompt_emphases()
        if not 0 <= index < len(emphases):
            return
        try:
            emphases[index] = replace(emphases[index], strength=float(strength))
        except ValueError as error:
            self.notification.emit(str(error))
            return
        self._set_active_prompt_emphases(emphases)
        self.refresh()

    @Slot(int)
    def removePromptEmphasis(self, index: int) -> None:
        emphases = self._active_prompt_emphases()
        if not 0 <= index < len(emphases):
            return
        removed = emphases.pop(index)
        self._set_active_prompt_emphases(emphases)
        self.notification.emit(f"Removed token emphasis for {removed.phrase!r}")
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
            binding = replace(self._active_lora_binding(lora_id), strength=float(strength))
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

    @Slot(str, str)
    def setLoraRoutingMode(self, lora_id: str, routing_mode: str) -> None:
        if routing_mode not in {STANDARD_LORA_ROUTING, CHARACTER_IDENTITY_LORA_ROUTING}:
            return
        try:
            current = self._active_lora_binding(lora_id)
            if routing_mode == CHARACTER_IDENTITY_LORA_ROUTING and (
                current.global_scope or not current.region_ids
            ):
                self.notification.emit(
                    "Assign this LoRA to one or more regions before using Character identity"
                )
                return
            binding = replace(current, routing_mode=routing_mode)
        except (KeyError, ValueError) as error:
            self.notification.emit(str(error))
            return
        if self._mode == self.IMAGE_EDIT:
            self._set_active_lora_binding(lora_id, binding)
        else:
            self.backend.lora_library.set_routing_mode(lora_id, routing_mode)
        self.refresh()

    @Slot(str, str)
    def setLoraTriggerPhrase(self, lora_id: str, trigger_phrase: str) -> None:
        phrase = str(trigger_phrase).strip()
        if not phrase:
            self.notification.emit("Character identity trigger cannot be empty")
            return
        try:
            binding = replace(self._active_lora_binding(lora_id), trigger_phrase=phrase)
        except (KeyError, ValueError) as error:
            self.notification.emit(str(error))
            return
        if self._mode == self.IMAGE_EDIT:
            self._set_active_lora_binding(lora_id, binding)
        else:
            self.backend.lora_library.set_trigger_phrase(lora_id, phrase)
        self.refresh()

    @Slot(str, result=bool)
    def loraUsesSelectedRegion(self, lora_id: str) -> bool:
        if not self._selected_region_id:
            return False
        try:
            return self._selected_region_id in self._active_lora_binding(lora_id).region_ids
        except KeyError:
            return False

    @Slot(str)
    def toggleLoraSelectedRegion(self, lora_id: str) -> None:
        if not self._selected_region_id:
            self.notification.emit("Select a region before changing regional LoRA scope")
            return
        try:
            current = self._active_lora_binding(lora_id)
            region_ids = list(current.region_ids)
            if self._selected_region_id in region_ids:
                region_ids.remove(self._selected_region_id)
            else:
                region_ids.append(self._selected_region_id)
            routing_mode = current.routing_mode
            if not region_ids:
                routing_mode = STANDARD_LORA_ROUTING
            binding = replace(
                current,
                global_scope=False,
                region_ids=tuple(region_ids),
                routing_mode=routing_mode,
            )
        except (KeyError, ValueError) as error:
            self.notification.emit(str(error))
            return
        self._set_active_lora_binding(lora_id, binding)
        self.refresh()

    @Slot(str)
    def diagnoseLora(self, lora_id: str) -> None:
        item = self.backend._lora_list_item(lora_id)
        if item is None:
            return
        self.backend.lora_list.setCurrentItem(item)
        self.backend._diagnose_selected_lora()

    @Slot(int, float)
    def setProjectorValue(self, index: int, value: float) -> None:
        if not 0 <= index < len(self.backend.projector_vector_inputs):
            return
        self.backend.projector_vector_inputs[index].setValue(float(value))
        self.refresh()

    @Slot()
    def previewUnifiedPrompt(self) -> None:
        self.backend._preview_unified_prompt()

    @Slot()
    def browseUpscaleModel(self) -> None:
        self.backend._browse_upscale_model()
        self.refresh()

    @Slot()
    def clearUpscaleModel(self) -> None:
        self.backend._clear_upscale_model()
        self.refresh()

    @Slot()
    def useLatestFaceSource(self) -> None:
        self.backend._use_latest_face_source()
        self.refresh()

    @Slot()
    def openFaceLasso(self) -> None:
        self.backend._open_face_lasso_dialog()

    @Slot()
    def undoFaceLasso(self) -> None:
        self.backend.face_source_preview.undo_lasso()
        self.refresh()

    @Slot()
    def clearFaceLassos(self) -> None:
        self.backend.face_source_preview.clear_lassos()
        self.refresh()

    @Slot(str)
    def removeLora(self, lora_id: str) -> None:
        try:
            entry = self.backend.lora_library.get(lora_id)
        except KeyError:
            return
        item = self.backend._lora_list_item(lora_id)
        if item is not None:
            self.backend.lora_list.takeItem(self.backend.lora_list.row(item))
        self.backend.lora_library.remove(lora_id)
        self.backend._edit_lora_bindings.pop(lora_id, None)
        self.backend._edit_reference_lora_bindings.pop(lora_id, None)
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
                item.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
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
                self._active_lora_binding(lora_id),
                global_scope=True,
                region_ids=(),
                routing_mode=STANDARD_LORA_ROUTING,
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
                region_ids=tuple(dict.fromkeys((*current.region_ids, self._selected_region_id))),
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
                "seedMode",
                "batchMode",
                "batchCount",
                "regionalPrompting",
                "insideBoost",
                "outsidePenalty",
                "spatialFalloff",
                "subjectCompetition",
                "subjectFill",
                "relaxation",
                "lateStepScale",
                "loraAdaptation",
                "loraResponse",
                "postUpscale",
                "upscaleScale",
                "upscaleMethod",
                "projectorEnabled",
                "projectorPreset",
                "projectorMultiplier",
                "projectorIdentityProtection",
            ),
            self.IMAGE_EDIT: (
                "steps",
                "seed",
                "sampler",
                "scheduler",
                "denoise",
                "latentFeather",
                "compositeFeather",
                "insideBoost",
                "outsidePenalty",
                "spatialFalloff",
                "lateStepScale",
                "subjectCompetition",
                "subjectFill",
                "relaxation",
                "editEntireImage",
                "loraAdaptation",
                "loraResponse",
            ),
            self.FACE_REFINEMENT: (
                "steps",
                "seed",
                "denoise",
                "padding",
                "cropSize",
                "feather",
                "blend",
                "loraScale",
                "detectorThreshold",
                "detectorProvider",
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
            tuple(
                (item.scope_id, item.phrase, item.strength, item.occurrence)
                for item in self._active_prompt_emphases()
            ),
            tuple(self.projectorVector),
            self.upscaleModelPath,
            self.backend.statusBar().currentMessage(),
            self.backend.memory_status.text(),
            self.resourceGpuText,
            self.resourceRamText,
            self.resourceActivityText,
            tuple(self.eventMessages),
            face_state,
            settings,
        )

    def _active_regions(self) -> list[RegionDefinition]:
        if self._mode == self.GENERATION or self._mode == self.FACE_REFINEMENT:
            return self.backend.regions
        if self._edit_layer == "reference":
            return self.backend.edit_reference_regions
        return self.backend.edit_regions

    def _active_prompt_emphases(self) -> list[PromptEmphasis]:
        if self._mode == self.GENERATION:
            return list(self.backend.prompt_emphases)
        if self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            return list(self.backend.edit_reference_prompt_emphases)
        return []

    def _set_active_prompt_emphases(self, emphases: list[PromptEmphasis]) -> None:
        if self._mode == self.GENERATION:
            self.backend.prompt_emphases = list(emphases)
            self.backend._refresh_prompt_emphases()
        elif self._mode == self.IMAGE_EDIT and self._edit_layer == "reference":
            self.backend.edit_reference_prompt_emphases = list(emphases)

    def _emphasis_scope_label(self, scope_id: str) -> str:
        if scope_id == GLOBAL_EMPHASIS_SCOPE:
            return "Global"
        region = self._find_active_region(scope_id)
        return region.name if region is not None else "Missing region"

    def _emphasis_matches(self, emphasis: PromptEmphasis) -> bool:
        if emphasis.scope_id == GLOBAL_EMPHASIS_SCOPE:
            source = self.globalPrompt
        else:
            region = self._find_active_region(emphasis.scope_id)
            if region is None:
                return False
            source = region.prompt.rstrip(".!? ")
        offset = -1
        start = 0
        for _occurrence in range(emphasis.occurrence + 1):
            offset = source.find(emphasis.phrase, start)
            if offset < 0:
                return False
            start = offset + len(emphasis.phrase)
        return True

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
        if name == "emphasisStrength":
            return self.backend.emphasis_strength_input
        generation = {
            "width": self.backend.width_input,
            "height": self.backend.height_input,
            "steps": self.backend.steps_input,
            "seed": self.backend.seed_input,
            "sampler": self.backend.sampler_input,
            "scheduler": self.backend.scheduler_input,
            "seedMode": self.backend.seed_mode_input,
            "batchMode": self.backend.batch_mode_input,
            "batchCount": self.backend.batch_count_input,
            "regionalPrompting": self.backend.regional_prompting_input,
            "insideBoost": self.backend.regional_prompt_strength_input,
            "outsidePenalty": self.backend.regional_outside_penalty_input,
            "spatialFalloff": self.backend.regional_feather_input,
            "subjectCompetition": self.backend.regional_subject_competition_input,
            "subjectFill": self.backend.regional_subject_fill_input,
            "relaxation": self.backend.regional_relaxation_input,
            "lateStepScale": self.backend.regional_late_step_scale_input,
            "loraAdaptation": self.backend.regional_lora_delta_adaptation_input,
            "loraResponse": self.backend.regional_lora_delta_adaptation_gain_input,
            "postUpscale": self.backend.post_upscale_input,
            "upscaleScale": self.backend.upscale_scale_input,
            "upscaleMethod": self.backend.upscale_method_input,
            "projectorEnabled": self.backend.projector_enabled_input,
            "projectorPreset": self.backend.projector_preset_input,
            "projectorMultiplier": self.backend.projector_multiplier_input,
            "projectorIdentityProtection": self.backend.projector_identity_protection_input,
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
            "subjectCompetition": self.backend.edit_subject_competition_input,
            "subjectFill": self.backend.edit_subject_fill_input,
            "relaxation": self.backend.edit_regional_relaxation_input,
            "preserveIdentity": self.backend.edit_preserve_identity_input,
            "editEntireImage": self.backend.edit_entire_image_input,
            "loraAdaptation": self.backend.edit_lora_adaptation_input,
            "loraResponse": self.backend.edit_lora_adaptation_gain_input,
        }
        face = {
            "steps": self.backend.face_detail_steps_input,
            "seed": self.backend.face_detail_seed_input,
            "denoise": self.backend.face_detail_denoise_input,
            "cropSize": self.backend.face_detail_crop_size_input,
            "padding": self.backend.face_detail_padding_input,
            "feather": self.backend.face_detail_feather_input,
            "blend": self.backend.face_detail_blend_input,
            "loraScale": self.backend.face_detail_lora_scale_input,
            "detectorThreshold": self.backend.face_detail_detector_threshold_input,
            "detectorProvider": self.backend.face_detail_detector_provider_input,
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
            self.backend.lora_library.set_strength(lora_id, binding.strength)
            if binding.trigger_phrase:
                self.backend.lora_library.set_trigger_phrase(lora_id, binding.trigger_phrase)
            self.backend.lora_library.set_routing_mode(lora_id, binding.routing_mode)
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
