from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CommandKind(StrEnum):
    PROBE = "probe"
    DIAGNOSE_ACCELERATOR = "diagnose_accelerator"
    DISCOVER_MODELS = "discover_models"
    VALIDATE_MODELS = "validate_models"
    VALIDATE_LORAS = "validate_loras"
    LOAD_MODEL = "load_model"
    INITIALIZE_RUN = "initialize_run"
    GENERATE_BASELINE = "generate_baseline"
    EDIT_IMAGE = "edit_image"
    REFINE_FACES = "refine_faces"
    NEXT_BLOCK = "next_block"
    NEXT_STEP = "next_step"
    CONTINUE = "continue"
    PAUSE = "pause"
    CANCEL = "cancel"
    SHUTDOWN = "shutdown"


class WorkerState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    PROBING = "probing"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    command_id: str
    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    command_id: str | None
    state: WorkerState
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
