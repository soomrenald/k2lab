from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from k2_region_lab.config import AppSettings
from k2_region_lab.worker.protocol import CommandKind


class ExternalWorkerClient(QObject):
    event_received = Signal(object)
    stderr_received = Signal(str)
    process_status = Signal(str)

    def __init__(self, settings: AppSettings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.started.connect(lambda: self.process_status.emit("started"))
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(
            lambda error: self.process_status.emit(f"process error: {error.name}")
        )
        self._stdout_buffer = ""

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self) -> bool:
        if self.running:
            return True
        if not self.settings.worker_python.is_file():
            self.process_status.emit(
                f"worker interpreter missing: {self.settings.worker_python}"
            )
            return False
        project_root = Path(__file__).resolve().parents[3]
        environment = QProcessEnvironment.systemEnvironment()
        existing_pythonpath = environment.value("PYTHONPATH")
        path_entries = [str(project_root / "src"), str(self.settings.comfyui_root)]
        if existing_pythonpath:
            path_entries.append(existing_pythonpath)
        environment.insert("PYTHONPATH", os.pathsep.join(path_entries))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(project_root))
        self.process.setProgram(str(self.settings.worker_python))
        self.process.setArguments(["-m", "k2_region_lab.worker.entrypoint"])
        self.process.start()
        self.process_status.emit("starting")
        return True

    def send(self, kind: CommandKind, payload: dict | None = None) -> str:
        if not self.running:
            raise RuntimeError("GPU worker is not running")
        command_id = uuid4().hex
        encoded = json.dumps(
            {
                "command_id": command_id,
                "kind": kind.value,
                "payload": payload or {},
            },
            separators=(",", ":"),
        )
        self.process.write((encoded + "\n").encode("utf-8"))
        return command_id

    def stop(self, timeout_ms: int = 3000) -> None:
        if not self.running:
            return
        try:
            self.send(CommandKind.SHUTDOWN)
            self.process.waitForFinished(timeout_ms)
        finally:
            if self.running:
                self.process.kill()
                self.process.waitForFinished(1000)

    def _read_stdout(self) -> None:
        self._stdout_buffer += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._stdout_buffer:
            encoded, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            if not encoded.strip():
                continue
            try:
                event = json.loads(encoded)
            except json.JSONDecodeError:
                self.stderr_received.emit(f"non-protocol worker output: {encoded}")
                continue
            self.event_received.emit(event)

    def _read_stderr(self) -> None:
        output = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if output:
            self.stderr_received.emit(output.rstrip())

    def _finished(self, exit_code: int, exit_status) -> None:
        self.process_status.emit(
            f"stopped (exit {exit_code}, {exit_status.name})"
        )
