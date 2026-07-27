from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import k2core
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from k2_region_lab.config import AppSettings
from k2_region_lab.worker.bootstrap import K2CORE_PACKAGE_ENV
from k2core.worker.protocol import CommandKind


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
        self._expected_stop_reason: str | None = None

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
        worker_environment = self.settings.worker_python.parent.parent
        environment.remove("PYTHONHOME")
        environment.insert("VIRTUAL_ENV", str(worker_environment))
        environment.insert("K2LAB_DATA_DIR", str(self.settings.data_directory))
        environment.insert(
            K2CORE_PACKAGE_ENV,
            str(Path(k2core.__file__).resolve().parent),
        )
        if not environment.contains("PYTORCH_ALLOC_CONF") and not environment.contains(
            "PYTORCH_CUDA_ALLOC_CONF"
        ):
            environment.insert("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        if not environment.contains("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"):
            # Recent PyTorch/ROCm releases otherwise disable the memory-efficient
            # attention backend on newer AMD architectures. The baseline does not
            # fit reliably in 16 GiB when it falls back to eager attention.
            environment.insert("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
        environment.insert(
            "PATH",
            os.pathsep.join(
                (
                    str(self.settings.worker_python.parent),
                    environment.value("PATH"),
                )
            ),
        )
        existing_pythonpath = environment.value("PYTHONPATH")
        path_entries = [str(project_root / "src"), str(self.settings.comfyui_root)]
        if existing_pythonpath:
            path_entries.append(existing_pythonpath)
        environment.insert("PYTHONPATH", os.pathsep.join(path_entries))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(project_root))
        self.process.setProgram(str(self.settings.worker_python))
        self.process.setArguments(["-m", "k2_region_lab.worker.bootstrap"])
        logging.getLogger(__name__).debug(
            "starting worker program=%s cwd=%s virtual_env=%s pythonpath=%s",
            self.settings.worker_python,
            project_root,
            worker_environment,
            os.pathsep.join(path_entries),
        )
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
        logging.getLogger(__name__).debug(
            "sent worker command id=%s kind=%s", command_id, kind.value
        )
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

    def kill_immediately(self) -> int | None:
        if not self.running:
            return None
        pid = int(self.process.processId())
        self.process.kill()
        self.process.waitForFinished(1500)
        return pid

    def cancel_generation(self, timeout_ms: int = 1500) -> int | None:
        """Terminate the isolated worker while leaving the desktop process alive."""
        if not self.running:
            return None
        pid = int(self.process.processId())
        self._expected_stop_reason = "generation cancelled"
        self.process.terminate()
        if not self.process.waitForFinished(timeout_ms):
            self.process.kill()
            self.process.waitForFinished(1500)
        return pid

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
            logging.getLogger(__name__).debug("worker event: %r", event)
            self.event_received.emit(event)

    def _read_stderr(self) -> None:
        output = bytes(self.process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if output:
            logging.getLogger(__name__).debug("worker stderr: %s", output.rstrip())
            self.stderr_received.emit(output.rstrip())

    def _finished(self, exit_code: int, exit_status) -> None:
        logging.getLogger(__name__).debug(
            "worker finished exit_code=%s exit_status=%s", exit_code, exit_status.name
        )
        if self._expected_stop_reason is not None:
            status = f"stopped ({self._expected_stop_reason})"
            self._expected_stop_reason = None
        else:
            status = f"stopped (exit {exit_code}, {exit_status.name})"
        self.process_status.emit(status)
