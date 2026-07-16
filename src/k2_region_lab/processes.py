from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


WORKER_MODULE = "k2_region_lab.worker.entrypoint"


@dataclass(frozen=True, slots=True)
class WorkerProcess:
    pid: int
    parent_pid: int
    command: tuple[str, ...]


def is_k2_worker_command(command: tuple[str, ...]) -> bool:
    return WORKER_MODULE in command or any(
        item.endswith("/k2_region_lab/worker/entrypoint.py") for item in command
    )


def _process_uid(status_path: Path) -> int | None:
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def _parent_pid(status_path: Path) -> int | None:
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def find_owned_k2_workers(proc_root: Path = Path("/proc")) -> tuple[WorkerProcess, ...]:
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    current_pid = os.getpid()
    workers: list[WorkerProcess] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == current_pid:
            continue
        status_path = process_dir / "status"
        if current_uid is not None and _process_uid(status_path) != current_uid:
            continue
        try:
            encoded = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = tuple(
            item.decode("utf-8", errors="replace")
            for item in encoded.split(b"\0")
            if item
        )
        if not is_k2_worker_command(command):
            continue
        parent_pid = _parent_pid(status_path)
        if parent_pid is None:
            continue
        workers.append(WorkerProcess(pid, parent_pid, command))
    return tuple(sorted(workers, key=lambda process: process.pid))


def terminate_workers(
    workers: tuple[WorkerProcess, ...], *, timeout_seconds: float = 1.5
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    requested = {process.pid for process in workers if process.pid > 1}
    for pid in requested:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue

    def still_running(pid: int) -> bool:
        process_path = Path(f"/proc/{pid}")
        try:
            fields = (process_path / "stat").read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return False
        return len(fields) < 3 or fields[2] != "Z"

    deadline = time.monotonic() + timeout_seconds
    remaining = set(requested)
    while remaining and time.monotonic() < deadline:
        remaining = {pid for pid in remaining if still_running(pid)}
        if remaining:
            time.sleep(0.05)

    failed: set[int] = set()
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError:
            failed.add(pid)
    kill_deadline = time.monotonic() + 0.5
    after_kill = remaining - failed
    while after_kill and time.monotonic() < kill_deadline:
        after_kill = {pid for pid in after_kill if still_running(pid)}
        if after_kill:
            time.sleep(0.05)
    failed.update(after_kill)
    terminated = requested - failed
    return tuple(sorted(terminated)), tuple(sorted(failed))
