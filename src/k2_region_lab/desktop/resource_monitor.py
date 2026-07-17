from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ResourceSample:
    gpu_used_bytes: int = 0
    gpu_total_bytes: int = 0
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0
    gpu_busy_percent: float | None = None

    @property
    def gpu_memory_percent(self) -> float:
        return _percent(self.gpu_used_bytes, self.gpu_total_bytes)

    @property
    def ram_percent(self) -> float:
        return _percent(self.ram_used_bytes, self.ram_total_bytes)


def _percent(used: int, total: int) -> float:
    return max(0.0, min(100.0, 100.0 * used / total)) if total > 0 else 0.0


def discover_gpu_device(drm_root: Path = Path("/sys/class/drm")) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for total_path in drm_root.glob("card*/device/mem_info_vram_total"):
        try:
            total = int(total_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        candidates.append((total, total_path.parent))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def read_resource_sample(
    gpu_device: Path | None,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> ResourceSample:
    gpu_total = _read_integer(gpu_device / "mem_info_vram_total") if gpu_device else 0
    gpu_used = _read_integer(gpu_device / "mem_info_vram_used") if gpu_device else 0
    gpu_busy = _read_integer(gpu_device / "gpu_busy_percent") if gpu_device else None
    memory = _read_meminfo(meminfo_path)
    ram_total = memory.get("MemTotal", 0) * 1024
    ram_available = memory.get("MemAvailable", memory.get("MemFree", 0)) * 1024
    return ResourceSample(
        gpu_used_bytes=gpu_used or 0,
        gpu_total_bytes=gpu_total or 0,
        ram_used_bytes=max(0, ram_total - ram_available),
        ram_total_bytes=ram_total,
        gpu_busy_percent=float(gpu_busy) if gpu_busy is not None else None,
    )


def _read_integer(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _read_meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return values
    for line in lines:
        name, separator, value = line.partition(":")
        if not separator:
            continue
        try:
            values[name] = int(value.strip().split()[0])
        except (IndexError, ValueError):
            continue
    return values


class UsageGraph(QWidget):
    def __init__(self, parent: QWidget | None = None, history_size: int = 120) -> None:
        super().__init__(parent)
        self.gpu_history: deque[float] = deque(maxlen=history_size)
        self.ram_history: deque[float] = deque(maxlen=history_size)
        self.setMinimumHeight(105)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_sample(self, gpu_percent: float, ram_percent: float) -> None:
        self.gpu_history.append(gpu_percent)
        self.ram_history.append(ram_percent)
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11161a"))
        plot = self.rect().adjusted(7, 7, -7, -7)
        painter.setPen(QPen(QColor("#2e383f"), 1))
        for fraction in (0.25, 0.5, 0.75):
            y = plot.bottom() - plot.height() * fraction
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))
        self._draw_series(painter, plot, self.gpu_history, QColor("#42c7f5"))
        self._draw_series(painter, plot, self.ram_history, QColor("#f0a34a"))

    @staticmethod
    def _draw_series(painter, plot, history, color: QColor) -> None:
        if not history:
            return
        values = tuple(history)
        denominator = max(1, history.maxlen - 1)
        path = QPainterPath()
        for index, value in enumerate(values):
            x = plot.left() + plot.width() * index / denominator
            y = plot.bottom() - plot.height() * value / 100.0
            point = QPointF(x, y)
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setPen(QPen(color, 2))
        painter.drawPath(path)


class ResourceMonitorWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, interval_ms: int = 1000) -> None:
        super().__init__(parent)
        self.setMinimumWidth(275)
        self.setMaximumWidth(380)
        self._gpu_device = discover_gpu_device()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        title = QLabel("Live memory — last 2 minutes")
        title.setStyleSheet("font-weight: 600")
        self.gpu_label = QLabel("GPU VRAM: unavailable")
        self.gpu_label.setStyleSheet("color: #42c7f5")
        self.ram_label = QLabel("System RAM: unavailable")
        self.ram_label.setStyleSheet("color: #f0a34a")
        self.busy_label = QLabel("GPU activity: unavailable")
        self.graph = UsageGraph(self)
        layout.addWidget(title)
        layout.addWidget(self.gpu_label)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.busy_label)
        layout.addWidget(self.graph, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self) -> ResourceSample:
        sample = read_resource_sample(self._gpu_device)
        if sample.gpu_total_bytes:
            self.gpu_label.setText(
                "GPU VRAM: "
                f"{sample.gpu_used_bytes / GIB:.1f}/{sample.gpu_total_bytes / GIB:.1f} GiB "
                f"({sample.gpu_memory_percent:.0f}%)"
            )
        else:
            self.gpu_label.setText("GPU VRAM: unavailable")
        if sample.ram_total_bytes:
            self.ram_label.setText(
                "System RAM: "
                f"{sample.ram_used_bytes / GIB:.1f}/{sample.ram_total_bytes / GIB:.1f} GiB "
                f"({sample.ram_percent:.0f}%)"
            )
        else:
            self.ram_label.setText("System RAM: unavailable")
        self.busy_label.setText(
            "GPU activity: unavailable"
            if sample.gpu_busy_percent is None
            else f"GPU activity: {sample.gpu_busy_percent:.0f}%"
        )
        self.graph.add_sample(sample.gpu_memory_percent, sample.ram_percent)
        return sample
