from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)


GeometryCallback = Callable[[str, QRectF], None]


class ResizableRegionItem(QGraphicsRectItem):
    """A region rectangle with movable body and four resize handles."""

    HANDLE_SIZE = 14.0
    MINIMUM_SIZE = 16.0

    def __init__(
        self,
        region_id: str,
        scene_rect: QRectF,
        geometry_changed: GeometryCallback,
    ) -> None:
        normalized = scene_rect.normalized()
        super().__init__(QRectF(0, 0, normalized.width(), normalized.height()))
        self.region_id = region_id
        self._geometry_changed = geometry_changed
        self._resize_corner: str | None = None
        self._resize_start: QRectF | None = None
        self._label = QGraphicsSimpleTextItem(self)
        self._label.setBrush(QBrush(QColor("#f2fbff")))
        self._label.setPos(6.0, 4.0)
        self._label.setZValue(1.0)
        self._label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._label.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        self.setPos(normalized.topLeft())
        self.setPen(QPen(QColor("#53d6ff"), 3))
        self.setBrush(QBrush(QColor(83, 214, 255, 35)))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    def set_label(self, label: str) -> None:
        self._label.setText(label)

    def scene_geometry(self) -> QRectF:
        return self.sceneTransform().mapRect(self.rect()).normalized()

    def _handles(self) -> dict[str, QRectF]:
        half = self.HANDLE_SIZE / 2.0
        rect = self.rect()
        return {
            "top_left": QRectF(rect.left() - half, rect.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "top_right": QRectF(rect.right() - half, rect.top() - half, self.HANDLE_SIZE, self.HANDLE_SIZE),
            "bottom_left": QRectF(
                rect.left() - half,
                rect.bottom() - half,
                self.HANDLE_SIZE,
                self.HANDLE_SIZE,
            ),
            "bottom_right": QRectF(
                rect.right() - half,
                rect.bottom() - half,
                self.HANDLE_SIZE,
                self.HANDLE_SIZE,
            ),
        }

    def _corner_at(self, point: QPointF) -> str | None:
        for name, handle in self._handles().items():
            if handle.contains(point):
                return name
        return None

    def boundingRect(self) -> QRectF:
        half = self.HANDLE_SIZE / 2.0
        return self.rect().adjusted(-half, -half, half, half)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        super().paint(painter, option, widget)
        if not self.isSelected():
            return
        painter.save()
        painter.setPen(QPen(QColor("#10242b"), 1))
        painter.setBrush(QBrush(QColor("#f2fbff")))
        for handle in self._handles().values():
            painter.drawRect(handle)
        painter.restore()

    def itemChange(self, change, value):
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene() is not None
        ):
            proposed = value
            bounds = self.scene().sceneRect()
            x = min(
                max(proposed.x(), bounds.left()),
                max(bounds.left(), bounds.right() - self.rect().width()),
            )
            y = min(
                max(proposed.y(), bounds.top()),
                max(bounds.top(), bounds.bottom() - self.rect().height()),
            )
            return QPointF(x, y)
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event) -> None:
        corner = self._corner_at(event.pos()) if self.isSelected() else None
        cursors = {
            "top_left": Qt.CursorShape.SizeFDiagCursor,
            "bottom_right": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(corner, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        corner = self._corner_at(event.pos()) if self.isSelected() else None
        if corner is not None and event.button() == Qt.MouseButton.LeftButton:
            self._resize_corner = corner
            self._resize_start = self.scene_geometry()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resize_corner is None or self._resize_start is None:
            super().mouseMoveEvent(event)
            return

        bounds = self.scene().sceneRect()
        start = self._resize_start
        point = event.scenePos()
        left, top, right, bottom = start.left(), start.top(), start.right(), start.bottom()
        if "left" in self._resize_corner:
            left = min(point.x(), right - self.MINIMUM_SIZE)
        else:
            right = max(point.x(), left + self.MINIMUM_SIZE)
        if "top" in self._resize_corner:
            top = min(point.y(), bottom - self.MINIMUM_SIZE)
        else:
            bottom = max(point.y(), top + self.MINIMUM_SIZE)

        left = max(bounds.left(), left)
        top = max(bounds.top(), top)
        right = min(bounds.right(), right)
        bottom = min(bounds.bottom(), bottom)
        self.set_scene_geometry(QRectF(QPointF(left, top), QPointF(right, bottom)), notify=False)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._resize_corner is not None:
            self._resize_corner = None
            self._resize_start = None
            self._notify_geometry()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self._notify_geometry()

    def set_scene_geometry(self, scene_rect: QRectF, *, notify: bool = True) -> None:
        normalized = scene_rect.normalized()
        if normalized.width() < self.MINIMUM_SIZE or normalized.height() < self.MINIMUM_SIZE:
            raise ValueError("region boxes must be at least 16×16 output pixels")
        bounds = self.scene().sceneRect() if self.scene() is not None else normalized
        width = min(normalized.width(), bounds.width())
        height = min(normalized.height(), bounds.height())
        left = min(max(normalized.left(), bounds.left()), bounds.right() - width)
        top = min(max(normalized.top(), bounds.top()), bounds.bottom() - height)
        self.prepareGeometryChange()
        self.setRect(QRectF(0, 0, width, height))
        self.setPos(QPointF(left, top))
        self.update()
        if notify:
            self._notify_geometry()

    def _notify_geometry(self) -> None:
        self._geometry_changed(self.region_id, self.scene_geometry())


class RegionCanvas(QGraphicsView):
    region_created = Signal(str, float, float, float, float)
    region_changed = Signal(str, float, float, float, float)
    region_deleted = Signal(str)
    region_selected = Signal(str)

    def __init__(self, width: int = 1024, height: int = 1024, parent=None) -> None:
        super().__init__(parent)
        self._canvas_width = width
        self._canvas_height = height
        self._origin: QPointF | None = None
        self._draft: QGraphicsRectItem | None = None
        self._drawing_enabled = False
        self._items: dict[str, ResizableRegionItem] = {}
        self._image_item: QGraphicsPixmapItem | None = None
        self.setScene(QGraphicsScene(self))
        self.scene().selectionChanged.connect(self._selection_changed)
        self.setBackgroundBrush(QBrush(QColor("#202225")))
        self.set_canvas_size(width, height)

    def set_canvas_size(self, width: int, height: int) -> None:
        self._canvas_width = width
        self._canvas_height = height
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        for item in self._items.values():
            geometry = item.scene_geometry()
            resized = QRectF(
                min(geometry.left(), max(0.0, width - min(geometry.width(), width))),
                min(geometry.top(), max(0.0, height - min(geometry.height(), height))),
                min(geometry.width(), float(width)),
                min(geometry.height(), float(height)),
            )
            item.set_scene_geometry(resized)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def begin_region(self) -> None:
        self._drawing_enabled = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, path: str) -> bool:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        scaled = pixmap.scaled(
            self._canvas_width,
            self._canvas_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._image_item is None:
            self._image_item = self.scene().addPixmap(scaled)
            self._image_item.setZValue(-100.0)
        else:
            self._image_item.setPixmap(scaled)
        return True

    def add_region_box(
        self, region_id: str, rect: QRectF, label: str = ""
    ) -> ResizableRegionItem:
        if region_id in self._items:
            raise ValueError(f"region already exists on canvas: {region_id}")
        item = ResizableRegionItem(region_id, rect, self._item_geometry_changed)
        item.set_label(label)
        self.scene().addItem(item)
        self._items[region_id] = item
        self.scene().clearSelection()
        item.setSelected(True)
        return item

    def region_item(self, region_id: str) -> ResizableRegionItem:
        return self._items[region_id]

    def set_region_name(self, region_id: str, name: str) -> None:
        self._items[region_id].set_label(name)

    def select_region(self, region_id: str) -> None:
        item = self._items.get(region_id)
        if item is None:
            return
        self.scene().clearSelection()
        item.setSelected(True)
        self.ensureVisible(item)

    def delete_selected_regions(self) -> None:
        selected = [item for item in self.scene().selectedItems() if isinstance(item, ResizableRegionItem)]
        for item in selected:
            self.remove_region(item.region_id, notify=True)

    def remove_region(self, region_id: str, *, notify: bool = False) -> None:
        item = self._items.pop(region_id, None)
        if item is None:
            return
        self.scene().removeItem(item)
        if notify:
            self.region_deleted.emit(region_id)

    def clear_regions(self) -> None:
        for region_id in tuple(self._items):
            self.remove_region(region_id)
        self.scene().clearSelection()

    def clear_image(self) -> None:
        if self._image_item is not None:
            self.scene().removeItem(self._image_item)
            self._image_item = None

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_regions()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self._origin = point
                self._draft = self.scene().addRect(
                    QRectF(point, point),
                    QPen(QColor("#ffcc66"), 3, Qt.PenStyle.DashLine),
                    QBrush(QColor(255, 204, 102, 30)),
                )
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._origin is not None and self._draft is not None:
            point = self.mapToScene(event.position().toPoint())
            point.setX(min(max(point.x(), 0.0), float(self._canvas_width)))
            point.setY(min(max(point.y(), 0.0), float(self._canvas_height)))
            self._draft.setRect(QRectF(self._origin, point).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._origin is not None and self._draft is not None:
            rect = self._draft.rect().normalized()
            self.scene().removeItem(self._draft)
            self._draft = None
            self._origin = None
            self._drawing_enabled = False
            self.unsetCursor()
            if rect.width() >= 16.0 and rect.height() >= 16.0:
                region_id = f"region-{uuid4().hex}"
                self.region_created.emit(
                    region_id, rect.left(), rect.top(), rect.right(), rect.bottom()
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _item_geometry_changed(self, region_id: str, rect: QRectF) -> None:
        self.region_changed.emit(
            region_id, rect.left(), rect.top(), rect.right(), rect.bottom()
        )

    def _selection_changed(self) -> None:
        for item in self.scene().selectedItems():
            if isinstance(item, ResizableRegionItem):
                self.region_selected.emit(item.region_id)
                return
