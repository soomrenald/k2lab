import QtQuick
import QtQuick.Controls

Item {
    id: root

    required property QtObject controller
    property real comparisonPosition: 1.0
    property color accent: "#7c8cff"
    property color referenceAccent: "#56d3b2"
    property color targetAccent: "#ffb65c"
    readonly property color layerColor: controller.mode === "edit"
                                        ? (controller.editLayer === "reference"
                                           ? referenceAccent : targetAccent)
                                        : accent
    readonly property real contentRatio: Math.max(1, controller.canvasWidth)
                                         / Math.max(1, controller.canvasHeight)
    readonly property real stageRatio: stage.width / Math.max(1, stage.height)
    readonly property real paintedWidth: stageRatio > contentRatio
                                         ? stage.height * contentRatio : stage.width
    readonly property real paintedHeight: stageRatio > contentRatio
                                          ? stage.height : stage.width / contentRatio
    readonly property real paintedX: (stage.width - paintedWidth) / 2
    readonly property real paintedY: (stage.height - paintedHeight) / 2

    function imageX(pixel) {
        return paintedX + pixel * paintedWidth / Math.max(1, controller.canvasWidth)
    }

    function imageY(pixel) {
        return paintedY + pixel * paintedHeight / Math.max(1, controller.canvasHeight)
    }

    function pixelX(display) {
        return Math.max(0, Math.min(controller.canvasWidth,
                                    (display - paintedX) * controller.canvasWidth
                                    / Math.max(1, paintedWidth)))
    }

    function pixelY(display) {
        return Math.max(0, Math.min(controller.canvasHeight,
                                    (display - paintedY) * controller.canvasHeight
                                    / Math.max(1, paintedHeight)))
    }

    function regionItemAt(index) {
        return regionRepeater.itemAt(index)
    }

    Rectangle {
        id: stage
        objectName: "canvasStage"
        anchors.fill: parent
        radius: 12
        color: "#080b12"
        border.color: "#252b3a"
        border.width: 1
        clip: true

        Rectangle {
            x: root.paintedX
            y: root.paintedY
            width: root.paintedWidth
            height: root.paintedHeight
            color: "#101521"

            Repeater {
                model: 20
                delegate: Rectangle {
                    required property int index
                    width: 1
                    height: parent.height
                    x: index * parent.width / 20
                    color: index % 5 === 0 ? "#1b2231" : "#151b28"
                }
            }
        }

        Image {
            id: sourceImage
            anchors.fill: parent
            source: controller.imageSource
            fillMode: Image.PreserveAspectFit
            asynchronous: true
            cache: false
            smooth: true
        }

        Item {
            id: comparisonResultClip
            objectName: "comparisonResultClip"
            x: 0
            y: 0
            height: stage.height
            width: stage.width * root.comparisonPosition
            clip: true
            visible: controller.resultSource.toString().length > 0
                     && root.comparisonPosition > 0

            Image {
                width: stage.width
                height: stage.height
                source: controller.resultSource
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
                smooth: true
            }
        }

        Rectangle {
            visible: controller.resultSource.toString().length > 0
                     && root.comparisonPosition > 0
                     && root.comparisonPosition < 1
            x: stage.width * root.comparisonPosition - 1
            width: 2
            height: stage.height
            color: "white"
            opacity: 0.8
        }

        MouseArea {
            anchors.fill: parent
            enabled: !controller.drawMode
            onClicked: controller.selectRegion("")
        }

        Repeater {
            id: regionRepeater
            model: controller.activeRegionModel

            delegate: Rectangle {
                id: regionBox
                required property string regionId
                required property string name
                required property real x0
                required property real y0
                required property real x1
                required property real y1
                required property bool regionEnabled
                required property int index

                objectName: "regionBox-" + regionId
                property bool resizePreviewActive: false
                property real resizePreviewX: 0
                property real resizePreviewY: 0
                property real resizePreviewWidth: 0
                property real resizePreviewHeight: 0
                property real resizeStartLeft: 0
                property real resizeStartTop: 0
                property real resizeStartRight: 0
                property real resizeStartBottom: 0
                readonly property real minimumResizeWidth: 16 * root.paintedWidth
                                                           / Math.max(1, controller.canvasWidth)
                readonly property real minimumResizeHeight: 16 * root.paintedHeight
                                                            / Math.max(1, controller.canvasHeight)

                function beginResize() {
                    resizeStartLeft = regionBox.x
                    resizeStartTop = regionBox.y
                    resizeStartRight = regionBox.x + regionBox.width
                    resizeStartBottom = regionBox.y + regionBox.height
                    resizePreviewX = resizeStartLeft
                    resizePreviewY = resizeStartTop
                    resizePreviewWidth = resizeStartRight - resizeStartLeft
                    resizePreviewHeight = resizeStartBottom - resizeStartTop
                    resizePreviewActive = true
                }

                function updateResize(horizontalEdge, verticalEdge, deltaX, deltaY) {
                    if (!resizePreviewActive)
                        return
                    let left = resizeStartLeft
                    let top = resizeStartTop
                    let right = resizeStartRight
                    let bottom = resizeStartBottom
                    if (horizontalEdge < 0)
                        left = Math.max(root.paintedX,
                                        Math.min(resizeStartRight - minimumResizeWidth,
                                                 resizeStartLeft + deltaX))
                    else if (horizontalEdge > 0)
                        right = Math.min(root.paintedX + root.paintedWidth,
                                         Math.max(resizeStartLeft + minimumResizeWidth,
                                                  resizeStartRight + deltaX))
                    if (verticalEdge < 0)
                        top = Math.max(root.paintedY,
                                      Math.min(resizeStartBottom - minimumResizeHeight,
                                               resizeStartTop + deltaY))
                    else if (verticalEdge > 0)
                        bottom = Math.min(root.paintedY + root.paintedHeight,
                                          Math.max(resizeStartTop + minimumResizeHeight,
                                                   resizeStartBottom + deltaY))
                    resizePreviewX = left
                    resizePreviewY = top
                    resizePreviewWidth = right - left
                    resizePreviewHeight = bottom - top
                }

                function finishResize() {
                    if (!resizePreviewActive)
                        return
                    let left = root.pixelX(resizePreviewX)
                    let top = root.pixelY(resizePreviewY)
                    let right = root.pixelX(resizePreviewX + resizePreviewWidth)
                    let bottom = root.pixelY(resizePreviewY + resizePreviewHeight)
                    resizePreviewActive = false
                    controller.updateRegionGeometry(regionId, left, top, right, bottom)
                }

                x: resizePreviewActive ? resizePreviewX : root.imageX(x0)
                y: resizePreviewActive ? resizePreviewY : root.imageY(y0)
                width: resizePreviewActive
                       ? resizePreviewWidth
                       : Math.max(2, root.imageX(x1) - root.imageX(x0))
                height: resizePreviewActive
                        ? resizePreviewHeight
                        : Math.max(2, root.imageY(y1) - root.imageY(y0))
                color: Qt.alpha(root.layerColor, regionEnabled ? 0.15 : 0.05)
                border.color: controller.selectedRegionId === regionId
                              ? "white" : root.layerColor
                border.width: controller.selectedRegionId === regionId ? 2 : 1
                opacity: regionEnabled ? 1 : 0.55
                z: controller.selectedRegionId === regionId ? 1000 : index

                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.leftMargin: -1
                    anchors.topMargin: -26
                    height: 24
                    width: Math.min(label.implicitWidth + 18, Math.max(80, stage.width - parent.x))
                    radius: 6
                    color: controller.selectedRegionId === regionId ? "#ffffff" : root.layerColor

                    Text {
                        id: label
                        anchors.centerIn: parent
                        width: parent.width - 12
                        text: regionBox.name
                        color: "#10131c"
                        elide: Text.ElideRight
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                    }
                }

                DragHandler {
                    id: moveHandler
                    enabled: !controller.drawMode
                             && controller.selectedRegionId === regionId
                    target: regionBox
                    xAxis.minimum: root.paintedX
                    xAxis.maximum: root.paintedX + root.paintedWidth - regionBox.width
                    yAxis.minimum: root.paintedY
                    yAxis.maximum: root.paintedY + root.paintedHeight - regionBox.height
                    onActiveChanged: {
                        if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x),
                                root.pixelY(regionBox.y),
                                root.pixelX(regionBox.x + regionBox.width),
                                root.pixelY(regionBox.y + regionBox.height))
                        } else {
                            controller.selectRegion(regionId)
                        }
                    }
                }

                TapHandler {
                    enabled: !controller.drawMode
                             && controller.selectedRegionId !== regionId
                    onTapped: controller.selectRegion(regionId)
                }

                Rectangle {
                    id: bottomRightHandle
                    visible: controller.selectedRegionId === regionId
                    z: 30
                    width: 13
                    height: 13
                    radius: 4
                    color: "white"
                    border.color: root.layerColor
                    border.width: 2
                    x: parent.width - 6
                    y: parent.height - 6
                    HoverHandler { cursorShape: Qt.SizeFDiagCursor }

                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(1, 1, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: topLeftHandle
                    visible: controller.selectedRegionId === regionId
                    z: 30
                    width: 13
                    height: 13
                    radius: 4
                    color: "white"
                    border.color: root.layerColor
                    border.width: 2
                    x: -6
                    y: -6
                    HoverHandler { cursorShape: Qt.SizeFDiagCursor }

                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(-1, -1, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: topRightHandle
                    visible: controller.selectedRegionId === regionId
                    z: 30
                    width: 13
                    height: 13
                    radius: 4
                    color: "white"
                    border.color: root.layerColor
                    border.width: 2
                    x: parent.width - 6
                    y: -6
                    HoverHandler { cursorShape: Qt.SizeBDiagCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(1, -1, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: bottomLeftHandle
                    visible: controller.selectedRegionId === regionId
                    z: 30
                    width: 13
                    height: 13
                    radius: 4
                    color: "white"
                    border.color: root.layerColor
                    border.width: 2
                    x: -6
                    y: parent.height - 6
                    HoverHandler { cursorShape: Qt.SizeBDiagCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(-1, 1, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: leftEdge
                    visible: controller.selectedRegionId === regionId
                    z: 20
                    x: -5
                    y: 8
                    width: 10
                    height: Math.max(0, parent.height - 16)
                    color: "transparent"
                    HoverHandler { cursorShape: Qt.SizeHorCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(-1, 0, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: rightEdge
                    visible: controller.selectedRegionId === regionId
                    z: 20
                    x: parent.width - 5
                    y: 8
                    width: 10
                    height: Math.max(0, parent.height - 16)
                    color: "transparent"
                    HoverHandler { cursorShape: Qt.SizeHorCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(1, 0, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: topEdge
                    visible: controller.selectedRegionId === regionId
                    z: 20
                    x: 8
                    y: -5
                    width: Math.max(0, parent.width - 16)
                    height: 10
                    color: "transparent"
                    HoverHandler { cursorShape: Qt.SizeVerCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(0, -1, translation.x, translation.y)
                    }
                }

                Rectangle {
                    id: bottomEdge
                    visible: controller.selectedRegionId === regionId
                    z: 20
                    x: 8
                    y: parent.height - 5
                    width: Math.max(0, parent.width - 16)
                    height: 10
                    color: "transparent"
                    HoverHandler { cursorShape: Qt.SizeVerCursor }
                    DragHandler {
                        target: null
                        onActiveChanged: {
                            if (active)
                                regionBox.beginResize()
                            else
                                regionBox.finishResize()
                        }
                        onTranslationChanged: if (active)
                            regionBox.updateResize(0, 1, translation.x, translation.y)
                    }
                }
            }
        }

        Repeater {
            model: root.controller.mode === "face" ? root.controller.faces : []
            delegate: Rectangle {
                id: faceBox
                required property var modelData
                x: root.imageX(modelData.x0)
                y: root.imageY(modelData.y0)
                width: root.imageX(modelData.x1) - root.imageX(modelData.x0)
                height: root.imageY(modelData.y1) - root.imageY(modelData.y0)
                color: Qt.alpha(modelData.selected ? "#56d3b2" : "#ff9d67", 0.13)
                border.color: modelData.selected ? "#56d3b2" : "#ff9d67"
                border.width: 2

                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.leftMargin: -1
                    anchors.topMargin: -24
                    width: faceLabel.implicitWidth + 16
                    height: 22
                    radius: 6
                    color: faceBox.modelData.selected ? "#56d3b2" : "#ff9d67"
                    Text {
                        id: faceLabel
                        anchors.centerIn: parent
                        text: faceBox.modelData.label
                        color: "#0b0e16"
                        font.pixelSize: 11
                        font.weight: Font.Bold
                    }
                }
                TapHandler {
                    onTapped: root.controller.setFaceSelected(
                                  faceBox.modelData.index, !faceBox.modelData.selected)
                }
            }
        }

        MouseArea {
            id: drawingArea
            anchors.fill: parent
            enabled: controller.drawMode
            cursorShape: Qt.CrossCursor
            property real startX: 0
            property real startY: 0

            onPressed: mouse => {
                startX = Math.max(root.paintedX,
                                  Math.min(root.paintedX + root.paintedWidth, mouse.x))
                startY = Math.max(root.paintedY,
                                  Math.min(root.paintedY + root.paintedHeight, mouse.y))
                draft.x = startX
                draft.y = startY
                draft.width = 0
                draft.height = 0
                draft.visible = true
            }
            onPositionChanged: mouse => {
                if (!pressed)
                    return
                let endX = Math.max(root.paintedX,
                                    Math.min(root.paintedX + root.paintedWidth, mouse.x))
                let endY = Math.max(root.paintedY,
                                    Math.min(root.paintedY + root.paintedHeight, mouse.y))
                draft.x = Math.min(startX, endX)
                draft.y = Math.min(startY, endY)
                draft.width = Math.abs(endX - startX)
                draft.height = Math.abs(endY - startY)
            }
            onReleased: {
                draft.visible = false
                controller.createRegion(root.pixelX(draft.x), root.pixelY(draft.y),
                                        root.pixelX(draft.x + draft.width),
                                        root.pixelY(draft.y + draft.height))
            }
        }

        Rectangle {
            id: draft
            visible: false
            color: Qt.alpha(root.layerColor, 0.18)
            border.color: "white"
            border.width: 2
        }

        Column {
            anchors.centerIn: parent
            spacing: 8
            visible: controller.imageSource.toString().length === 0

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: controller.mode === "generation" ? "Blank generation canvas" : "No image loaded"
                color: "#d6d9e4"
                font.pixelSize: 18
                font.weight: Font.DemiBold
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: controller.mode === "generation"
                      ? "Draw regions or add a reference image"
                      : "Use Load image in the toolbar to begin"
                color: "#7f8799"
                font.pixelSize: 13
            }
        }
    }
}
