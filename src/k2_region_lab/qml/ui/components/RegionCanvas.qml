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

    Rectangle {
        id: stage
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
            anchors.fill: parent
            clip: true
            visible: controller.resultSource.toString().length > 0
                     && root.comparisonPosition > 0
            width: parent.width * root.comparisonPosition

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

                x: root.imageX(x0)
                y: root.imageY(y0)
                width: Math.max(2, root.imageX(x1) - root.imageX(x0))
                height: Math.max(2, root.imageY(y1) - root.imageY(y0))
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
                        target: bottomRightHandle
                        xAxis.minimum: 18
                        xAxis.maximum: root.paintedX + root.paintedWidth - regionBox.x - 6.5
                        yAxis.minimum: 18
                        yAxis.maximum: root.paintedY + root.paintedHeight - regionBox.y - 6.5
                        onActiveChanged: {
                            if (!active) {
                                controller.updateRegionGeometry(
                                    regionId,
                                    root.pixelX(regionBox.x),
                                    root.pixelY(regionBox.y),
                                    root.pixelX(regionBox.x + bottomRightHandle.x + 6.5),
                                    root.pixelY(regionBox.y + bottomRightHandle.y + 6.5))
                            }
                        }
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
                        target: topLeftHandle
                        xAxis.minimum: root.paintedX - regionBox.x - 6.5
                        xAxis.maximum: regionBox.width - 18
                        yAxis.minimum: root.paintedY - regionBox.y - 6.5
                        yAxis.maximum: regionBox.height - 18
                        onActiveChanged: {
                            if (!active) {
                                controller.updateRegionGeometry(
                                    regionId,
                                    root.pixelX(regionBox.x + topLeftHandle.x + 6.5),
                                    root.pixelY(regionBox.y + topLeftHandle.y + 6.5),
                                    root.pixelX(regionBox.x + regionBox.width),
                                    root.pixelY(regionBox.y + regionBox.height))
                            }
                        }
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
                        target: topRightHandle
                        xAxis.minimum: 18
                        xAxis.maximum: root.paintedX + root.paintedWidth - regionBox.x - 6.5
                        yAxis.minimum: root.paintedY - regionBox.y - 6.5
                        yAxis.maximum: regionBox.height - 18
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x),
                                root.pixelY(regionBox.y + topRightHandle.y + 6.5),
                                root.pixelX(regionBox.x + topRightHandle.x + 6.5),
                                root.pixelY(regionBox.y + regionBox.height))
                        }
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
                        target: bottomLeftHandle
                        xAxis.minimum: root.paintedX - regionBox.x - 6.5
                        xAxis.maximum: regionBox.width - 18
                        yAxis.minimum: 18
                        yAxis.maximum: root.paintedY + root.paintedHeight - regionBox.y - 6.5
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x + bottomLeftHandle.x + 6.5),
                                root.pixelY(regionBox.y),
                                root.pixelX(regionBox.x + regionBox.width),
                                root.pixelY(regionBox.y + bottomLeftHandle.y + 6.5))
                        }
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
                        target: leftEdge
                        xAxis.minimum: root.paintedX - regionBox.x - 5
                        xAxis.maximum: regionBox.width - 21
                        yAxis.enabled: false
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x + leftEdge.x + 5),
                                root.pixelY(regionBox.y),
                                root.pixelX(regionBox.x + regionBox.width),
                                root.pixelY(regionBox.y + regionBox.height))
                        }
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
                        target: rightEdge
                        xAxis.minimum: 11
                        xAxis.maximum: root.paintedX + root.paintedWidth - regionBox.x - 5
                        yAxis.enabled: false
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x),
                                root.pixelY(regionBox.y),
                                root.pixelX(regionBox.x + rightEdge.x + 5),
                                root.pixelY(regionBox.y + regionBox.height))
                        }
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
                        target: topEdge
                        xAxis.enabled: false
                        yAxis.minimum: root.paintedY - regionBox.y - 5
                        yAxis.maximum: regionBox.height - 21
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x),
                                root.pixelY(regionBox.y + topEdge.y + 5),
                                root.pixelX(regionBox.x + regionBox.width),
                                root.pixelY(regionBox.y + regionBox.height))
                        }
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
                        target: bottomEdge
                        xAxis.enabled: false
                        yAxis.minimum: 11
                        yAxis.maximum: root.paintedY + root.paintedHeight - regionBox.y - 5
                        onActiveChanged: if (!active) {
                            controller.updateRegionGeometry(
                                regionId,
                                root.pixelX(regionBox.x),
                                root.pixelY(regionBox.y),
                                root.pixelX(regionBox.x + regionBox.width),
                                root.pixelY(regionBox.y + bottomEdge.y + 5))
                        }
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
