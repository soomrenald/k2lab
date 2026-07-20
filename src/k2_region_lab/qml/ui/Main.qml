import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "components"

ApplicationWindow {
    id: window

    width: 1480
    height: 920
    minimumWidth: 1040
    minimumHeight: 680
    visible: true
    title: "K2 Region Lab"
    color: "#090c13"

    property bool inspectorVisible: true
    property real compareValue: 0.5
    property string lastWorkspaceMode: ""
    property string lastEditSource: ""
    property var setupWindowInstance: null
    readonly property QtObject studio: controller
    property real comparisonPosition: resultMode.currentIndex === 0 ? 0
                                      : (resultMode.currentIndex === 1 ? 1 : compareValue)
    readonly property color accent: "#7c8cff"

    function openSetupWindow() {
        if (setupWindowInstance === null) {
            let component = Qt.createComponent("SetupWindow.qml")
            if (component.status !== Component.Ready) {
                toastMessage.text = "Could not open setup: " + component.errorString()
                toast.visible = true
                toastTimer.restart()
                return
            }
            setupWindowInstance = component.createObject(window, {
                "controller": window.studio.setupController
            })
        }
        setupWindowInstance.openWindow()
    }

    component StudioButton: Button {
        id: studioButton
        property bool primary: false
        implicitHeight: 34
        leftPadding: 14
        rightPadding: 14
        font.pixelSize: 12
        font.weight: Font.DemiBold
        contentItem: Text {
            text: studioButton.text
            color: studioButton.enabled
                   ? (studioButton.primary ? "#0b0e16" : "#e5e8f1") : "#656c7d"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font: studioButton.font
        }
        background: Rectangle {
            color: {
                if (!studioButton.enabled)
                    return studioButton.primary ? "#41475a" : "#161b26"
                if (studioButton.down)
                    return studioButton.primary ? "#aab3ff" : "#30394d"
                if (studioButton.hovered)
                    return studioButton.primary ? "#95a1ff" : "#252d3d"
                return studioButton.primary ? "#7c8cff" : "#191f2c"
            }
            border.color: studioButton.primary ? "transparent" : "#30384a"
            border.width: 1
            radius: 8
        }
    }

    component RailButton: Button {
        id: railButton
        required property string modeName
        property string glyph: ""
        checkable: true
        checked: controller.mode === modeName
        implicitWidth: 66
        implicitHeight: 58
        font.pixelSize: 11
        font.weight: Font.DemiBold
        contentItem: Column {
            anchors.centerIn: parent
            spacing: 4
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: railButton.glyph
                color: railButton.checked ? "#ffffff" : "#8d96a9"
                font.pixelSize: 18
                font.weight: Font.Bold
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: railButton.text
                color: railButton.checked ? "#ffffff" : "#8d96a9"
                font.pixelSize: 10
                font.weight: Font.DemiBold
            }
        }
        background: Rectangle {
            color: railButton.checked ? "#252d42" : (railButton.hovered ? "#171d29" : "transparent")
            border.color: railButton.checked ? window.accent : "transparent"
            border.width: 1
            radius: 10
        }
        onClicked: controller.setMode(modeName)
    }

    Connections {
        target: controller
        function onNotification(message) {
            toastMessage.text = message
            toast.visible = true
            toastTimer.restart()
        }
        function onStateChanged() {
            let source = controller.imageSource.toString()
            if (controller.mode === "edit"
                    && (window.lastWorkspaceMode !== "edit"
                        || source !== window.lastEditSource)) {
                resultMode.currentIndex = 0
            }
            window.lastWorkspaceMode = controller.mode
            window.lastEditSource = source
        }
    }

    Timer {
        id: toastTimer
        interval: 3200
        onTriggered: toast.visible = false
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 54
            color: "#0e121b"
            border.color: "#222938"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 14
                spacing: 8

                Rectangle {
                    width: 28
                    height: 28
                    radius: 8
                    gradient: Gradient {
                        GradientStop { position: 0; color: "#8f72ff" }
                        GradientStop { position: 1; color: "#5f8dff" }
                    }
                    Text {
                        anchors.centerIn: parent
                        text: "K2"
                        color: "white"
                        font.pixelSize: 10
                        font.weight: Font.Bold
                    }
                }
                Text {
                    text: "Region Lab"
                    color: "#f1f3f8"
                    font.pixelSize: 15
                    font.weight: Font.DemiBold
                }
                Rectangle { width: 1; height: 22; color: "#2b3242" }
                Text {
                    text: controller.modeTitle
                    color: "#8992a5"
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }

                StudioButton { text: "New"; onClicked: controller.newProject() }
                StudioButton { text: "Open"; onClicked: controller.openProject() }
                StudioButton { text: "Import PNG"; onClicked: controller.importProjectImage() }
                StudioButton { text: "Save"; onClicked: controller.saveProject() }
                StudioButton { text: "Save as"; onClicked: controller.saveProjectAs() }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.preferredWidth: 82
                Layout.fillHeight: true
                color: "#0c1018"
                border.color: "#202635"

                Column {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.top: parent.top
                    anchors.topMargin: 14
                    spacing: 8

                    RailButton { modeName: "generation"; text: "Generate"; glyph: "✦" }
                    RailButton { modeName: "edit"; text: "Edit"; glyph: "▣" }
                    RailButton { modeName: "face"; text: "Faces"; glyph: "◉" }
                }

                Column {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 16
                    spacing: 8

                    ToolButton {
                        width: 42
                        height: 42
                        text: "⚙"
                        ToolTip.visible: hovered
                        ToolTip.text: "Runtime and model setup"
                        onClicked: window.openSetupWindow()
                    }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.margins: 12
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    color: "#111621"
                    border.color: "#262d3c"
                    radius: 10

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 12
                        anchors.rightMargin: 10
                        spacing: 8

                        Text {
                            text: controller.canvasCaption
                            color: "#e9ecf4"
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                        }
                        Text {
                            text: controller.canvasWidth + " × " + controller.canvasHeight
                            color: "#737c90"
                            font.pixelSize: 11
                        }
                        Rectangle { width: 1; height: 24; color: "#2b3242" }

                        RowLayout {
                            visible: controller.mode === "edit"
                            spacing: 3
                            Rectangle {
                                width: refLayerText.implicitWidth + 22
                                height: 32
                                radius: 7
                                color: controller.editLayer === "reference" ? "#25433d" : "#171c27"
                                border.color: controller.editLayer === "reference" ? "#56d3b2" : "#2a3242"
                                Text {
                                    id: refLayerText
                                    anchors.centerIn: parent
                                    text: "Reference layer"
                                    color: controller.editLayer === "reference" ? "#c8fff0" : "#8b94a7"
                                    font.pixelSize: 11
                                    font.weight: Font.DemiBold
                                }
                                MouseArea { anchors.fill: parent; onClicked: controller.setEditLayer("reference") }
                            }
                            Rectangle {
                                width: targetLayerText.implicitWidth + 22
                                height: 32
                                radius: 7
                                color: controller.editLayer === "targets" ? "#49351f" : "#171c27"
                                border.color: controller.editLayer === "targets" ? "#ffb65c" : "#2a3242"
                                Text {
                                    id: targetLayerText
                                    anchors.centerIn: parent
                                    text: "Edit targets"
                                    color: controller.editLayer === "targets" ? "#ffdfb7" : "#8b94a7"
                                    font.pixelSize: 11
                                    font.weight: Font.DemiBold
                                }
                                MouseArea { anchors.fill: parent; onClicked: controller.setEditLayer("targets") }
                            }
                        }

                        Item { Layout.fillWidth: true }

                        StudioButton {
                            visible: controller.mode !== "generation"
                            text: "Load image"
                            onClicked: controller.loadCanvasImage()
                        }
                        StudioButton {
                            visible: controller.mode === "generation"
                            text: "Reference image"
                            onClicked: controller.loadCanvasImage()
                        }
                        StudioButton {
                            visible: controller.mode === "generation"
                            text: "Clear image"
                            onClicked: controller.clearCanvasImage()
                        }
                        StudioButton {
                            visible: controller.mode !== "face"
                            text: controller.drawMode ? "Drawing…" : "Draw region"
                            primary: controller.drawMode
                            onClicked: controller.setDrawMode(!controller.drawMode)
                        }
                        StudioButton {
                            visible: controller.mode !== "face"
                            text: "Delete selected"
                            enabled: controller.selectedRegionId.length > 0
                            onClicked: controller.deleteRegion(controller.selectedRegionId)
                        }
                        StudioButton {
                            visible: controller.mode === "face"
                            text: "Detect faces"
                            enabled: controller.imageSource.toString().length > 0 && !controller.busy
                            onClicked: controller.detectFaces()
                        }
                        ToolButton {
                            text: window.inspectorVisible ? "◧" : "◨"
                            ToolTip.visible: hovered
                            ToolTip.text: window.inspectorVisible ? "Hide inspector" : "Show inspector"
                            onClicked: window.inspectorVisible = !window.inspectorVisible
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 10

                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        RegionCanvas {
                            id: canvas
                            objectName: "mainRegionCanvas"
                            anchors.fill: parent
                            controller: window.studio
                            comparisonPosition: window.comparisonPosition
                        }

                        Rectangle {
                            anchors.left: parent.left
                            anchors.bottom: parent.bottom
                            anchors.margins: 14
                            visible: controller.resultSource.toString().length > 0
                            width: comparisonControls.implicitWidth + 20
                            height: 42
                            radius: 9
                            color: "#db0b0f18"
                            border.color: "#394156"

                            RowLayout {
                                id: comparisonControls
                                anchors.centerIn: parent
                                spacing: 7
                                ComboBox {
                                    id: resultMode
                                    objectName: "comparisonMode"
                                    model: ["Source", "Result", "Compare"]
                                    currentIndex: 1
                                    implicitWidth: 92
                                }
                                ValueSlider {
                                    visible: resultMode.currentIndex === 2
                                    from: 0
                                    to: 1
                                    value: window.compareValue
                                    stepSize: 0.01
                                    decimals: 2
                                    Layout.preferredWidth: 205
                                    onValueEdited: value => window.compareValue = value
                                }
                            }
                        }

                        Rectangle {
                            id: toast
                            visible: false
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 18
                            width: Math.min(parent.width - 30, toastMessage.implicitWidth + 32)
                            height: 42
                            radius: 9
                            color: "#e9293040"
                            border.color: "#6578ff"
                            Text {
                                id: toastMessage
                                anchors.centerIn: parent
                                color: "white"
                                font.pixelSize: 12
                            }
                        }
                    }

                    RegionStrip {
                        visible: controller.imageSource.toString().length > 0
                                 && controller.activeRegionCount > 0
                        Layout.preferredWidth: visible ? 184 : 0
                        Layout.minimumWidth: visible ? 168 : 0
                        Layout.fillHeight: true
                        controller: window.studio
                    }

                    InspectorPanel {
                        id: inspector
                        visible: window.inspectorVisible
                        Layout.preferredWidth: visible ? 370 : 0
                        Layout.minimumWidth: visible ? 340 : 0
                        Layout.fillHeight: true
                        controller: window.studio
                        onOpenSetupRequested: window.openSetupWindow()
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 62
            color: "#0e121b"
            border.color: "#222938"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: 12

                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: controller.busy ? "#ffb65c" : (controller.runEnabled ? "#56d3b2" : "#697287")
                }
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        Layout.fillWidth: true
                        text: controller.statusText
                        color: "#d6dae5"
                        elide: Text.ElideRight
                        font.pixelSize: 12
                    }
                    Text {
                        Layout.fillWidth: true
                        text: controller.memoryText
                        color: "#747e92"
                        elide: Text.ElideRight
                        font.pixelSize: 10
                    }
                }
                ProgressBar {
                    visible: controller.busy || controller.progress > 0
                    from: 0
                    to: 1
                    value: controller.progress
                    Layout.preferredWidth: 210
                }
                StudioButton {
                    visible: controller.busy
                    text: "Stop"
                    onClicked: controller.stopActive()
                }
                StudioButton {
                    visible: !controller.busy
                    text: controller.runLabel
                    primary: true
                    enabled: controller.runEnabled
                    implicitWidth: 170
                    onClicked: controller.runActive()
                }
            }
        }
    }
}
