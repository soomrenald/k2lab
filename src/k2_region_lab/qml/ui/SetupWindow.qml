import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "components"

Window {
    id: setupWindow
    objectName: "setupWindow"

    required property QtObject controller
    property bool bypassDirtyCheck: false
    property string message: ""
    width: 820
    height: 760
    minimumWidth: 700
    minimumHeight: 620
    visible: false
    title: "K2 Region Lab · Setup"
    color: "#090c13"
    flags: Qt.Window

    function openWindow() {
        controller.reset()
        bypassDirtyCheck = false
        show()
        raise()
        requestActivate()
    }

    function finishClose() {
        bypassDirtyCheck = true
        hide()
        Qt.callLater(() => bypassDirtyCheck = false)
    }

    function requestClose() {
        contentItem.forceActiveFocus()
        if (controller.dirty)
            unsavedDialog.open()
        else
            finishClose()
    }

    onClosing: close => {
        contentItem.forceActiveFocus()
        close.accepted = false
        if (!bypassDirtyCheck && controller.dirty) {
            unsavedDialog.open()
        } else {
            hide()
        }
    }

    Connections {
        target: setupWindow.controller
        function onNotification(text) {
            setupWindow.message = text
            messageTimer.restart()
        }
    }

    Timer {
        id: messageTimer
        interval: 4000
        onTriggered: setupWindow.message = ""
    }

    component SectionLabel: Text {
        color: "#939caf"
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.7
        font.capitalization: Font.AllUppercase
    }

    component SetupButton: Button {
        id: setupButton
        property bool primary: false
        implicitHeight: 34
        leftPadding: 13
        rightPadding: 13
        font.pixelSize: 12
        font.weight: Font.DemiBold
        contentItem: Text {
            text: setupButton.text
            color: setupButton.enabled
                   ? (setupButton.primary ? "#0a0d15" : "#e4e7ef") : "#626a7a"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font: setupButton.font
        }
        background: Rectangle {
            color: setupButton.primary
                   ? (setupButton.hovered ? "#95a1ff" : "#7c8cff")
                   : (setupButton.hovered ? "#283044" : "#1a202d")
            border.color: setupButton.primary ? "transparent" : "#323a4c"
            radius: 8
        }
    }

    component SetupField: TextField {
        id: setupField
        required property string settingName
        color: "#e8ebf4"
        placeholderTextColor: "#687185"
        selectionColor: "#6578ff"
        selectedTextColor: "white"
        font.pixelSize: 12
        leftPadding: 10
        rightPadding: 10
        text: {
            let revision = setupWindow.controller.revision
            return String(setupWindow.controller.value(settingName) ?? "")
        }
        onTextEdited: setupWindow.controller.setValue(settingName, text)
        onEditingFinished: setupWindow.controller.setValue(settingName, text)
        onActiveFocusChanged: {
            if (!activeFocus)
                setupWindow.controller.setValue(settingName, text)
        }
        background: Rectangle {
            color: "#0b0f18"
            border.color: setupField.activeFocus ? "#6578ff" : "#2b3344"
            radius: 7
        }
    }

    component PathRow: ColumnLayout {
        id: pathRow
        required property string label
        required property string settingName
        property bool directory: false
        property bool automatic: false
        Layout.fillWidth: true
        spacing: 5

        SectionLabel { text: pathRow.label }
        RowLayout {
            Layout.fillWidth: true
            spacing: 7
            SetupField {
                Layout.fillWidth: true
                settingName: pathRow.settingName
            }
            SetupButton {
                text: "Choose…"
                onClicked: {
                    if (pathRow.directory)
                        setupWindow.controller.browseDirectory(pathRow.settingName)
                    else
                        setupWindow.controller.browseFile(pathRow.settingName)
                }
            }
            SetupButton {
                visible: pathRow.automatic
                text: "Auto"
                onClicked: setupWindow.controller.useAutomatic(pathRow.settingName)
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 58
            color: "#0e121b"
            border.color: "#242b3a"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: 10
                Rectangle {
                    Layout.preferredWidth: 30
                    Layout.preferredHeight: 30
                    radius: 8
                    color: "#7c8cff"
                    Text {
                        anchors.centerIn: parent
                        text: "⚙"
                        color: "white"
                        font.pixelSize: 15
                    }
                }
                Column {
                    Layout.fillWidth: true
                    spacing: 2
                    Text {
                        text: "Runtime & model setup"
                        color: "#f0f2f8"
                        font.pixelSize: 15
                        font.weight: Font.DemiBold
                    }
                    Text {
                        text: "Staged configuration · generation controls remain in the main workspace"
                        color: "#788195"
                        font.pixelSize: 10
                    }
                }
                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: setupWindow.controller.dirty ? "#ffb65c" : "#56d3b2"
                }
                Text {
                    text: setupWindow.controller.dirty ? "Unsaved changes" : "Up to date"
                    color: setupWindow.controller.dirty ? "#ffca8a" : "#8ce5ce"
                    font.pixelSize: 11
                }
            }
        }

        ScrollView {
            id: setupScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ColumnLayout {
                width: setupScroll.availableWidth
                spacing: 12

                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16
                    Layout.rightMargin: 16
                    Layout.topMargin: 14
                    implicitHeight: runtimeContent.implicitHeight + 28
                    radius: 11
                    color: "#111621"
                    border.color: "#293143"

                    ColumnLayout {
                        id: runtimeContent
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        SectionLabel { text: "Runtime" }
                        PathRow {
                            label: "ComfyUI checkout"
                            settingName: "comfyuiRoot"
                            directory: true
                        }
                        PathRow {
                            label: "GPU worker Python"
                            settingName: "workerPython"
                            automatic: true
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16
                    Layout.rightMargin: 16
                    implicitHeight: modelContent.implicitHeight + 28
                    radius: 11
                    color: "#111621"
                    border.color: "#293143"

                    ColumnLayout {
                        id: modelContent
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        SectionLabel { text: "Exact model files · blank uses discovery" }
                        PathRow { label: "Krea transformer"; settingName: "transformer"; automatic: true }
                        PathRow { label: "Text encoder"; settingName: "textEncoder"; automatic: true }
                        PathRow { label: "VAE"; settingName: "vae"; automatic: true }
                        PathRow { label: "Face detector"; settingName: "faceDetector"; automatic: true }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16
                    Layout.rightMargin: 16
                    implicitHeight: memoryContent.implicitHeight + 28
                    radius: 11
                    color: "#111621"
                    border.color: "#293143"

                    ColumnLayout {
                        id: memoryContent
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        SectionLabel { text: "Memory policy" }
                        ComboBox {
                            id: memoryPolicy
                            Layout.fillWidth: true
                            model: setupWindow.controller.memoryPolicyOptions
                            textRole: "label"
                            valueRole: "value"
                            currentIndex: {
                                let revision = setupWindow.controller.revision
                                let selected = String(setupWindow.controller.value("memoryPolicy"))
                                for (let index = 0; index < count; ++index) {
                                    if (valueAt(index) === selected)
                                        return index
                                }
                                return 0
                            }
                            onActivated: setupWindow.controller.setValue(
                                             "memoryPolicy", currentValue)
                        }
                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2
                            columnSpacing: 16
                            rowSpacing: 7
                            SectionLabel { text: "Keep VRAM free" }
                            SectionLabel { text: "Minimum free RAM" }
                            ValueSlider {
                                objectName: "reserveVramSlider"
                                Layout.fillWidth: true
                                from: 0.5
                                to: 128
                                stepSize: 0.5
                                decimals: 1
                                suffix: "GB"
                                liveUpdate: true
                                value: {
                                    let revision = setupWindow.controller.revision
                                    return Number(setupWindow.controller.value("reserveVram"))
                                }
                                onValueEdited: value => setupWindow.controller.setValue(
                                                   "reserveVram", value)
                            }
                            ValueSlider {
                                objectName: "minimumRamSlider"
                                Layout.fillWidth: true
                                from: 4
                                to: 256
                                stepSize: 1
                                decimals: 0
                                suffix: "GB"
                                liveUpdate: true
                                value: {
                                    let revision = setupWindow.controller.revision
                                    return Number(setupWindow.controller.value("minimumRam"))
                                }
                                onValueEdited: value => setupWindow.controller.setValue(
                                                   "minimumRam", value)
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            CheckBox {
                                objectName: "cpuVaeCheckBox"
                                text: "Decode with CPU VAE"
                                checked: {
                                    let revision = setupWindow.controller.revision
                                    return Boolean(setupWindow.controller.value("cpuVae"))
                                }
                                onClicked: setupWindow.controller.setValue("cpuVae", checked)
                            }
                            CheckBox {
                                objectName: "oomRecoveryCheckBox"
                                text: "Retry once after OOM"
                                checked: {
                                    let revision = setupWindow.controller.revision
                                    return Boolean(setupWindow.controller.value("oomRecovery"))
                                }
                                onClicked: setupWindow.controller.setValue(
                                               "oomRecovery", checked)
                            }
                            Item { Layout.fillWidth: true }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16
                    Layout.rightMargin: 16
                    implicitHeight: outputContent.implicitHeight + 28
                    radius: 11
                    color: "#111621"
                    border.color: "#293143"

                    ColumnLayout {
                        id: outputContent
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 10
                        SectionLabel { text: "Output" }
                        PathRow {
                            label: "Output directory"
                            settingName: "outputDirectory"
                            directory: true
                        }
                        SectionLabel { text: "Filename prefix" }
                        SetupField {
                            Layout.fillWidth: true
                            settingName: "filenamePrefix"
                            placeholderText: "baseline"
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 16
                    Layout.rightMargin: 16
                    Layout.bottomMargin: 16
                    implicitHeight: statusContent.implicitHeight + 28
                    radius: 11
                    color: "#111621"
                    border.color: "#293143"

                    ColumnLayout {
                        id: statusContent
                        anchors.fill: parent
                        anchors.margins: 14
                        spacing: 8
                        SectionLabel { text: "Worker & validation" }
                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 5
                            Text { text: "Worker"; color: "#788195"; font.pixelSize: 11 }
                            Text { text: setupWindow.controller.workerStatus; color: "#e2e5ed"; font.pixelSize: 11 }
                            Text { text: "Accelerator"; color: "#788195"; font.pixelSize: 11 }
                            Text { text: setupWindow.controller.acceleratorStatus; color: "#e2e5ed"; font.pixelSize: 11 }
                            Text { text: "Models"; color: "#788195"; font.pixelSize: 11 }
                            Text { text: setupWindow.controller.modelStatus; color: "#e2e5ed"; font.pixelSize: 11 }
                            Text { text: "Memory"; color: "#788195"; font.pixelSize: 11 }
                            Text {
                                Layout.fillWidth: true
                                text: setupWindow.controller.memoryStatus
                                color: "#e2e5ed"
                                font.pixelSize: 11
                                wrapMode: Text.Wrap
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            SetupButton { text: "Discover"; onClicked: setupWindow.controller.discoverModels() }
                            SetupButton { text: "Start worker"; onClicked: setupWindow.controller.startWorker() }
                            SetupButton { text: "Validate"; onClicked: setupWindow.controller.validateModels() }
                            SetupButton { text: "Load model"; onClicked: setupWindow.controller.loadModel() }
                            SetupButton { text: "Diagnose"; onClicked: setupWindow.controller.diagnoseAccelerator() }
                            Item { Layout.fillWidth: true }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 64
            color: "#0e121b"
            border.color: "#242b3a"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: 10
                Text {
                    Layout.fillWidth: true
                    text: setupWindow.message
                    color: "#9fa8bb"
                    font.pixelSize: 11
                    elide: Text.ElideRight
                }
                SetupButton {
                    text: "Close window"
                    onClicked: setupWindow.requestClose()
                }
                SetupButton {
                    objectName: "applySettingsButton"
                    text: "Apply settings"
                    primary: true
                    enabled: setupWindow.controller.dirty
                    onClicked: setupWindow.controller.apply()
                }
            }
        }
    }

    Dialog {
        id: unsavedDialog
        parent: setupWindow.contentItem
        modal: true
        anchors.centerIn: parent
        width: 430
        title: "Unsaved setup changes"
        closePolicy: Popup.NoAutoClose

        contentItem: ColumnLayout {
            spacing: 8
            Text {
                Layout.fillWidth: true
                text: "Do you want to apply the staged settings before closing?"
                color: "#e8ebf4"
                wrapMode: Text.Wrap
                font.pixelSize: 13
            }
            Text {
                Layout.fillWidth: true
                text: "Discard closes this setup window without changing the running application."
                color: "#8992a5"
                wrapMode: Text.Wrap
                font.pixelSize: 11
            }
        }

        footer: DialogButtonBox {
            Button {
                text: "Cancel"
                DialogButtonBox.buttonRole: DialogButtonBox.RejectRole
                onClicked: unsavedDialog.close()
            }
            Button {
                text: "Discard"
                DialogButtonBox.buttonRole: DialogButtonBox.DestructiveRole
                onClicked: {
                    setupWindow.controller.reset()
                    unsavedDialog.close()
                    setupWindow.finishClose()
                }
            }
            Button {
                text: "Apply settings"
                DialogButtonBox.buttonRole: DialogButtonBox.AcceptRole
                onClicked: {
                    if (setupWindow.controller.apply()) {
                        unsavedDialog.close()
                        setupWindow.finishClose()
                    }
                }
            }
        }
    }
}
