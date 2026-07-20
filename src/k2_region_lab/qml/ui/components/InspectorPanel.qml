import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    required property QtObject controller
    signal openSetupRequested()
    color: "#111621"
    border.color: "#262d3c"
    border.width: 1
    radius: 12

    component LabelText: Text {
        color: "#9099ad"
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.6
        font.capitalization: Font.AllUppercase
    }

    component StudioTextField: TextField {
        color: "#e8ebf4"
        placeholderTextColor: "#697287"
        selectionColor: "#6578ff"
        selectedTextColor: "white"
        font.pixelSize: 13
        leftPadding: 11
        rightPadding: 11
        background: Rectangle {
            color: "#0b0f18"
            border.color: parent.activeFocus ? "#6578ff" : "#2a3243"
            border.width: 1
            radius: 7
        }
    }

    component StudioTextArea: TextArea {
        color: "#e8ebf4"
        placeholderTextColor: "#697287"
        selectionColor: "#6578ff"
        selectedTextColor: "white"
        wrapMode: TextEdit.Wrap
        font.pixelSize: 13
        leftPadding: 11
        rightPadding: 11
        topPadding: 9
        bottomPadding: 9
        background: Rectangle {
            color: "#0b0f18"
            border.color: parent.activeFocus ? "#6578ff" : "#2a3243"
            border.width: 1
            radius: 7
        }
    }

    component MiniButton: Button {
        id: miniButton
        implicitHeight: 30
        leftPadding: 11
        rightPadding: 11
        font.pixelSize: 12
        font.weight: Font.DemiBold
        contentItem: Text {
            text: miniButton.text
            color: miniButton.enabled ? "#dfe3ee" : "#666d7d"
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font: miniButton.font
        }
        background: Rectangle {
            color: miniButton.down ? "#30394d" : (miniButton.hovered ? "#252d3d" : "#1a202d")
            border.color: "#30394d"
            radius: 7
        }
    }

    component NumericSetting: ColumnLayout {
        id: numeric
        required property string label
        required property string settingName
        property string suffix: ""
        property real minimum: -999999999
        property real maximum: 999999999
        property int decimals: 0
        spacing: 5
        Layout.fillWidth: true

        LabelText { text: numeric.label }
        StudioTextField {
            id: numericInput
            Layout.fillWidth: true
            text: {
                let revision = root.controller.stateRevision
                let value = root.controller.setting(numeric.settingName)
                if (value === null || value === undefined)
                    return ""
                return numeric.decimals > 0 ? Number(value).toFixed(numeric.decimals) : String(value)
            }
            validator: DoubleValidator {
                bottom: numeric.minimum
                top: numeric.maximum
                decimals: numeric.decimals
                notation: DoubleValidator.StandardNotation
            }
            onEditingFinished: {
                let parsed = numeric.decimals > 0 ? Number(text) : parseInt(text)
                if (!isNaN(parsed))
                    root.controller.setSetting(numeric.settingName, parsed)
            }
            rightPadding: numeric.suffix.length > 0 ? 42 : 11
            Text {
                visible: numeric.suffix.length > 0
                anchors.right: parent.right
                anchors.rightMargin: 10
                anchors.verticalCenter: parent.verticalCenter
                text: numeric.suffix
                color: "#697287"
                font.pixelSize: 11
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 1
        spacing: 0

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 14
            Layout.rightMargin: 14
            Layout.topMargin: 12
            Layout.bottomMargin: 8
            spacing: 8

            Text {
                text: "Inspector"
                color: "#f0f2f8"
                font.pixelSize: 15
                font.weight: Font.DemiBold
                Layout.fillWidth: true
            }
            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: controller.busy ? "#ffb65c" : "#56d3b2"
            }
        }

        TabBar {
            id: inspectorTabs
            objectName: "inspectorTabs"
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            background: Rectangle { color: "transparent" }

            Repeater {
                model: ["Prompt", "Regions", "LoRAs", "Advanced"]
                TabButton {
                    required property string modelData
                    text: modelData
                    font.pixelSize: 12
                    contentItem: Text {
                        text: parent.text
                        color: parent.checked ? "#ffffff" : "#838da1"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        font: parent.font
                    }
                    background: Rectangle {
                        color: parent.checked ? "#252c3c" : "transparent"
                        radius: 7
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#262d3c"
        }

        StackLayout {
            currentIndex: inspectorTabs.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            ScrollView {
                id: promptScroll
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ColumnLayout {
                    width: promptScroll.availableWidth
                    spacing: 9

                    Item { height: 4 }
                    LabelText {
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        text: controller.promptLabel
                    }
                    StudioTextArea {
                        id: globalPrompt
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        implicitHeight: 118
                        text: controller.globalPrompt
                        placeholderText: controller.mode === "edit"
                                         ? "Describe the overall edit. Leave blank to preserve everything outside edit boxes."
                                         : "Describe the complete image..."
                        onActiveFocusChanged: {
                            if (!activeFocus)
                                controller.setGlobalPrompt(text)
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        height: 1
                        color: "#262d3c"
                        visible: controller.mode !== "face"
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        visible: controller.mode !== "face"

                        RowLayout {
                            Layout.fillWidth: true
                            LabelText {
                                text: controller.selectedRegionId.length > 0
                                      ? "Selected region" : "Region prompt"
                                Layout.fillWidth: true
                            }
                            Text {
                                text: controller.selectedRegionId.length > 0 ? "ACTIVE" : "NONE"
                                color: controller.selectedRegionId.length > 0 ? "#7c8cff" : "#697287"
                                font.pixelSize: 10
                                font.weight: Font.Bold
                            }
                        }

                        Text {
                            visible: controller.selectedRegionId.length === 0
                            Layout.fillWidth: true
                            text: "Select a box on the canvas to edit its prompt and identity controls."
                            color: "#778095"
                            wrapMode: Text.Wrap
                            font.pixelSize: 13
                        }

                        StudioTextField {
                            id: selectedName
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            text: controller.selectedRegion.name || ""
                            placeholderText: "Region name"
                            onEditingFinished: controller.updateSelectedRegion("name", text)
                        }

                        RowLayout {
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            spacing: 8
                            ComboBox {
                                id: spatialRole
                                Layout.fillWidth: true
                                model: ["auto", "subject", "background", "edit"]
                                currentIndex: Math.max(0, indexOfValue(controller.selectedRegion.spatialRole || "auto"))
                                onActivated: controller.updateSelectedRegion("spatialRole", currentValue)
                            }
                            CheckBox {
                                text: "Enabled"
                                checked: controller.selectedRegion.enabled === undefined
                                         ? true : controller.selectedRegion.enabled
                                onToggled: controller.updateSelectedRegion("enabled", checked)
                            }
                        }

                        StudioTextArea {
                            id: selectedPrompt
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            implicitHeight: 112
                            text: controller.selectedRegion.prompt || ""
                            placeholderText: controller.mode === "edit" && controller.editLayer === "targets"
                                             ? "Describe the edit inside this box..."
                                             : "Describe the content inside this box..."
                            onActiveFocusChanged: {
                                if (!activeFocus)
                                    controller.updateSelectedRegion("prompt", text)
                            }
                        }

                        StudioTextArea {
                            id: identityPrompt
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            implicitHeight: 78
                            text: controller.selectedRegion.facePrompt || ""
                            placeholderText: "Optional face / identity anchor..."
                            onActiveFocusChanged: {
                                if (!activeFocus)
                                    controller.updateSelectedRegion("facePrompt", text)
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        visible: controller.mode === "face"

                        RowLayout {
                            Layout.fillWidth: true
                            LabelText { text: "Detected faces"; Layout.fillWidth: true }
                            MiniButton { text: "All"; onClicked: controller.setAllFacesSelected(true) }
                            MiniButton { text: "None"; onClicked: controller.setAllFacesSelected(false) }
                        }
                        Text {
                            visible: controller.faceCount === 0
                            Layout.fillWidth: true
                            text: "Load an image, then run face detection. Faces matched to regional LoRAs are selected automatically."
                            color: "#778095"
                            wrapMode: Text.Wrap
                            font.pixelSize: 13
                        }
                        Repeater {
                            model: controller.faces
                            delegate: CheckBox {
                                required property var modelData
                                Layout.fillWidth: true
                                text: modelData.label + "  ·  " + modelData.regionName
                                      + "  ·  " + Number(modelData.score).toFixed(3)
                                checked: modelData.selected
                                onToggled: controller.setFaceSelected(modelData.index, checked)
                            }
                        }
                    }
                    Item { height: 14 }
                }
            }

            ScrollView {
                id: regionsScroll
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ColumnLayout {
                    width: regionsScroll.availableWidth
                    spacing: 8

                    RowLayout {
                        visible: controller.mode !== "face"
                        Layout.fillWidth: true
                        Layout.margins: 14
                        LabelText { text: "Canvas regions"; Layout.fillWidth: true }
                        MiniButton {
                            text: "+ Draw"
                            enabled: controller.canDrawRegions
                            onClicked: controller.setDrawMode(true)
                        }
                    }

                    Repeater {
                        model: controller.activeRegionModel
                        delegate: Rectangle {
                            id: regionRow
                            required property string regionId
                            required property string name
                            required property string prompt
                            required property string spatialRole
                            required property bool regionEnabled
                            Layout.fillWidth: true
                            Layout.leftMargin: 14
                            Layout.rightMargin: 14
                            implicitHeight: 70
                            radius: 8
                            color: controller.selectedRegionId === regionId ? "#252d40" : "#171d29"
                            border.color: controller.selectedRegionId === regionId ? "#6578ff" : "#293143"

                            MouseArea {
                                anchors.fill: parent
                                onClicked: controller.selectRegion(regionId)
                            }
                            Column {
                                anchors.left: parent.left
                                anchors.right: removeButton.left
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.leftMargin: 12
                                spacing: 4
                                Text {
                                    width: parent.width
                                    text: regionRow.name
                                    color: regionRow.regionEnabled ? "#edf0f7" : "#7a8294"
                                    elide: Text.ElideRight
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                }
                                Text {
                                    width: parent.width
                                    text: regionRow.prompt.length > 0 ? regionRow.prompt : "No regional prompt"
                                    color: "#7e879a"
                                    elide: Text.ElideRight
                                    font.pixelSize: 11
                                }
                                Text {
                                    text: regionRow.spatialRole.toUpperCase()
                                    color: "#6578ff"
                                    font.pixelSize: 9
                                    font.weight: Font.Bold
                                }
                            }
                            MiniButton {
                                id: removeButton
                                anchors.right: parent.right
                                anchors.rightMargin: 8
                                anchors.verticalCenter: parent.verticalCenter
                                text: "Delete"
                                onClicked: controller.deleteRegion(regionId)
                            }
                        }
                    }

                    Text {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        visible: controller.mode === "face"
                        text: "Face refinement uses detected faces instead of prompt regions."
                        color: "#778095"
                        wrapMode: Text.Wrap
                        font.pixelSize: 13
                    }
                }
            }

            ScrollView {
                id: loraScroll
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ColumnLayout {
                    width: loraScroll.availableWidth
                    spacing: 9

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.margins: 14
                        LabelText { text: "LoRA routing"; Layout.fillWidth: true }
                        MiniButton { text: "+ Add LoRA"; onClicked: controller.addLora() }
                    }

                    Text {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        text: controller.mode === "edit"
                              ? "Assignments apply only to the active edit layer."
                              : "Assign each LoRA globally or to the selected region."
                        color: "#778095"
                        wrapMode: Text.Wrap
                        font.pixelSize: 12
                    }

                    Repeater {
                        model: controller.loraModel
                        delegate: Rectangle {
                            id: loraCard
                            required property string loraId
                            required property string name
                            required property string scope
                            required property real strength
                            required property string routingMode
                            required property bool active
                            Layout.fillWidth: true
                            Layout.leftMargin: 14
                            Layout.rightMargin: 14
                            implicitHeight: 148
                            radius: 8
                            color: "#171d29"
                            border.color: "#293143"

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 5
                                RowLayout {
                                    Layout.fillWidth: true
                                    Switch {
                                        checked: loraCard.active
                                        text: checked ? "Active" : "Inactive"
                                        onToggled: controller.setLoraActive(
                                                       loraCard.loraId, checked)
                                    }
                                    Text {
                                        text: loraCard.name
                                        color: "#edf0f7"
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                    Text {
                                        text: loraCard.strength.toFixed(2)
                                        color: "#aeb6ca"
                                        font.pixelSize: 11
                                    }
                                }
                                Text {
                                    text: loraCard.scope
                                    color: loraCard.scope === "Unassigned" ? "#ffb65c" : "#56d3b2"
                                    font.pixelSize: 11
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                                ValueSlider {
                                    objectName: "loraStrengthSlider-" + loraCard.loraId
                                    Layout.fillWidth: true
                                    from: -4
                                    to: 4
                                    stepSize: 0.05
                                    value: loraCard.strength
                                    decimals: 2
                                    onValueEdited: value => controller.setLoraStrength(
                                                       loraCard.loraId, value)
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    MiniButton {
                                        text: "Global"
                                        Layout.fillWidth: true
                                        onClicked: controller.assignLoraGlobal(loraCard.loraId)
                                    }
                                    MiniButton {
                                        text: "Selected region"
                                        Layout.fillWidth: true
                                        enabled: controller.selectedRegionId.length > 0
                                        onClicked: controller.assignLoraToSelectedRegion(loraCard.loraId)
                                    }
                                    MiniButton {
                                        text: "Remove"
                                        onClicked: controller.removeLora(loraCard.loraId)
                                    }
                                }
                            }
                        }
                    }
                    Item { implicitHeight: 28 }
                }
            }

            ScrollView {
                id: advancedScroll
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ColumnLayout {
                    width: advancedScroll.availableWidth
                    spacing: 10

                    Item { height: 4 }
                    LabelText {
                        Layout.leftMargin: 14
                        text: controller.mode === "edit" ? "Denoising" : "Sampling"
                    }

                    RowLayout {
                        visible: controller.mode !== "face"
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 9

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 5
                            LabelText { text: "Sampler" }
                            ComboBox {
                                Layout.fillWidth: true
                                model: root.controller.samplerOptions
                                currentIndex: {
                                    let revision = root.controller.stateRevision
                                    return model.indexOf(String(root.controller.setting("sampler")))
                                }
                                onActivated: root.controller.setSetting("sampler", currentValue)
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 5
                            LabelText { text: "Scheduler" }
                            ComboBox {
                                Layout.fillWidth: true
                                model: root.controller.schedulerOptions
                                currentIndex: {
                                    let revision = root.controller.stateRevision
                                    return model.indexOf(String(root.controller.setting("scheduler")))
                                }
                                onActivated: root.controller.setSetting("scheduler", currentValue)
                            }
                        }
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        columns: 2
                        columnSpacing: 9
                        rowSpacing: 10

                        NumericSetting {
                            label: "Steps"
                            settingName: "steps"
                            minimum: 1
                            maximum: 200
                        }
                        NumericSetting {
                            label: "Seed"
                            settingName: "seed"
                            minimum: -2147483648
                            maximum: 2147483647
                        }
                        NumericSetting {
                            visible: controller.mode === "generation"
                            label: "Width"
                            settingName: "width"
                            minimum: 64
                            maximum: 4096
                        }
                        NumericSetting {
                            visible: controller.mode === "generation"
                            label: "Height"
                            settingName: "height"
                            minimum: 64
                            maximum: 4096
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Denoise"
                            settingName: "denoise"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Retention"
                            settingName: "referenceRetention"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Latent feather"
                            settingName: "latentFeather"
                            minimum: 0
                            maximum: 1024
                            suffix: "px"
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Composite feather"
                            settingName: "compositeFeather"
                            minimum: 0
                            maximum: 1024
                            suffix: "px"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Denoise"
                            settingName: "denoise"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Padding"
                            settingName: "padding"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Detector threshold"
                            settingName: "detectorThreshold"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Blend"
                            settingName: "blend"
                            minimum: 0
                            maximum: 1
                            decimals: 2
                        }
                    }

                    LabelText {
                        visible: controller.mode !== "face"
                        Layout.leftMargin: 14
                        text: "Regional attention"
                    }
                    GridLayout {
                        visible: controller.mode !== "face"
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        columns: 2
                        columnSpacing: 9
                        rowSpacing: 10

                        NumericSetting {
                            label: "Inside boost"
                            settingName: "insideBoost"
                            decimals: 2
                        }
                        NumericSetting {
                            label: "Outside penalty"
                            settingName: "outsidePenalty"
                            decimals: 2
                        }
                        NumericSetting {
                            label: "Spatial falloff"
                            settingName: "spatialFalloff"
                            decimals: 2
                        }
                        NumericSetting {
                            label: "Late scale"
                            settingName: "lateStepScale"
                            decimals: 2
                        }
                    }

                    CheckBox {
                        visible: controller.mode === "edit"
                        Layout.leftMargin: 14
                        text: "Preserve identity"
                        checked: Boolean(controller.setting("preserveIdentity"))
                        onToggled: controller.setSetting("preserveIdentity", checked)
                    }
                    CheckBox {
                        visible: controller.mode === "edit"
                        Layout.leftMargin: 14
                        text: "Edit entire image"
                        checked: Boolean(controller.setting("editEntireImage"))
                        onToggled: controller.setSetting("editEntireImage", checked)
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        implicitHeight: setupColumn.implicitHeight + 20
                        radius: 8
                        color: "#171d29"
                        border.color: "#293143"

                        ColumnLayout {
                            id: setupColumn
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.margins: 10
                            spacing: 7
                            Text {
                                Layout.fillWidth: true
                                text: "Runtime & model setup"
                                color: "#edf0f7"
                                font.pixelSize: 13
                                font.weight: Font.DemiBold
                            }
                            Text {
                                Layout.fillWidth: true
                                text: "Configure runtime paths, model files, memory policy, and output defaults in the dedicated setup window."
                                color: "#778095"
                                wrapMode: Text.Wrap
                                font.pixelSize: 11
                            }
                            MiniButton {
                                text: "Open setup window"
                                Layout.fillWidth: true
                                onClicked: root.openSetupRequested()
                            }
                        }
                    }
                    Item { height: 14 }
                }
            }
        }
    }
}
