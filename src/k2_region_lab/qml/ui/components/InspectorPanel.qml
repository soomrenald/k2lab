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

    function commitPendingText() {
        controller.setGlobalPrompt(globalPrompt.text)
        if (controller.selectedRegionId.length > 0) {
            controller.updateSelectedRegion("name", selectedName.text)
            controller.updateSelectedRegion("prompt", selectedPrompt.text)
            controller.updateSelectedRegion("facePrompt", identityPrompt.text)
        }
    }

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

    component StudioTextArea: ScrollView {
        id: textAreaFrame
        property alias text: textEditor.text
        property alias placeholderText: textEditor.placeholderText
        property alias selectedText: textEditor.selectedText
        property alias selectionStart: textEditor.selectionStart
        signal editingFinished()
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ScrollBar.vertical: ScrollBar {
            objectName: textAreaFrame.objectName + "VerticalScrollBar"
            policy: ScrollBar.AsNeeded
        }
        background: Rectangle {
            color: "#0b0f18"
            border.color: textEditor.activeFocus ? "#6578ff" : "#2a3243"
            border.width: 1
            radius: 7
        }
        TextArea {
            id: textEditor
            width: textAreaFrame.availableWidth
            color: "#e8ebf4"
            placeholderTextColor: "#697287"
            selectionColor: "#6578ff"
            selectedTextColor: "white"
            wrapMode: TextEdit.Wrap
            persistentSelection: true
            font.pixelSize: 13
            leftPadding: 11
            rightPadding: 18
            topPadding: 9
            bottomPadding: 9
            background: null
            onActiveFocusChanged: {
                if (!activeFocus)
                    textAreaFrame.editingFinished()
            }
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
        property var controlSpec: {
            let revision = root.controller.stateRevision
            return root.controller.settingSpec(settingName)
        }
        property string suffix: String(controlSpec.suffix ?? "")
        property real minimum: Number(controlSpec.minimum ?? -999999999)
        property real maximum: Number(controlSpec.maximum ?? 999999999)
        property int decimals: Number(controlSpec.decimals ?? 0)
        property bool editing: false
        property string modelText: {
            let revision = root.controller.stateRevision
            let value = root.controller.setting(settingName)
            if (value === null || value === undefined)
                return ""
            return decimals > 0 ? Number(value).toFixed(decimals) : String(value)
        }
        spacing: 5
        Layout.fillWidth: true
        objectName: "numericSetting-" + settingName

        function beginEditing() {
            editing = true
        }

        function commitPendingText() {
            editing = true
            let parsed = decimals > 0 ? Number(numericInput.text) : parseInt(numericInput.text)
            if (!isNaN(parsed))
                root.controller.setSetting(settingName, parsed)
            editing = numericInput.activeFocus
        }

        LabelText { text: numeric.label }
        StudioTextField {
            id: numericInput
            objectName: "numericInput-" + numeric.settingName
            Layout.fillWidth: true
            enabled: Boolean(numeric.controlSpec.enabled ?? true)
            text: ""
            validator: DoubleValidator {
                bottom: numeric.minimum
                top: numeric.maximum
                decimals: numeric.decimals
                notation: DoubleValidator.StandardNotation
            }
            onActiveFocusChanged: {
                if (activeFocus)
                    numeric.beginEditing()
            }
            onTextEdited: numeric.beginEditing()
            onEditingFinished: numeric.commitPendingText()
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
        Binding {
            target: numericInput
            property: "text"
            value: numeric.modelText
            when: !numeric.editing && !numericInput.activeFocus
            restoreMode: Binding.RestoreNone
        }
    }

    component SettingCombo: ColumnLayout {
        id: settingCombo
        required property string label
        required property string settingName
        property string comboObjectName: "settingCombo-" + settingName
        readonly property var controlSpec: {
            let revision = root.controller.stateRevision
            return root.controller.settingSpec(settingName)
        }
        spacing: 5
        Layout.fillWidth: true

        LabelText { visible: settingCombo.label.length > 0; text: settingCombo.label }
        ComboBox {
            id: combo
            objectName: settingCombo.comboObjectName
            Layout.fillWidth: true
            enabled: Boolean(settingCombo.controlSpec.enabled ?? true)
            model: settingCombo.controlSpec.options ?? []
            textRole: "label"
            valueRole: "value"
            currentIndex: {
                let revision = root.controller.stateRevision
                let selected = root.controller.setting(settingCombo.settingName)
                for (let index = 0; index < count; ++index) {
                    if (valueAt(index) === selected)
                        return index
                }
                return count > 0 ? 0 : -1
            }
            delegate: ItemDelegate {
                required property var modelData
                width: combo.width
                text: String(modelData.label)
                enabled: Boolean(modelData.enabled)
            }
            onActivated: root.controller.setSetting(settingCombo.settingName, currentValue)
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
                        objectName: "globalPromptEditor"
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        implicitHeight: 118
                        text: controller.globalPrompt
                        placeholderText: controller.mode === "edit"
                                         ? "Describe the overall edit. Leave blank to preserve everything outside edit boxes."
                                         : "Describe the complete image..."
                        onTextChanged: controller.setGlobalPrompt(text)
                        onEditingFinished: controller.setGlobalPrompt(text)
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
                                objectName: "spatialRoleCombo"
                                Layout.fillWidth: true
                                model: controller.spatialRoleOptions
                                textRole: "label"
                                valueRole: "value"
                                currentIndex: {
                                    let selected = controller.selectedRegion.spatialRole || "auto"
                                    for (let index = 0; index < count; ++index) {
                                        if (valueAt(index) === selected)
                                            return index
                                    }
                                    return 0
                                }
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
                            objectName: "selectedPromptEditor"
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            implicitHeight: 112
                            text: controller.selectedRegion.prompt || ""
                            placeholderText: controller.mode === "edit" && controller.editLayer === "targets"
                                             ? "Describe the edit inside this box..."
                                             : "Describe the content inside this box..."
                            onTextChanged: {
                                if (visible)
                                    controller.updateSelectedRegion("prompt", text)
                            }
                            onEditingFinished: controller.updateSelectedRegion("prompt", text)
                        }

                        StudioTextArea {
                            id: identityPrompt
                            objectName: "identityPromptEditor"
                            visible: controller.selectedRegionId.length > 0
                            Layout.fillWidth: true
                            implicitHeight: 78
                            text: controller.selectedRegion.facePrompt || ""
                            placeholderText: "Optional face / identity anchor..."
                            onTextChanged: {
                                if (visible)
                                    controller.updateSelectedRegion("facePrompt", text)
                            }
                            onEditingFinished: controller.updateSelectedRegion("facePrompt", text)
                        }

                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        visible: controller.promptEmphasisAvailable

                        Rectangle { Layout.fillWidth: true; height: 1; color: "#262d3c" }
                        LabelText { text: "Phrase emphasis" }
                        Text {
                            Layout.fillWidth: true
                            text: "Highlight an exact phrase in a prompt, then add an additive token-attention boost. Invalid saved matches are shown in red."
                            color: "#778095"
                            wrapMode: Text.Wrap
                            font.pixelSize: 11
                        }
                        ValueSlider {
                            id: emphasisStrength
                            objectName: "promptEmphasisStrength"
                            Layout.fillWidth: true
                            from: 0
                            to: 2
                            stepSize: 0.1
                            decimals: 2
                            value: Number(controller.setting("emphasisStrength"))
                            onValueEdited: value => controller.setSetting("emphasisStrength", value)
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            MiniButton {
                                text: "Emphasize global selection"
                                Layout.fillWidth: true
                                enabled: globalPrompt.selectedText.length > 0
                                onClicked: controller.addPromptEmphasis(
                                    controller.globalEmphasisScope,
                                    globalPrompt.selectedText,
                                    globalPrompt.selectionStart,
                                    emphasisStrength.currentValue)
                            }
                            MiniButton {
                                visible: controller.selectedRegionId.length > 0
                                text: "Region selection"
                                Layout.fillWidth: true
                                enabled: selectedPrompt.selectedText.length > 0
                                onClicked: controller.addPromptEmphasis(
                                    controller.selectedRegionId,
                                    selectedPrompt.selectedText,
                                    selectedPrompt.selectionStart,
                                    emphasisStrength.currentValue)
                            }
                        }
                        Repeater {
                            model: controller.promptEmphases
                            delegate: Rectangle {
                                id: emphasisCard
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: emphasisColumn.implicitHeight + 18
                                radius: 7
                                color: "#171d29"
                                border.color: modelData.matches ? "#293143" : "#df5f67"
                                ColumnLayout {
                                    id: emphasisColumn
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.margins: 9
                                    spacing: 5
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: emphasisCard.modelData.scopeLabel + ": “"
                                                  + emphasisCard.modelData.phrase + "”"
                                            color: emphasisCard.modelData.matches ? "#e8ebf4" : "#ff8f96"
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                            font.pixelSize: 11
                                        }
                                        MiniButton {
                                            text: "Remove"
                                            onClicked: controller.removePromptEmphasis(
                                                emphasisCard.modelData.index)
                                        }
                                    }
                                    Text {
                                        visible: !emphasisCard.modelData.matches
                                        text: "Phrase no longer matches this prompt; remove it or restore the exact text."
                                        color: "#ff8f96"
                                        wrapMode: Text.Wrap
                                        Layout.fillWidth: true
                                        font.pixelSize: 10
                                    }
                                    ValueSlider {
                                        Layout.fillWidth: true
                                        from: 0
                                        to: 2
                                        stepSize: 0.1
                                        decimals: 2
                                        value: emphasisCard.modelData.strength
                                        onValueEdited: value => controller.setPromptEmphasisStrength(
                                            emphasisCard.modelData.index, value)
                                    }
                                }
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

                    RowLayout {
                        visible: controller.mode !== "face"
                                 && controller.selectedRegionId.length > 0
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        LabelText {
                            text: "Selected depth"
                            Layout.fillWidth: true
                        }
                        MiniButton {
                            objectName: "regionForwardButton"
                            text: "↑ Forward"
                            onClicked: controller.moveRegion(controller.selectedRegionId, -1)
                        }
                        MiniButton {
                            objectName: "regionBackwardButton"
                            text: "↓ Backward"
                            onClicked: controller.moveRegion(controller.selectedRegionId, 1)
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
                        id: loraRepeater
                        objectName: "loraRepeater"
                        model: controller.loraModel
                        delegate: Rectangle {
                            id: loraCard
                            required property string loraId
                            required property string name
                            required property string scope
                            required property real strength
                            required property string routingMode
                            required property string triggerPhrase
                            required property bool active
                            Layout.fillWidth: true
                            Layout.leftMargin: 14
                            Layout.rightMargin: 14
                            implicitHeight: loraColumn.implicitHeight + 20
                            radius: 8
                            color: "#171d29"
                            border.color: "#293143"

                            ColumnLayout {
                                id: loraColumn
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
                                        property bool assigned: {
                                            let revision = controller.stateRevision
                                            return controller.loraUsesSelectedRegion(loraCard.loraId)
                                        }
                                        text: assigned ? "Remove selected" : "Add selected"
                                        Layout.fillWidth: true
                                        enabled: controller.selectedRegionId.length > 0
                                        onClicked: controller.toggleLoraSelectedRegion(loraCard.loraId)
                                    }
                                    MiniButton {
                                        text: "Remove"
                                        onClicked: controller.removeLora(loraCard.loraId)
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    LabelText { text: "Routing" }
                                    ComboBox {
                                        id: routingMode
                                        objectName: "loraRouting-" + loraCard.loraId
                                        Layout.fillWidth: true
                                        model: ["Standard regional", "Character identity (face)"]
                                        currentIndex: loraCard.routingMode === "character_identity" ? 1 : 0
                                        enabled: loraCard.scope !== "Global"
                                                 && loraCard.scope !== "Unassigned"
                                        onActivated: controller.setLoraRoutingMode(
                                            loraCard.loraId,
                                            currentIndex === 1 ? "character_identity" : "standard")
                                    }
                                }
                                StudioTextField {
                                    objectName: "loraTrigger-" + loraCard.loraId
                                    visible: loraCard.routingMode === "character_identity"
                                    Layout.fillWidth: true
                                    text: loraCard.triggerPhrase
                                    placeholderText: "Training trigger, for example lface"
                                    onEditingFinished: controller.setLoraTriggerPhrase(
                                        loraCard.loraId, text)
                                }
                                Text {
                                    visible: loraCard.routingMode === "character_identity"
                                    Layout.fillWidth: true
                                    text: "The identity trigger is inserted automatically into each assigned region's face anchor; it does not need to be duplicated in the visible prompt."
                                    color: "#778095"
                                    wrapMode: Text.Wrap
                                    font.pixelSize: 10
                                }
                                MiniButton {
                                    text: "Inspect Krea compatibility"
                                    Layout.fillWidth: true
                                    onClicked: controller.diagnoseLora(loraCard.loraId)
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

                        SettingCombo { label: "Sampler"; settingName: "sampler" }
                        SettingCombo { label: "Scheduler"; settingName: "scheduler" }
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
                        }
                        NumericSetting {
                            label: "Seed"
                            settingName: "seed"
                        }
                        SettingCombo {
                            visible: controller.mode === "generation"
                            label: "Seed behavior"
                            settingName: "seedMode"
                            comboObjectName: "seedModeCombo"
                        }
                        NumericSetting {
                            visible: controller.mode === "generation"
                                     && Boolean(controller.setting("batchMode"))
                            label: "Batch runs"
                            settingName: "batchCount"
                        }
                        NumericSetting {
                            visible: controller.mode === "generation"
                            label: "Width"
                            settingName: "width"
                        }
                        NumericSetting {
                            visible: controller.mode === "generation"
                            label: "Height"
                            settingName: "height"
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Denoise"
                            settingName: "denoise"
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Retention"
                            settingName: "referenceRetention"
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Latent feather"
                            settingName: "latentFeather"
                        }
                        NumericSetting {
                            visible: controller.mode === "edit"
                            label: "Composite feather"
                            settingName: "compositeFeather"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Denoise"
                            settingName: "denoise"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Padding"
                            settingName: "padding"
                        }
                        SettingCombo {
                            visible: controller.mode === "face"
                            label: "Crop size"
                            settingName: "cropSize"
                            comboObjectName: "faceCropSizeCombo"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Edge feather"
                            settingName: "feather"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Regional LoRA scale"
                            settingName: "loraScale"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Detector threshold"
                            settingName: "detectorThreshold"
                        }
                        NumericSetting {
                            visible: controller.mode === "face"
                            label: "Blend"
                            settingName: "blend"
                        }
                    }

                    CheckBox {
                        objectName: "batchModeCheckBox"
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Run generation in batch mode"
                        checked: Boolean(controller.setting("batchMode"))
                        onToggled: controller.setSetting("batchMode", checked)
                    }

                    ColumnLayout {
                        visible: controller.mode === "face"
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 5
                        LabelText { text: "Detector device" }
                        SettingCombo {
                            label: ""
                            settingName: "detectorProvider"
                            comboObjectName: "detectorProviderCombo"
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            MiniButton { text: "Use latest first pass"; onClicked: controller.useLatestFaceSource() }
                            MiniButton { text: "Draw lasso…"; onClicked: controller.openFaceLasso() }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            MiniButton { text: "Undo lasso"; Layout.fillWidth: true; onClicked: controller.undoFaceLasso() }
                            MiniButton { text: "Clear lassos"; Layout.fillWidth: true; onClicked: controller.clearFaceLassos() }
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
                        }
                        NumericSetting {
                            label: "Outside penalty"
                            settingName: "outsidePenalty"
                        }
                        NumericSetting {
                            label: "Spatial falloff"
                            settingName: "spatialFalloff"
                        }
                        NumericSetting {
                            label: "Late scale"
                            settingName: "lateStepScale"
                        }
                    }

                    CheckBox {
                        objectName: "regionalPromptingCheckBox"
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Use unified spatial prompting"
                        checked: Boolean(controller.setting("regionalPrompting"))
                        onToggled: controller.setSetting("regionalPrompting", checked)
                    }
                    CheckBox {
                        objectName: "subjectCompetitionCheckBox"
                        visible: controller.mode !== "face"
                        Layout.leftMargin: 14
                        text: "Separate overlapping subject targets"
                        checked: Boolean(controller.setting("subjectCompetition"))
                        onToggled: controller.setSetting("subjectCompetition", checked)
                    }
                    CheckBox {
                        objectName: "subjectFillCheckBox"
                        visible: controller.mode !== "face"
                        Layout.leftMargin: 14
                        text: "Make subjects fill their boxes"
                        checked: Boolean(controller.setting("subjectFill"))
                        onToggled: controller.setSetting("subjectFill", checked)
                    }
                    CheckBox {
                        objectName: "relaxationCheckBox"
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Relax spatial guidance during late steps"
                        checked: Boolean(controller.setting("relaxation"))
                        onToggled: controller.setSetting("relaxation", checked)
                    }
                    CheckBox {
                        objectName: "loraAdaptationCheckBox"
                        visible: controller.mode !== "face"
                        Layout.leftMargin: 14
                        text: "Adapt spatial guidance from regional LoRA delta"
                        checked: Boolean(controller.setting("loraAdaptation"))
                        onToggled: controller.setSetting("loraAdaptation", checked)
                    }
                    NumericSetting {
                        visible: controller.mode !== "face"
                                 && Boolean(controller.setting("loraAdaptation"))
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        label: "LoRA delta response"
                        settingName: "loraResponse"
                    }

                    CheckBox {
                        objectName: "preserveIdentityCheckBox"
                        visible: controller.mode === "edit"
                        Layout.leftMargin: 14
                        text: "Preserve identity"
                        checked: Boolean(controller.setting("preserveIdentity"))
                        onToggled: controller.setSetting("preserveIdentity", checked)
                    }
                    CheckBox {
                        objectName: "editEntireImageCheckBox"
                        visible: controller.mode === "edit"
                        Layout.leftMargin: 14
                        text: "Edit entire image"
                        checked: Boolean(controller.setting("editEntireImage"))
                        onToggled: controller.setSetting("editEntireImage", checked)
                    }

                    LabelText {
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Post-upscale"
                    }
                    CheckBox {
                        objectName: "postUpscaleCheckBox"
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Post-upscale after releasing Krea VRAM"
                        checked: Boolean(controller.setting("postUpscale"))
                        onToggled: controller.setSetting("postUpscale", checked)
                    }
                    RowLayout {
                        visible: controller.mode === "generation"
                                 && Boolean(controller.setting("postUpscale"))
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        SettingCombo { label: "Scale"; settingName: "upscaleScale" }
                        SettingCombo { label: "Method"; settingName: "upscaleMethod" }
                    }
                    ColumnLayout {
                        visible: controller.mode === "generation"
                                 && Boolean(controller.setting("postUpscale"))
                                 && String(controller.setting("upscaleMethod")) === "model"
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        StudioTextField {
                            Layout.fillWidth: true
                            readOnly: true
                            text: controller.upscaleModelPath
                            placeholderText: "Select ESRGAN-compatible model..."
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            MiniButton { text: "Choose model…"; Layout.fillWidth: true; onClicked: controller.browseUpscaleModel() }
                            MiniButton { text: "Clear"; onClicked: controller.clearUpscaleModel() }
                        }
                    }

                    LabelText {
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Projector"
                    }
                    CheckBox {
                        objectName: "projectorEnabledCheckBox"
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        text: "Apply global projector vector"
                        checked: Boolean(controller.setting("projectorEnabled"))
                        onToggled: controller.setSetting("projectorEnabled", checked)
                    }
                    ColumnLayout {
                        visible: controller.mode === "generation"
                                 && Boolean(controller.setting("projectorEnabled"))
                        Layout.fillWidth: true
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        spacing: 8
                        SettingCombo { label: "Preset"; settingName: "projectorPreset" }
                        GridLayout {
                            Layout.fillWidth: true
                            columns: 3
                            columnSpacing: 6
                            rowSpacing: 6
                            Repeater {
                                model: controller.projectorVector
                                delegate: StudioTextField {
                                    required property int index
                                    required property real modelData
                                    Layout.fillWidth: true
                                    text: Number(modelData).toFixed(4)
                                    validator: DoubleValidator { bottom: -1000; top: 1000; decimals: 4 }
                                    onEditingFinished: controller.setProjectorValue(index, Number(text))
                                }
                            }
                        }
                        NumericSetting {
                            label: "Global multiplier"
                            settingName: "projectorMultiplier"
                        }
                        NumericSetting {
                            label: "Face identity protection"
                            settingName: "projectorIdentityProtection"
                        }
                    }

                    MiniButton {
                        visible: controller.mode === "generation"
                        Layout.leftMargin: 14
                        Layout.rightMargin: 14
                        Layout.fillWidth: true
                        text: "Preview unified prompt…"
                        onClicked: controller.previewUnifiedPrompt()
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
