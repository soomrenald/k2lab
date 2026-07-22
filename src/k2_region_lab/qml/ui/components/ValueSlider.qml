import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

RowLayout {
    id: root

    property real from: 0
    property real to: 1
    property real stepSize: 0.01
    property real value: 0
    property real currentValue: bounded(value)
    property int decimals: 2
    property string suffix: ""
    property bool liveUpdate: false
    property bool textEditing: false
    signal valueEdited(real value)

    function bounded(candidate) {
        return Math.max(root.from, Math.min(root.to, Number(candidate)))
    }

    function updateFromUser(candidate, commit) {
        currentValue = bounded(candidate)
        syncText()
        if (root.liveUpdate || commit)
            root.valueEdited(currentValue)
    }

    function syncText() {
        if (!textEditing && !valueInput.activeFocus)
            valueInput.text = Number(currentValue).toFixed(root.decimals)
    }

    function commitText() {
        textEditing = true
        let parsed = Number(valueInput.text)
        if (isNaN(parsed)) {
            textEditing = valueInput.activeFocus
            if (!textEditing)
                valueInput.text = Number(root.currentValue).toFixed(root.decimals)
            return
        }
        root.updateFromUser(parsed, true)
        textEditing = valueInput.activeFocus
        valueInput.text = Number(root.currentValue).toFixed(root.decimals)
    }

    onValueChanged: {
        currentValue = bounded(value)
        syncText()
    }
    onCurrentValueChanged: syncText()
    Component.onCompleted: syncText()

    spacing: 8

    Slider {
        id: slider
        objectName: "valueSliderTrack"
        Layout.fillWidth: true
        from: root.from
        to: root.to
        stepSize: root.stepSize
        value: root.currentValue
        live: true
        onMoved: root.updateFromUser(value, false)
        onPressedChanged: {
            if (!pressed)
                root.updateFromUser(value, true)
        }
    }

    TextField {
        id: valueInput
        objectName: "valueSliderInput"
        Layout.preferredWidth: root.suffix.length > 0 ? 76 : 62
        implicitHeight: 30
        text: ""
        color: "#e8ebf4"
        selectionColor: "#6578ff"
        selectedTextColor: "white"
        horizontalAlignment: TextInput.AlignRight
        rightPadding: root.suffix.length > 0 ? 28 : 9
        validator: DoubleValidator {
            bottom: root.from
            top: root.to
            decimals: root.decimals
            notation: DoubleValidator.StandardNotation
        }
        onActiveFocusChanged: {
            if (activeFocus)
                root.textEditing = true
        }
        onTextEdited: root.textEditing = true
        onEditingFinished: root.commitText()
        background: Rectangle {
            color: "#0b0f18"
            border.color: parent.activeFocus ? "#6578ff" : "#2a3243"
            radius: 6
        }
        Text {
            visible: root.suffix.length > 0
            anchors.right: parent.right
            anchors.rightMargin: 7
            anchors.verticalCenter: parent.verticalCenter
            text: root.suffix
            color: "#697287"
            font.pixelSize: 10
        }
    }

    Binding {
        target: valueInput
        property: "text"
        value: Number(root.currentValue).toFixed(root.decimals)
        when: !root.textEditing && !valueInput.activeFocus
        restoreMode: Binding.RestoreNone
    }
}
