import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

RowLayout {
    id: root

    property real from: 0
    property real to: 1
    property real stepSize: 0.01
    property real value: 0
    property int decimals: 2
    property string suffix: ""
    signal valueEdited(real value)

    function commitText() {
        let parsed = Number(valueInput.text)
        if (isNaN(parsed)) {
            valueInput.text = Number(root.value).toFixed(root.decimals)
            return
        }
        let bounded = Math.max(root.from, Math.min(root.to, parsed))
        root.valueEdited(bounded)
        valueInput.text = Number(bounded).toFixed(root.decimals)
    }

    spacing: 8

    Slider {
        id: slider
        Layout.fillWidth: true
        from: root.from
        to: root.to
        stepSize: root.stepSize
        value: root.value
        live: true
        onMoved: root.valueEdited(value)
    }

    TextField {
        id: valueInput
        Layout.preferredWidth: root.suffix.length > 0 ? 76 : 62
        implicitHeight: 30
        text: Number(root.value).toFixed(root.decimals)
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
}
