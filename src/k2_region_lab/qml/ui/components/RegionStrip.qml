import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root

    required property QtObject controller
    color: "#111621"
    border.color: "#262d3c"
    radius: 10

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Text {
            Layout.fillWidth: true
            text: root.controller.mode === "face"
                  ? "Detected faces"
                  : (root.controller.mode === "edit"
                     && root.controller.editLayer === "reference"
                     ? "Reference regions" : "Regions")
            color: "#eef1f7"
            font.pixelSize: 12
            font.weight: Font.DemiBold
            elide: Text.ElideRight
        }

        Text {
            Layout.fillWidth: true
            text: "Click a name, then drag its interior or edges."
            color: "#737d91"
            wrapMode: Text.Wrap
            font.pixelSize: 10
        }

        ScrollView {
            id: regionScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ColumnLayout {
                width: regionScroll.availableWidth
                spacing: 6

                Repeater {
                    model: root.controller.mode === "face" ? []
                                                               : root.controller.activeRegionModel
                    delegate: Rectangle {
                        id: regionEntry
                        required property string regionId
                        required property string name
                        required property bool regionEnabled
                        Layout.fillWidth: true
                        implicitHeight: 40
                        radius: 7
                        color: root.controller.selectedRegionId === regionId
                               ? "#29334c" : "#181e2a"
                        border.color: root.controller.selectedRegionId === regionId
                                      ? "#7c8cff" : "#2b3344"
                        opacity: regionEnabled ? 1 : 0.55

                        Text {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.margins: 10
                            text: regionEntry.name
                            color: "#e5e8f1"
                            font.pixelSize: 11
                            font.weight: Font.DemiBold
                            elide: Text.ElideRight
                        }
                        TapHandler {
                            onTapped: root.controller.selectRegion(regionEntry.regionId)
                        }
                    }
                }

                Repeater {
                    model: root.controller.mode === "face" ? root.controller.faces : []
                    delegate: Rectangle {
                        id: faceEntry
                        required property var modelData
                        Layout.fillWidth: true
                        implicitHeight: 46
                        radius: 7
                        color: modelData.selected ? "#234039" : "#241d1c"
                        border.color: modelData.selected ? "#56d3b2" : "#ff9d67"

                        Column {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.margins: 9
                            spacing: 2
                            Text {
                                width: parent.width
                                text: faceEntry.modelData.label
                                color: "#e5e8f1"
                                font.pixelSize: 11
                                font.weight: Font.DemiBold
                            }
                            Text {
                                width: parent.width
                                text: faceEntry.modelData.regionName
                                color: "#7f899d"
                                font.pixelSize: 9
                                elide: Text.ElideRight
                            }
                        }
                        TapHandler {
                            onTapped: root.controller.setFaceSelected(
                                          faceEntry.modelData.index,
                                          !faceEntry.modelData.selected)
                        }
                    }
                }
            }
        }
    }
}
