import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#0a0a0a88"
    border.color: "#00ffff"
    border.width: 1
    radius: 10
    
    property real cpuUsage: 0.0
    property real memoryUsage: 0.0
    property real gpuUsage: 0.0
    property int activeAgents: 0
    property string systemMode: "NORMAL"
    
    Column {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 5
        
        // Header
        Text {
            text: "SYSTEM STATUS"
            color: "#00ffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 12
            font.bold: true
        }
        
        // System mode indicator
        Rectangle {
            width: parent.width
            height: 20
            color: systemMode === "NORMAL" ? "#002200" : 
                   systemMode === "BOOST" ? "#222200" : "#220000"
            border.color: systemMode === "NORMAL" ? "#00ff00" : 
                         systemMode === "BOOST" ? "#ffff00" : "#ff0000"
            border.width: 1
            radius: 3
            
            Text {
                anchors.centerIn: parent
                text: systemMode
                color: parent.border.color
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                font.bold: true
            }
        }
        
        // Resource monitors
        ResourceBar {
            label: "CPU"
            value: cpuUsage
            color: "#00ff00"
        }
        
        ResourceBar {
            label: "MEM"
            value: memoryUsage
            color: "#0088ff"
        }
        
        ResourceBar {
            label: "GPU"
            value: gpuUsage
            color: "#ff6600"
        }
        
        // Agent count
        Row {
            spacing: 5
            
            Text {
                text: "AGENTS:"
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
            
            Text {
                text: activeAgents.toString()
                color: "#00ffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                font.bold: true
            }
        }
    }
    
    // Component for resource bars
    component ResourceBar : Item {
        property string label: ""
        property real value: 0.0
        property color color: "#00ffff"
        
        width: parent.width
        height: 15
        
        Row {
            anchors.fill: parent
            spacing: 5
            
            Text {
                width: 30
                text: parent.parent.label
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
            
            Rectangle {
                width: parent.width - 70
                height: 10
                anchors.verticalCenter: parent.verticalCenter
                color: "#111111"
                border.color: "#333333"
                border.width: 1
                radius: 2
                
                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    anchors.margins: 1
                    width: (parent.width - 2) * parent.parent.parent.value
                    color: parent.parent.parent.color
                    radius: 1
                    
                    Behavior on width {
                        NumberAnimation { duration: 200 }
                    }
                }
            }
            
            Text {
                width: 35
                text: Math.round(parent.parent.value * 100) + "%"
                color: parent.parent.color
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                horizontalAlignment: Text.AlignRight
            }
        }
    }
    
    // Simulate resource usage
    Timer {
        interval: 1000
        running: true
        repeat: true
        onTriggered: {
            cpuUsage = 0.3 + Math.random() * 0.4
            memoryUsage = 0.5 + Math.random() * 0.3
            gpuUsage = 0.4 + Math.random() * 0.5
            activeAgents = Math.floor(3 + Math.random() * 3)
        }
    }
}