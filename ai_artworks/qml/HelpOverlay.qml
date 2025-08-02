import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#000000dd"
    visible: false
    
    MouseArea {
        anchors.fill: parent
        onClicked: root.visible = false
    }
    
    Rectangle {
        anchors.centerIn: parent
        width: 800
        height: 600
        color: "#0a0a0a"
        border.color: "#00ffff"
        border.width: 2
        radius: 20
        
        // Header
        Rectangle {
            id: header
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: 60
            color: "#001122"
            radius: parent.radius
            
            Text {
                anchors.centerIn: parent
                text: "NEURAL INTERFACE COMMAND REFERENCE"
                color: "#00ffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 20
                font.bold: true
            }
            
            // Close button
            Text {
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.rightMargin: 20
                text: "✕"
                color: "#ff0000"
                font.pixelSize: 24
                
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.visible = false
                }
            }
        }
        
        // Content
        ScrollView {
            anchors.top: header.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 20
            
            Column {
                width: parent.width
                spacing: 20
                
                // Keyboard shortcuts section
                HelpSection {
                    title: "KEYBOARD SHORTCUTS"
                    items: [
                        { key: "SPACE", description: "Toggle voice listening" },
                        { key: "ESC", description: "Exit fullscreen / Quit" },
                        { key: "F1", description: "Show/hide this help" },
                        { key: "F11", description: "Toggle fullscreen" },
                        { key: "Ctrl+Q", description: "Toggle quantum field overlay" },
                        { key: "Ctrl+M", description: "Unlock memory vault" },
                        { key: "Ctrl+S", description: "Synchronize all agents" },
                        { key: "Ctrl+B", description: "Boost performance mode" }
                    ]
                }
                
                // Voice commands section
                HelpSection {
                    title: "VOICE COMMANDS"
                    items: [
                        { key: "Generate [description]", description: "Create AI artwork" },
                        { key: "Apply style transfer", description: "Transfer artistic style" },
                        { key: "Show agent status", description: "Display agent states" },
                        { key: "Analyze document", description: "Process document with RAG" },
                        { key: "Unlock memory vault", description: "Access stored memories" },
                        { key: "Activate shield", description: "Enable security mode" },
                        { key: "Run diagnostics", description: "System health check" },
                        { key: "Emergency shutdown", description: "Safe system halt" }
                    ]
                }
                
                // Agent commands section
                HelpSection {
                    title: "AGENT CONTROL"
                    items: [
                        { key: "RenderOps", description: "GPU rendering and image generation" },
                        { key: "DataDaemon", description: "Analytics and data processing" },
                        { key: "SecSentinel", description: "Security monitoring and alerts" },
                        { key: "VoiceNav", description: "Voice command processing" },
                        { key: "Autopilot", description: "Autonomous task planning" },
                        { key: "Athena", description: "Central orchestration brain" }
                    ]
                }
                
                // System indicators
                HelpSection {
                    title: "SYSTEM INDICATORS"
                    items: [
                        { key: "🟢 Green", description: "System normal, optimal performance" },
                        { key: "🟡 Yellow", description: "Warning, reduced performance" },
                        { key: "🔴 Red", description: "Critical, immediate attention required" },
                        { key: "🔵 Blue", description: "Processing, operation in progress" },
                        { key: "⚡ Lightning", description: "Boost mode active" },
                        { key: "🛡️ Shield", description: "Security mode enabled" }
                    ]
                }
            }
        }
    }
    
    // Help section component
    component HelpSection : Column {
        property string title: ""
        property var items: []
        
        width: parent.width
        spacing: 10
        
        Text {
            text: parent.title
            color: "#00ffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 16
            font.bold: true
        }
        
        Rectangle {
            width: parent.width
            height: 1
            color: "#00ffff"
            opacity: 0.3
        }
        
        Repeater {
            model: parent.items
            
            Row {
                spacing: 20
                
                Text {
                    width: 150
                    text: modelData.key
                    color: "#ffff00"
                    font.family: "Consolas, Monaco, monospace"
                    font.pixelSize: 12
                    font.bold: true
                }
                
                Text {
                    width: parent.parent.width - 170
                    text: modelData.description
                    color: "#ffffff"
                    font.family: "Consolas, Monaco, monospace"
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}