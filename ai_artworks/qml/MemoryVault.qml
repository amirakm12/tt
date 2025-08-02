import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

Rectangle {
    id: root
    color: "#0a0a0a"
    border.color: "#00ffff"
    border.width: 2
    radius: 10
    clip: true
    
    property bool locked: true
    property var memories: []
    property string accessCode: ""
    
    // Vault door effect
    Rectangle {
        id: vaultDoor
        anchors.fill: parent
        color: "#001122"
        radius: parent.radius
        visible: locked
        
        // Lock mechanism
        Item {
            anchors.centerIn: parent
            width: 100
            height: 100
            
            // Rotating lock rings
            Repeater {
                model: 3
                
                Rectangle {
                    anchors.centerIn: parent
                    width: 80 - index * 20
                    height: 80 - index * 20
                    color: "transparent"
                    border.color: "#00ffff"
                    border.width: 2
                    radius: width / 2
                    
                    NumberAnimation on rotation {
                        from: 0
                        to: 360 * (index % 2 === 0 ? 1 : -1)
                        duration: 5000 + index * 1000
                        loops: Animation.Infinite
                    }
                    
                    // Lock segments
                    Repeater {
                        model: 8
                        
                        Rectangle {
                            x: parent.width / 2 + Math.cos(index * Math.PI / 4) * (parent.width / 2 - 5)
                            y: parent.height / 2 + Math.sin(index * Math.PI / 4) * (parent.height / 2 - 5)
                            width: 6
                            height: 6
                            color: "#00ffff"
                            radius: 3
                        }
                    }
                }
            }
            
            // Central lock icon
            Text {
                anchors.centerIn: parent
                text: "🔒"
                font.pixelSize: 30
                color: "#00ffff"
                
                SequentialAnimation on opacity {
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.5; duration: 1000 }
                    NumberAnimation { to: 1.0; duration: 1000 }
                }
            }
        }
        
        // Access panel
        Column {
            anchors.bottom: parent.bottom
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottomMargin: 20
            spacing: 10
            
            Text {
                text: "MEMORY VAULT LOCKED"
                color: "#ff0000"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 12
                font.bold: true
                horizontalAlignment: Text.AlignHCenter
            }
            
            // Voice command hint
            Text {
                text: "Say 'unlock memory vault' to access"
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                horizontalAlignment: Text.AlignHCenter
            }
        }
        
        // Unlock animation
        ParallelAnimation {
            id: unlockAnimation
            
            NumberAnimation {
                target: vaultDoor
                property: "opacity"
                to: 0
                duration: 1000
                easing.type: Easing.InOutQuad
            }
            
            NumberAnimation {
                target: vaultDoor
                property: "scale"
                to: 0.8
                duration: 1000
                easing.type: Easing.InOutQuad
            }
            
            onFinished: {
                vaultDoor.visible = false
                locked = false
            }
        }
    }
    
    // Memory grid (visible when unlocked)
    GridView {
        id: memoryGrid
        anchors.fill: parent
        anchors.margins: 10
        cellWidth: 80
        cellHeight: 80
        visible: !locked
        opacity: locked ? 0 : 1
        
        Behavior on opacity {
            NumberAnimation { duration: 500 }
        }
        
        model: ListModel {
            id: memoryModel
            
            Component.onCompleted: {
                // Add sample memories
                for (var i = 0; i < 12; i++) {
                    append({
                        id: i,
                        type: ["conversation", "image", "task", "learning"][i % 4],
                        timestamp: Date.now() - Math.random() * 86400000,
                        encrypted: Math.random() > 0.7
                    })
                }
            }
        }
        
        delegate: Rectangle {
            width: 70
            height: 70
            color: model.encrypted ? "#220000" : "#002222"
            border.color: getMemoryColor(model.type)
            border.width: 2
            radius: 10
            
            // Memory icon
            Text {
                anchors.centerIn: parent
                text: getMemoryIcon(model.type)
                font.pixelSize: 30
                color: getMemoryColor(model.type)
                opacity: model.encrypted ? 0.3 : 1.0
            }
            
            // Encryption overlay
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: "#ff0000"
                border.width: 1
                radius: parent.radius
                visible: model.encrypted
                
                Text {
                    anchors.centerIn: parent
                    text: "🔐"
                    font.pixelSize: 20
                    color: "#ff0000"
                }
            }
            
            // Hover effect
            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                
                onEntered: {
                    parent.scale = 1.1
                    if (!model.encrypted) {
                        memoryPreview.showMemory(model)
                    }
                }
                
                onExited: {
                    parent.scale = 1.0
                }
                
                onClicked: {
                    if (!model.encrypted) {
                        // Access memory
                        console.log("Accessing memory:", model.id)
                    }
                }
            }
            
            Behavior on scale {
                NumberAnimation { duration: 200 }
            }
            
            // Pulse animation for recent memories
            SequentialAnimation on opacity {
                running: (Date.now() - model.timestamp) < 3600000 // Less than 1 hour old
                loops: Animation.Infinite
                NumberAnimation { to: 0.6; duration: 1000 }
                NumberAnimation { to: 1.0; duration: 1000 }
            }
        }
    }
    
    // Memory preview
    Rectangle {
        id: memoryPreview
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        height: 0
        color: "#001122"
        border.color: "#00ffff"
        border.width: 1
        visible: height > 0
        
        property var currentMemory: null
        
        Behavior on height {
            NumberAnimation { duration: 200 }
        }
        
        Column {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 5
            
            Text {
                text: memoryPreview.currentMemory ? 
                      "Memory Type: " + memoryPreview.currentMemory.type.toUpperCase() : ""
                color: "#00ffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                font.bold: true
            }
            
            Text {
                text: memoryPreview.currentMemory ? 
                      "Timestamp: " + new Date(memoryPreview.currentMemory.timestamp).toLocaleString() : ""
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 9
            }
        }
        
        function showMemory(memory) {
            currentMemory = memory
            height = 60
        }
        
        Timer {
            interval: 3000
            running: memoryPreview.visible
            onTriggered: {
                memoryPreview.height = 0
            }
        }
    }
    
    // Functions
    function getMemoryColor(type) {
        switch(type) {
            case "conversation": return "#00ff00"
            case "image": return "#ff6600"
            case "task": return "#ffff00"
            case "learning": return "#ff00ff"
            default: return "#ffffff"
        }
    }
    
    function getMemoryIcon(type) {
        switch(type) {
            case "conversation": return "💬"
            case "image": return "🖼️"
            case "task": return "📋"
            case "learning": return "🧠"
            default: return "📄"
        }
    }
    
    function unlock(code) {
        if (code === "neural" || code === "") { // Simple unlock for demo
            unlockAnimation.start()
        }
    }
    
    // Voice command connection
    Connections {
        target: voiceHUD
        function onCommandRecognized(command) {
            if (command.toLowerCase().includes("unlock memory vault")) {
                unlock("")
            }
        }
    }
}