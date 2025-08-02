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
    
    property var tasks: []
    property real currentTime: 0
    
    // Timeline header
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 30
        color: "#001122"
        
        Text {
            anchors.centerIn: parent
            text: "EXECUTION TIMELINE - TACTICAL VIEW"
            color: "#00ffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 14
            font.bold: true
        }
        
        // Time indicator
        Text {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.rightMargin: 10
            text: formatTime(currentTime)
            color: "#00ff00"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 12
            
            function formatTime(seconds) {
                var mins = Math.floor(seconds / 60)
                var secs = Math.floor(seconds % 60)
                return mins.toString().padStart(2, '0') + ":" + secs.toString().padStart(2, '0')
            }
        }
    }
    
    // Main timeline view
    Item {
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 10
        
        // Grid background
        Canvas {
            anchors.fill: parent
            opacity: 0.2
            
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                
                ctx.strokeStyle = "#003333"
                ctx.lineWidth = 1
                
                // Vertical lines (time markers)
                for (var x = 0; x < width; x += 50) {
                    ctx.beginPath()
                    ctx.moveTo(x, 0)
                    ctx.lineTo(x, height)
                    ctx.stroke()
                }
                
                // Horizontal lines (agent lanes)
                for (var y = 0; y < height; y += 30) {
                    ctx.beginPath()
                    ctx.moveTo(0, y)
                    ctx.lineTo(width, y)
                    ctx.stroke()
                }
            }
        }
        
        // Agent lanes
        Column {
            anchors.fill: parent
            spacing: 5
            
            Repeater {
                model: ["RenderOps", "DataDaemon", "SecSentinel", "VoiceNav", "Autopilot"]
                
                Rectangle {
                    width: parent.width
                    height: 25
                    color: "transparent"
                    
                    // Agent label
                    Rectangle {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        width: 100
                        height: 20
                        color: "#001122"
                        border.color: getAgentColor(modelData)
                        border.width: 1
                        radius: 3
                        
                        Text {
                            anchors.centerIn: parent
                            text: modelData
                            color: getAgentColor(modelData)
                            font.family: "Consolas, Monaco, monospace"
                            font.pixelSize: 10
                            font.bold: true
                        }
                    }
                    
                    // Task bars
                    Item {
                        anchors.left: parent.left
                        anchors.leftMargin: 110
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        
                        Repeater {
                            model: getAgentTasks(modelData)
                            
                            Rectangle {
                                x: modelData.startTime * 10
                                y: 2
                                width: (modelData.endTime - modelData.startTime) * 10
                                height: parent.height - 4
                                color: getAgentColor(parent.parent.modelData)
                                opacity: modelData.status === "completed" ? 0.5 : 0.8
                                radius: 3
                                
                                // Progress indicator
                                Rectangle {
                                    anchors.left: parent.left
                                    anchors.top: parent.top
                                    anchors.bottom: parent.bottom
                                    width: parent.width * modelData.progress
                                    color: Qt.lighter(parent.color, 1.5)
                                    radius: parent.radius
                                }
                                
                                // Task name
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.name
                                    color: "#000000"
                                    font.family: "Consolas, Monaco, monospace"
                                    font.pixelSize: 9
                                    font.bold: true
                                }
                                
                                // Status indicator
                                Rectangle {
                                    anchors.right: parent.right
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.rightMargin: 5
                                    width: 6
                                    height: 6
                                    radius: 3
                                    color: modelData.status === "running" ? "#00ff00" :
                                          modelData.status === "completed" ? "#0088ff" :
                                          modelData.status === "failed" ? "#ff0000" : "#ffff00"
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Current time indicator
        Rectangle {
            x: currentTime * 10
            y: 0
            width: 2
            height: parent.height
            color: "#ff0000"
            opacity: 0.8
            
            Rectangle {
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                width: 20
                height: 10
                color: "#ff0000"
                rotation: 180
                transformOrigin: Item.Bottom
                
                Shape {
                    anchors.fill: parent
                    ShapePath {
                        fillColor: "#ff0000"
                        PathMove { x: 0; y: 0 }
                        PathLine { x: 10; y: 10 }
                        PathLine { x: 20; y: 0 }
                        PathLine { x: 0; y: 0 }
                    }
                }
            }
            
            SequentialAnimation on opacity {
                loops: Animation.Infinite
                NumberAnimation { to: 0.3; duration: 500 }
                NumberAnimation { to: 0.8; duration: 500 }
            }
        }
        
        // Dependency connections
        Canvas {
            id: dependencyCanvas
            anchors.fill: parent
            z: -1
            
            property var dependencies: []
            
            onPaint: {
                var ctx = getContext("2d")
                ctx.clearRect(0, 0, width, height)
                
                ctx.strokeStyle = "#00ffff"
                ctx.lineWidth = 1
                ctx.setLineDash([5, 5])
                ctx.globalAlpha = 0.3
                
                // Draw dependency arrows
                for (var i = 0; i < dependencies.length; i++) {
                    var dep = dependencies[i]
                    ctx.beginPath()
                    ctx.moveTo(dep.fromX, dep.fromY)
                    ctx.bezierCurveTo(
                        dep.fromX + 50, dep.fromY,
                        dep.toX - 50, dep.toY,
                        dep.toX, dep.toY
                    )
                    ctx.stroke()
                }
            }
        }
    }
    
    // Status bar
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        height: 25
        color: "#001122"
        
        Row {
            anchors.centerIn: parent
            spacing: 20
            
            StatusIndicator {
                label: "ACTIVE"
                value: getActiveTaskCount()
                color: "#00ff00"
            }
            
            StatusIndicator {
                label: "QUEUED"
                value: getQueuedTaskCount()
                color: "#ffff00"
            }
            
            StatusIndicator {
                label: "COMPLETED"
                value: getCompletedTaskCount()
                color: "#0088ff"
            }
            
            StatusIndicator {
                label: "FAILED"
                value: getFailedTaskCount()
                color: "#ff0000"
            }
        }
    }
    
    // Component for status indicators
    component StatusIndicator : Row {
        property string label: ""
        property int value: 0
        property color color: "#00ffff"
        
        spacing: 5
        
        Text {
            text: label + ":"
            color: "#666666"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 10
        }
        
        Text {
            text: value.toString()
            color: parent.color
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 10
            font.bold: true
        }
    }
    
    // Functions
    function getAgentColor(agent) {
        switch(agent) {
            case "RenderOps": return "#ff6600"
            case "DataDaemon": return "#00ff66"
            case "SecSentinel": return "#ff0066"
            case "VoiceNav": return "#6600ff"
            case "Autopilot": return "#ffff00"
            default: return "#ffffff"
        }
    }
    
    function getAgentTasks(agent) {
        // Filter tasks by agent
        return tasks.filter(function(task) {
            return task.agent === agent
        })
    }
    
    function getActiveTaskCount() {
        return tasks.filter(function(task) {
            return task.status === "running"
        }).length
    }
    
    function getQueuedTaskCount() {
        return tasks.filter(function(task) {
            return task.status === "queued"
        }).length
    }
    
    function getCompletedTaskCount() {
        return tasks.filter(function(task) {
            return task.status === "completed"
        }).length
    }
    
    function getFailedTaskCount() {
        return tasks.filter(function(task) {
            return task.status === "failed"
        }).length
    }
    
    // Time progression
    Timer {
        interval: 100
        running: true
        repeat: true
        onTriggered: {
            currentTime += 0.1
            
            // Update task progress
            for (var i = 0; i < tasks.length; i++) {
                if (tasks[i].status === "running") {
                    tasks[i].progress = Math.min(1.0, tasks[i].progress + 0.01)
                    if (tasks[i].progress >= 1.0) {
                        tasks[i].status = "completed"
                    }
                }
            }
        }
    }
}