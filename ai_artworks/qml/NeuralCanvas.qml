import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

Rectangle {
    id: root
    color: "transparent"
    
    property var pulses: []
    property real renderProgress: 0.0
    property string currentTask: ""
    
    // Glass effect background
    Rectangle {
        anchors.fill: parent
        color: "#001122"
        opacity: 0.3
        radius: 20
        
        border.color: "#00ffff"
        border.width: 1
    }
    
    // Neural network visualization
    Canvas {
        id: neuralCanvas
        anchors.fill: parent
        anchors.margins: 20
        
        property real time: 0
        NumberAnimation on time {
            from: 0
            to: 1
            duration: 5000
            loops: Animation.Infinite
        }
        
        onTimeChanged: requestPaint()
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            
            // Draw neural connections
            ctx.strokeStyle = "#006666"
            ctx.lineWidth = 0.5
            ctx.globalAlpha = 0.3
            
            // Grid of nodes
            var nodeSize = 40
            var nodesX = Math.floor(width / nodeSize)
            var nodesY = Math.floor(height / nodeSize)
            
            for (var x = 0; x < nodesX; x++) {
                for (var y = 0; y < nodesY; y++) {
                    var posX = x * nodeSize + nodeSize/2
                    var posY = y * nodeSize + nodeSize/2
                    
                    // Connect to neighbors with probability
                    if (x < nodesX - 1 && Math.random() > 0.5) {
                        ctx.beginPath()
                        ctx.moveTo(posX, posY)
                        ctx.lineTo(posX + nodeSize, posY)
                        ctx.stroke()
                    }
                    
                    if (y < nodesY - 1 && Math.random() > 0.5) {
                        ctx.beginPath()
                        ctx.moveTo(posX, posY)
                        ctx.lineTo(posX, posY + nodeSize)
                        ctx.stroke()
                    }
                }
            }
            
            // Draw active pulses
            ctx.globalAlpha = 1.0
            for (var i = 0; i < pulses.length; i++) {
                var pulse = pulses[i]
                var age = (Date.now() - pulse.timestamp) / 1000
                
                if (age < 2) {
                    var radius = age * 50
                    var opacity = 1 - (age / 2)
                    
                    ctx.beginPath()
                    ctx.arc(pulse.x * width, pulse.y * height, radius, 0, Math.PI * 2)
                    ctx.strokeStyle = "#00ffff"
                    ctx.lineWidth = 3 * opacity
                    ctx.globalAlpha = opacity
                    ctx.stroke()
                }
            }
        }
    }
    
    // Central display area
    Item {
        anchors.centerIn: parent
        width: 400
        height: 300
        
        // Task indicator
        Rectangle {
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            width: 300
            height: 30
            color: "#001122"
            border.color: "#00ffff"
            border.width: 1
            radius: 15
            
            Text {
                anchors.centerIn: parent
                text: currentTask || "NEURAL NETWORK IDLE"
                color: "#00ffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 12
                font.bold: true
            }
        }
        
        // Render preview area
        Rectangle {
            anchors.centerIn: parent
            width: 300
            height: 200
            color: "#000000"
            border.color: "#00ffff"
            border.width: 2
            radius: 10
            
            // Scanning effect
            Rectangle {
                width: parent.width
                height: 2
                color: "#00ffff"
                opacity: 0.8
                
                SequentialAnimation on y {
                    loops: Animation.Infinite
                    NumberAnimation {
                        from: 0
                        to: parent.height
                        duration: 2000
                    }
                    PauseAnimation { duration: 500 }
                }
            }
            
            // Progress indicator
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: 4
                color: "#003333"
                
                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: parent.width * renderProgress
                    color: "#00ffff"
                    
                    Behavior on width {
                        NumberAnimation { duration: 200 }
                    }
                }
            }
        }
        
        // Status text
        Column {
            anchors.bottom: parent.bottom
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 5
            
            Text {
                text: "NEURAL PROCESSING: " + Math.round(renderProgress * 100) + "%"
                color: "#00ffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                horizontalAlignment: Text.AlignHCenter
            }
            
            Text {
                text: "ACTIVE NODES: " + pulses.length
                color: "#006666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
                horizontalAlignment: Text.AlignHCenter
            }
        }
    }
    
    // Particle effects
    Repeater {
        model: 20
        
        Rectangle {
            x: Math.random() * root.width
            y: Math.random() * root.height
            width: 2
            height: 2
            color: "#00ffff"
            opacity: 0.5
            
            SequentialAnimation on opacity {
                loops: Animation.Infinite
                NumberAnimation {
                    to: 0
                    duration: Math.random() * 2000 + 1000
                }
                NumberAnimation {
                    to: 0.5
                    duration: Math.random() * 2000 + 1000
                }
            }
            
            NumberAnimation on y {
                from: y
                to: y - 50
                duration: 3000
                loops: Animation.Infinite
            }
        }
    }
    
    // Functions
    function addPulse(x, y, intensity) {
        var pulse = {
            x: x,
            y: y,
            intensity: intensity,
            timestamp: Date.now()
        }
        
        pulses.push(pulse)
        
        // Keep only recent pulses
        pulses = pulses.filter(function(p) {
            return (Date.now() - p.timestamp) < 2000
        })
        
        // Update render progress simulation
        renderProgress = Math.min(1.0, renderProgress + 0.05)
        
        // Trigger canvas repaint
        neuralCanvas.requestPaint()
    }
    
    function setTask(task) {
        currentTask = task
        renderProgress = 0
    }
    
    // Cleanup timer
    Timer {
        interval: 100
        running: true
        repeat: true
        onTriggered: {
            // Remove old pulses
            pulses = pulses.filter(function(p) {
                return (Date.now() - p.timestamp) < 2000
            })
            neuralCanvas.requestPaint()
        }
    }
}