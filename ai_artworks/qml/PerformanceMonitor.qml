import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#0a0a0a88"
    border.color: "#00ffff"
    border.width: 1
    radius: 10
    
    property real fps: 60
    property real frameTime: 16.67
    property real latency: 0
    property int droppedFrames: 0
    
    Column {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 3
        
        // FPS Display
        Row {
            spacing: 5
            
            Text {
                text: "FPS:"
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
            
            Text {
                text: Math.round(fps).toString()
                color: fps >= 50 ? "#00ff00" : fps >= 30 ? "#ffff00" : "#ff0000"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 12
                font.bold: true
            }
        }
        
        // Frame time
        Row {
            spacing: 5
            
            Text {
                text: "Frame:"
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
            
            Text {
                text: frameTime.toFixed(1) + "ms"
                color: frameTime <= 20 ? "#00ff00" : frameTime <= 33 ? "#ffff00" : "#ff0000"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
        }
        
        // Network latency
        Row {
            spacing: 5
            
            Text {
                text: "Latency:"
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
            
            Text {
                text: latency.toFixed(0) + "ms"
                color: latency <= 50 ? "#00ff00" : latency <= 150 ? "#ffff00" : "#ff0000"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
        }
        
        // Dropped frames indicator
        Rectangle {
            width: parent.width
            height: 3
            color: droppedFrames > 0 ? "#ff0000" : "#00ff00"
            opacity: droppedFrames > 0 ? 0.8 : 0.3
            
            Behavior on color {
                ColorAnimation { duration: 200 }
            }
        }
    }
    
    // FPS Graph
    Canvas {
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 5
        height: 30
        
        property var fpsHistory: []
        property int maxSamples: 50
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            
            if (fpsHistory.length < 2) return
            
            // Draw graph
            ctx.strokeStyle = "#00ffff"
            ctx.lineWidth = 1
            ctx.globalAlpha = 0.5
            
            ctx.beginPath()
            for (var i = 0; i < fpsHistory.length; i++) {
                var x = (i / maxSamples) * width
                var y = height - (fpsHistory[i] / 120) * height
                
                if (i === 0) {
                    ctx.moveTo(x, y)
                } else {
                    ctx.lineTo(x, y)
                }
            }
            ctx.stroke()
            
            // Draw 60 FPS line
            ctx.strokeStyle = "#00ff00"
            ctx.globalAlpha = 0.3
            ctx.setLineDash([5, 5])
            ctx.beginPath()
            ctx.moveTo(0, height - (60 / 120) * height)
            ctx.lineTo(width, height - (60 / 120) * height)
            ctx.stroke()
        }
        
        function addSample(value) {
            fpsHistory.push(value)
            if (fpsHistory.length > maxSamples) {
                fpsHistory.shift()
            }
            requestPaint()
        }
    }
    
    // Performance tracking
    property real lastFrameTime: 0
    property var frameTimes: []
    
    Timer {
        id: fpsTimer
        interval: 16
        running: true
        repeat: true
        
        onTriggered: {
            var currentTime = Date.now()
            if (lastFrameTime > 0) {
                var delta = currentTime - lastFrameTime
                frameTimes.push(delta)
                
                // Keep last 60 samples
                if (frameTimes.length > 60) {
                    frameTimes.shift()
                }
                
                // Calculate average
                var sum = 0
                for (var i = 0; i < frameTimes.length; i++) {
                    sum += frameTimes[i]
                }
                frameTime = sum / frameTimes.length
                fps = 1000 / frameTime
                
                // Detect dropped frames
                if (delta > 33) {
                    droppedFrames++
                }
            }
            lastFrameTime = currentTime
            
            // Update graph
            parent.children[1].addSample(fps)
        }
    }
    
    // Simulate network latency
    Timer {
        interval: 1000
        running: true
        repeat: true
        onTriggered: {
            latency = 20 + Math.random() * 80
        }
    }
    
    // Reset dropped frames counter
    Timer {
        interval: 5000
        running: true
        repeat: true
        onTriggered: {
            droppedFrames = 0
        }
    }
}