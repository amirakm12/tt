import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#0a0a0a"
    border.color: "#00ffff"
    border.width: 2
    radius: 10
    clip: true
    
    property var thoughts: []
    property real entropy: 0.5
    
    // Header
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 40
        color: "#001122"
        radius: parent.radius
        
        Text {
            anchors.centerIn: parent
            text: "ATHENA CONSCIOUSNESS STREAM"
            color: "#00ffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 14
            font.bold: true
        }
        
        // Entropy indicator
        Rectangle {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            anchors.rightMargin: 10
            width: 60
            height: 20
            color: "transparent"
            border.color: "#00ffff"
            border.width: 1
            radius: 3
            
            Rectangle {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.margins: 2
                width: (parent.width - 4) * root.entropy
                color: Qt.rgba(1.0, 1.0 - root.entropy, 0, 0.8)
                radius: 2
                
                Behavior on width {
                    NumberAnimation { duration: 200 }
                }
            }
            
            Text {
                anchors.centerIn: parent
                text: Math.round(root.entropy * 100) + "%"
                color: "#ffffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
        }
    }
    
    // Thought stream
    ListView {
        id: thoughtList
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 10
        spacing: 5
        clip: true
        
        model: ListModel {
            id: thoughtModel
        }
        
        delegate: Rectangle {
            width: thoughtList.width
            height: thoughtText.height + 20
            color: "#001a1a"
            border.color: "#006666"
            border.width: 1
            radius: 5
            opacity: 1.0 - (index * 0.1)
            
            // Thought indicator
            Rectangle {
                id: indicator
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 10
                width: 4
                height: parent.height - 10
                color: "#00ffff"
                radius: 2
                
                SequentialAnimation on opacity {
                    running: index === 0
                    loops: 3
                    NumberAnimation { to: 0.3; duration: 200 }
                    NumberAnimation { to: 1.0; duration: 200 }
                }
            }
            
            Text {
                id: thoughtText
                anchors.left: indicator.right
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                text: model.text
                color: "#ffffff"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 12
                wrapMode: Text.WordWrap
                
                // Typewriter effect for new thoughts
                property int displayLength: index === 0 ? 0 : text.length
                text: model.text.substring(0, displayLength)
                
                NumberAnimation on displayLength {
                    from: 0
                    to: model.text.length
                    duration: model.text.length * 20
                    running: index === 0
                }
            }
            
            // Timestamp
            Text {
                anchors.bottom: parent.bottom
                anchors.right: parent.right
                anchors.margins: 5
                text: new Date(model.timestamp).toLocaleTimeString()
                color: "#666666"
                font.family: "Consolas, Monaco, monospace"
                font.pixelSize: 10
            }
        }
        
        // Auto-scroll to new thoughts
        onCountChanged: {
            positionViewAtEnd()
        }
    }
    
    // Neural activity background
    Canvas {
        anchors.fill: parent
        z: -1
        opacity: 0.1
        
        property real phase: 0
        NumberAnimation on phase {
            from: 0
            to: Math.PI * 2
            duration: 10000
            loops: Animation.Infinite
        }
        
        onPhaseChanged: requestPaint()
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            
            ctx.strokeStyle = "#00ffff"
            ctx.lineWidth = 1
            
            // Draw neural pathways
            for (var i = 0; i < 5; i++) {
                ctx.beginPath()
                var y = height * (i + 1) / 6
                ctx.moveTo(0, y)
                
                for (var x = 0; x < width; x += 5) {
                    var wave = Math.sin((x / 50) + phase + (i * 0.5)) * 20
                    ctx.lineTo(x, y + wave)
                }
                
                ctx.stroke()
            }
        }
    }
    
    // Functions
    function addThought(thought) {
        thoughtModel.insert(0, {
            text: thought,
            timestamp: Date.now()
        })
        
        // Keep only recent thoughts
        while (thoughtModel.count > 20) {
            thoughtModel.remove(thoughtModel.count - 1)
        }
        
        // Update entropy based on thought complexity
        root.entropy = Math.min(0.9, Math.max(0.1, thought.length / 100))
    }
}