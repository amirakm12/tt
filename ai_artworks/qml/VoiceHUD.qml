import QtQuick
import QtQuick.Controls
import QtQuick.Shapes

Rectangle {
    id: root
    color: "transparent"
    border.color: "#00ffff"
    border.width: 2
    radius: 10
    
    property bool listening: false
    property var waveformData: []
    property string currentCommand: ""
    property string decodedText: ""
    
    // Glow effect
    Rectangle {
        anchors.fill: parent
        anchors.margins: -10
        color: "transparent"
        radius: parent.radius + 5
        
        layer.enabled: true
        layer.effect: ShaderEffect {
            property real glowIntensity: listening ? 0.8 : 0.3
            Behavior on glowIntensity { NumberAnimation { duration: 200 } }
            
            fragmentShader: "
                #version 440
                layout(location = 0) in vec2 qt_TexCoord0;
                layout(location = 0) out vec4 fragColor;
                layout(std140, binding = 0) uniform buf {
                    mat4 qt_Matrix;
                    float qt_Opacity;
                    float glowIntensity;
                };
                
                void main() {
                    vec2 uv = qt_TexCoord0 - 0.5;
                    float dist = length(uv);
                    float glow = exp(-dist * 5.0) * glowIntensity;
                    fragColor = vec4(0.0, 1.0, 1.0, glow * qt_Opacity);
                }
            "
        }
    }
    
    // Voice waveform
    Item {
        id: waveformContainer
        anchors.fill: parent
        anchors.margins: 20
        
        Shape {
            id: waveformShape
            anchors.fill: parent
            
            ShapePath {
                id: waveformPath
                strokeColor: listening ? "#00ffff" : "#006666"
                strokeWidth: 2
                fillColor: "transparent"
                
                PathSvg {
                    id: wavePath
                    path: generateWaveformPath()
                }
            }
        }
        
        // Animated pulse when listening
        Rectangle {
            anchors.centerIn: parent
            width: listening ? parent.width : 0
            height: 2
            color: "#00ffff"
            opacity: 0.5
            
            Behavior on width {
                NumberAnimation {
                    duration: 1000
                    easing.type: Easing.OutElastic
                }
            }
        }
    }
    
    // Command text display
    Column {
        anchors.centerIn: parent
        spacing: 10
        
        Text {
            text: listening ? "LISTENING..." : "PRESS SPACE TO SPEAK"
            color: "#00ffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 14
            font.bold: true
            horizontalAlignment: Text.AlignHCenter
            
            SequentialAnimation on opacity {
                running: listening
                loops: Animation.Infinite
                NumberAnimation { to: 0.3; duration: 500 }
                NumberAnimation { to: 1.0; duration: 500 }
            }
        }
        
        Text {
            text: decodedText
            color: "#ffffff"
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 18
            horizontalAlignment: Text.AlignHCenter
            visible: decodedText.length > 0
            
            // Typewriter effect
            property int displayLength: 0
            text: decodedText.substring(0, displayLength)
            
            NumberAnimation on displayLength {
                from: 0
                to: decodedText.length
                duration: decodedText.length * 50
                running: decodedText.length > 0
            }
        }
    }
    
    // Intent indicator
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 10
        width: parent.width - 40
        height: 4
        color: "transparent"
        border.color: "#00ffff"
        border.width: 1
        radius: 2
        
        Rectangle {
            id: intentProgress
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 0
            color: "#00ffff"
            radius: parent.radius
            
            SequentialAnimation on width {
                running: decodedText.length > 0
                NumberAnimation {
                    to: parent.width
                    duration: 1000
                    easing.type: Easing.OutQuad
                }
                PauseAnimation { duration: 500 }
                NumberAnimation {
                    to: 0
                    duration: 200
                }
            }
        }
    }
    
    // Functions
    function addCommand(command) {
        currentCommand = command
        decodedText = command
    }
    
    function updateWaveform(data) {
        waveformData = data
        wavePath.path = generateWaveformPath()
    }
    
    function generateWaveformPath() {
        if (waveformData.length === 0) {
            return "M 0," + (waveformContainer.height / 2) + " L " + waveformContainer.width + "," + (waveformContainer.height / 2)
        }
        
        var path = "M 0," + (waveformContainer.height / 2)
        var step = waveformContainer.width / waveformData.length
        
        for (var i = 0; i < waveformData.length; i++) {
            var x = i * step
            var y = (waveformContainer.height / 2) + (waveformData[i] * waveformContainer.height * 0.4)
            path += " L " + x + "," + y
        }
        
        return path
    }
}