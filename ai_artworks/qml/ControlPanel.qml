import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    color: "#0a0a0a"
    border.color: "#00ffff"
    border.width: 2
    radius: 10
    
    // Holographic effect
    Rectangle {
        anchors.fill: parent
        anchors.margins: 2
        color: "transparent"
        radius: parent.radius
        
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#00ffff10" }
            GradientStop { position: 0.5; color: "#00ffff05" }
            GradientStop { position: 1.0; color: "#00ffff10" }
        }
    }
    
    Row {
        anchors.centerIn: parent
        spacing: 15
        
        HolographicButton {
            icon: "⚡"
            label: "BOOST"
            color: "#ffff00"
            onClicked: {
                console.log("Boost activated")
                hudController.boostPerformance()
            }
        }
        
        HolographicButton {
            icon: "🛡️"
            label: "SHIELD"
            color: "#00ff00"
            onClicked: {
                console.log("Shield activated")
                hudController.activateShield()
            }
        }
        
        HolographicButton {
            icon: "🎯"
            label: "TARGET"
            color: "#ff0066"
            onClicked: {
                console.log("Targeting mode")
                hudController.enterTargetingMode()
            }
        }
        
        HolographicButton {
            icon: "🔄"
            label: "SYNC"
            color: "#00ffff"
            onClicked: {
                console.log("Synchronizing")
                hudController.synchronizeAgents()
            }
        }
        
        HolographicButton {
            icon: "⚠️"
            label: "ALERT"
            color: "#ff6600"
            onClicked: {
                console.log("Alert mode")
                hudController.toggleAlertMode()
            }
        }
    }
    
    // Component for holographic buttons
    component HolographicButton : Item {
        property string icon: ""
        property string label: ""
        property color color: "#00ffff"
        signal clicked()
        
        width: 60
        height: 80
        
        // Button base
        Rectangle {
            id: buttonBase
            anchors.centerIn: parent
            width: 50
            height: 50
            color: "transparent"
            border.color: parent.color
            border.width: 2
            radius: 25
            
            // Inner glow
            Rectangle {
                anchors.centerIn: parent
                width: parent.width - 10
                height: parent.height - 10
                color: "transparent"
                border.color: parent.parent.color
                border.width: 1
                radius: width / 2
                opacity: 0.5
            }
            
            // Icon
            Text {
                anchors.centerIn: parent
                text: parent.parent.icon
                font.pixelSize: 24
                color: parent.parent.color
            }
            
            // Hover effect
            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                
                onEntered: {
                    buttonBase.scale = 1.1
                    glowEffect.visible = true
                }
                
                onExited: {
                    buttonBase.scale = 1.0
                    glowEffect.visible = false
                }
                
                onClicked: {
                    pulseAnimation.start()
                    parent.parent.clicked()
                }
            }
            
            Behavior on scale {
                NumberAnimation { duration: 200 }
            }
            
            // Pulse animation
            SequentialAnimation {
                id: pulseAnimation
                
                ParallelAnimation {
                    NumberAnimation {
                        target: buttonBase
                        property: "scale"
                        to: 1.2
                        duration: 100
                    }
                    NumberAnimation {
                        target: buttonBase
                        property: "opacity"
                        to: 0.7
                        duration: 100
                    }
                }
                
                ParallelAnimation {
                    NumberAnimation {
                        target: buttonBase
                        property: "scale"
                        to: 1.0
                        duration: 200
                    }
                    NumberAnimation {
                        target: buttonBase
                        property: "opacity"
                        to: 1.0
                        duration: 200
                    }
                }
            }
        }
        
        // Label
        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottom: parent.bottom
            text: parent.label
            color: parent.color
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 9
            font.bold: true
        }
        
        // Glow effect
        Rectangle {
            id: glowEffect
            anchors.centerIn: buttonBase
            width: buttonBase.width + 20
            height: buttonBase.height + 20
            color: "transparent"
            radius: width / 2
            visible: false
            
            Rectangle {
                anchors.fill: parent
                color: "transparent"
                radius: parent.radius
                
                layer.enabled: true
                layer.effect: ShaderEffect {
                    property color glowColor: parent.parent.parent.color
                    
                    fragmentShader: "
                        #version 440
                        layout(location = 0) in vec2 qt_TexCoord0;
                        layout(location = 0) out vec4 fragColor;
                        layout(std140, binding = 0) uniform buf {
                            mat4 qt_Matrix;
                            float qt_Opacity;
                            vec4 glowColor;
                        };
                        
                        void main() {
                            vec2 uv = qt_TexCoord0 - 0.5;
                            float dist = length(uv);
                            float glow = exp(-dist * 10.0);
                            fragColor = vec4(glowColor.rgb, glow * qt_Opacity * 0.5);
                        }
                    "
                }
            }
            
            NumberAnimation on opacity {
                from: 0.5
                to: 1.0
                duration: 500
                loops: Animation.Infinite
                easing.type: Easing.InOutQuad
            }
        }
        
        // Idle animation
        SequentialAnimation on opacity {
            loops: Animation.Infinite
            PauseAnimation { duration: Math.random() * 3000 }
            NumberAnimation { to: 0.8; duration: 1000 }
            NumberAnimation { to: 1.0; duration: 1000 }
        }
    }
    
    // System status indicator
    Rectangle {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 5
        width: 10
        height: 10
        radius: 5
        color: "#00ff00"
        
        SequentialAnimation on color {
            loops: Animation.Infinite
            ColorAnimation { to: "#00ff00"; duration: 2000 }
            ColorAnimation { to: "#00ff88"; duration: 2000 }
        }
    }
}