import QtQuick
import QtQuick.Controls

Item {
    id: root
    
    property string alertMessage: ""
    property string alertType: "warning" // warning, error, critical
    property bool active: false
    
    // Dark overlay
    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: active ? 0.7 : 0
        
        Behavior on opacity {
            NumberAnimation { duration: 300 }
        }
        
        MouseArea {
            anchors.fill: parent
            onClicked: root.dismiss()
        }
    }
    
    // Alert container
    Rectangle {
        id: alertBox
        anchors.centerIn: parent
        width: 600
        height: 300
        color: "#0a0a0a"
        border.color: alertType === "critical" ? "#ff0000" :
                      alertType === "error" ? "#ff6600" : "#ffff00"
        border.width: 3
        radius: 20
        visible: active
        scale: active ? 1 : 0.8
        opacity: active ? 1 : 0
        
        Behavior on scale {
            NumberAnimation { duration: 300; easing.type: Easing.OutBack }
        }
        
        Behavior on opacity {
            NumberAnimation { duration: 300 }
        }
        
        // Glitch effect for critical alerts
        ShaderEffect {
            anchors.fill: parent
            visible: alertType === "critical"
            opacity: 0.5
            
            property real time: 0
            NumberAnimation on time {
                from: 0
                to: 1
                duration: 100
                loops: Animation.Infinite
            }
            
            fragmentShader: "
                #version 440
                layout(location = 0) in vec2 qt_TexCoord0;
                layout(location = 0) out vec4 fragColor;
                layout(std140, binding = 0) uniform buf {
                    mat4 qt_Matrix;
                    float qt_Opacity;
                    float time;
                };
                
                void main() {
                    vec2 uv = qt_TexCoord0;
                    float glitch = fract(sin(time * 12.9898) * 43758.5453);
                    uv.x += (glitch - 0.5) * 0.02;
                    vec3 color = vec3(1.0, 0.0, 0.0) * glitch;
                    fragColor = vec4(color, qt_Opacity * 0.5);
                }
            "
        }
        
        Column {
            anchors.fill: parent
            anchors.margins: 30
            spacing: 20
            
            // Alert icon and type
            Row {
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 15
                
                // Animated icon
                Text {
                    text: alertType === "critical" ? "⚠️" :
                          alertType === "error" ? "❌" : "⚡"
                    font.pixelSize: 48
                    
                    SequentialAnimation on scale {
                        running: alertType === "critical"
                        loops: Animation.Infinite
                        NumberAnimation { to: 1.2; duration: 200 }
                        NumberAnimation { to: 1.0; duration: 200 }
                    }
                }
                
                Text {
                    text: alertType.toUpperCase() + " ALERT"
                    color: alertBox.border.color
                    font.family: "Consolas, Monaco, monospace"
                    font.pixelSize: 24
                    font.bold: true
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
            
            // Alert message
            Rectangle {
                width: parent.width
                height: 100
                color: "#001122"
                border.color: alertBox.border.color
                border.width: 1
                radius: 10
                
                ScrollView {
                    anchors.fill: parent
                    anchors.margins: 10
                    
                    Text {
                        text: alertMessage
                        color: "#ffffff"
                        font.family: "Consolas, Monaco, monospace"
                        font.pixelSize: 14
                        wrapMode: Text.WordWrap
                        width: parent.width
                    }
                }
            }
            
            // Action buttons
            Row {
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 20
                
                AlertButton {
                    text: "ACKNOWLEDGE"
                    color: "#00ff00"
                    onClicked: root.acknowledge()
                }
                
                AlertButton {
                    text: "INVESTIGATE"
                    color: "#00ffff"
                    onClicked: root.investigate()
                    visible: alertType !== "warning"
                }
                
                AlertButton {
                    text: "DISMISS"
                    color: "#ff6600"
                    onClicked: root.dismiss()
                }
            }
        }
        
        // Pulse effect for critical alerts
        Rectangle {
            anchors.fill: parent
            color: "transparent"
            border.color: "#ff0000"
            border.width: 5
            radius: parent.radius
            opacity: 0
            visible: alertType === "critical"
            
            SequentialAnimation on opacity {
                running: visible
                loops: Animation.Infinite
                NumberAnimation { to: 0.5; duration: 500 }
                NumberAnimation { to: 0; duration: 500 }
            }
        }
    }
    
    // Component for alert buttons
    component AlertButton : Rectangle {
        property string text: ""
        property color color: "#00ffff"
        signal clicked()
        
        width: 120
        height: 40
        color: "transparent"
        border.color: parent.color
        border.width: 2
        radius: 20
        
        Text {
            anchors.centerIn: parent
            text: parent.text
            color: parent.color
            font.family: "Consolas, Monaco, monospace"
            font.pixelSize: 12
            font.bold: true
        }
        
        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            
            onEntered: {
                parent.color = Qt.rgba(parent.color.r, parent.color.g, parent.color.b, 0.2)
            }
            
            onExited: {
                parent.color = "transparent"
            }
            
            onClicked: {
                parent.clicked()
            }
        }
    }
    
    // Functions
    function showAlert(message, type) {
        alertMessage = message
        alertType = type || "warning"
        active = true
        visible = true
        
        // Auto-dismiss warnings after 5 seconds
        if (type === "warning") {
            autoDismissTimer.start()
        }
    }
    
    function acknowledge() {
        console.log("Alert acknowledged:", alertMessage)
        dismiss()
    }
    
    function investigate() {
        console.log("Investigating alert:", alertMessage)
        hudController.investigateAlert(alertMessage, alertType)
        dismiss()
    }
    
    function dismiss() {
        active = false
        hideTimer.start()
    }
    
    Timer {
        id: hideTimer
        interval: 300
        onTriggered: {
            visible = false
        }
    }
    
    Timer {
        id: autoDismissTimer
        interval: 5000
        onTriggered: {
            if (alertType === "warning") {
                dismiss()
            }
        }
    }
}