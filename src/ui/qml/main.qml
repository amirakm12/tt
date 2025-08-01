import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Material
import QtGraphicalEffects
import Qt5Compat.GraphicalEffects

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1400
    height: 900
    title: qsTr("AI System - Neural Command Center")
    
    // GPU-accelerated rendering
    flags: Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint
    
    // Theme properties
    property var currentTheme: themeManager.getThemeColors(themeManager.currentTheme)
    
    Material.theme: Material.Dark
    Material.primary: currentTheme.primary
    Material.accent: currentTheme.accent
    Material.background: currentTheme.background
    
    // Background with animated gradient
    Rectangle {
        anchors.fill: parent
        
        // GPU-accelerated gradient animation
        LinearGradient {
            anchors.fill: parent
            start: Qt.point(0, 0)
            end: Qt.point(parent.width, parent.height)
            
            gradient: Gradient {
                GradientStop { 
                    position: 0.0
                    color: Qt.darker(currentTheme.background, 1.2)
                    
                    Behavior on color {
                        ColorAnimation { duration: 500 }
                    }
                }
                GradientStop { 
                    position: 0.5
                    color: currentTheme.background
                    
                    Behavior on color {
                        ColorAnimation { duration: 500 }
                    }
                }
                GradientStop { 
                    position: 1.0
                    color: Qt.lighter(currentTheme.background, 1.1)
                    
                    Behavior on color {
                        ColorAnimation { duration: 500 }
                    }
                }
            }
        }
        
        // Animated particles background
        ParticleSystem {
            id: particleSystem
            anchors.fill: parent
            
            ImageParticle {
                source: "qrc:/assets/particle.png"
                alpha: 0.1
                alphaVariation: 0.1
                color: currentTheme.primary
                colorVariation: 0.3
                rotationVariation: 360
                rotationVelocityVariation: 180
            }
            
            Emitter {
                anchors.fill: parent
                emitRate: 20
                lifeSpan: 10000
                size: 4
                sizeVariation: 2
                velocity: AngleDirection {
                    angle: -90
                    angleVariation: 30
                    magnitude: 20
                    magnitudeVariation: 10
                }
            }
        }
    }
    
    // Main layout
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Header bar
        HeaderBar {
            id: headerBar
            Layout.fillWidth: true
            Layout.preferredHeight: 70
        }
        
        // Main content area
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0
            
            // Sidebar
            SideBar {
                id: sideBar
                Layout.preferredWidth: 250
                Layout.fillHeight: true
                
                onPageChanged: function(page) {
                    stackView.currentIndex = page
                }
            }
            
            // Content area with stack view
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "transparent"
                
                StackLayout {
                    id: stackView
                    anchors.fill: parent
                    anchors.margins: 20
                    currentIndex: 0
                    
                    // Dashboard page
                    DashboardPage {
                        id: dashboardPage
                    }
                    
                    // AI Agents page
                    AIAgentsPage {
                        id: agentsPage
                    }
                    
                    // Neural Network Visualization
                    NeuralNetworkPage {
                        id: neuralPage
                    }
                    
                    // Chat Interface
                    ChatPage {
                        id: chatPage
                    }
                    
                    // Settings page
                    SettingsPage {
                        id: settingsPage
                    }
                }
            }
        }
    }
    
    // Glassmorphism effect overlay
    Rectangle {
        anchors.fill: parent
        color: "transparent"
        visible: false // Enable for glassmorphism effect
        
        FastBlur {
            anchors.fill: parent
            source: mainWindow.contentItem
            radius: 32
            transparentBorder: true
        }
    }
    
    // System tray notification
    Connections {
        target: systemMetrics
        
        function onCpuUsageChanged(usage) {
            if (usage > 80) {
                showNotification("High CPU Usage", "CPU usage is above 80%")
            }
        }
    }
    
    // Notification popup
    function showNotification(title, message) {
        notificationPopup.showNotification(title, message)
    }
    
    NotificationPopup {
        id: notificationPopup
    }
    
    // Window animations
    PropertyAnimation {
        id: showAnimation
        target: mainWindow
        property: "opacity"
        from: 0
        to: 1
        duration: 300
        easing.type: Easing.OutCubic
    }
    
    Component.onCompleted: {
        showAnimation.start()
    }
}