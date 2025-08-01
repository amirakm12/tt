import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick3D
import QtQuick3D.Effects
import QtQuick.Particles
import AIArtworks 1.0

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1920
    height: 1080
    title: "AI-ARTWORKS Neural Interface"
    color: "black"
    
    // Full screen by default
    visibility: Window.FullScreen
    
    // HUD Controller
    HUDController {
        id: hudController
        
        onVoiceCommandReceived: (command) => {
            voiceHUD.addCommand(command)
        }
        
        onVoiceWaveformUpdated: (waveform) => {
            voiceHUD.updateWaveform(waveform)
        }
        
        onAgentStatusChanged: (agentId, agentType, status) => {
            agentSwarm.updateAgent(agentId, agentType, status)
        }
        
        onThoughtStreamUpdated: (thought) => {
            thoughtInspector.addThought(thought)
        }
        
        onNeuralActivityPulse: (x, y, intensity) => {
            neuralCanvas.addPulse(x, y, intensity)
        }
    }
    
    // 3D Scene for background effects
    View3D {
        id: view3d
        anchors.fill: parent
        
        environment: SceneEnvironment {
            clearColor: "#000000"
            backgroundMode: SceneEnvironment.Color
            aoEnabled: true
            aoDarkness: 0.5
            aoDistance: 10
            
            effects: [
                DepthOfFieldBlur {
                    focusDistance: 100
                    focusRange: 50
                    blurAmount: 0.2
                },
                Glow {
                    threshold: 0.3
                    intensity: 0.8
                    bloom: 0.3
                }
            ]
        }
        
        // Animated neural network background
        NeuralNetworkBackground {
            id: neuralBg
        }
        
        // Particle effects
        ParticleSystem3D {
            id: particleSystem
            
            SpriteParticle3D {
                id: neuralParticles
                sprite: Texture {
                    source: "assets/particle_glow.png"
                }
                maxAmount: 1000
                color: "#00ffff"
                colorVariation: Qt.vector3d(0.2, 0.2, 0.2)
                fadeInDuration: 200
                fadeOutDuration: 500
                billboard: true
                blendMode: SpriteParticle3D.Additive
            }
            
            ParticleEmitter3D {
                particle: neuralParticles
                position: Qt.vector3d(0, 0, -100)
                emitRate: 50
                lifeSpan: 3000
                lifeSpanVariation: 1000
                
                velocity: VectorDirection3D {
                    direction: Qt.vector3d(0, 1, 0)
                    directionVariation: Qt.vector3d(100, 100, 100)
                    magnitude: 50
                    magnitudeVariation: 25
                }
            }
        }
    }
    
    // Main HUD Layer
    Item {
        anchors.fill: parent
        
        // Scanline effect overlay
        ShaderEffect {
            anchors.fill: parent
            opacity: 0.1
            
            property real time: 0
            NumberAnimation on time {
                from: 0
                to: 1
                duration: 1000
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
                    float scanline = sin(uv.y * 800.0 + time * 10.0) * 0.04;
                    vec3 color = vec3(0.0, 1.0, 1.0) * (1.0 + scanline);
                    fragColor = vec4(color, qt_Opacity);
                }
            "
        }
        
        // Voice HUD Component
        VoiceHUD {
            id: voiceHUD
            anchors.top: parent.top
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.topMargin: 50
            width: 600
            height: 200
        }
        
        // Agent Swarm Visualizer
        AgentSwarmMap {
            id: agentSwarm
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            anchors.leftMargin: 50
            width: 400
            height: 400
        }
        
        // Thought Inspector
        ThoughtInspector {
            id: thoughtInspector
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.rightMargin: 50
            anchors.topMargin: 150
            width: 350
            height: 300
        }
        
        // Neural Canvas
        NeuralCanvas {
            id: neuralCanvas
            anchors.centerIn: parent
            width: 800
            height: 600
            opacity: 0.8
        }
        
        // Execution Timeline
        ExecutionTimeline {
            id: timeline
            anchors.bottom: parent.bottom
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottomMargin: 50
            width: parent.width - 100
            height: 150
        }
        
        // Memory Vault Access
        MemoryVault {
            id: memoryVault
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.rightMargin: 50
            anchors.bottomMargin: 200
            width: 300
            height: 200
        }
        
        // Control Panel
        ControlPanel {
            id: controlPanel
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            anchors.leftMargin: 50
            anchors.bottomMargin: 50
            width: 300
            height: 100
        }
    }
    
    // Keyboard shortcuts
    Shortcut {
        sequence: "Space"
        onActivated: {
            if (voiceHUD.listening) {
                hudController.stopListening()
                voiceHUD.listening = false
            } else {
                hudController.startListening()
                voiceHUD.listening = true
            }
        }
    }
    
    Shortcut {
        sequence: "Escape"
        onActivated: {
            if (mainWindow.visibility === Window.FullScreen) {
                mainWindow.visibility = Window.Windowed
            } else {
                Qt.quit()
            }
        }
    }
}