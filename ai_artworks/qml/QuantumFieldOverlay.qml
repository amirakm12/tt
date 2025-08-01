import QtQuick
import QtQuick3D
import QtQuick3D.Particles3D

Item {
    id: root
    anchors.fill: parent
    
    // Quantum field visualization
    View3D {
        anchors.fill: parent
        
        environment: SceneEnvironment {
            clearColor: "transparent"
            backgroundMode: SceneEnvironment.Transparent
            aoEnabled: false
            
            effects: [
                DistortionSpiral {
                    center: Qt.point(0.5, 0.5)
                    distortionStrength: 0.1
                    radius: 2.0
                    
                    NumberAnimation on distortionStrength {
                        from: 0.05
                        to: 0.15
                        duration: 4000
                        loops: Animation.Infinite
                        easing.type: Easing.SineCurve
                    }
                }
            ]
        }
        
        PerspectiveCamera {
            position: Qt.vector3d(0, 0, 600)
            eulerRotation.x: 0
        }
        
        DirectionalLight {
            eulerRotation.x: -30
            eulerRotation.y: -70
            color: "#404040"
            ambientColor: "#101010"
        }
        
        // Quantum probability clouds
        Repeater3D {
            model: 20
            
            Node {
                position: Qt.vector3d(
                    Math.random() * 800 - 400,
                    Math.random() * 600 - 300,
                    Math.random() * 400 - 200
                )
                
                Model {
                    source: "#Sphere"
                    scale: Qt.vector3d(0.5, 0.5, 0.5)
                    opacity: 0.3
                    
                    materials: [
                        PrincipledMaterial {
                            baseColor: Qt.hsla(Math.random(), 1.0, 0.5, 1.0)
                            metalness: 0.0
                            roughness: 1.0
                            emissiveColor: baseColor
                            emissiveFactor: 0.5
                            opacity: 0.3
                        }
                    ]
                    
                    SequentialAnimation on scale {
                        loops: Animation.Infinite
                        Vector3dAnimation {
                            to: Qt.vector3d(1.0, 1.0, 1.0)
                            duration: Math.random() * 3000 + 2000
                            easing.type: Easing.InOutQuad
                        }
                        Vector3dAnimation {
                            to: Qt.vector3d(0.5, 0.5, 0.5)
                            duration: Math.random() * 3000 + 2000
                            easing.type: Easing.InOutQuad
                        }
                    }
                    
                    NumberAnimation on eulerRotation.y {
                        from: 0
                        to: 360
                        duration: Math.random() * 10000 + 5000
                        loops: Animation.Infinite
                    }
                }
                
                // Quantum entanglement connections
                Repeater3D {
                    model: 2
                    
                    Model {
                        property vector3d targetPos: Qt.vector3d(
                            Math.random() * 800 - 400,
                            Math.random() * 600 - 300,
                            Math.random() * 400 - 200
                        )
                        
                        geometry: LineGeometry {
                            startPos: Qt.vector3d(0, 0, 0)
                            endPos: parent.targetPos
                        }
                        
                        materials: [
                            PrincipledMaterial {
                                baseColor: "#00ffff"
                                emissiveColor: "#00ffff"
                                emissiveFactor: 0.3
                                opacity: 0.2
                            }
                        ]
                        
                        SequentialAnimation on opacity {
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.4; duration: 2000 }
                            NumberAnimation { to: 0.1; duration: 2000 }
                        }
                    }
                }
            }
        }
        
        // Quantum wave function
        Model {
            position: Qt.vector3d(0, 0, 0)
            scale: Qt.vector3d(10, 10, 10)
            
            geometry: GridGeometry {
                horizontalLines: 50
                verticalLines: 50
                horizontalStep: 20
                verticalStep: 20
            }
            
            materials: [
                PrincipledMaterial {
                    baseColor: "#00ffff"
                    metalness: 0.5
                    roughness: 0.5
                    emissiveColor: "#00ffff"
                    emissiveFactor: 0.2
                    opacity: 0.1
                }
            ]
            
            property real phase: 0
            NumberAnimation on phase {
                from: 0
                to: Math.PI * 2
                duration: 5000
                loops: Animation.Infinite
            }
            
            onPhaseChanged: {
                // Update wave function shape
                geometry.updateWave(phase)
            }
        }
        
        // Quantum particles
        ParticleSystem3D {
            SpriteParticle3D {
                id: quantumParticle
                sprite: Texture {
                    source: "assets/quantum_particle.png"
                }
                maxAmount: 1000
                color: "#00ffff"
                colorVariation: Qt.vector3d(0.5, 0.5, 0.5)
                fadeInDuration: 1000
                fadeOutDuration: 1000
                billboard: true
                blendMode: SpriteParticle3D.Additive
                particleScale: 0.5
            }
            
            ParticleEmitter3D {
                particle: quantumParticle
                position: Qt.vector3d(0, 0, 0)
                emitRate: 50
                lifeSpan: 5000
                lifeSpanVariation: 2000
                
                shape: ParticleShape3D {
                    type: ParticleShape3D.Sphere
                    extents: Qt.vector3d(400, 400, 400)
                }
                
                velocity: VectorDirection3D {
                    direction: Qt.vector3d(0, 0, 0)
                    directionVariation: Qt.vector3d(50, 50, 50)
                    magnitude: 20
                    magnitudeVariation: 10
                }
            }
        }
    }
    
    // 2D Overlay effects
    ShaderEffect {
        anchors.fill: parent
        opacity: 0.3
        
        property real time: 0
        NumberAnimation on time {
            from: 0
            to: 1
            duration: 10000
            loops: Animation.Infinite
        }
        
        property real interference: 0.5
        SequentialAnimation on interference {
            loops: Animation.Infinite
            NumberAnimation { to: 1.0; duration: 3000 }
            NumberAnimation { to: 0.0; duration: 3000 }
        }
        
        fragmentShader: "
            #version 440
            layout(location = 0) in vec2 qt_TexCoord0;
            layout(location = 0) out vec4 fragColor;
            layout(std140, binding = 0) uniform buf {
                mat4 qt_Matrix;
                float qt_Opacity;
                float time;
                float interference;
            };
            
            void main() {
                vec2 uv = qt_TexCoord0;
                
                // Quantum interference pattern
                float wave1 = sin(uv.x * 20.0 + time * 10.0) * sin(uv.y * 20.0 + time * 10.0);
                float wave2 = sin(uv.x * 30.0 - time * 15.0) * sin(uv.y * 30.0 - time * 15.0);
                float pattern = mix(wave1, wave2, interference);
                
                // Probability density visualization
                float density = abs(pattern);
                vec3 color = vec3(0.0, density, density);
                
                // Quantum noise
                float noise = fract(sin(dot(uv + time, vec2(12.9898, 78.233))) * 43758.5453);
                color += vec3(0.0, noise * 0.1, noise * 0.1);
                
                fragColor = vec4(color, density * qt_Opacity);
            }
        "
    }
}

// Custom grid geometry for wave function
component GridGeometry : Geometry {
    property int horizontalLines: 50
    property int verticalLines: 50
    property real horizontalStep: 10
    property real verticalStep: 10
    
    function updateWave(phase) {
        // Update vertex positions to create wave effect
        // This would be implemented in C++ for performance
    }
}