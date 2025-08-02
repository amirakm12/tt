import QtQuick
import QtQuick3D
import QtQuick3D.Particles3D

Node {
    id: root
    
    // Rotating neural network structure
    Node {
        id: networkContainer
        
        NumberAnimation on eulerRotation.y {
            from: 0
            to: 360
            duration: 60000
            loops: Animation.Infinite
        }
        
        NumberAnimation on eulerRotation.x {
            from: -15
            to: 15
            duration: 10000
            loops: Animation.Infinite
            easing.type: Easing.SineCurve
        }
        
        // Neural nodes
        Repeater3D {
            model: 150
            
            Node {
                position: generateNodePosition(index)
                
                Model {
                    id: nodeModel
                    source: "#Sphere"
                    scale: Qt.vector3d(0.05, 0.05, 0.05)
                    
                    materials: [
                        PrincipledMaterial {
                            baseColor: "#00ffff"
                            metalness: 0.8
                            roughness: 0.2
                            emissiveColor: "#00ffff"
                            emissiveFactor: Math.random() * 0.5 + 0.1
                            
                            SequentialAnimation on emissiveFactor {
                                loops: Animation.Infinite
                                NumberAnimation {
                                    to: Math.random() * 0.8 + 0.2
                                    duration: Math.random() * 3000 + 2000
                                }
                                NumberAnimation {
                                    to: Math.random() * 0.3
                                    duration: Math.random() * 3000 + 2000
                                }
                            }
                        }
                    ]
                }
                
                // Connections to nearby nodes
                Repeater3D {
                    model: 3
                    
                    Model {
                        property vector3d endPos: generateNodePosition(Math.floor(Math.random() * 150))
                        
                        geometry: LineGeometry {
                            startPos: Qt.vector3d(0, 0, 0)
                            endPos: parent.endPos
                        }
                        
                        materials: [
                            PrincipledMaterial {
                                baseColor: "#006666"
                                emissiveColor: "#00ffff"
                                emissiveFactor: 0.2
                                opacity: 0.3
                            }
                        ]
                    }
                }
            }
        }
        
        function generateNodePosition(index) {
            // Create multiple layers of nodes
            var layer = Math.floor(index / 50)
            var angleStep = (Math.PI * 2) / 50
            var angle = (index % 50) * angleStep
            var radius = 200 + layer * 100
            var height = (layer - 1) * 150
            
            // Add some randomness
            radius += Math.random() * 50 - 25
            height += Math.random() * 50 - 25
            
            return Qt.vector3d(
                Math.cos(angle) * radius,
                height,
                Math.sin(angle) * radius
            )
        }
    }
    
    // Data flow particles
    ParticleSystem3D {
        id: dataFlow
        
        SpriteParticle3D {
            id: dataParticle
            sprite: Texture {
                source: "assets/data_particle.png"
            }
            maxAmount: 2000
            color: "#00ffff"
            colorVariation: Qt.vector3d(0.2, 0.2, 0.2)
            fadeInDuration: 500
            fadeOutDuration: 1000
            billboard: true
            blendMode: SpriteParticle3D.Additive
            particleScale: 1.5
        }
        
        ParticleEmitter3D {
            particle: dataParticle
            position: Qt.vector3d(0, -300, 0)
            emitRate: 100
            lifeSpan: 10000
            
            shape: ParticleShape3D {
                type: ParticleShape3D.Sphere
                extents: Qt.vector3d(500, 10, 500)
            }
            
            velocity: VectorDirection3D {
                direction: Qt.vector3d(0, 1, 0)
                directionVariation: Qt.vector3d(0.3, 0, 0.3)
                magnitude: 100
                magnitudeVariation: 50
            }
        }
    }
    
    // Quantum field effect
    Model {
        source: "#Cube"
        scale: Qt.vector3d(10, 10, 10)
        position: Qt.vector3d(0, 0, -500)
        opacity: 0.1
        
        materials: [
            PrincipledMaterial {
                baseColor: "#001144"
                emissiveColor: "#0044ff"
                emissiveFactor: 0.3
                opacity: 0.1
            }
        ]
        
        NumberAnimation on eulerRotation.x {
            from: 0
            to: 360
            duration: 20000
            loops: Animation.Infinite
        }
        
        NumberAnimation on eulerRotation.y {
            from: 0
            to: -360
            duration: 30000
            loops: Animation.Infinite
        }
        
        SequentialAnimation on scale {
            loops: Animation.Infinite
            Vector3dAnimation {
                to: Qt.vector3d(12, 12, 12)
                duration: 5000
                easing.type: Easing.InOutQuad
            }
            Vector3dAnimation {
                to: Qt.vector3d(10, 10, 10)
                duration: 5000
                easing.type: Easing.InOutQuad
            }
        }
    }
    
    // Energy waves
    Repeater3D {
        model: 5
        
        Model {
            property real phase: index * 0.4
            property real radius: 300 + index * 100
            
            y: Math.sin(phase + (Date.now() / 1000)) * 50
            
            source: "#Cylinder"
            scale: Qt.vector3d(radius / 50, 0.1, radius / 50)
            eulerRotation.x: -90
            opacity: 0.3 - (index * 0.05)
            
            materials: [
                PrincipledMaterial {
                    baseColor: "#00ffff"
                    emissiveColor: "#00ffff"
                    emissiveFactor: 0.5
                    opacity: parent.opacity
                }
            ]
            
            NumberAnimation on y {
                from: -200
                to: 200
                duration: 3000 + index * 500
                loops: Animation.Infinite
            }
            
            NumberAnimation on opacity {
                from: 0.3 - (index * 0.05)
                to: 0
                duration: 3000 + index * 500
                loops: Animation.Infinite
            }
        }
    }
}

// Custom line geometry component
component LineGeometry : Geometry {
    property vector3d startPos: Qt.vector3d(0, 0, 0)
    property vector3d endPos: Qt.vector3d(100, 100, 100)
    
    positions: [startPos, endPos]
    
    Buffer {
        id: vertexBuffer
        type: Buffer.VertexBuffer
        data: new Float32Array([
            startPos.x, startPos.y, startPos.z,
            endPos.x, endPos.y, endPos.z
        ])
    }
    
    Buffer {
        id: indexBuffer
        type: Buffer.IndexBuffer
        data: new Uint16Array([0, 1])
    }
    
    Attribute {
        attributeType: Attribute.PositionAttribute
        vertexBaseType: Attribute.F32Type
        vertexSize: 3
        count: 2
        buffer: vertexBuffer
    }
    
    Attribute {
        attributeType: Attribute.IndexAttribute
        vertexBaseType: Attribute.U16Type
        vertexSize: 1
        count: 2
        buffer: indexBuffer
    }
}