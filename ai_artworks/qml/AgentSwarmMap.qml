import QtQuick
import QtQuick3D
import QtQuick3D.Particles3D

Rectangle {
    id: root
    color: "#0a0a0a"
    border.color: "#00ffff"
    border.width: 2
    radius: 10
    
    property var agents: ({})
    property var connections: []
    
    // Title
    Text {
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 10
        text: "AGENT SWARM NETWORK"
        color: "#00ffff"
        font.family: "Consolas, Monaco, monospace"
        font.pixelSize: 16
        font.bold: true
    }
    
    View3D {
        anchors.fill: parent
        anchors.margins: 20
        anchors.topMargin: 40
        
        environment: SceneEnvironment {
            clearColor: "transparent"
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
        }
        
        // Camera
        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0, 0, 300)
            eulerRotation.x: -10
            
            // Auto-rotate
            SequentialAnimation on eulerRotation.y {
                loops: Animation.Infinite
                NumberAnimation {
                    from: 0
                    to: 360
                    duration: 30000
                }
            }
        }
        
        // Ambient light
        DirectionalLight {
            eulerRotation.x: -30
            eulerRotation.y: -70
            color: "#404040"
            ambientColor: "#202020"
        }
        
        // Agent nodes
        Repeater3D {
            id: agentRepeater
            model: Object.keys(agents)
            
            Node {
                id: agentNode
                property string agentId: modelData
                property var agentData: agents[agentId]
                
                position: calculatePosition(index)
                
                // Agent sphere
                Model {
                    source: "#Sphere"
                    scale: Qt.vector3d(0.2, 0.2, 0.2)
                    
                    materials: [
                        PrincipledMaterial {
                            baseColor: getAgentColor(agentData.type)
                            metalness: 0.5
                            roughness: 0.2
                            emissiveColor: getAgentColor(agentData.type)
                            emissiveFactor: agentData.status === "busy" ? 0.8 : 0.3
                            
                            Behavior on emissiveFactor {
                                NumberAnimation { duration: 200 }
                            }
                        }
                    ]
                    
                    // Pulse animation when busy
                    SequentialAnimation on scale {
                        running: agentData.status === "busy"
                        loops: Animation.Infinite
                        Vector3dAnimation {
                            to: Qt.vector3d(0.25, 0.25, 0.25)
                            duration: 500
                            easing.type: Easing.OutQuad
                        }
                        Vector3dAnimation {
                            to: Qt.vector3d(0.2, 0.2, 0.2)
                            duration: 500
                            easing.type: Easing.InQuad
                        }
                    }
                }
                
                // Agent label
                Node {
                    y: 30
                    
                    Text {
                        text: agentData.type.toUpperCase()
                        color: getAgentColor(agentData.type)
                        font.family: "Consolas, Monaco, monospace"
                        font.pixelSize: 10
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
                
                // Status indicator
                Model {
                    y: -20
                    source: "#Rectangle"
                    scale: Qt.vector3d(0.5, 0.05, 0.05)
                    
                    materials: [
                        PrincipledMaterial {
                            baseColor: agentData.status === "idle" ? "#00ff00" : 
                                      agentData.status === "busy" ? "#ffff00" : "#ff0000"
                            emissiveColor: baseColor
                            emissiveFactor: 1.0
                        }
                    ]
                }
            }
        }
        
        // Connection lines
        Node {
            id: connectionContainer
            
            Component {
                id: connectionLine
                
                Model {
                    geometry: LineGeometry {
                        id: lineGeo
                        property vector3d startPos: Qt.vector3d(0, 0, 0)
                        property vector3d endPos: Qt.vector3d(100, 100, 0)
                    }
                    
                    materials: [
                        PrincipledMaterial {
                            baseColor: "#00ffff"
                            emissiveColor: "#00ffff"
                            emissiveFactor: 0.5
                            opacity: 0.5
                        }
                    ]
                }
            }
        }
        
        // Particle effects for data flow
        ParticleSystem3D {
            id: dataFlowParticles
            
            SpriteParticle3D {
                id: dataParticle
                sprite: Texture {
                    source: "assets/data_particle.png"
                }
                maxAmount: 500
                color: "#00ffff"
                colorVariation: Qt.vector3d(0.2, 0.2, 0.2)
                fadeInDuration: 100
                fadeOutDuration: 200
                billboard: true
                blendMode: SpriteParticle3D.Additive
                particleScale: 2
            }
            
            ParticleEmitter3D {
                id: dataEmitter
                particle: dataParticle
                position: Qt.vector3d(0, 0, 0)
                emitRate: 20
                lifeSpan: 2000
                
                shape: ParticleShape3D {
                    type: ParticleShape3D.Sphere
                    extents: Qt.vector3d(10, 10, 10)
                }
                
                velocity: VectorDirection3D {
                    direction: Qt.vector3d(1, 0, 0)
                    magnitude: 50
                }
            }
        }
    }
    
    // Functions
    function updateAgent(agentId, agentType, status) {
        if (!agents[agentId]) {
            agents[agentId] = {}
        }
        agents[agentId].type = agentType
        agents[agentId].status = status
        agents = agents // Trigger binding update
    }
    
    function calculatePosition(index) {
        var angle = (index / Object.keys(agents).length) * Math.PI * 2
        var radius = 100
        var x = Math.cos(angle) * radius
        var z = Math.sin(angle) * radius
        var y = Math.sin(index * 0.5) * 30
        return Qt.vector3d(x, y, z)
    }
    
    function getAgentColor(type) {
        switch(type) {
            case "render_ops": return "#ff6600"
            case "data_daemon": return "#00ff66"
            case "sec_sentinel": return "#ff0066"
            case "voice_nav": return "#6600ff"
            case "autopilot": return "#ffff00"
            case "athena": return "#00ffff"
            default: return "#ffffff"
        }
    }
    
    // Custom line geometry
    Component {
        id: lineGeometryComponent
        
        Geometry {
            id: lineGeometry
            property vector3d start: Qt.vector3d(0, 0, 0)
            property vector3d end: Qt.vector3d(100, 0, 0)
            
            positions: [start, end]
            
            GeometryAttribute {
                id: positionAttribute
                attributeType: GeometryAttribute.PositionAttribute
                vertexSize: 3
                count: 2
                byteStride: 3 * 4
                buffer: Buffer {
                    type: Buffer.VertexBuffer
                    data: new Float32Array([
                        start.x, start.y, start.z,
                        end.x, end.y, end.z
                    ])
                }
            }
            
            GeometryAttribute {
                attributeType: GeometryAttribute.IndexAttribute
                vertexSize: 1
                count: 2
                buffer: Buffer {
                    type: Buffer.IndexBuffer
                    data: new Uint16Array([0, 1])
                }
            }
        }
    }
}