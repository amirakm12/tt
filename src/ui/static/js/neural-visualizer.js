// Neural Network 3D Visualizer
// Advanced Three.js visualization for neural network architecture

class NeuralNetworkVisualizer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.neurons = [];
        this.connections = [];
        this.animationId = null;
        this.clock = new THREE.Clock();
        
        this.config = {
            neuronSize: 0.3,
            connectionOpacity: 0.6,
            animationSpeed: 1,
            glowIntensity: 2,
            particleCount: 1000
        };
        
        this.init();
    }
    
    init() {
        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
        
        // Setup camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 15);
        
        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);
        
        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.rotateSpeed = 0.5;
        this.controls.zoomSpeed = 0.8;
        
        // Add lights
        this.setupLights();
        
        // Create neural network
        this.createNeuralNetwork();
        
        // Add particles
        this.createParticles();
        
        // Add post-processing effects
        this.setupPostProcessing();
        
        // Load neural network data
        this.loadNetworkData();
    }
    
    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        
        // Point lights for neurons
        const light1 = new THREE.PointLight(0x00d4ff, 1, 20);
        light1.position.set(5, 5, 5);
        this.scene.add(light1);
        
        const light2 = new THREE.PointLight(0xff00ff, 1, 20);
        light2.position.set(-5, -5, 5);
        this.scene.add(light2);
        
        // Directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(0, 10, 5);
        dirLight.castShadow = true;
        this.scene.add(dirLight);
    }
    
    createNeuralNetwork() {
        // Create layers
        const layers = [
            { size: 5, z: -6 },
            { size: 8, z: -3 },
            { size: 10, z: 0 },
            { size: 8, z: 3 },
            { size: 4, z: 6 }
        ];
        
        // Create neurons
        layers.forEach((layer, layerIndex) => {
            for (let i = 0; i < layer.size; i++) {
                const angle = (i / layer.size) * Math.PI * 2;
                const radius = 3;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                const z = layer.z;
                
                const neuron = this.createNeuron(x, y, z, layerIndex);
                this.neurons.push(neuron);
                this.scene.add(neuron.mesh);
                
                // Add glow effect
                const glow = this.createGlow(neuron.mesh);
                this.scene.add(glow);
            }
        });
        
        // Create connections
        this.createConnections();
    }
    
    createNeuron(x, y, z, layer) {
        // Create geometry
        const geometry = new THREE.SphereGeometry(this.config.neuronSize, 32, 32);
        
        // Create material with emissive glow
        const material = new THREE.MeshPhongMaterial({
            color: this.getNeuronColor(layer),
            emissive: this.getNeuronColor(layer),
            emissiveIntensity: 0.5,
            shininess: 100,
            specular: 0xffffff
        });
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        return {
            mesh,
            layer,
            activation: Math.random(),
            pulsePhase: Math.random() * Math.PI * 2
        };
    }
    
    getNeuronColor(layer) {
        const colors = [0x00ff88, 0x00d4ff, 0xff00ff, 0xffaa00, 0xff0044];
        return colors[layer % colors.length];
    }
    
    createGlow(neuronMesh) {
        const glowGeometry = new THREE.SphereGeometry(this.config.neuronSize * 2, 16, 16);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: neuronMesh.material.color,
            transparent: true,
            opacity: 0.2,
            side: THREE.BackSide
        });
        
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.position.copy(neuronMesh.position);
        
        return glow;
    }
    
    createConnections() {
        // Connect neurons between adjacent layers
        for (let i = 0; i < this.neurons.length; i++) {
            for (let j = i + 1; j < this.neurons.length; j++) {
                const n1 = this.neurons[i];
                const n2 = this.neurons[j];
                
                // Only connect adjacent layers
                if (Math.abs(n1.layer - n2.layer) === 1) {
                    // Random connection probability
                    if (Math.random() > 0.3) {
                        const connection = this.createConnection(n1, n2);
                        this.connections.push(connection);
                        this.scene.add(connection.mesh);
                    }
                }
            }
        }
    }
    
    createConnection(neuron1, neuron2) {
        const points = [];
        points.push(neuron1.mesh.position);
        
        // Add curve control point for organic look
        const midPoint = new THREE.Vector3(
            (neuron1.mesh.position.x + neuron2.mesh.position.x) / 2,
            (neuron1.mesh.position.y + neuron2.mesh.position.y) / 2,
            (neuron1.mesh.position.z + neuron2.mesh.position.z) / 2
        );
        midPoint.x += (Math.random() - 0.5) * 0.5;
        midPoint.y += (Math.random() - 0.5) * 0.5;
        
        points.push(midPoint);
        points.push(neuron2.mesh.position);
        
        const curve = new THREE.CatmullRomCurve3(points);
        const geometry = new THREE.TubeGeometry(curve, 20, 0.02, 8, false);
        
        const material = new THREE.MeshBasicMaterial({
            color: 0x00d4ff,
            transparent: true,
            opacity: this.config.connectionOpacity,
            blending: THREE.AdditiveBlending
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        return {
            mesh,
            neuron1,
            neuron2,
            strength: Math.random(),
            pulseProgress: 0
        };
    }
    
    createParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
        for (let i = 0; i < this.config.particleCount; i++) {
            positions.push(
                (Math.random() - 0.5) * 30,
                (Math.random() - 0.5) * 30,
                (Math.random() - 0.5) * 30
            );
            
            const color = new THREE.Color();
            color.setHSL(Math.random(), 0.7, 0.5);
            colors.push(color.r, color.g, color.b);
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }
    
    setupPostProcessing() {
        // This would typically use Three.js post-processing pipeline
        // For now, we'll use basic renderer settings
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.renderer.toneMappingExposure = 2;
    }
    
    async loadNetworkData() {
        try {
            const response = await fetch('/api/viz/neural-network');
            const data = await response.json();
            
            // Update network based on data
            this.updateNetwork(data);
        } catch (error) {
            console.error('Failed to load neural network data:', error);
        }
    }
    
    updateNetwork(data) {
        // Update neuron activations
        if (data.layers) {
            let neuronIndex = 0;
            data.layers.forEach(layer => {
                layer.neurons.forEach(neuronData => {
                    if (this.neurons[neuronIndex]) {
                        this.neurons[neuronIndex].activation = neuronData.activation;
                        neuronIndex++;
                    }
                });
            });
        }
        
        // Update connection strengths
        if (data.connections) {
            data.connections.forEach((connData, index) => {
                if (this.connections[index]) {
                    this.connections[index].strength = connData.weight;
                }
            });
        }
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();
        
        // Update controls
        this.controls.update();
        
        // Animate neurons
        this.neurons.forEach((neuron, index) => {
            // Pulse effect
            const scale = 1 + Math.sin(time * 2 + neuron.pulsePhase) * 0.1 * neuron.activation;
            neuron.mesh.scale.set(scale, scale, scale);
            
            // Update emissive intensity based on activation
            neuron.mesh.material.emissiveIntensity = 0.3 + neuron.activation * 0.7;
            
            // Slight rotation
            neuron.mesh.rotation.y += delta * 0.5;
        });
        
        // Animate connections
        this.connections.forEach(connection => {
            // Pulse along connection
            connection.pulseProgress += delta * 2;
            if (connection.pulseProgress > 1) {
                connection.pulseProgress = 0;
            }
            
            // Update opacity based on strength
            connection.mesh.material.opacity = 
                this.config.connectionOpacity * connection.strength * 
                (0.5 + Math.sin(time * 3) * 0.5);
        });
        
        // Rotate particles
        if (this.particles) {
            this.particles.rotation.y += delta * 0.1;
            this.particles.rotation.x += delta * 0.05;
        }
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    startAnimation() {
        if (!this.animationId) {
            this.animate();
        }
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    dispose() {
        this.stopAnimation();
        
        // Dispose of Three.js resources
        this.neurons.forEach(neuron => {
            neuron.mesh.geometry.dispose();
            neuron.mesh.material.dispose();
        });
        
        this.connections.forEach(connection => {
            connection.mesh.geometry.dispose();
            connection.mesh.material.dispose();
        });
        
        if (this.particles) {
            this.particles.geometry.dispose();
            this.particles.material.dispose();
        }
        
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}

// System Topology Visualizer
class SystemTopologyVisualizer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.nodes = {};
        this.edges = [];
        this.dataFlows = [];
        
        this.init();
    }
    
    init() {
        // Setup scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        
        // Setup camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(10, 10, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Setup renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Setup controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(10, 10, 5);
        this.scene.add(directionalLight);
        
        // Create grid
        this.createGrid();
        
        // Load topology
        this.loadTopology();
    }
    
    createGrid() {
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(gridHelper);
    }
    
    async loadTopology() {
        try {
            const response = await fetch('/api/viz/system-topology');
            const data = await response.json();
            
            this.createTopology(data);
        } catch (error) {
            console.error('Failed to load topology:', error);
        }
    }
    
    createTopology(data) {
        // Create nodes
        data.nodes.forEach(nodeData => {
            const node = this.createNode(nodeData);
            this.nodes[nodeData.id] = node;
            this.scene.add(node.group);
        });
        
        // Create edges
        data.edges.forEach(edgeData => {
            const edge = this.createEdge(edgeData);
            if (edge) {
                this.edges.push(edge);
                this.scene.add(edge.mesh);
            }
        });
    }
    
    createNode(nodeData) {
        const group = new THREE.Group();
        
        // Create main geometry
        const geometry = new THREE.BoxGeometry(2, 2, 2);
        const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color(nodeData.color),
            emissive: new THREE.Color(nodeData.color),
            emissiveIntensity: 0.3
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        group.add(mesh);
        
        // Add label
        // In a real implementation, you'd use a text geometry or sprite
        
        // Position
        group.position.set(
            nodeData.position.x,
            nodeData.position.y,
            nodeData.position.z
        );
        
        return {
            group,
            mesh,
            data: nodeData
        };
    }
    
    createEdge(edgeData) {
        const source = this.nodes[edgeData.source];
        const target = this.nodes[edgeData.target];
        
        if (!source || !target) return null;
        
        const points = [
            source.group.position,
            target.group.position
        ];
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x00d4ff,
            transparent: true,
            opacity: 0.6
        });
        
        const mesh = new THREE.Line(geometry, material);
        
        return {
            mesh,
            source: edgeData.source,
            target: edgeData.target
        };
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Rotate nodes
        Object.values(this.nodes).forEach(node => {
            node.mesh.rotation.y += 0.01;
        });
        
        // Update controls
        this.controls.update();
        
        // Render
        this.renderer.render(this.scene, this.camera);
    }
    
    updateAgent(agentData) {
        const node = this.nodes[agentData.id];
        if (node) {
            // Update node appearance based on agent status
            const color = agentData.status === 'active' ? 0x00ff00 : 0xff0000;
            node.mesh.material.color.setHex(color);
            node.mesh.material.emissive.setHex(color);
        }
    }
    
    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
}