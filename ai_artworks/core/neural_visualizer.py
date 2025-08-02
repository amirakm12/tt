"""
Neural Network Visualizer
Real-time visualization of agent neural activity
"""

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer
import random
import time

class NeuralNetworkVisualizer(QObject):
    """Visualizes neural network activity across agents"""
    
    # Signals
    activity_updated = Signal(dict)
    connection_pulse = Signal(str, str, float)  # from_agent, to_agent, intensity
    
    def __init__(self):
        super().__init__()
        
        # Neural network state
        self.nodes = {}
        self.connections = []
        self.activity_history = []
        
        # Simulation parameters
        self.pulse_frequency = 0.1
        self.decay_rate = 0.95
        self.noise_level = 0.1
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_activity)
        self.update_timer.start(50)  # 20 FPS
        
    def register_agent(self, agent_id: str, agent_type: str):
        """Register an agent as a neural node"""
        self.nodes[agent_id] = {
            'type': agent_type,
            'activation': 0.0,
            'position': self._calculate_position(len(self.nodes)),
            'connections': [],
            'last_pulse': 0
        }
        
    def create_connection(self, from_agent: str, to_agent: str, weight: float = 1.0):
        """Create a connection between agents"""
        if from_agent in self.nodes and to_agent in self.nodes:
            connection = {
                'from': from_agent,
                'to': to_agent,
                'weight': weight,
                'activity': 0.0
            }
            self.connections.append(connection)
            self.nodes[from_agent]['connections'].append(to_agent)
            
    def pulse_activity(self, agent_id: str, intensity: float = 1.0):
        """Send an activity pulse through an agent"""
        if agent_id in self.nodes:
            self.nodes[agent_id]['activation'] = min(1.0, intensity)
            self.nodes[agent_id]['last_pulse'] = time.time()
            
            # Propagate to connected nodes
            for target in self.nodes[agent_id]['connections']:
                self._propagate_pulse(agent_id, target, intensity * 0.7)
                
    def _propagate_pulse(self, from_agent: str, to_agent: str, intensity: float):
        """Propagate activity pulse through connection"""
        # Find connection
        for conn in self.connections:
            if conn['from'] == from_agent and conn['to'] == to_agent:
                conn['activity'] = intensity
                self.connection_pulse.emit(from_agent, to_agent, intensity)
                
                # Activate target node with delay
                QTimer.singleShot(
                    int(100 * (1 + random.random())),  # Random delay 100-200ms
                    lambda: self.pulse_activity(to_agent, intensity * conn['weight'])
                )
                break
                
    def _update_activity(self):
        """Update neural activity simulation"""
        # Decay activations
        for node_id, node in self.nodes.items():
            node['activation'] *= self.decay_rate
            
            # Add noise
            if random.random() < self.noise_level:
                node['activation'] += random.uniform(0, 0.2)
                
            # Clamp values
            node['activation'] = max(0, min(1, node['activation']))
            
        # Decay connection activities
        for conn in self.connections:
            conn['activity'] *= self.decay_rate
            
        # Random spontaneous activity
        if random.random() < self.pulse_frequency:
            random_node = random.choice(list(self.nodes.keys())) if self.nodes else None
            if random_node:
                self.pulse_activity(random_node, random.uniform(0.5, 1.0))
                
        # Emit update
        self.activity_updated.emit(self._get_state())
        
    def _calculate_position(self, index: int):
        """Calculate 3D position for node"""
        # Arrange nodes in a sphere
        phi = np.arccos(1 - 2 * (index + 0.5) / max(1, len(self.nodes)))
        theta = np.pi * (1 + 5**0.5) * index
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        return {'x': x * 100, 'y': y * 100, 'z': z * 100}
        
    def _get_state(self):
        """Get current neural network state"""
        return {
            'nodes': self.nodes,
            'connections': self.connections,
            'timestamp': time.time()
        }
        
    def get_activity_pulse(self):
        """Get a random activity pulse for visualization"""
        if not self.nodes:
            return None
            
        # Find most active node
        active_nodes = [(nid, n['activation']) for nid, n in self.nodes.items() if n['activation'] > 0.1]
        
        if active_nodes:
            node_id, activation = max(active_nodes, key=lambda x: x[1])
            pos = self.nodes[node_id]['position']
            return {
                'x': pos['x'],
                'y': pos['y'],
                'intensity': activation
            }
            
        return None