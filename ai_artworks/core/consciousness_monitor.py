"""
Consciousness Monitor
Monitors and visualizes Athena's internal thought processes
"""

from PySide6.QtCore import QObject, Signal, QTimer
import random
import time
from collections import deque
import json

class ConsciousnessMonitor(QObject):
    """Monitor for AI consciousness and decision-making processes"""
    
    # Signals
    thought_generated = Signal(str)
    decision_made = Signal(dict)
    entropy_changed = Signal(float)
    
    def __init__(self):
        super().__init__()
        
        # Thought stream
        self.thought_history = deque(maxlen=100)
        self.current_thought = ""
        
        # Decision tracking
        self.pending_decisions = []
        self.decision_history = deque(maxlen=50)
        
        # Consciousness metrics
        self.entropy_level = 0.5
        self.coherence = 0.8
        self.focus_level = 0.6
        
        # Thought patterns
        self.thought_patterns = [
            "Analyzing voice command: {}",
            "Evaluating agent availability for task: {}",
            "Calculating optimal execution path...",
            "Memory retrieval: searching for similar patterns",
            "Hypothesis: {} might require {} agent",
            "Confidence level: {}% for current approach",
            "Alternative path detected: considering {} strategy",
            "Resource allocation: {} units required",
            "Parallel processing opportunity identified",
            "Synchronizing agent states...",
            "Predictive model suggests: {}",
            "Anomaly detected in {}: investigating",
            "Learning from previous execution: adjusting weights",
            "Creative solution emerging: combining {} with {}",
            "Quantum superposition of strategies resolved to: {}"
        ]
        
        # Simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self._simulate_thought)
        self.sim_timer.start(2000)  # Generate thought every 2 seconds
        
    def inject_context(self, context: dict):
        """Inject external context into consciousness"""
        if 'command' in context:
            self._generate_thought(f"Processing command: '{context['command']}'")
        elif 'agent_status' in context:
            self._generate_thought(f"Agent {context['agent_id']} status: {context['agent_status']}")
        elif 'task' in context:
            self._generate_thought(f"New task received: {context['task']['type']}")
            
    def _simulate_thought(self):
        """Simulate internal thought generation"""
        # Random thought generation
        pattern = random.choice(self.thought_patterns)
        
        # Fill in placeholders
        if '{}' in pattern:
            placeholders = pattern.count('{}')
            values = []
            
            for _ in range(placeholders):
                value_type = random.choice(['agent', 'task', 'metric', 'concept'])
                
                if value_type == 'agent':
                    values.append(random.choice(['RenderOps', 'DataDaemon', 'VoiceNav', 'SecSentinel', 'Autopilot']))
                elif value_type == 'task':
                    values.append(random.choice(['image generation', 'voice processing', 'security scan', 'data analysis']))
                elif value_type == 'metric':
                    values.append(f"{random.randint(60, 95)}")
                else:
                    values.append(random.choice(['neural pathway', 'quantum state', 'memory cluster', 'decision tree']))
                    
            thought = pattern.format(*values)
        else:
            thought = pattern
            
        self._generate_thought(thought)
        
        # Occasionally make decisions
        if random.random() < 0.3:
            self._make_decision()
            
        # Update entropy
        self._update_entropy()
        
    def _generate_thought(self, thought: str):
        """Generate and emit a thought"""
        timestamp = time.time()
        thought_data = {
            'text': thought,
            'timestamp': timestamp,
            'entropy': self.entropy_level,
            'coherence': self.coherence
        }
        
        self.current_thought = thought
        self.thought_history.append(thought_data)
        self.thought_generated.emit(thought)
        
    def _make_decision(self):
        """Simulate decision-making process"""
        decisions = [
            {
                'type': 'agent_selection',
                'options': ['RenderOps', 'DataDaemon', 'VoiceNav'],
                'selected': 'RenderOps',
                'confidence': random.uniform(0.7, 0.95)
            },
            {
                'type': 'strategy_choice',
                'options': ['parallel', 'sequential', 'hybrid'],
                'selected': 'parallel',
                'confidence': random.uniform(0.6, 0.9)
            },
            {
                'type': 'resource_allocation',
                'options': ['minimal', 'balanced', 'maximum'],
                'selected': 'balanced',
                'confidence': random.uniform(0.8, 0.95)
            }
        ]
        
        decision = random.choice(decisions)
        decision['timestamp'] = time.time()
        
        self.decision_history.append(decision)
        self.decision_made.emit(decision)
        
        # Generate thought about decision
        self._generate_thought(
            f"Decision made: {decision['type']} -> {decision['selected']} "
            f"(confidence: {decision['confidence']:.1%})"
        )
        
    def _update_entropy(self):
        """Update consciousness entropy level"""
        # Random walk with mean reversion
        delta = random.uniform(-0.1, 0.1)
        self.entropy_level += delta
        
        # Mean reversion
        self.entropy_level += (0.5 - self.entropy_level) * 0.1
        
        # Clamp
        self.entropy_level = max(0.1, min(0.9, self.entropy_level))
        
        # Update coherence (inverse of entropy)
        self.coherence = 1.0 - self.entropy_level * 0.5
        
        self.entropy_changed.emit(self.entropy_level)
        
    def get_current_thought(self):
        """Get the current thought"""
        return self.current_thought
        
    def get_thought_stream(self, count: int = 10):
        """Get recent thoughts"""
        return list(self.thought_history)[-count:]
        
    def get_consciousness_state(self):
        """Get full consciousness state"""
        return {
            'current_thought': self.current_thought,
            'entropy': self.entropy_level,
            'coherence': self.coherence,
            'focus': self.focus_level,
            'recent_decisions': list(self.decision_history)[-5:],
            'thought_count': len(self.thought_history)
        }