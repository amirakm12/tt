#!/usr/bin/env python3
"""
AI-ARTWORKS Cyberpunk HUD
Futuristic neural interface with voice control
"""

import sys
import os
from pathlib import Path

# Configure Qt to use Vulkan
os.environ['QSG_RHI_BACKEND'] = 'vulkan'
os.environ['QT_QUICK_CONTROLS_STYLE'] = 'Material'
os.environ['QT_QUICK_CONTROLS_MATERIAL_THEME'] = 'Dark'

from PySide6.QtCore import QUrl, QObject, Signal, Slot, Property, QTimer
from PySide6.QtGui import QGuiApplication, QSurfaceFormat, QVulkanInstance
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide6.QtQuick import QQuickView
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras

# Import backend systems
from core.multi_agent_system import get_athena, AgentType
from core.voice_interface import VoiceController
from core.neural_visualizer import NeuralNetworkVisualizer
from core.consciousness_monitor import ConsciousnessMonitor

class HUDController(QObject):
    """Main controller for the cyberpunk HUD interface"""
    
    # Signals for QML
    voiceCommandReceived = Signal(str)
    voiceWaveformUpdated = Signal(list)
    agentStatusChanged = Signal(str, str, str)  # agent_id, type, status
    thoughtStreamUpdated = Signal(str)
    taskProgressUpdated = Signal(str, float)
    neuralActivityPulse = Signal(float, float, float)  # x, y, intensity
    
    def __init__(self):
        super().__init__()
        
        # Initialize Athena orchestrator
        self.athena = get_athena()
        
        # Voice controller
        self.voice_controller = VoiceController()
        self.voice_controller.command_recognized.connect(self._on_voice_command)
        self.voice_controller.waveform_updated.connect(self._on_waveform_update)
        
        # Neural visualizer
        self.neural_viz = NeuralNetworkVisualizer()
        
        # Consciousness monitor
        self.consciousness = ConsciousnessMonitor()
        
        # Start monitoring loops
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_monitors)
        self.monitor_timer.start(50)  # 20 FPS update
        
    @Slot(str)
    def processVoiceCommand(self, command: str):
        """Process voice command from QML"""
        self.voice_controller.process_command(command)
        
    @Slot()
    def startListening(self):
        """Start voice listening"""
        self.voice_controller.start_listening()
        
    @Slot()
    def stopListening(self):
        """Stop voice listening"""
        self.voice_controller.stop_listening()
        
    def _on_voice_command(self, command: str):
        """Handle recognized voice command"""
        self.voiceCommandReceived.emit(command)
        
        # Submit to Athena
        task = {
            'type': 'voice_command',
            'agent_type': 'voice_nav',
            'payload': {'command': command}
        }
        self.athena.submit_task(task)
        
    def _on_waveform_update(self, waveform_data: list):
        """Update voice waveform visualization"""
        self.voiceWaveformUpdated.emit(waveform_data)
        
    def _update_monitors(self):
        """Update all monitoring systems"""
        # Update agent statuses
        for agent_id, agent in self.athena.agents.items():
            self.agentStatusChanged.emit(
                agent_id,
                agent.agent_type.value,
                agent.status.value
            )
            
        # Update thought stream
        thought = self.consciousness.get_current_thought()
        if thought:
            self.thoughtStreamUpdated.emit(thought)
            
        # Update neural activity
        activity = self.neural_viz.get_activity_pulse()
        if activity:
            self.neuralActivityPulse.emit(
                activity['x'],
                activity['y'], 
                activity['intensity']
            )

    @Slot()
    def boostPerformance(self):
        """Activate performance boost mode"""
        self.athena.set_performance_mode("boost")
        self.systemModeChanged.emit("BOOST")
        
    @Slot()
    def activateShield(self):
        """Activate security shield"""
        for agent_id, agent in self.athena.agents.items():
            if agent.agent_type == AgentType.SEC_SENTINEL:
                agent.activate_shield()
                
    @Slot()
    def enterTargetingMode(self):
        """Enter targeting mode for precise operations"""
        self.targetingModeActivated.emit()
        
    @Slot()
    def synchronizeAgents(self):
        """Synchronize all agents"""
        self.athena.synchronize_all()
        
    @Slot()
    def toggleAlertMode(self):
        """Toggle alert monitoring mode"""
        self.alertModeToggled.emit()
        
    @Slot(str, str)
    def investigateAlert(self, message: str, alert_type: str):
        """Investigate an alert"""
        task = {
            'type': 'investigate_alert',
            'agent_type': 'sec_sentinel',
            'payload': {
                'message': message,
                'alert_type': alert_type
            }
        }
        self.athena.submit_task(task)
        
    @Slot()
    def startupSequence(self):
        """Run startup sequence"""
        # Initialize agents
        self.athena.initialize_agents()
        
        # Start monitoring
        self.consciousness.inject_context({'command': 'System startup initiated'})
        
        # Emit startup complete
        self.startupComplete.emit()
    
    # Additional signals
    systemModeChanged = Signal(str)
    targetingModeActivated = Signal()
    alertModeToggled = Signal()
    startupComplete = Signal()

def setup_vulkan():
    """Configure Vulkan for Qt"""
    # Set up surface format for Vulkan
    format = QSurfaceFormat()
    format.setVersion(4, 6)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setRenderableType(QSurfaceFormat.OpenGL)
    format.setSwapBehavior(QSurfaceFormat.TripleBuffer)
    format.setSamples(4)  # Anti-aliasing
    QSurfaceFormat.setDefaultFormat(format)
    
    # Create Vulkan instance if available
    vulkan_instance = QVulkanInstance()
    if vulkan_instance.create():
        print("Vulkan initialized successfully")
        return vulkan_instance
    else:
        print("Falling back to OpenGL")
        return None

def main():
    """Main application entry point"""
    # Create application
    app = QGuiApplication(sys.argv)
    app.setApplicationName("AI-ARTWORKS Neural Interface")
    app.setOrganizationName("Cyberpunk Systems")
    
    # Setup Vulkan
    vulkan = setup_vulkan()
    
    # Register QML types
    qmlRegisterType(HUDController, "AIArtworks", 1, 0, "HUDController")
    
    # Create QML engine
    engine = QQmlApplicationEngine()
    
    # Set up QML path
    qml_path = Path(__file__).parent / "qml"
    engine.addImportPath(str(qml_path))
    
    # Load main QML
    qml_file = qml_path / "main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_file)))
    
    if not engine.rootObjects():
        print("Failed to load QML")
        sys.exit(1)
        
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()