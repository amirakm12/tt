#!/usr/bin/env python3
"""
AI System Qt Application
GPU-Accelerated UI with QML and QtQuick
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtCore import (
    QObject, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot, 
    pyqtProperty, QPropertyAnimation, QEasingCurve, Qt
)
from PyQt6.QtGui import QGuiApplication, QIcon, QFontDatabase
from PyQt6.QtQml import QQmlApplicationEngine, qmlRegisterType, QQmlContext
from PyQt6.QtQuick import QQuickView
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu

# Import our AI system components
try:
    from ..core.orchestrator import SystemOrchestrator
    from ..core.config import SystemConfig
    from ..monitoring.system_monitor import SystemMonitor
except ImportError:
    # Handle absolute imports for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.orchestrator import SystemOrchestrator
    from src.core.config import SystemConfig
    from src.monitoring.system_monitor import SystemMonitor


class SystemMetrics(QObject):
    """Backend model for system metrics with GPU acceleration support"""
    
    # Signals for real-time updates
    cpuUsageChanged = pyqtSignal(float)
    memoryUsageChanged = pyqtSignal(float)
    gpuUsageChanged = pyqtSignal(float)
    networkInChanged = pyqtSignal(float)
    networkOutChanged = pyqtSignal(float)
    aiOperationsChanged = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self._cpu_usage = 0.0
        self._memory_usage = 0.0
        self._gpu_usage = 0.0
        self._network_in = 0.0
        self._network_out = 0.0
        self._ai_operations = 0
        
        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)  # Update every second
        
    @pyqtProperty(float, notify=cpuUsageChanged)
    def cpuUsage(self):
        return self._cpu_usage
        
    @pyqtProperty(float, notify=memoryUsageChanged)
    def memoryUsage(self):
        return self._memory_usage
        
    @pyqtProperty(float, notify=gpuUsageChanged)
    def gpuUsage(self):
        return self._gpu_usage
        
    @pyqtProperty(float, notify=networkInChanged)
    def networkIn(self):
        return self._network_in
        
    @pyqtProperty(float, notify=networkOutChanged)
    def networkOut(self):
        return self._network_out
        
    @pyqtProperty(int, notify=aiOperationsChanged)
    def aiOperations(self):
        return self._ai_operations
        
    @pyqtSlot()
    def update_metrics(self):
        """Update metrics with simulated data"""
        import random
        
        # Simulate metric changes
        self._cpu_usage = 40 + random.random() * 20
        self._memory_usage = 50 + random.random() * 10
        self._gpu_usage = 60 + random.random() * 30
        self._network_in = 100 + random.random() * 50
        self._network_out = 80 + random.random() * 40
        self._ai_operations = int(1000 + random.random() * 500)
        
        # Emit signals
        self.cpuUsageChanged.emit(self._cpu_usage)
        self.memoryUsageChanged.emit(self._memory_usage)
        self.gpuUsageChanged.emit(self._gpu_usage)
        self.networkInChanged.emit(self._network_in)
        self.networkOutChanged.emit(self._network_out)
        self.aiOperationsChanged.emit(self._ai_operations)


class AIAgentModel(QObject):
    """Model for AI agents"""
    
    agentsChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._agents = [
            {
                "id": "agent-001",
                "name": "Triage Agent",
                "status": "active",
                "task": "Processing user queries",
                "performance": 95.5,
                "color": "#00ff88"
            },
            {
                "id": "agent-002",
                "name": "Research Agent",
                "status": "active",
                "task": "Analyzing data patterns",
                "performance": 88.2,
                "color": "#ff00ff"
            },
            {
                "id": "agent-003",
                "name": "Orchestration Agent",
                "status": "active",
                "task": "Coordinating workflows",
                "performance": 92.7,
                "color": "#00d4ff"
            }
        ]
        
    @pyqtProperty(list, notify=agentsChanged)
    def agents(self):
        return self._agents
        
    @pyqtSlot(str, str)
    def updateAgentStatus(self, agent_id: str, status: str):
        """Update agent status"""
        for agent in self._agents:
            if agent["id"] == agent_id:
                agent["status"] = status
                self.agentsChanged.emit()
                break


class ChatBackend(QObject):
    """Backend for AI chat functionality"""
    
    messageReceived = pyqtSignal(str, str, str)  # role, message, timestamp
    
    def __init__(self):
        super().__init__()
        
    @pyqtSlot(str)
    def sendMessage(self, message: str):
        """Process user message"""
        # Add user message
        timestamp = datetime.now().isoformat()
        self.messageReceived.emit("user", message, timestamp)
        
        # Simulate AI response after a delay
        QTimer.singleShot(500, lambda: self._generate_response(message))
        
    def _generate_response(self, user_message: str):
        """Generate AI response"""
        response = f"I received your message: '{user_message}'. Processing with neural networks..."
        timestamp = datetime.now().isoformat()
        self.messageReceived.emit("ai", response, timestamp)


class ThemeManager(QObject):
    """Manage application themes"""
    
    themeChanged = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._current_theme = "dark"
        self._themes = {
            "dark": {
                "primary": "#00d4ff",
                "secondary": "#ff00ff",
                "accent": "#00ff88",
                "background": "#0a0a0a",
                "surface": "#1a1a1a",
                "text": "#ffffff"
            },
            "light": {
                "primary": "#0066cc",
                "secondary": "#9933ff",
                "accent": "#00aa44",
                "background": "#ffffff",
                "surface": "#f5f5f5",
                "text": "#000000"
            },
            "cyberpunk": {
                "primary": "#ff0080",
                "secondary": "#00ffff",
                "accent": "#ffff00",
                "background": "#0a0014",
                "surface": "#1f0029",
                "text": "#ffffff"
            }
        }
        
    @pyqtProperty(str, notify=themeChanged)
    def currentTheme(self):
        return self._current_theme
        
    @pyqtProperty(dict)
    def themes(self):
        return self._themes
        
    @pyqtSlot(str)
    def setTheme(self, theme_name: str):
        """Change application theme"""
        if theme_name in self._themes:
            self._current_theme = theme_name
            self.themeChanged.emit(theme_name)
            
    @pyqtSlot(str, result=dict)
    def getThemeColors(self, theme_name: str):
        """Get colors for a specific theme"""
        return self._themes.get(theme_name, self._themes["dark"])


class AISystemQt(QObject):
    """Main Qt application class"""
    
    def __init__(self):
        super().__init__()
        self.app = None
        self.engine = None
        self.system_tray = None
        
        # Backend models
        self.metrics = SystemMetrics()
        self.agents = AIAgentModel()
        self.chat = ChatBackend()
        self.theme_manager = ThemeManager()
        
        # AI System components
        self.config = SystemConfig()
        self.orchestrator = None
        
    def initialize(self):
        """Initialize the Qt application"""
        # Set up application
        self.app = QGuiApplication(sys.argv)
        self.app.setApplicationName("AI System")
        self.app.setOrganizationName("Neural Systems")
        
        # Enable GPU acceleration
        os.environ["QSG_RENDER_LOOP"] = "threaded"
        os.environ["QT_QUICK_BACKEND"] = "software"  # Change to "d3d12" on Windows for better GPU support
        
        # Set application icon
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            self.app.setWindowIcon(QIcon(str(icon_path)))
            
        # Load custom fonts
        self._load_fonts()
        
        # Initialize QML engine
        self.engine = QQmlApplicationEngine()
        
        # Register Python types with QML
        self._register_types()
        
        # Set context properties
        context = self.engine.rootContext()
        context.setContextProperty("systemMetrics", self.metrics)
        context.setContextProperty("agentModel", self.agents)
        context.setContextProperty("chatBackend", self.chat)
        context.setContextProperty("themeManager", self.theme_manager)
        
        # Load main QML file
        qml_file = Path(__file__).parent / "qml" / "main.qml"
        self.engine.load(QUrl.fromLocalFile(str(qml_file)))
        
        # Check if loading was successful
        if not self.engine.rootObjects():
            print("Failed to load QML file")
            return False
            
        # Set up system tray
        self._setup_system_tray()
        
        return True
        
    def _load_fonts(self):
        """Load custom fonts"""
        fonts_dir = Path(__file__).parent / "assets" / "fonts"
        if fonts_dir.exists():
            for font_file in fonts_dir.glob("*.ttf"):
                QFontDatabase.addApplicationFont(str(font_file))
                
    def _register_types(self):
        """Register custom types with QML"""
        # Register any custom QML types here
        pass
        
    def _setup_system_tray(self):
        """Set up system tray icon"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.system_tray = QSystemTrayIcon(self.app)
            
            # Set tray icon
            icon_path = Path(__file__).parent / "assets" / "icon.png"
            if icon_path.exists():
                self.system_tray.setIcon(QIcon(str(icon_path)))
                
            # Create context menu
            menu = QMenu()
            
            show_action = menu.addAction("Show Dashboard")
            show_action.triggered.connect(self._show_dashboard)
            
            menu.addSeparator()
            
            quit_action = menu.addAction("Quit")
            quit_action.triggered.connect(self.app.quit)
            
            self.system_tray.setContextMenu(menu)
            self.system_tray.show()
            
            # Connect double-click
            self.system_tray.activated.connect(self._tray_activated)
            
    def _tray_activated(self, reason):
        """Handle system tray activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_dashboard()
            
    def _show_dashboard(self):
        """Show the main dashboard window"""
        if self.engine.rootObjects():
            window = self.engine.rootObjects()[0]
            window.show()
            window.raise_()
            window.requestActivate()
            
    def run(self):
        """Run the application"""
        if not self.initialize():
            return 1
            
        # Start the Qt event loop
        return self.app.exec()


def main():
    """Main entry point"""
    app = AISystemQt()
    sys.exit(app.run())


if __name__ == "__main__":
    main()