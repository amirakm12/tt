"""
Main Application Window
Production-grade Qt6 interface with dockable panels and modular design
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from PySide6.QtCore import Qt, Signal, Slot, QSettings, QTimer, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence, QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QDockWidget, QToolBar, QStatusBar,
    QMenuBar, QMenu, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QSlider, QSpinBox, QComboBox, QListWidget, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar, QSplitter
)
from PySide6.QtAsyncio import QAsyncioEventLoopPolicy

from ..core.gpu_canvas import GPUCanvas
from ..core.plugin_manager import PluginManager
from ..core.ai_agents import VoiceCommandSystem
from .panels import LayerPanel, ToolPanel, PropertiesPanel, CommandPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.canvas = None
        self.plugin_manager = PluginManager()
        self.voice_system = VoiceCommandSystem()
        self.settings = QSettings("AI-ARTWORKS", "MainWindow")
        
        # UI components
        self.dock_widgets = {}
        self.toolbars = {}
        self.recent_files = []
        
        # Setup UI
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_dock_widgets()
        self.setup_statusbar()
        
        # Connect signals
        self.connect_signals()
        
        # Load plugins
        self.plugin_manager.load_all_plugins()
        
        # Restore window state
        self.restore_state()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        logger.info("Main window initialized")
        
    def setup_ui(self):
        """Setup the main UI"""
        self.setWindowTitle("AI-ARTWORKS - Neural Creative Suite")
        self.setMinimumSize(1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QDockWidget {
                color: #ffffff;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
            }
            QDockWidget::title {
                background-color: #2d2d2d;
                padding: 5px;
                border-bottom: 2px solid #007ACC;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 3px;
                padding: 3px;
            }
            QToolBar::separator {
                background-color: #3d3d3d;
                width: 1px;
                margin: 3px;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #007ACC;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #007ACC;
            }
            QStatusBar {
                background-color: #007ACC;
                color: #ffffff;
            }
        """)
        
        # Create central widget with canvas
        self.canvas = GPUCanvas()
        self.setCentralWidget(self.canvas)
        
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Recent files
        self.recent_menu = file_menu.addMenu("Recent Files")
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        edit_menu.addAction(paste_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_window_action = QAction("&Fit to Window", self)
        fit_window_action.setShortcut("Ctrl+0")
        fit_window_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_window_action)
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        view_menu.addAction(zoom_out_action)
        
        view_menu.addSeparator()
        
        # Dock widgets visibility
        self.dock_menu = view_menu.addMenu("Panels")
        
        # Plugins menu
        plugins_menu = menubar.addMenu("&Plugins")
        
        reload_plugins_action = QAction("&Reload Plugins", self)
        reload_plugins_action.setShortcut("F5")
        reload_plugins_action.triggered.connect(self.reload_plugins)
        plugins_menu.addAction(reload_plugins_action)
        
        plugins_menu.addSeparator()
        
        # Dynamic plugin actions will be added here
        self.plugins_menu = plugins_menu
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbars(self):
        """Setup toolbars"""
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setObjectName("MainToolbar")
        self.toolbars["main"] = main_toolbar
        
        # File operations
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_project)
        main_toolbar.addWidget(new_btn)
        
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_image)
        main_toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_image)
        main_toolbar.addWidget(save_btn)
        
        main_toolbar.addSeparator()
        
        # Voice command
        self.voice_btn = QPushButton("🎤 Voice")
        self.voice_btn.setCheckable(True)
        self.voice_btn.toggled.connect(self.toggle_voice_recording)
        main_toolbar.addWidget(self.voice_btn)
        
        # Command input
        self.command_input = QComboBox()
        self.command_input.setEditable(True)
        self.command_input.setMinimumWidth(300)
        self.command_input.lineEdit().setPlaceholderText("Type a command...")
        self.command_input.lineEdit().returnPressed.connect(self.process_text_command)
        main_toolbar.addWidget(self.command_input)
        
        main_toolbar.addSeparator()
        
        # Tools toolbar
        tools_toolbar = self.addToolBar("Tools")
        tools_toolbar.setObjectName("ToolsToolbar")
        self.toolbars["tools"] = tools_toolbar
        
        # Tool buttons will be added dynamically from plugins
        
    def setup_dock_widgets(self):
        """Setup dockable panels"""
        # Layers panel
        layers_dock = QDockWidget("Layers", self)
        layers_dock.setObjectName("LayersDock")
        self.layer_panel = LayerPanel()
        layers_dock.setWidget(self.layer_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, layers_dock)
        self.dock_widgets["layers"] = layers_dock
        
        # Tools panel
        tools_dock = QDockWidget("Tools", self)
        tools_dock.setObjectName("ToolsDock")
        self.tool_panel = ToolPanel()
        tools_dock.setWidget(self.tool_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, tools_dock)
        self.dock_widgets["tools"] = tools_dock
        
        # Properties panel
        properties_dock = QDockWidget("Properties", self)
        properties_dock.setObjectName("PropertiesDock")
        self.properties_panel = PropertiesPanel()
        properties_dock.setWidget(self.properties_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)
        self.dock_widgets["properties"] = properties_dock
        
        # Command panel
        command_dock = QDockWidget("Command Output", self)
        command_dock.setObjectName("CommandDock")
        self.command_panel = CommandPanel()
        command_dock.setWidget(self.command_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, command_dock)
        self.dock_widgets["command"] = command_dock
        
        # Tab docks
        self.tabifyDockWidget(layers_dock, properties_dock)
        layers_dock.raise_()
        
        # Update dock menu
        for name, dock in self.dock_widgets.items():
            action = dock.toggleViewAction()
            self.dock_menu.addAction(action)
            
    def setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = self.statusBar()
        
        # Permanent widgets
        self.fps_label = QLabel("FPS: 0")
        self.statusbar.addPermanentWidget(self.fps_label)
        
        self.zoom_label = QLabel("Zoom: 100%")
        self.statusbar.addPermanentWidget(self.zoom_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # Update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(100)
        
    def connect_signals(self):
        """Connect all signals"""
        # Canvas signals
        self.canvas.layer_added.connect(self.layer_panel.add_layer)
        self.canvas.layer_removed.connect(self.layer_panel.remove_layer)
        self.canvas.zoom_changed.connect(self.on_zoom_changed)
        
        # Layer panel signals
        self.layer_panel.layer_selected.connect(self.canvas.set_active_layer)
        self.layer_panel.layer_visibility_changed.connect(self.canvas.set_layer_visibility)
        self.layer_panel.layer_opacity_changed.connect(self.canvas.set_layer_opacity)
        
        # Tool panel signals
        self.tool_panel.tool_selected.connect(self.on_tool_selected)
        
        # Voice system signals
        self.voice_system.transcription_ready.connect(self.on_transcription)
        self.voice_system.command_ready.connect(self.on_command_ready)
        self.voice_system.status_changed.connect(self.statusbar.showMessage)
        
        # Plugin manager signals
        self.plugin_manager.plugin_loaded.connect(self.on_plugin_loaded)
        self.plugin_manager.plugin_error.connect(self.on_plugin_error)
        
    @Slot(bool)
    def toggle_voice_recording(self, checked: bool):
        """Toggle voice recording"""
        if checked:
            self.voice_system.start_listening()
            self.voice_btn.setText("🔴 Recording...")
        else:
            self.voice_system.stop_listening()
            self.voice_btn.setText("🎤 Voice")
            
    @Slot()
    def process_text_command(self):
        """Process text command from input"""
        text = self.command_input.currentText()
        if text:
            self.command_input.addItem(text)
            self.command_input.clearEditText()
            self.voice_system.process_text_command(text)
            
    @Slot(str)
    def on_transcription(self, text: str):
        """Handle voice transcription"""
        self.command_panel.add_message(f"You said: {text}", "user")
        
    @Slot(object)
    def on_command_ready(self, command):
        """Handle parsed command"""
        self.command_panel.add_message(
            f"Command: {command.action} (confidence: {command.confidence:.2f})",
            "system"
        )
        
    @Slot(str)
    def on_plugin_loaded(self, plugin_id: str):
        """Handle plugin loaded"""
        plugin = self.plugin_manager.get_plugin(plugin_id)
        if plugin:
            # Add to appropriate UI
            if plugin.metadata.category == "tool":
                self.tool_panel.add_tool(plugin)
            elif plugin.metadata.category == "filter":
                # Add to filters menu
                action = QAction(plugin.metadata.name, self)
                action.triggered.connect(lambda: self.apply_plugin(plugin_id))
                self.plugins_menu.addAction(action)
                
    @Slot(str, str)
    def on_plugin_error(self, plugin_id: str, error: str):
        """Handle plugin error"""
        QMessageBox.warning(self, "Plugin Error", f"Plugin {plugin_id} error: {error}")
        
    @Slot(str)
    def on_tool_selected(self, tool_id: str):
        """Handle tool selection"""
        plugin = self.plugin_manager.get_plugin(tool_id)
        if plugin:
            self.canvas.set_tool(plugin)
            
    @Slot(float)
    def on_zoom_changed(self, zoom: float):
        """Handle zoom change"""
        self.zoom_label.setText(f"Zoom: {int(zoom * 100)}%")
        
    def update_status(self):
        """Update status bar"""
        self.fps_label.setText(f"FPS: {self.canvas.get_fps()}")
        
    def new_project(self):
        """Create new project"""
        # Clear canvas
        while self.canvas.layers:
            self.canvas.remove_layer(0)
            
        # Add default layer
        import numpy as np
        default_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        self.canvas.add_layer("Background", default_image)
        
    def open_image(self):
        """Open image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*.*)"
        )
        
        if file_path:
            self.load_image(file_path)
            
    def load_image(self, file_path: str):
        """Load image from file"""
        try:
            import cv2
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Failed to load image")
                
            # Convert to RGBA if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                
            # Add as new layer
            layer_name = Path(file_path).stem
            self.canvas.add_layer(layer_name, image)
            
            # Add to recent files
            self.add_recent_file(file_path)
            
            self.statusbar.showMessage(f"Loaded: {file_path}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
            
    def save_image(self):
        """Save current image"""
        if hasattr(self, "current_file"):
            self.save_image_to_file(self.current_file)
        else:
            self.save_image_as()
            
    def save_image_as(self):
        """Save image with new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            self.save_image_to_file(file_path)
            self.current_file = file_path
            
    def save_image_to_file(self, file_path: str):
        """Save image to file"""
        try:
            import cv2
            composite = self.canvas.composite_layers()
            
            # Convert from RGBA to BGR for OpenCV
            if composite.shape[2] == 4:
                bgr = cv2.cvtColor(composite, cv2.COLOR_RGBA2BGR)
            else:
                bgr = composite
                
            cv2.imwrite(file_path, bgr)
            self.statusbar.showMessage(f"Saved: {file_path}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
            
    def add_recent_file(self, file_path: str):
        """Add file to recent files list"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # Keep last 10
        self.update_recent_files_menu()
        
    def update_recent_files_menu(self):
        """Update recent files menu"""
        self.recent_menu.clear()
        
        for file_path in self.recent_files:
            action = QAction(Path(file_path).name, self)
            action.setData(file_path)
            action.triggered.connect(lambda checked, fp=file_path: self.load_image(fp))
            self.recent_menu.addAction(action)
            
    def reload_plugins(self):
        """Reload all plugins"""
        self.plugin_manager.load_all_plugins()
        self.statusbar.showMessage("Plugins reloaded", 3000)
        
    def apply_plugin(self, plugin_id: str):
        """Apply a plugin"""
        plugin = self.plugin_manager.get_plugin(plugin_id)
        if plugin and plugin.metadata.category == "filter":
            # Get active layer
            layer = self.canvas.get_active_layer()
            if layer:
                # Apply filter
                params = plugin.get_parameters()
                result = plugin.apply(layer.image, params)
                
                # Add as new layer
                self.canvas.add_layer(f"{layer.name} - {plugin.metadata.name}", result)
                
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About AI-ARTWORKS",
            "AI-ARTWORKS - Neural Creative Suite\n\n"
            "A production-grade AI-powered image editing application\n"
            "with GPU acceleration and voice control.\n\n"
            "Version 1.0.0"
        )
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """Handle drop"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.load_image(file_path)
                break
                
    def closeEvent(self, event: QCloseEvent):
        """Handle window close"""
        # Save state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("recentFiles", self.recent_files)
        
        # Shutdown systems
        self.voice_system.shutdown()
        
        event.accept()
        
    def restore_state(self):
        """Restore window state"""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
            
        recent = self.settings.value("recentFiles", [])
        if recent:
            self.recent_files = recent
            self.update_recent_files_menu()


def main():
    """Main entry point"""
    # Set up asyncio event loop
    asyncio.set_event_loop_policy(QAsyncioEventLoopPolicy())
    
    app = QApplication(sys.argv)
    app.setApplicationName("AI-ARTWORKS")
    app.setOrganizationName("AI-ARTWORKS")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()