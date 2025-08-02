"""
UI Panels
Dockable panels for the main application window
"""

from typing import Optional, Dict, Any, List
import logging

from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QSlider, QLabel, QSpinBox, QCheckBox, QGroupBox,
    QScrollArea, QTextEdit, QComboBox, QToolButton, QGridLayout,
    QSizePolicy, QFrame
)

logger = logging.getLogger(__name__)


class LayerItem(QListWidgetItem):
    """Custom list item for layers"""
    
    def __init__(self, name: str, index: int):
        super().__init__()
        self.layer_name = name
        self.layer_index = index
        self.setText(name)
        
        # Create thumbnail icon
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.gray)
        self.setIcon(QIcon(pixmap))


class LayerPanel(QWidget):
    """Panel for managing layers"""
    
    # Signals
    layer_selected = Signal(int)
    layer_visibility_changed = Signal(int, bool)
    layer_opacity_changed = Signal(int, float)
    layer_renamed = Signal(int, str)
    layer_deleted = Signal(int)
    layer_reordered = Signal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Layer list
        self.layer_list = QListWidget()
        self.layer_list.setIconSize(QSize(64, 64))
        self.layer_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.layer_list.currentRowChanged.connect(self.on_selection_changed)
        layout.addWidget(self.layer_list)
        
        # Layer controls
        controls_layout = QHBoxLayout()
        
        self.visibility_checkbox = QCheckBox("Visible")
        self.visibility_checkbox.setChecked(True)
        self.visibility_checkbox.stateChanged.connect(self.on_visibility_changed)
        controls_layout.addWidget(self.visibility_checkbox)
        
        controls_layout.addWidget(QLabel("Opacity:"))
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        controls_layout.addWidget(self.opacity_slider)
        
        self.opacity_spinbox = QSpinBox()
        self.opacity_spinbox.setRange(0, 100)
        self.opacity_spinbox.setValue(100)
        self.opacity_spinbox.setSuffix("%")
        self.opacity_spinbox.valueChanged.connect(self.opacity_slider.setValue)
        self.opacity_slider.valueChanged.connect(self.opacity_spinbox.setValue)
        controls_layout.addWidget(self.opacity_spinbox)
        
        layout.addLayout(controls_layout)
        
        # Layer buttons
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_layer_clicked)
        button_layout.addWidget(self.add_button)
        
        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self.duplicate_layer_clicked)
        button_layout.addWidget(self.duplicate_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_layer_clicked)
        button_layout.addWidget(self.delete_button)
        
        layout.addLayout(button_layout)
        
        # Blend mode
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("Blend:"))
        
        self.blend_combo = QComboBox()
        self.blend_combo.addItems([
            "Normal", "Multiply", "Screen", "Overlay",
            "Soft Light", "Hard Light", "Color Dodge", "Color Burn"
        ])
        blend_layout.addWidget(self.blend_combo)
        
        layout.addLayout(blend_layout)
        
    @Slot(str)
    def add_layer(self, name: str):
        """Add a new layer to the list"""
        index = self.layer_list.count()
        item = LayerItem(name, index)
        self.layer_list.addItem(item)
        self.layer_list.setCurrentItem(item)
        
    @Slot(str)
    def remove_layer(self, name: str):
        """Remove a layer from the list"""
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if isinstance(item, LayerItem) and item.layer_name == name:
                self.layer_list.takeItem(i)
                break
                
    @Slot(int)
    def on_selection_changed(self, row: int):
        """Handle layer selection change"""
        if row >= 0:
            self.layer_selected.emit(row)
            
    @Slot(int)
    def on_visibility_changed(self, state: int):
        """Handle visibility change"""
        row = self.layer_list.currentRow()
        if row >= 0:
            visible = state == Qt.CheckState.Checked.value
            self.layer_visibility_changed.emit(row, visible)
            
    @Slot(int)
    def on_opacity_changed(self, value: int):
        """Handle opacity change"""
        row = self.layer_list.currentRow()
        if row >= 0:
            opacity = value / 100.0
            self.layer_opacity_changed.emit(row, opacity)
            
    def add_layer_clicked(self):
        """Handle add layer button"""
        # This would typically open a dialog or create a new layer
        pass
        
    def duplicate_layer_clicked(self):
        """Handle duplicate layer button"""
        row = self.layer_list.currentRow()
        if row >= 0:
            # Emit signal to duplicate layer
            pass
            
    def delete_layer_clicked(self):
        """Handle delete layer button"""
        row = self.layer_list.currentRow()
        if row >= 0:
            self.layer_deleted.emit(row)
            self.layer_list.takeItem(row)


class ToolButton(QToolButton):
    """Custom tool button"""
    
    def __init__(self, tool_id: str, name: str, icon: Optional[QIcon] = None):
        super().__init__()
        self.tool_id = tool_id
        self.setToolTip(name)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        
        if icon:
            self.setIcon(icon)
        else:
            # Create default icon
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.darkGray)
            painter = QPainter(pixmap)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, name[0])
            painter.end()
            self.setIcon(QIcon(pixmap))
            
        self.setIconSize(QSize(32, 32))
        self.setMinimumSize(48, 48)


class ToolPanel(QWidget):
    """Panel for tools"""
    
    # Signals
    tool_selected = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.tools = {}
        self.setup_ui()
        self.add_default_tools()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tool grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.tool_widget = QWidget()
        self.tool_layout = QGridLayout(self.tool_widget)
        self.tool_layout.setSpacing(5)
        
        scroll.setWidget(self.tool_widget)
        layout.addWidget(scroll)
        
        # Tool options area
        self.options_group = QGroupBox("Tool Options")
        self.options_layout = QVBoxLayout(self.options_group)
        layout.addWidget(self.options_group)
        
    def add_default_tools(self):
        """Add default tools"""
        default_tools = [
            ("select", "Select"),
            ("move", "Move"),
            ("brush", "Brush"),
            ("eraser", "Eraser"),
            ("text", "Text"),
            ("shape", "Shape"),
            ("crop", "Crop"),
            ("wand", "Magic Wand")
        ]
        
        for i, (tool_id, name) in enumerate(default_tools):
            button = ToolButton(tool_id, name)
            button.clicked.connect(lambda checked, tid=tool_id: self.on_tool_clicked(tid))
            
            row = i // 2
            col = i % 2
            self.tool_layout.addWidget(button, row, col)
            self.tools[tool_id] = button
            
    def add_tool(self, plugin):
        """Add a tool from a plugin"""
        tool_id = plugin.metadata.id
        name = plugin.metadata.name
        
        button = ToolButton(tool_id, name)
        button.clicked.connect(lambda checked, tid=tool_id: self.on_tool_clicked(tid))
        
        # Add to grid
        count = len(self.tools)
        row = count // 2
        col = count % 2
        self.tool_layout.addWidget(button, row, col)
        self.tools[tool_id] = button
        
    @Slot(str)
    def on_tool_clicked(self, tool_id: str):
        """Handle tool selection"""
        self.tool_selected.emit(tool_id)
        
        # Clear options
        while self.options_layout.count():
            child = self.options_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Add tool-specific options
        # This would be populated based on the selected tool


class PropertiesPanel(QWidget):
    """Panel for object properties"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Properties scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.properties_widget = QWidget()
        self.properties_layout = QVBoxLayout(self.properties_widget)
        
        scroll.setWidget(self.properties_widget)
        layout.addWidget(scroll)
        
        # Add default properties
        self.add_section("Transform")
        self.add_property("X", 0, -9999, 9999, "px")
        self.add_property("Y", 0, -9999, 9999, "px")
        self.add_property("Width", 100, 1, 9999, "px")
        self.add_property("Height", 100, 1, 9999, "px")
        self.add_property("Rotation", 0, -360, 360, "°")
        
        self.add_section("Appearance")
        self.add_property("Opacity", 100, 0, 100, "%")
        
    def add_section(self, title: str):
        """Add a property section"""
        label = QLabel(f"<b>{title}</b>")
        self.properties_layout.addWidget(label)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.properties_layout.addWidget(line)
        
    def add_property(self, name: str, value: int, min_val: int, max_val: int, suffix: str = ""):
        """Add a property control"""
        layout = QHBoxLayout()
        
        label = QLabel(name + ":")
        label.setMinimumWidth(80)
        layout.addWidget(label)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        if suffix:
            spinbox.setSuffix(suffix)
        layout.addWidget(spinbox)
        
        self.properties_layout.addLayout(layout)
        
    def clear_properties(self):
        """Clear all properties"""
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
                
    def clear_layout(self, layout):
        """Clear a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class CommandPanel(QWidget):
    """Panel for command output and history"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(200)
        
        # Style for different message types
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        
        layout.addWidget(self.output_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.output_text.clear)
        button_layout.addWidget(clear_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
    def add_message(self, message: str, msg_type: str = "info"):
        """Add a message to the output"""
        colors = {
            "info": "#ffffff",
            "user": "#00ff00",
            "system": "#00ffff",
            "error": "#ff0000",
            "warning": "#ffff00"
        }
        
        color = colors.get(msg_type, "#ffffff")
        formatted_msg = f'<span style="color: {color}">{message}</span><br>'
        
        self.output_text.append(formatted_msg)
        
        # Auto-scroll to bottom
        scrollbar = self.output_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())