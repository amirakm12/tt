"""
Advanced Custom Widgets
High-performance, visually stunning widgets for the UI
"""

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from PySide6.QtCore import (
    Qt, Signal, Slot, QTimer, QPropertyAnimation, QEasingCurve,
    QRectF, QPointF, Property, QParallelAnimationGroup, QSequentialAnimationGroup
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient,
    QPainterPath, QFont, QFontMetrics, QPolygonF, QConicalGradient,
    QPixmap, QImage, QPainterPathStroker
)
from PySide6.QtWidgets import (
    QWidget, QGraphicsDropShadowEffect, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDial, QPushButton, QFrame
)


class NeuralButton(QPushButton):
    """Futuristic neural network-inspired button"""
    
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        
        # Visual properties
        self._glow_intensity = 0.0
        self._ripple_radius = 0.0
        self._neural_phase = 0.0
        
        # Colors
        self.base_color = QColor(0, 122, 204)
        self.glow_color = QColor(0, 200, 255)
        self.text_color = QColor(255, 255, 255)
        
        # Animations
        self.glow_animation = QPropertyAnimation(self, b"glow_intensity")
        self.glow_animation.setDuration(200)
        self.glow_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.ripple_animation = QPropertyAnimation(self, b"ripple_radius")
        self.ripple_animation.setDuration(600)
        self.ripple_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Neural network animation timer
        self.neural_timer = QTimer()
        self.neural_timer.timeout.connect(self._update_neural)
        self.neural_timer.start(50)
        
        # Style
        self.setMinimumSize(120, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(self.glow_color)
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
        
    def get_glow_intensity(self) -> float:
        return self._glow_intensity
        
    def set_glow_intensity(self, value: float):
        self._glow_intensity = value
        self.update()
        
        # Update shadow
        shadow = self.graphicsEffect()
        if shadow:
            shadow.setBlurRadius(20 + value * 30)
            
    def get_ripple_radius(self) -> float:
        return self._ripple_radius
        
    def set_ripple_radius(self, value: float):
        self._ripple_radius = value
        self.update()
        
    glow_intensity = Property(float, get_glow_intensity, set_glow_intensity)
    ripple_radius = Property(float, get_ripple_radius, set_ripple_radius)
    
    def _update_neural(self):
        """Update neural network animation"""
        self._neural_phase += 0.1
        self.update()
        
    def enterEvent(self, event):
        """Mouse enter"""
        super().enterEvent(event)
        self.glow_animation.setStartValue(self._glow_intensity)
        self.glow_animation.setEndValue(1.0)
        self.glow_animation.start()
        
    def leaveEvent(self, event):
        """Mouse leave"""
        super().leaveEvent(event)
        self.glow_animation.setStartValue(self._glow_intensity)
        self.glow_animation.setEndValue(0.0)
        self.glow_animation.start()
        
    def mousePressEvent(self, event):
        """Mouse press"""
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.ripple_animation.setStartValue(0)
            self.ripple_animation.setEndValue(self.width())
            self.ripple_animation.start()
            
    def paintEvent(self, event):
        """Custom paint"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Draw neural network background
        self._draw_neural_network(painter, rect)
        
        # Draw button background
        gradient = QLinearGradient(0, 0, 0, rect.height())
        base = self.base_color
        gradient.setColorAt(0, base.lighter(120))
        gradient.setColorAt(1, base.darker(110))
        
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 10, 10)
        
        painter.fillPath(path, gradient)
        
        # Draw glow
        if self._glow_intensity > 0:
            glow_pen = QPen(self.glow_color, 2 + self._glow_intensity * 2)
            glow_pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(glow_pen)
            painter.drawPath(path)
            
        # Draw ripple effect
        if self._ripple_radius > 0:
            ripple_color = QColor(self.glow_color)
            ripple_color.setAlpha(int(100 * (1 - self._ripple_radius / self.width())))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(ripple_color)
            
            center = rect.center()
            painter.drawEllipse(center, self._ripple_radius, self._ripple_radius)
            
        # Draw text
        painter.setPen(self.text_color)
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())
        
    def _draw_neural_network(self, painter: QPainter, rect: QRectF):
        """Draw animated neural network pattern"""
        painter.save()
        
        # Neural nodes
        node_count = 5
        connections = []
        
        for i in range(node_count):
            angle = (i / node_count) * 2 * math.pi + self._neural_phase
            radius = min(rect.width(), rect.height()) * 0.3
            
            x = rect.center().x() + radius * math.cos(angle)
            y = rect.center().y() + radius * math.sin(angle)
            
            # Draw connections
            for j in range(i + 1, node_count):
                angle2 = (j / node_count) * 2 * math.pi + self._neural_phase
                x2 = rect.center().x() + radius * math.cos(angle2)
                y2 = rect.center().y() + radius * math.sin(angle2)
                
                # Animated opacity
                opacity = (math.sin(self._neural_phase + i + j) + 1) / 2 * 50
                pen = QPen(QColor(0, 200, 255, int(opacity)), 1)
                painter.setPen(pen)
                painter.drawLine(QPointF(x, y), QPointF(x2, y2))
                
        painter.restore()


class CircularProgressBar(QWidget):
    """Advanced circular progress bar with animations"""
    
    valueChanged = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Properties
        self._value = 0
        self._maximum = 100
        self._thickness = 10
        self._animated_value = 0.0
        
        # Colors
        self.background_color = QColor(50, 50, 50)
        self.progress_color = QColor(0, 200, 255)
        self.text_color = QColor(255, 255, 255)
        
        # Animation
        self.animation = QPropertyAnimation(self, b"animated_value")
        self.animation.setDuration(500)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Size
        self.setMinimumSize(100, 100)
        
    def get_animated_value(self) -> float:
        return self._animated_value
        
    def set_animated_value(self, value: float):
        self._animated_value = value
        self.update()
        
    animated_value = Property(float, get_animated_value, set_animated_value)
    
    def value(self) -> int:
        return self._value
        
    def setValue(self, value: int):
        """Set progress value with animation"""
        value = max(0, min(value, self._maximum))
        if value != self._value:
            self._value = value
            
            # Animate to new value
            self.animation.setStartValue(self._animated_value)
            self.animation.setEndValue(float(value))
            self.animation.start()
            
            self.valueChanged.emit(value)
            
    def maximum(self) -> int:
        return self._maximum
        
    def setMaximum(self, maximum: int):
        self._maximum = max(1, maximum)
        self.update()
        
    def paintEvent(self, event):
        """Custom paint"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate dimensions
        rect = self.rect()
        side = min(rect.width(), rect.height())
        painter.setViewport((rect.width() - side) // 2, (rect.height() - side) // 2, side, side)
        painter.setWindow(0, 0, 100, 100)
        
        # Draw background circle
        pen = QPen(self.background_color, self._thickness)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(10, 10, 80, 80, 90 * 16, -360 * 16)
        
        # Draw progress arc
        if self._animated_value > 0:
            gradient = QConicalGradient(50, 50, 90)
            gradient.setColorAt(0, self.progress_color.darker(150))
            gradient.setColorAt(1, self.progress_color)
            
            pen = QPen(QBrush(gradient), self._thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            
            span_angle = int(-360 * 16 * (self._animated_value / self._maximum))
            painter.drawArc(10, 10, 80, 80, 90 * 16, span_angle)
            
        # Draw percentage text
        painter.setPen(self.text_color)
        font = QFont("Arial", 20, QFont.Weight.Bold)
        painter.setFont(font)
        
        percentage = int((self._animated_value / self._maximum) * 100)
        painter.drawText(0, 0, 100, 100, Qt.AlignmentFlag.AlignCenter, f"{percentage}%")


class WaveformWidget(QWidget):
    """Real-time waveform visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Waveform data
        self.samples = 256
        self.waveform_data = np.zeros(self.samples)
        self.history = []
        self.max_history = 50
        
        # Colors
        self.waveform_color = QColor(0, 255, 200)
        self.background_color = QColor(20, 20, 20)
        self.grid_color = QColor(40, 40, 40)
        
        # Animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)  # 33 FPS
        
        self.setMinimumHeight(100)
        
    def update_waveform(self, data: np.ndarray):
        """Update waveform data"""
        if len(data) != self.samples:
            data = np.interp(
                np.linspace(0, len(data), self.samples),
                np.arange(len(data)),
                data
            )
            
        self.waveform_data = data
        
        # Add to history for persistence effect
        self.history.append(data.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def paintEvent(self, event):
        """Custom paint"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.background_color)
        
        # Grid
        self._draw_grid(painter)
        
        # Draw history (fading effect)
        for i, hist_data in enumerate(self.history[:-1]):
            alpha = int(50 * (i / len(self.history)))
            color = QColor(self.waveform_color)
            color.setAlpha(alpha)
            self._draw_waveform(painter, hist_data, color, 1)
            
        # Draw current waveform
        self._draw_waveform(painter, self.waveform_data, self.waveform_color, 2)
        
        # Draw glow effect
        self._draw_glow(painter)
        
    def _draw_grid(self, painter: QPainter):
        """Draw background grid"""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Horizontal lines
        h_lines = 5
        for i in range(h_lines):
            y = int(self.height() * i / (h_lines - 1))
            painter.drawLine(0, y, self.width(), y)
            
        # Vertical lines
        v_lines = 10
        for i in range(v_lines):
            x = int(self.width() * i / (v_lines - 1))
            painter.drawLine(x, 0, x, self.height())
            
    def _draw_waveform(self, painter: QPainter, data: np.ndarray, color: QColor, width: int):
        """Draw waveform data"""
        painter.setPen(QPen(color, width))
        
        path = QPainterPath()
        
        for i in range(len(data)):
            x = (i / len(data)) * self.width()
            y = self.height() / 2 - (data[i] * self.height() / 2)
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
                
        painter.drawPath(path)
        
    def _draw_glow(self, painter: QPainter):
        """Draw glow effect on peaks"""
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Find peaks
        threshold = 0.7
        for i in range(len(self.waveform_data)):
            if abs(self.waveform_data[i]) > threshold:
                x = (i / len(self.waveform_data)) * self.width()
                y = self.height() / 2 - (self.waveform_data[i] * self.height() / 2)
                
                # Radial gradient for glow
                gradient = QRadialGradient(x, y, 20)
                glow_color = QColor(self.waveform_color)
                glow_color.setAlpha(100)
                gradient.setColorAt(0, glow_color)
                gradient.setColorAt(1, Qt.GlobalColor.transparent)
                
                painter.setBrush(gradient)
                painter.drawEllipse(QPointF(x, y), 20, 20)


class NeuralNetworkVisualization(QWidget):
    """Interactive neural network visualization"""
    
    def __init__(self, layers: List[int] = None, parent=None):
        super().__init__(parent)
        
        self.layers = layers or [4, 6, 6, 3]
        self.node_radius = 15
        self.activation_values = {}
        self.connection_weights = {}
        self.selected_node = None
        
        # Animation
        self.pulse_phase = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        
    def _animate(self):
        """Update animation"""
        self.pulse_phase += 0.1
        
        # Simulate random activations
        for layer_idx in range(len(self.layers)):
            for node_idx in range(self.layers[layer_idx]):
                key = (layer_idx, node_idx)
                if key not in self.activation_values:
                    self.activation_values[key] = 0
                    
                # Random walk
                change = (np.random.random() - 0.5) * 0.1
                self.activation_values[key] = max(0, min(1, 
                    self.activation_values[key] + change))
                    
        self.update()
        
    def paintEvent(self, event):
        """Custom paint"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(10, 10, 10))
        
        # Calculate positions
        positions = self._calculate_positions()
        
        # Draw connections
        self._draw_connections(painter, positions)
        
        # Draw nodes
        self._draw_nodes(painter, positions)
        
    def _calculate_positions(self) -> Dict[Tuple[int, int], QPointF]:
        """Calculate node positions"""
        positions = {}
        
        layer_width = self.width() / (len(self.layers) + 1)
        
        for layer_idx, layer_size in enumerate(self.layers):
            x = layer_width * (layer_idx + 1)
            layer_height = self.height() / (layer_size + 1)
            
            for node_idx in range(layer_size):
                y = layer_height * (node_idx + 1)
                positions[(layer_idx, node_idx)] = QPointF(x, y)
                
        return positions
        
    def _draw_connections(self, painter: QPainter, positions: Dict):
        """Draw connections between nodes"""
        for layer_idx in range(len(self.layers) - 1):
            for node_idx in range(self.layers[layer_idx]):
                for next_node_idx in range(self.layers[layer_idx + 1]):
                    start = positions[(layer_idx, node_idx)]
                    end = positions[(layer_idx + 1, next_node_idx)]
                    
                    # Get activation values
                    start_activation = self.activation_values.get((layer_idx, node_idx), 0)
                    end_activation = self.activation_values.get((layer_idx + 1, next_node_idx), 0)
                    
                    # Connection strength based on activations
                    strength = (start_activation + end_activation) / 2
                    
                    # Animated pulse
                    pulse = (math.sin(self.pulse_phase + layer_idx + node_idx) + 1) / 2
                    
                    # Color and width based on strength
                    color = QColor(0, int(100 + 155 * strength), int(200 + 55 * strength))
                    color.setAlpha(int(50 + 150 * strength * pulse))
                    
                    pen = QPen(color, 1 + strength * 2)
                    painter.setPen(pen)
                    painter.drawLine(start, end)
                    
    def _draw_nodes(self, painter: QPainter, positions: Dict):
        """Draw neural network nodes"""
        for (layer_idx, node_idx), pos in positions.items():
            activation = self.activation_values.get((layer_idx, node_idx), 0)
            
            # Node color based on activation
            base_color = QColor(0, 150, 255)
            node_color = QColor(
                int(base_color.red() + (255 - base_color.red()) * activation),
                int(base_color.green() + (255 - base_color.green()) * activation),
                int(base_color.blue())
            )
            
            # Outer glow
            if activation > 0.5:
                glow_radius = self.node_radius + 10 * activation
                gradient = QRadialGradient(pos, glow_radius)
                glow_color = QColor(node_color)
                glow_color.setAlpha(100)
                gradient.setColorAt(0, glow_color)
                gradient.setColorAt(1, Qt.GlobalColor.transparent)
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(gradient)
                painter.drawEllipse(pos, glow_radius, glow_radius)
                
            # Node circle
            painter.setPen(QPen(node_color.lighter(150), 2))
            painter.setBrush(node_color)
            painter.drawEllipse(pos, self.node_radius, self.node_radius)
            
            # Inner highlight
            highlight_pos = pos - QPointF(self.node_radius * 0.3, self.node_radius * 0.3)
            highlight_radius = self.node_radius * 0.3
            
            gradient = QRadialGradient(highlight_pos, highlight_radius)
            gradient.setColorAt(0, QColor(255, 255, 255, 150))
            gradient.setColorAt(1, Qt.GlobalColor.transparent)
            
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(highlight_pos, highlight_radius, highlight_radius)
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement"""
        # Update node activations based on mouse proximity
        positions = self._calculate_positions()
        mouse_pos = event.position()
        
        for key, pos in positions.items():
            distance = math.sqrt(
                (mouse_pos.x() - pos.x())**2 + 
                (mouse_pos.y() - pos.y())**2
            )
            
            if distance < 100:
                influence = 1 - (distance / 100)
                self.activation_values[key] = min(1, 
                    self.activation_values.get(key, 0) + influence * 0.1)


class GradientSlider(QSlider):
    """Custom slider with gradient track"""
    
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        
        self.gradient_colors = [
            QColor(0, 100, 200),
            QColor(0, 200, 255),
            QColor(0, 255, 200)
        ]
        
        self.setMinimumHeight(30)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 2px solid #00C8FF;
                width: 20px;
                height: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #00C8FF;
            }
        """)
        
    def paintEvent(self, event):
        """Custom paint for gradient track"""
        # First draw the default slider
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw gradient track
        rect = self.rect()
        track_height = 10
        track_y = (rect.height() - track_height) // 2
        track_rect = QRectF(10, track_y, rect.width() - 20, track_height)
        
        # Create gradient
        gradient = QLinearGradient(track_rect.left(), 0, track_rect.right(), 0)
        for i, color in enumerate(self.gradient_colors):
            gradient.setColorAt(i / (len(self.gradient_colors) - 1), color)
            
        # Draw track
        path = QPainterPath()
        path.addRoundedRect(track_rect, 5, 5)
        painter.fillPath(path, gradient)
        
        # Draw value indicator
        value_percent = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
        indicator_x = track_rect.left() + value_percent * track_rect.width()
        
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(
            QPointF(indicator_x, track_rect.top() - 5),
            QPointF(indicator_x, track_rect.bottom() + 5)
        )