"""
GPU-Accelerated Canvas
High-performance OpenGL-based canvas for real-time image rendering and editing
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging

from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QImage, QPainter, QColor, QTransform, QMouseEvent, 
    QWheelEvent, QKeyEvent, QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat, QMatrix4x4, QVector3D
)
from PySide6.QtWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture
from PySide6.QtOpenGLWidgets import QOpenGLWidget as ModernGLWidget

import OpenGL.GL as GL
from OpenGL.arrays import vbo
import cv2

logger = logging.getLogger(__name__)


class Layer:
    """Represents a single layer in the canvas"""
    
    def __init__(self, name: str, image: np.ndarray):
        self.name = name
        self.image = image
        self.visible = True
        self.opacity = 1.0
        self.blend_mode = "normal"
        self.transform = QTransform()
        self.texture_id = None
        self.dirty = True
        
    def update_texture(self):
        """Mark texture for update"""
        self.dirty = True


class GPUCanvas(ModernGLWidget):
    """GPU-accelerated canvas for real-time image editing"""
    
    # Signals
    layer_added = Signal(str)
    layer_removed = Signal(str)
    layer_changed = Signal(str)
    zoom_changed = Signal(float)
    tool_used = Signal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Canvas state
        self.layers: List[Layer] = []
        self.active_layer_index = -1
        self.background_color = QColor(32, 32, 32)
        
        # View state
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.rotation = 0.0
        
        # Interaction state
        self.is_panning = False
        self.last_mouse_pos = QPointF()
        self.current_tool = None
        
        # OpenGL resources
        self.shader_program = None
        self.vao = None
        self.vbo_vertices = None
        self.framebuffer = None
        self.render_texture = None
        
        # Performance
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.frame_count = 0
        self.current_fps = 0
        
        # Enable multisampling for better quality
        format = self.format()
        format.setSamples(4)
        self.setFormat(format)
        
    def initializeGL(self):
        """Initialize OpenGL resources"""
        # Set up OpenGL state
        GL.glClearColor(
            self.background_color.redF(),
            self.background_color.greenF(),
            self.background_color.blueF(),
            1.0
        )
        
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        
        # Create shader program
        self.create_shaders()
        
        # Create vertex buffer for quad
        self.create_geometry()
        
        # Create framebuffer for offscreen rendering
        self.create_framebuffer()
        
        logger.info("GPU Canvas initialized")
        
    def create_shaders(self):
        """Create and compile shaders"""
        vertex_shader_source = """
        #version 330 core
        
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 texCoord;
        
        out vec2 fragTexCoord;
        
        uniform mat4 mvp_matrix;
        
        void main() {
            gl_Position = mvp_matrix * vec4(position, 1.0);
            fragTexCoord = texCoord;
        }
        """
        
        fragment_shader_source = """
        #version 330 core
        
        in vec2 fragTexCoord;
        out vec4 fragColor;
        
        uniform sampler2D texture0;
        uniform float opacity;
        uniform int blend_mode;
        
        vec3 blend_normal(vec3 base, vec3 blend, float opacity) {
            return mix(base, blend, opacity);
        }
        
        vec3 blend_multiply(vec3 base, vec3 blend, float opacity) {
            return mix(base, base * blend, opacity);
        }
        
        vec3 blend_screen(vec3 base, vec3 blend, float opacity) {
            return mix(base, 1.0 - (1.0 - base) * (1.0 - blend), opacity);
        }
        
        vec3 blend_overlay(vec3 base, vec3 blend, float opacity) {
            vec3 result;
            for (int i = 0; i < 3; i++) {
                if (base[i] < 0.5) {
                    result[i] = 2.0 * base[i] * blend[i];
                } else {
                    result[i] = 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - blend[i]);
                }
            }
            return mix(base, result, opacity);
        }
        
        void main() {
            vec4 texColor = texture(texture0, fragTexCoord);
            
            // Apply blend mode
            vec3 blended = texColor.rgb;
            
            // Apply opacity
            fragColor = vec4(blended, texColor.a * opacity);
        }
        """
        
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, vertex_shader_source
        )
        self.shader_program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, fragment_shader_source
        )
        self.shader_program.link()
        
    def create_geometry(self):
        """Create vertex buffer for rendering quads"""
        # Quad vertices with texture coordinates
        vertices = np.array([
            # Position        # TexCoord
            -1.0, -1.0, 0.0,  0.0, 1.0,
             1.0, -1.0, 0.0,  1.0, 1.0,
             1.0,  1.0, 0.0,  1.0, 0.0,
            -1.0,  1.0, 0.0,  0.0, 0.0,
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        
        # Create VBO
        self.vbo_vertices = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_vertices)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        
        # Position attribute
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, None)
        GL.glEnableVertexAttribArray(0)
        
        # TexCoord attribute
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(1)
        
        # Create EBO
        self.ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        
        GL.glBindVertexArray(0)
        
    def create_framebuffer(self):
        """Create framebuffer for offscreen rendering"""
        # This will be created when we know the size
        pass
        
    def resizeGL(self, width: int, height: int):
        """Handle canvas resize"""
        GL.glViewport(0, 0, width, height)
        
        # Recreate framebuffer with new size
        if self.framebuffer:
            self.framebuffer.release()
            
        format = QOpenGLFramebufferObjectFormat()
        format.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
        format.setSamples(4)
        
        self.framebuffer = QOpenGLFramebufferObject(width, height, format)
        
    def paintGL(self):
        """Render the canvas"""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        if not self.shader_program:
            return
            
        self.shader_program.bind()
        GL.glBindVertexArray(self.vao)
        
        # Calculate view matrix
        view_matrix = QMatrix4x4()
        view_matrix.translate(self.pan_offset.x(), self.pan_offset.y(), 0)
        view_matrix.scale(self.zoom)
        view_matrix.rotate(self.rotation, 0, 0, 1)
        
        # Render each visible layer
        for layer in self.layers:
            if not layer.visible:
                continue
                
            # Update texture if needed
            if layer.dirty or layer.texture_id is None:
                self.update_layer_texture(layer)
                
            # Bind texture
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, layer.texture_id)
            
            # Set uniforms
            self.shader_program.setUniformValue("texture0", 0)
            self.shader_program.setUniformValue("opacity", layer.opacity)
            self.shader_program.setUniformValue("mvp_matrix", view_matrix)
            
            # Draw quad
            GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
            
        GL.glBindVertexArray(0)
        self.shader_program.release()
        
        self.frame_count += 1
        
    def update_layer_texture(self, layer: Layer):
        """Update OpenGL texture from layer image"""
        if layer.texture_id is None:
            layer.texture_id = GL.glGenTextures(1)
            
        # Convert image to OpenGL format
        height, width = layer.image.shape[:2]
        if len(layer.image.shape) == 2:
            # Grayscale
            image_data = cv2.cvtColor(layer.image, cv2.COLOR_GRAY2RGBA)
        elif layer.image.shape[2] == 3:
            # RGB
            image_data = cv2.cvtColor(layer.image, cv2.COLOR_BGR2RGBA)
        else:
            # RGBA
            image_data = layer.image.copy()
            
        # Flip vertically for OpenGL
        image_data = np.flipud(image_data)
        
        # Upload to GPU
        GL.glBindTexture(GL.GL_TEXTURE_2D, layer.texture_id)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
            width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
            image_data
        )
        
        # Set texture parameters
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        
        layer.dirty = False
        
    def add_layer(self, name: str, image: np.ndarray, index: Optional[int] = None):
        """Add a new layer"""
        layer = Layer(name, image)
        
        if index is None:
            self.layers.append(layer)
            self.active_layer_index = len(self.layers) - 1
        else:
            self.layers.insert(index, layer)
            self.active_layer_index = index
            
        self.layer_added.emit(name)
        self.update()
        
    def remove_layer(self, index: int):
        """Remove a layer"""
        if 0 <= index < len(self.layers):
            layer = self.layers.pop(index)
            if layer.texture_id:
                GL.glDeleteTextures(1, [layer.texture_id])
                
            self.layer_removed.emit(layer.name)
            
            # Update active layer index
            if self.active_layer_index >= len(self.layers):
                self.active_layer_index = len(self.layers) - 1
                
            self.update()
            
    def get_active_layer(self) -> Optional[Layer]:
        """Get the currently active layer"""
        if 0 <= self.active_layer_index < len(self.layers):
            return self.layers[self.active_layer_index]
        return None
        
    def set_layer_visibility(self, index: int, visible: bool):
        """Toggle layer visibility"""
        if 0 <= index < len(self.layers):
            self.layers[index].visible = visible
            self.layer_changed.emit(self.layers[index].name)
            self.update()
            
    def set_layer_opacity(self, index: int, opacity: float):
        """Set layer opacity"""
        if 0 <= index < len(self.layers):
            self.layers[index].opacity = max(0.0, min(1.0, opacity))
            self.layer_changed.emit(self.layers[index].name)
            self.update()
            
    def composite_layers(self) -> np.ndarray:
        """Composite all visible layers into a single image"""
        if not self.layers:
            return np.zeros((512, 512, 4), dtype=np.uint8)
            
        # Find canvas size
        max_height = max(layer.image.shape[0] for layer in self.layers)
        max_width = max(layer.image.shape[1] for layer in self.layers)
        
        # Create composite image
        composite = np.zeros((max_height, max_width, 4), dtype=np.float32)
        
        for layer in self.layers:
            if not layer.visible:
                continue
                
            # Convert to RGBA if needed
            if len(layer.image.shape) == 2:
                rgba = cv2.cvtColor(layer.image, cv2.COLOR_GRAY2RGBA)
            elif layer.image.shape[2] == 3:
                rgba = cv2.cvtColor(layer.image, cv2.COLOR_BGR2RGBA)
            else:
                rgba = layer.image.copy()
                
            # Apply opacity
            rgba = rgba.astype(np.float32) / 255.0
            rgba[:, :, 3] *= layer.opacity
            
            # Blend with composite
            alpha = rgba[:, :, 3:4]
            composite[:rgba.shape[0], :rgba.shape[1]] = (
                composite[:rgba.shape[0], :rgba.shape[1]] * (1 - alpha) +
                rgba * alpha
            )
            
        # Convert back to uint8
        return (composite * 255).astype(np.uint8)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool:
                self.current_tool.on_mouse_press(event)
                
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move"""
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.position()
            self.update()
        elif self.current_tool:
            self.current_tool.on_mouse_move(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool:
                self.current_tool.on_mouse_release(event)
                
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10.0, self.zoom))
        self.zoom_changed.emit(self.zoom)
        self.update()
        
    def set_tool(self, tool):
        """Set the current tool"""
        if self.current_tool:
            self.current_tool.deactivate()
        self.current_tool = tool
        if tool:
            tool.activate()
            
    def fit_to_window(self):
        """Fit the canvas content to window"""
        if not self.layers:
            return
            
        # Get canvas bounds
        max_height = max(layer.image.shape[0] for layer in self.layers)
        max_width = max(layer.image.shape[1] for layer in self.layers)
        
        # Calculate zoom to fit
        widget_width = self.width()
        widget_height = self.height()
        
        zoom_x = widget_width / max_width
        zoom_y = widget_height / max_height
        
        self.zoom = min(zoom_x, zoom_y) * 0.9  # 90% to add some padding
        self.pan_offset = QPointF(0, 0)
        self.rotation = 0.0
        
        self.zoom_changed.emit(self.zoom)
        self.update()
        
    def update_fps(self):
        """Update FPS counter"""
        self.current_fps = self.frame_count
        self.frame_count = 0
        
    def get_fps(self) -> int:
        """Get current FPS"""
        return self.current_fps