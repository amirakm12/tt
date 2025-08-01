"""
Real-ESRGAN Upscale Plugin
High-quality image upscaling using Real-ESRGAN
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QSpinBox

from ..core.plugin_manager import FilterPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class RealESRGANPlugin(FilterPlugin):
    """Real-ESRGAN upscaling plugin"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            id="realesrgan_upscale",
            name="Real-ESRGAN Upscale",
            version="1.0.0",
            author="AI-ARTWORKS",
            description="High-quality image upscaling using Real-ESRGAN",
            category="filter",
            tags=["upscale", "enhance", "AI"]
        )
        
    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            # Import Real-ESRGAN
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Model paths
            model_name = "RealESRGAN_x4plus"
            model_path = Path(__file__).parent / "models" / f"{model_name}.pth"
            
            # Initialize model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            
            # Initialize upsampler
            self.model = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            
            logger.info(f"Real-ESRGAN initialized on {self.device}")
            return True
            
        except ImportError:
            logger.warning("Real-ESRGAN not installed, using fallback upscaling")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            return False
            
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
            
    def get_widget(self) -> Optional[QWidget]:
        """Get the plugin's UI widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scale selection
        layout.addWidget(QLabel("Scale Factor:"))
        scale_combo = QComboBox()
        scale_combo.addItems(["2x", "4x", "8x"])
        scale_combo.setCurrentText("4x")
        scale_combo.currentTextChanged.connect(self._on_scale_changed)
        layout.addWidget(scale_combo)
        
        # Model selection
        layout.addWidget(QLabel("Model:"))
        model_combo = QComboBox()
        model_combo.addItems([
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime",
            "RealESRGAN_x2plus"
        ])
        model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(model_combo)
        
        # Tile size (for GPU memory management)
        layout.addWidget(QLabel("Tile Size:"))
        tile_spin = QSpinBox()
        tile_spin.setRange(0, 1024)
        tile_spin.setValue(0)
        tile_spin.setSpecialValueText("Auto")
        tile_spin.setToolTip("0 = Auto, larger values use less memory but may be slower")
        layout.addWidget(tile_spin)
        
        layout.addStretch()
        
        self.widget = widget
        return widget
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        params = {
            "scale": 4,
            "model": "RealESRGAN_x4plus",
            "tile": 0
        }
        
        if hasattr(self, "widget"):
            # Extract from UI
            pass
            
        return params
        
    def apply(self, image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply the upscaling filter"""
        if self.model is None:
            # Fallback to basic upscaling
            return self._fallback_upscale(image, parameters.get("scale", 4))
            
        try:
            # Ensure image is in the right format
            if len(image.shape) == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
                
            # Apply Real-ESRGAN
            output, _ = self.model.enhance(image, outscale=parameters.get("scale", 4))
            
            # Convert back to original format
            if len(image.shape) == 2:
                output = np.mean(output, axis=-1).astype(image.dtype)
            elif image.shape[2] == 4:
                # Add alpha channel back
                alpha = cv2.resize(
                    image[:, :, 3],
                    (output.shape[1], output.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )
                output = np.dstack([output, alpha])
                
            return output
            
        except Exception as e:
            logger.error(f"Real-ESRGAN processing failed: {e}")
            return self._fallback_upscale(image, parameters.get("scale", 4))
            
    def _fallback_upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        """Fallback upscaling using OpenCV"""
        import cv2
        
        height, width = image.shape[:2]
        new_height = height * scale
        new_width = width * scale
        
        # Use Lanczos interpolation for better quality
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
    @Slot(str)
    def _on_scale_changed(self, scale_text: str):
        """Handle scale change"""
        scale = int(scale_text[0])
        # Update model if needed
        
    @Slot(str)
    def _on_model_changed(self, model_name: str):
        """Handle model change"""
        # Reload model with new weights
        pass
        
    def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            # Clear GPU memory
            try:
                import torch
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except:
                pass
            self.model = None