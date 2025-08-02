"""
Image Processing Module
Core image manipulation functionality
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Tuple, Union
import cv2
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Basic image processing functionality"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load an image from file"""
        try:
            image = Image.open(path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
            
    def save_image(self, image: np.ndarray, path: str, quality: int = 95) -> bool:
        """Save an image to file"""
        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            pil_image.save(path, quality=quality)
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {path}: {e}")
            return False
            
    def resize_image(self, image: np.ndarray, size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """Resize an image"""
        h, w = image.shape[:2]
        target_w, target_h = size
        
        if maintain_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w, new_h = target_w, target_h
            
        # Use cv2 for better quality resizing
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized
        
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
        
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
        
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
        
    def apply_blur(self, image: np.ndarray, radius: int = 2) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
        
    def apply_sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) * amount
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    def detect_edges(self, image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """Detect edges using Canny edge detection"""
        gray = self.convert_to_grayscale(image)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
        
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
        
    def flip_image(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Flip image horizontally or vertically"""
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
            
    def crop_image(self, image: np.ndarray, x: int, y: int, 
                  width: int, height: int) -> np.ndarray:
        """Crop image to specified region"""
        return image[y:y+height, x:x+width]
        
    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """Automatically enhance image"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced