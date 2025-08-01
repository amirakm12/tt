"""
AI-ARTWORKS - Production-Grade Creative Suite
GPU-Accelerated Desktop Application with Qt6
"""

__version__ = "1.0.0"
__author__ = "AI-ARTWORKS Team"

# Core modules
from .core import *
from .plugins import *
from .ui import *

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)