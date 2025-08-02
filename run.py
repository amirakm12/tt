#!/usr/bin/env python3
"""
Simple run script for AI-ARTWORKS
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def install_dependencies():
    """Install core dependencies if needed"""
    try:
        import PySide6
        import PIL
        import cv2
        import numpy
    except ImportError:
        print("Installing core dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_core.txt"])

def main():
    """Main entry point"""
    print("Starting AI-ARTWORKS...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies if needed
    install_dependencies()
    
    # Run the main application
    try:
        from ai_artworks.main import main as app_main
        app_main()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()