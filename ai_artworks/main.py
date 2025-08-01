#!/usr/bin/env python3
"""
AI-ARTWORKS Main Application
Core functionality entry point
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    try:
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("AI-ARTWORKS")
        app.setOrganizationName("AI-ARTWORKS")
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        logger.info("AI-ARTWORKS application started")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()