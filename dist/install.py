#!/usr/bin/env python3
"""
AI System Universal Installer
Automatically detects platform and installs appropriate package
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path

class UniversalInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.base_url = "https://github.com/ai-system/ai-system/releases/latest/download"
        
    def detect_platform(self):
        """Detect the current platform."""
        if self.system == "windows":
            return "windows"
        elif self.system == "darwin":
            return "macos"
        elif self.system == "linux":
            return "linux"
        else:
            raise Exception(f"Unsupported platform: {self.system}")
            
    def download_package(self, platform):
        """Download the appropriate package."""
        if platform == "windows":
            filename = "AI-System-1.0.0-windows.zip"
        elif platform == "linux":
            filename = "AI-System-1.0.0-linux.tar.gz"
        elif platform == "macos":
            filename = "AI-System-1.0.0-macos.zip"
            
        url = f"{self.base_url}/{filename}"
        print(f"Downloading {filename}...")
        
        urllib.request.urlretrieve(url, filename)
        return filename
        
    def extract_package(self, filename):
        """Extract the downloaded package."""
        print(f"Extracting {filename}...")
        
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall()
        elif filename.endswith('.tar.gz'):
            with tarfile.open(filename, 'r:gz') as tar_ref:
                tar_ref.extractall()
                
    def install(self):
        """Main installation process."""
        try:
            platform = self.detect_platform()
            print(f"Detected platform: {platform}")
            
            filename = self.download_package(platform)
            self.extract_package(filename)
            
            # Run platform-specific installation
            if platform == "windows":
                print("Please run the extracted setup.exe file to complete installation.")
            elif platform == "linux":
                os.system("chmod +x install.sh && ./install.sh")
            elif platform == "macos":
                print("Please drag the AI-System.app to your Applications folder.")
                
            print("Installation completed successfully!")
            
        except Exception as e:
            print(f"Installation failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    installer = UniversalInstaller()
    installer.install()
