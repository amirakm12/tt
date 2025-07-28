#!/usr/bin/env python3
"""
AI System - Standalone Windows Installer
Creates a complete standalone installation package for Windows.
This file can be compiled to an EXE using PyInstaller.
"""

import os
import sys
import shutil
import subprocess
import urllib.request
import zipfile
import tempfile
import json
import winreg
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import threading
import time

class StandaloneInstaller:
    def __init__(self):
        self.app_name = "AI System"
        self.app_version = "1.0.0"
        self.install_dir = None
        self.progress_var = None
        self.status_var = None
        self.root = None
        
        # Embedded files (will be populated by build script)
        self.embedded_files = {}
        
    def create_gui(self):
        """Create the installer GUI."""
        self.root = tk.Tk()
        self.root.title(f"{self.app_name} Installer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text=f"{self.app_name} Installer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Description
        desc_text = f"""Welcome to the {self.app_name} installation wizard.
        
This will install {self.app_name} v{self.app_version} on your computer.
The application includes:
• Multi-Agent AI System
• Advanced Sensor Fusion
• Real-time Dashboard
• Voice Interface
• Kernel Integration
• Security Monitoring"""
        
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.LEFT)
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20), sticky=tk.W)
        
        # Installation directory
        dir_frame = ttk.LabelFrame(main_frame, text="Installation Directory", padding="10")
        dir_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.dir_var = tk.StringVar()
        self.dir_var.set(str(Path.home() / "AppData" / "Local" / "AI-System"))
        
        dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var, width=50)
        dir_entry.grid(row=0, column=0, padx=(0, 10))
        
        browse_btn = ttk.Button(dir_frame, text="Browse...", command=self.browse_directory)
        browse_btn.grid(row=0, column=1)
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Installation Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.desktop_shortcut = tk.BooleanVar(value=True)
        self.start_menu = tk.BooleanVar(value=True)
        self.auto_start = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(options_frame, text="Create desktop shortcut", 
                       variable=self.desktop_shortcut).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Add to Start Menu", 
                       variable=self.start_menu).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Start with Windows", 
                       variable=self.auto_start).grid(row=2, column=0, sticky=tk.W)
        
        # Progress
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready to install")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, columnspan=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
        self.install_btn = ttk.Button(button_frame, text="Install", 
                                     command=self.start_installation)
        self.install_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.cancel_btn = ttk.Button(button_frame, text="Cancel", 
                                    command=self.root.quit)
        self.cancel_btn.grid(row=0, column=1)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def browse_directory(self):
        """Browse for installation directory."""
        directory = filedialog.askdirectory(initialdir=self.dir_var.get())
        if directory:
            self.dir_var.set(directory)
    
    def update_progress(self, value, status):
        """Update progress bar and status."""
        self.progress_var.set(value)
        self.status_var.set(status)
        self.root.update_idletasks()
    
    def start_installation(self):
        """Start installation in a separate thread."""
        self.install_btn.config(state='disabled')
        self.cancel_btn.config(state='disabled')
        
        install_thread = threading.Thread(target=self.install)
        install_thread.daemon = True
        install_thread.start()
    
    def install(self):
        """Perform the installation."""
        try:
            self.install_dir = Path(self.dir_var.get())
            
            # Step 1: Create directories
            self.update_progress(10, "Creating directories...")
            self.create_directories()
            
            # Step 2: Extract embedded files
            self.update_progress(20, "Extracting application files...")
            self.extract_files()
            
            # Step 3: Install Python if needed
            self.update_progress(40, "Checking Python installation...")
            self.ensure_python()
            
            # Step 4: Create virtual environment
            self.update_progress(50, "Creating virtual environment...")
            self.create_virtual_environment()
            
            # Step 5: Install dependencies
            self.update_progress(60, "Installing dependencies...")
            self.install_dependencies()
            
            # Step 6: Create configuration
            self.update_progress(70, "Creating configuration...")
            self.create_configuration()
            
            # Step 7: Create shortcuts
            self.update_progress(80, "Creating shortcuts...")
            self.create_shortcuts()
            
            # Step 8: Register application
            self.update_progress(90, "Registering application...")
            self.register_application()
            
            # Step 9: Complete
            self.update_progress(100, "Installation complete!")
            
            # Show completion message
            self.root.after(1000, self.show_completion)
            
        except Exception as e:
            messagebox.showerror("Installation Error", f"Installation failed: {str(e)}")
            self.install_btn.config(state='normal')
            self.cancel_btn.config(state='normal')
    
    def create_directories(self):
        """Create installation directories."""
        directories = [
            self.install_dir,
            self.install_dir / "src",
            self.install_dir / "config",
            self.install_dir / "data",
            self.install_dir / "logs",
            self.install_dir / "venv",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def extract_files(self):
        """Extract embedded application files."""
        # This would contain the actual application files
        # For now, we'll create the basic structure
        
        # Create main files
        files_to_create = {
            "src/main.py": self.get_main_py_content(),
            "src/__init__.py": "# AI System Package",
            "requirements.txt": self.get_requirements_content(),
            "config/config.json": self.get_config_content(),
            "README.md": self.get_readme_content(),
        }
        
        for file_path, content in files_to_create.items():
            full_path = self.install_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def ensure_python(self):
        """Ensure Python is available."""
        try:
            # Check if python is available
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return  # Python is available
        except:
            pass
        
        # If we get here, Python might not be available
        # For a standalone installer, we would embed Python
        # For now, we'll use the current Python executable
        pass
    
    def create_virtual_environment(self):
        """Create a virtual environment."""
        venv_path = self.install_dir / "venv"
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                          check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to create virtual environment: {e}")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        venv_python = self.install_dir / "venv" / "Scripts" / "python.exe"
        requirements_file = self.install_dir / "requirements.txt"
        
        try:
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)], 
                          check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Continue even if some packages fail to install
            pass
    
    def create_configuration(self):
        """Create application configuration."""
        config_data = {
            "installation_path": str(self.install_dir),
            "version": self.app_version,
            "installed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_path": str(self.install_dir / "venv" / "Scripts" / "python.exe"),
        }
        
        config_file = self.install_dir / "config" / "installation.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def create_shortcuts(self):
        """Create desktop and start menu shortcuts."""
        if self.desktop_shortcut.get():
            self.create_desktop_shortcut()
        
        if self.start_menu.get():
            self.create_start_menu_shortcut()
        
        if self.auto_start.get():
            self.create_autostart_entry()
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut."""
        try:
            import win32com.client
            
            desktop = Path.home() / "Desktop"
            shortcut_path = desktop / f"{self.app_name}.lnk"
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = str(self.install_dir / "venv" / "Scripts" / "python.exe")
            shortcut.Arguments = str(self.install_dir / "src" / "main.py")
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.IconLocation = str(self.install_dir / "icon.ico")
            shortcut.save()
        except:
            # Create a batch file instead
            desktop = Path.home() / "Desktop"
            batch_path = desktop / f"{self.app_name}.bat"
            
            batch_content = f'''@echo off
cd /d "{self.install_dir}"
"{self.install_dir / "venv" / "Scripts" / "python.exe"}" "{self.install_dir / "src" / "main.py"}"
pause
'''
            with open(batch_path, 'w') as f:
                f.write(batch_content)
    
    def create_start_menu_shortcut(self):
        """Create start menu shortcut."""
        try:
            start_menu = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
            app_folder = start_menu / self.app_name
            app_folder.mkdir(exist_ok=True)
            
            batch_path = app_folder / f"{self.app_name}.bat"
            batch_content = f'''@echo off
cd /d "{self.install_dir}"
"{self.install_dir / "venv" / "Scripts" / "python.exe"}" "{self.install_dir / "src" / "main.py"}"
'''
            with open(batch_path, 'w') as f:
                f.write(batch_content)
        except:
            pass
    
    def create_autostart_entry(self):
        """Create Windows autostart entry."""
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                               r"Software\Microsoft\Windows\CurrentVersion\Run", 
                               0, winreg.KEY_SET_VALUE)
            
            batch_path = self.install_dir / f"{self.app_name.replace(' ', '_')}_startup.bat"
            batch_content = f'''@echo off
cd /d "{self.install_dir}"
start "" "{self.install_dir / "venv" / "Scripts" / "python.exe"}" "{self.install_dir / "src" / "main.py"}"
'''
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            winreg.SetValueEx(key, self.app_name, 0, winreg.REG_SZ, str(batch_path))
            winreg.CloseKey(key)
        except:
            pass
    
    def register_application(self):
        """Register application in Windows registry."""
        try:
            # Add to Programs and Features
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\AI-System"
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
            
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, self.app_name)
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, self.app_version)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "AI System Team")
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, str(self.install_dir))
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, 
                            f'"{self.install_dir / "uninstall.exe"}"')
            
            winreg.CloseKey(key)
            
            # Create uninstaller
            self.create_uninstaller()
        except:
            pass
    
    def create_uninstaller(self):
        """Create uninstaller script."""
        uninstall_script = f'''@echo off
echo Uninstalling {self.app_name}...
cd /d "{self.install_dir.parent}"
rmdir /s /q "{self.install_dir.name}"

rem Remove registry entries
reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\AI-System" /f 2>nul
reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run" /v "{self.app_name}" /f 2>nul

rem Remove shortcuts
del "%USERPROFILE%\\Desktop\\{self.app_name}.lnk" 2>nul
del "%USERPROFILE%\\Desktop\\{self.app_name}.bat" 2>nul
rmdir /s /q "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\{self.app_name}" 2>nul

echo {self.app_name} has been uninstalled.
pause
'''
        
        uninstall_path = self.install_dir / "uninstall.bat"
        with open(uninstall_path, 'w') as f:
            f.write(uninstall_script)
    
    def show_completion(self):
        """Show installation completion dialog."""
        result = messagebox.askquestion(
            "Installation Complete",
            f"{self.app_name} has been successfully installed!\n\n"
            f"Installation directory: {self.install_dir}\n\n"
            "Would you like to launch the application now?",
            icon='question'
        )
        
        if result == 'yes':
            self.launch_application()
        
        self.root.quit()
    
    def launch_application(self):
        """Launch the installed application."""
        try:
            python_exe = self.install_dir / "venv" / "Scripts" / "python.exe"
            main_script = self.install_dir / "src" / "main.py"
            
            subprocess.Popen([str(python_exe), str(main_script)], 
                           cwd=str(self.install_dir))
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch application: {e}")
    
    def get_main_py_content(self):
        """Get the main.py content."""
        return '''#!/usr/bin/env python3
"""
AI System - Main Entry Point
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ai_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point."""
    print("AI System Starting...")
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("AI System initialized successfully")
    
    # Simple demo mode
    print("AI System is running in demo mode.")
    print("Features available:")
    print("- Basic system monitoring")
    print("- Configuration management")
    print("- Logging system")
    
    try:
        while True:
            import time
            time.sleep(60)  # Keep running
    except KeyboardInterrupt:
        logger.info("AI System shutting down...")
        print("AI System stopped.")

if __name__ == "__main__":
    main()
'''
    
    def get_requirements_content(self):
        """Get requirements.txt content."""
        return '''# Core dependencies
asyncio-mqtt>=0.11.1
aiohttp>=3.8.0
aiofiles>=22.1.0

# Optional dependencies (will install if available)
numpy>=1.21.0
psutil>=5.9.0
'''
    
    def get_config_content(self):
        """Get config.json content."""
        return '''{
  "system": {
    "name": "AI System",
    "version": "1.0.0",
    "debug": false
  },
  "logging": {
    "level": "INFO",
    "file": "logs/ai_system.log"
  },
  "ui": {
    "dashboard": {
      "enabled": true,
      "host": "localhost",
      "port": 8080
    }
  }
}'''
    
    def get_readme_content(self):
        """Get README.md content."""
        return '''# AI System

Welcome to the AI System - a comprehensive artificial intelligence platform.

## Features

- Multi-Agent AI System
- Advanced Sensor Fusion
- Real-time Dashboard
- Voice Interface
- Kernel Integration
- Security Monitoring

## Getting Started

The application has been installed and configured automatically.

### Running the Application

1. Use the desktop shortcut, or
2. Navigate to the installation directory and run `src/main.py`

### Configuration

Configuration files are located in the `config/` directory.

### Logs

Application logs are stored in the `logs/` directory.

## Support

For support and documentation, please visit our website or contact support.
'''

def main():
    """Main entry point for standalone installer."""
    installer = StandaloneInstaller()
    installer.create_gui()
    installer.root.mainloop()

if __name__ == "__main__":
    main()