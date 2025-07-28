#!/usr/bin/env python3
"""
AI System - Complete Installation Verification Script
Tests all components, imports, and functionality.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class InstallationVerifier:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def log_success(self, message):
        self.successes.append(message)
        logger.info(f"✓ {message}")
        
    def log_warning(self, message):
        self.warnings.append(message)
        logger.warning(f"⚠ {message}")
        
    def log_error(self, message):
        self.errors.append(message)
        logger.error(f"✗ {message}")
    
    def test_python_version(self):
        """Test Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log_success(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        else:
            self.log_error(f"Python {version.major}.{version.minor}.{version.micro} is not supported (requires 3.8+)")
    
    def test_core_imports(self):
        """Test core Python module imports."""
        core_modules = [
            'asyncio', 'logging', 'json', 'pathlib', 'typing',
            'dataclasses', 'enum', 'threading', 'collections',
            'subprocess', 'time', 'datetime', 'os', 'sys'
        ]
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                self.log_success(f"Core module '{module}' imports successfully")
            except ImportError as e:
                self.log_error(f"Core module '{module}' failed to import: {e}")
    
    def test_optional_dependencies(self):
        """Test optional third-party dependencies."""
        optional_deps = [
            ('uvloop', 'High-performance event loop'),
            ('cryptography', 'Encryption support'),
            ('psutil', 'System monitoring'),
            ('aiohttp', 'Web server'),
            ('jinja2', 'Template engine'),
            ('aiofiles', 'Async file operations'),
            ('numpy', 'Scientific computing'),
            ('torch', 'Deep learning'),
            ('transformers', 'NLP models'),
            ('sentence_transformers', 'Embeddings'),
            ('openai', 'OpenAI API'),
            ('langchain', 'LLM framework'),
            ('chromadb', 'Vector database'),
            ('scikit-learn', 'Machine learning'),
            ('pandas', 'Data analysis'),
            ('matplotlib', 'Plotting'),
            ('seaborn', 'Statistical plots'),
            ('redis', 'Database'),
            ('requests', 'HTTP client'),
            ('websockets', 'WebSocket support'),
            ('pyaudio', 'Audio processing'),
            ('pyttsx3', 'Text-to-speech'),
            ('speech_recognition', 'Speech recognition'),
            ('GPUtil', 'GPU monitoring'),
            ('pillow', 'Image processing'),
        ]
        
        for module_name, description in optional_deps:
            try:
                importlib.import_module(module_name)
                self.log_success(f"Optional dependency '{module_name}' ({description}) is available")
            except ImportError:
                self.log_warning(f"Optional dependency '{module_name}' ({description}) is not available")
    
    def test_ai_system_modules(self):
        """Test AI System module imports."""
        ai_modules = [
            'src.main',
            'src.core.config',
            'src.core.orchestrator',
            'src.ai.rag_engine',
            'src.ai.speculative_decoder',
            'src.agents.triage_agent',
            'src.agents.research_agent',
            'src.agents.orchestration_agent',
            'src.sensors.fusion',
            'src.kernel.integration',
            'src.ui.dashboard',
            'src.ui.voice_interface',
            'src.monitoring.system_monitor',
            'src.monitoring.security_monitor',
        ]
        
        for module in ai_modules:
            try:
                importlib.import_module(module)
                self.log_success(f"AI System module '{module}' imports successfully")
            except ImportError as e:
                self.log_error(f"AI System module '{module}' failed to import: {e}")
            except Exception as e:
                self.log_error(f"AI System module '{module}' has runtime error: {e}")
    
    def test_file_structure(self):
        """Test required files and directories exist."""
        required_files = [
            'src/__init__.py',
            'src/main.py',
            'src/core/__init__.py',
            'src/core/config.py',
            'src/core/orchestrator.py',
            'src/ai/__init__.py',
            'src/ai/rag_engine.py',
            'src/ai/speculative_decoder.py',
            'src/agents/__init__.py',
            'src/agents/triage_agent.py',
            'src/agents/research_agent.py',
            'src/agents/orchestration_agent.py',
            'src/sensors/__init__.py',
            'src/sensors/fusion.py',
            'src/kernel/__init__.py',
            'src/kernel/integration.py',
            'src/ui/__init__.py',
            'src/ui/dashboard.py',
            'src/ui/voice_interface.py',
            'src/ui/templates/dashboard.html',
            'src/ui/static/css/dashboard.css',
            'src/ui/static/js/dashboard.js',
            'src/monitoring/__init__.py',
            'src/monitoring/system_monitor.py',
            'src/monitoring/security_monitor.py',
            'config/config.json',
            'requirements.txt',
            'setup.py',
            'README.md',
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                self.log_success(f"Required file '{file_path}' exists")
            else:
                self.log_error(f"Required file '{file_path}' is missing")
    
    def test_configuration(self):
        """Test configuration loading."""
        try:
            from src.core.config import SystemConfig
            config = SystemConfig()
            self.log_success("Configuration system loads successfully")
            
            # Test config file loading
            config_file = Path("config/config.json")
            if config_file.exists():
                config.load_config()
                self.log_success("Configuration file loads successfully")
            else:
                self.log_warning("Configuration file not found, using defaults")
                
        except Exception as e:
            self.log_error(f"Configuration system failed: {e}")
    
    def test_entry_points(self):
        """Test setup.py entry points."""
        try:
            from src.main import run_system
            self.log_success("Main entry point 'run_system' is accessible")
        except ImportError as e:
            self.log_error(f"Main entry point failed to import: {e}")
    
    def test_installers(self):
        """Test installer scripts."""
        installer_files = [
            'install_executable.py',
            'create_installer.py',
            'create_exe.py',
            'install_windows_standalone.exe.py',
            'build_standalone_installer.py',
            'build_windows_installer.bat',
            'install_windows.bat',
            'install_windows.ps1',
        ]
        
        for installer in installer_files:
            path = Path(installer)
            if path.exists():
                self.log_success(f"Installer '{installer}' exists")
            else:
                self.log_warning(f"Installer '{installer}' is missing")
    
    def test_syntax_errors(self):
        """Test all Python files for syntax errors."""
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
                self.log_success(f"Python file '{py_file}' has valid syntax")
            except SyntaxError as e:
                self.log_error(f"Python file '{py_file}' has syntax error: {e}")
            except Exception as e:
                self.log_warning(f"Python file '{py_file}' could not be checked: {e}")
    
    def generate_report(self):
        """Generate final verification report."""
        print("\n" + "="*60)
        print("AI SYSTEM INSTALLATION VERIFICATION REPORT")
        print("="*60)
        
        print(f"\n✓ SUCCESSES: {len(self.successes)}")
        for success in self.successes[-5:]:  # Show last 5
            print(f"  • {success}")
        if len(self.successes) > 5:
            print(f"  ... and {len(self.successes) - 5} more")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print(f"\n✗ ERRORS: {len(self.errors)}")
            for error in self.errors:
                print(f"  • {error}")
        
        print(f"\nOVERALL STATUS:")
        if not self.errors:
            if not self.warnings:
                print("🟢 PERFECT - All components verified successfully!")
            else:
                print("🟡 GOOD - Core components work, some optional features unavailable")
        else:
            print("🔴 ISSUES FOUND - Some components need attention")
        
        print(f"\nSUMMARY:")
        print(f"  Total checks: {len(self.successes) + len(self.warnings) + len(self.errors)}")
        print(f"  Successes: {len(self.successes)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors: {len(self.errors)}")
        
        return len(self.errors) == 0

def main():
    """Run complete verification."""
    print("AI System - Complete Installation Verification")
    print("=" * 50)
    
    verifier = InstallationVerifier()
    
    print("\n1. Testing Python version...")
    verifier.test_python_version()
    
    print("\n2. Testing core imports...")
    verifier.test_core_imports()
    
    print("\n3. Testing optional dependencies...")
    verifier.test_optional_dependencies()
    
    print("\n4. Testing AI System modules...")
    verifier.test_ai_system_modules()
    
    print("\n5. Testing file structure...")
    verifier.test_file_structure()
    
    print("\n6. Testing configuration...")
    verifier.test_configuration()
    
    print("\n7. Testing entry points...")
    verifier.test_entry_points()
    
    print("\n8. Testing installer scripts...")
    verifier.test_installers()
    
    print("\n9. Testing syntax errors...")
    verifier.test_syntax_errors()
    
    # Generate final report
    success = verifier.generate_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())