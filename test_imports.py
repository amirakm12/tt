#!/usr/bin/env python3
"""
Test script to check all imports and identify missing dependencies
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, optional=False):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - Available")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠️  {module_name} - Optional (missing: {e})")
        else:
            print(f"❌ {module_name} - Required (missing: {e})")
        return False

def main():
    print("🔍 Testing Python Module Imports")
    print("=" * 50)
    
    # Core Python modules (should always be available)
    core_modules = [
        'asyncio', 'logging', 'json', 'os', 'sys', 'pathlib', 'typing',
        'dataclasses', 'enum', 'time', 'threading', 'subprocess', 'platform',
        'shutil', 'tempfile', 'collections', 'functools', 'itertools'
    ]
    
    print("\n📦 Core Python Modules:")
    for module in core_modules:
        test_import(module)
    
    # Required third-party modules
    required_modules = [
        'aiohttp', 'jinja2', 'yaml', 'toml'
    ]
    
    print("\n📦 Required Third-Party Modules:")
    missing_required = []
    for module in required_modules:
        if not test_import(module):
            missing_required.append(module)
    
    # Optional third-party modules
    optional_modules = [
        'uvloop', 'psutil', 'GPUtil', 'torch', 'transformers', 
        'sentence_transformers', 'chromadb', 'langchain', 'openai',
        'scikit_learn', 'sklearn', 'numpy', 'pandas', 'matplotlib',
        'seaborn', 'cryptography', 'redis', 'requests', 'websockets',
        'speech_recognition', 'pyttsx3', 'pyaudio'
    ]
    
    print("\n📦 Optional Third-Party Modules:")
    for module in optional_modules:
        test_import(module, optional=True)
    
    # Test our own modules
    print("\n📦 AI System Modules:")
    sys.path.insert(0, 'src')
    
    our_modules = [
        'src.core.config',
        'src.core.orchestrator', 
        'src.kernel.integration',
        'src.sensors.fusion',
        'src.ai.rag_engine',
        'src.ai.speculative_decoder',
        'src.agents.triage_agent',
        'src.agents.research_agent',
        'src.agents.orchestration_agent',
        'src.ui.dashboard',
        'src.ui.voice_interface',
        'src.monitoring.system_monitor',
        'src.monitoring.security_monitor'
    ]
    
    our_module_errors = []
    for module in our_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} - OK")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
            our_module_errors.append((module, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    if missing_required:
        print(f"❌ Missing required modules: {', '.join(missing_required)}")
        print("   Install with: pip install " + " ".join(missing_required))
    else:
        print("✅ All required modules available")
    
    if our_module_errors:
        print(f"\n❌ {len(our_module_errors)} AI System module errors:")
        for module, error in our_module_errors:
            print(f"   {module}: {error}")
    else:
        print("✅ All AI System modules import successfully")
    
    return len(missing_required) == 0 and len(our_module_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)