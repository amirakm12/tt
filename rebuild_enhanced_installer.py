#!/usr/bin/env python3
"""
Rebuild Windows Installer with Enhanced Modern UI
Creates a new installer package with the modernized dashboard
"""

import os
import sys
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime

def print_header():
    """Print script header"""
    print("=" * 60)
    print("🚀 AI System Enhanced Installer Builder")
    print("   Building Windows installer with modern UI...")
    print("=" * 60)
    print()

def check_requirements():
    """Check if all requirements are met"""
    print("📋 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print("✅ PyInstaller available")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0.0"], check=True)
    
    # Check for NSIS (optional for advanced installer)
    nsis_path = shutil.which("makensis")
    if nsis_path:
        print("✅ NSIS found (optional)")
    else:
        print("⚠️  NSIS not found (optional, for advanced installer features)")
    
    return True

def prepare_build_directory():
    """Prepare the build directory"""
    print("\n📁 Preparing build directory...")
    
    build_dir = Path("build_enhanced")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    # Copy source files
    print("   Copying source files...")
    shutil.copytree("src", build_dir / "src")
    
    # Copy configuration
    if Path("config").exists():
        shutil.copytree("config", build_dir / "config")
    
    # Copy assets
    if Path("assets").exists():
        shutil.copytree("assets", build_dir / "assets")
    
    # Copy requirements
    shutil.copy("requirements.txt", build_dir / "requirements.txt")
    
    print("✅ Build directory prepared")
    return build_dir

def create_enhanced_spec():
    """Create PyInstaller spec file with enhanced features"""
    print("\n📝 Creating enhanced installer specification...")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# Add all Python files
python_files = []
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/ui/static', 'src/ui/static'),
        ('config', 'config'),
        ('assets', 'assets'),
        ('requirements', 'requirements'),
    ],
    hiddenimports=[
        'aiohttp',
        'aiohttp_cors',
        'numpy',
        'asyncio',
        'multiprocessing',
        'src.ui.modern_dashboard',
        'src.ui.voice_interface',
        'src.core.orchestrator',
        'src.agents.triage_agent',
        'src.agents.research_agent',
        'src.agents.orchestration_agent',
        'src.ai.rag_engine',
        'src.ai.speculative_decoder',
        'src.kernel.integration',
        'src.sensors.fusion',
        'src.monitoring.system_monitor',
        'src.monitoring.security_monitor',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI-System-Enhanced',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
    uac_admin=True,  # Request admin privileges
)

# Create installer package
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI-System-Enhanced',
)
'''
    
    spec_file = Path("AI-System-Enhanced.spec")
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print("✅ Enhanced spec file created")
    return spec_file

def create_version_info():
    """Create version information file"""
    print("\n📋 Creating version information...")
    
    version_info = '''VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(2, 0, 0, 0),
    prodvers=(2, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'AI Systems Inc.'),
        StringStruct(u'FileDescription', u'AI System - Neural Command Center'),
        StringStruct(u'FileVersion', u'2.0.0.0'),
        StringStruct(u'InternalName', u'AI-System-Enhanced'),
        StringStruct(u'LegalCopyright', u'Copyright (c) 2025 AI Systems Inc.'),
        StringStruct(u'OriginalFilename', u'AI-System-Enhanced.exe'),
        StringStruct(u'ProductName', u'AI System Enhanced'),
        StringStruct(u'ProductVersion', u'2.0.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)'''
    
    with open('version_info.txt', 'w') as f:
        f.write(version_info)
    
    print("✅ Version information created")

def create_icon():
    """Create application icon if not exists"""
    icon_path = Path("assets/icon.ico")
    if not icon_path.exists():
        print("\n🎨 Creating application icon...")
        icon_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In a real scenario, we'd create a proper icon
        # For now, we'll just note it's missing
        print("⚠️  Icon file not found. Using default icon.")
    else:
        print("✅ Using existing icon")

def build_executable():
    """Build the executable using PyInstaller"""
    print("\n🔨 Building enhanced executable...")
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "AI-System-Enhanced.spec"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Executable built successfully")
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def create_installer_package():
    """Create the final installer package"""
    print("\n📦 Creating installer package...")
    
    # Create installer output directory
    output_dir = Path("installer_output_enhanced")
    output_dir.mkdir(exist_ok=True)
    
    # Find the built executable
    exe_path = Path("dist/AI-System-Enhanced.exe")
    if not exe_path.exists():
        exe_path = Path("dist/AI-System-Enhanced/AI-System-Enhanced.exe")
    
    if not exe_path.exists():
        print("❌ Built executable not found")
        return None
    
    # Copy to output directory
    output_exe = output_dir / "AI-System-Enhanced-Setup.exe"
    shutil.copy2(exe_path, output_exe)
    
    # Calculate checksum
    print("   Calculating checksum...")
    sha256_hash = hashlib.sha256()
    with open(output_exe, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    checksum = sha256_hash.hexdigest()
    
    # Write checksum file
    with open(output_dir / "AI-System-Enhanced-Setup.exe.sha256", "w") as f:
        f.write(f"{checksum}  AI-System-Enhanced-Setup.exe\n")
    
    # Create README
    readme_content = f"""AI System Enhanced - Neural Command Center
=========================================

Version: 2.0.0
Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Size: {output_exe.stat().st_size / (1024*1024):.1f} MB
SHA256: {checksum}

What's New in Version 2.0:
-------------------------
✨ Modern React-based UI with glassmorphism design
✨ 3D neural network visualizations with Three.js
✨ Real-time monitoring with WebSocket streaming
✨ AI chat interface with markdown support
✨ Voice command visualization
✨ Multiple theme options (Dark, Light, Cyberpunk, Matrix, Ocean, Sunset)
✨ Advanced animations and transitions
✨ Responsive design for all screen sizes

Installation Instructions:
-------------------------
1. Run AI-System-Enhanced-Setup.exe as Administrator
2. Follow the installation wizard
3. Launch from Start Menu or Desktop shortcut
4. Access the dashboard at http://localhost:8080

System Requirements:
-------------------
- Windows 10 or later (64-bit)
- 8 GB RAM (16 GB recommended)
- 1 GB free disk space
- Modern web browser (Chrome, Firefox, Edge)
- Internet connection for AI features

Support:
--------
For help and documentation, visit the Neural Command Center
after installation or check our online documentation.
"""
    
    with open(output_dir / "README.txt", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Installer package created: {output_exe}")
    print(f"   Size: {output_exe.stat().st_size / (1024*1024):.1f} MB")
    print(f"   SHA256: {checksum}")
    
    return output_exe

def cleanup():
    """Clean up temporary files"""
    print("\n🧹 Cleaning up...")
    
    dirs_to_remove = ["build", "dist", "__pycache__", "build_enhanced"]
    files_to_remove = ["AI-System-Enhanced.spec", "version_info.txt"]
    
    for dir_name in dirs_to_remove:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
    
    for file_name in files_to_remove:
        if Path(file_name).exists():
            os.remove(file_name)
    
    print("✅ Cleanup complete")

def main():
    """Main build process"""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed")
        return 1
    
    try:
        # Prepare build
        build_dir = prepare_build_directory()
        
        # Create version info
        create_version_info()
        
        # Create icon
        create_icon()
        
        # Create spec file
        spec_file = create_enhanced_spec()
        
        # Build executable
        if not build_executable():
            print("\n❌ Build failed")
            return 1
        
        # Create installer package
        installer_path = create_installer_package()
        if not installer_path:
            print("\n❌ Failed to create installer package")
            return 1
        
        # Cleanup
        cleanup()
        
        print("\n" + "=" * 60)
        print("✅ BUILD SUCCESSFUL!")
        print(f"   Enhanced installer created: {installer_path}")
        print("   The installer includes the modern Neural Command Center UI")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())