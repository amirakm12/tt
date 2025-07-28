#!/usr/bin/env python3
"""
Build script for creating standalone Windows installer EXE
"""

import os
import sys
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

def install_build_dependencies():
    """Install required build dependencies."""
    print("Installing build dependencies...")
    
    dependencies = [
        'pyinstaller>=5.0',
        'pywin32>=305',
        'pillow>=9.0.0'  # For icon creation
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")

def create_icon():
    """Create application icon."""
    icon_path = Path("assets/icon.ico")
    
    if icon_path.exists():
        return str(icon_path)
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple icon
        size = (64, 64)
        image = Image.new('RGBA', size, (37, 99, 235, 255))  # Blue background
        draw = ImageDraw.Draw(image)
        
        # Draw AI symbol
        draw.ellipse([16, 16, 48, 48], fill=(255, 255, 255, 255))
        draw.text((24, 20), "AI", fill=(37, 99, 235, 255))
        
        # Save as ICO
        icon_path.parent.mkdir(exist_ok=True)
        image.save(icon_path, format='ICO')
        print(f"✓ Created icon: {icon_path}")
        return str(icon_path)
        
    except ImportError:
        print("! PIL not available, using default icon")
        return None

def embed_source_files():
    """Embed source files into the installer."""
    print("Embedding source files...")
    
    # Create a zip file with all source files
    source_zip = Path("temp_source.zip")
    
    with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add all Python files from src/
        src_dir = Path("src")
        if src_dir.exists():
            for file_path in src_dir.rglob("*.py"):
                arcname = str(file_path.relative_to("."))
                zf.write(file_path, arcname)
                print(f"  Added: {arcname}")
        
        # Add configuration files
        config_files = [
            "config/config.json",
            "requirements.txt",
            "README.md",
            "setup.py"
        ]
        
        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                zf.write(file_path, config_file)
                print(f"  Added: {config_file}")
    
    print(f"✓ Created source archive: {source_zip}")
    return source_zip

def create_pyinstaller_spec():
    """Create PyInstaller spec file."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['install_windows_standalone.exe.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('temp_source.zip', '.'),
        ('assets/icon.ico', 'assets'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'winreg',
        'win32com.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='AI-System-Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
    version_file='version_info.txt'
)
'''
    
    spec_file = Path("installer.spec")
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"✓ Created PyInstaller spec: {spec_file}")
    return spec_file

def create_version_info():
    """Create version info file for Windows EXE."""
    version_info = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
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
        [StringStruct(u'CompanyName', u'AI System Team'),
        StringStruct(u'FileDescription', u'AI System Installer'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'AI-System-Installer'),
        StringStruct(u'LegalCopyright', u'Copyright (C) 2024 AI System Team'),
        StringStruct(u'OriginalFilename', u'AI-System-Installer.exe'),
        StringStruct(u'ProductName', u'AI System'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    
    version_file = Path("version_info.txt")
    with open(version_file, 'w', encoding='utf-8') as f:
        f.write(version_info)
    
    print(f"✓ Created version info: {version_file}")
    return version_file

def build_installer():
    """Build the installer using PyInstaller."""
    print("Building installer with PyInstaller...")
    
    try:
        # Run PyInstaller
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            'installer.spec'
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ PyInstaller build completed")
        
        # Check if the EXE was created
        exe_path = Path("dist/AI-System-Installer.exe")
        if exe_path.exists():
            print(f"✓ Installer created: {exe_path}")
            print(f"  Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
            return exe_path
        else:
            print("✗ Installer EXE not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"✗ PyInstaller failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_installer_package():
    """Create final installer package."""
    print("Creating installer package...")
    
    # Create output directory
    output_dir = Path("installer_output")
    output_dir.mkdir(exist_ok=True)
    
    # Copy the installer EXE
    exe_path = Path("dist/AI-System-Installer.exe")
    if exe_path.exists():
        final_exe = output_dir / "AI-System-Setup.exe"
        shutil.copy2(exe_path, final_exe)
        print(f"✓ Final installer: {final_exe}")
        
        # Create checksum
        import hashlib
        with open(final_exe, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
        
        checksum_file = output_dir / "AI-System-Setup.exe.sha256"
        with open(checksum_file, 'w') as f:
            f.write(f"{checksum}  AI-System-Setup.exe\n")
        
        print(f"✓ Checksum: {checksum_file}")
        
        # Create info file
        info_content = f"""AI System Installer
==================

Version: 1.0.0
Built: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Size: {final_exe.stat().st_size / 1024 / 1024:.1f} MB
SHA256: {checksum}

Installation Instructions:
1. Run AI-System-Setup.exe as Administrator (recommended)
2. Follow the installation wizard
3. The application will be installed to %LOCALAPPDATA%\\AI-System by default
4. Desktop and Start Menu shortcuts will be created

System Requirements:
- Windows 10 or later
- 500 MB free disk space
- Internet connection (for dependency installation)

Support:
For support and documentation, please visit our website.
"""
        
        info_file = output_dir / "README.txt"
        with open(info_file, 'w') as f:
            f.write(info_content)
        
        print(f"✓ Info file: {info_file}")
        
        return final_exe
    else:
        print("✗ Installer EXE not found")
        return None

def cleanup():
    """Clean up temporary files."""
    print("Cleaning up...")
    
    temp_files = [
        "temp_source.zip",
        "installer.spec",
        "version_info.txt",
        "build",
        "dist",
        "__pycache__"
    ]
    
    for temp_file in temp_files:
        path = Path(temp_file)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  Removed: {temp_file}")

def main():
    """Main build process."""
    print("AI System - Standalone Installer Builder")
    print("=" * 50)
    
    try:
        # Step 1: Install dependencies
        install_build_dependencies()
        
        # Step 2: Create icon
        create_icon()
        
        # Step 3: Embed source files
        embed_source_files()
        
        # Step 4: Create PyInstaller spec
        create_pyinstaller_spec()
        
        # Step 5: Create version info
        create_version_info()
        
        # Step 6: Build installer
        installer_exe = build_installer()
        
        if installer_exe:
            # Step 7: Create final package
            final_installer = create_installer_package()
            
            if final_installer:
                print("\n" + "=" * 50)
                print("✓ BUILD SUCCESSFUL!")
                print(f"✓ Installer: {final_installer}")
                print(f"✓ Size: {final_installer.stat().st_size / 1024 / 1024:.1f} MB")
                print("\nThe standalone installer is ready for distribution.")
            else:
                print("\n✗ Failed to create final package")
        else:
            print("\n✗ Failed to build installer")
        
    except Exception as e:
        print(f"\n✗ Build failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always cleanup
        cleanup()

if __name__ == "__main__":
    main()