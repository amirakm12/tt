"""
Build Script for AI-ARTWORKS Qt Application
Creates a standalone Windows executable with all dependencies
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import PyInstaller.__main__
import platform


class AppBuilder:
    """Build manager for AI-ARTWORKS"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.build_dir = self.root_dir / "build"
        self.dist_dir = self.root_dir / "dist"
        self.app_name = "AI-ARTWORKS"
        self.main_script = "ai_artworks/ui/main_window.py"
        
    def clean(self):
        """Clean build directories"""
        print("Cleaning build directories...")
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                
    def check_requirements(self):
        """Check build requirements"""
        print("Checking requirements...")
        
        required_packages = [
            "PySide6",
            "PyOpenGL",
            "numpy",
            "opencv-python",
            "whisper",
            "torch",
            "sounddevice",
            "aiohttp",
            "pyinstaller"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
                
        if missing:
            print(f"Missing packages: {', '.join(missing)}")
            print("Installing missing packages...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing)
            
    def create_spec_file(self):
        """Create PyInstaller spec file"""
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# Paths
root_dir = Path(SPECPATH)
ai_artworks_dir = root_dir / "ai_artworks"

# Analysis
a = Analysis(
    ['{self.main_script}'],
    pathex=[str(root_dir)],
    binaries=[],
    datas=[
        # Include all QML files
        (str(ai_artworks_dir / "ui" / "qml"), "ai_artworks/ui/qml"),
        # Include plugin directory
        (str(ai_artworks_dir / "plugins"), "ai_artworks/plugins"),
        # Include models directory (create if needed)
        (str(ai_artworks_dir / "models"), "ai_artworks/models"),
        # Include assets
        (str(ai_artworks_dir / "assets"), "ai_artworks/assets"),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtOpenGL',
        'PySide6.QtOpenGLWidgets',
        'PySide6.QtNetwork',
        'PySide6.QtMultimedia',
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.arrays',
        'cv2',
        'whisper',
        'torch',
        'sounddevice',
        'aiohttp',
        'asyncio',
        'numpy',
        'PIL',
        'scipy.signal',
        'realesrgan',
        'basicsr',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'test',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# EXE
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='{self.app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ai_artworks/assets/icon.ico',
    version_file='version_info.txt',
)

# COLLECT
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{self.app_name}',
)
'''
        
        spec_path = self.root_dir / f"{self.app_name}.spec"
        spec_path.write_text(spec_content)
        print(f"Created spec file: {spec_path}")
        return spec_path
        
    def create_version_info(self):
        """Create version info file for Windows"""
        version_info = '''
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
        [StringStruct(u'CompanyName', u'AI-ARTWORKS'),
        StringStruct(u'FileDescription', u'AI-ARTWORKS Neural Creative Suite'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'AI-ARTWORKS'),
        StringStruct(u'LegalCopyright', u'Copyright (c) 2024 AI-ARTWORKS'),
        StringStruct(u'OriginalFilename', u'AI-ARTWORKS.exe'),
        StringStruct(u'ProductName', u'AI-ARTWORKS'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
        version_path = self.root_dir / "version_info.txt"
        version_path.write_text(version_info)
        print(f"Created version info: {version_path}")
        
    def create_icon(self):
        """Create application icon if it doesn't exist"""
        icon_path = self.root_dir / "ai_artworks" / "assets" / "icon.ico"
        if not icon_path.exists():
            print("Creating default icon...")
            icon_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple icon using PIL
            try:
                from PIL import Image, ImageDraw
                
                # Create icon at multiple sizes
                sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
                images = []
                
                for size in sizes:
                    img = Image.new('RGBA', size, (30, 30, 30, 255))
                    draw = ImageDraw.Draw(img)
                    
                    # Draw AI text
                    text_size = size[0] // 3
                    draw.text(
                        (size[0] // 2, size[1] // 2),
                        "AI",
                        fill=(0, 122, 204, 255),
                        anchor="mm"
                    )
                    
                    images.append(img)
                    
                # Save as ICO
                images[0].save(
                    icon_path,
                    format='ICO',
                    sizes=[img.size for img in images],
                    append_images=images[1:]
                )
                
            except ImportError:
                print("PIL not available, skipping icon creation")
                
    def download_models(self):
        """Download required AI models"""
        print("Checking AI models...")
        
        models_dir = self.root_dir / "ai_artworks" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Whisper models
        print("Downloading Whisper models...")
        import whisper
        whisper.load_model("base", download_root=str(models_dir))
        
        # Real-ESRGAN models would be downloaded here
        # For now, we'll use the fallback upscaling
        
    def build(self):
        """Build the application"""
        print(f"Building {self.app_name}...")
        
        # Check requirements
        self.check_requirements()
        
        # Clean previous builds
        self.clean()
        
        # Create assets
        self.create_icon()
        self.create_version_info()
        
        # Download models
        self.download_models()
        
        # Create spec file
        spec_path = self.create_spec_file()
        
        # Run PyInstaller
        print("Running PyInstaller...")
        PyInstaller.__main__.run([
            str(spec_path),
            '--clean',
            '--noconfirm',
            '--windowed',
            '--name', self.app_name,
        ])
        
        # Create installer
        self.create_installer()
        
        print(f"\nBuild complete! Output: {self.dist_dir / self.app_name}")
        
    def create_installer(self):
        """Create NSIS installer for Windows"""
        if platform.system() != "Windows":
            print("Installer creation is only supported on Windows")
            return
            
        nsis_script = f'''
!define APP_NAME "AI-ARTWORKS"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "AI-ARTWORKS"
!define APP_URL "https://ai-artworks.com"
!define APP_EXE "${{APP_NAME}}.exe"

Name "${{APP_NAME}} ${{APP_VERSION}}"
OutFile "../${{APP_NAME}}-Setup-${{APP_VERSION}}.exe"
InstallDir "$PROGRAMFILES64\\${{APP_NAME}}"
InstallDirRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "InstallDir"

RequestExecutionLevel admin
ShowInstDetails show
ShowUninstDetails show

!include "MUI2.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON "..\\ai_artworks\\assets\\icon.ico"
!define MUI_UNICON "..\\ai_artworks\\assets\\icon.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Main Application" SecMain
    SetOutPath "$INSTDIR"
    
    ; Copy all files
    File /r "${{APP_NAME}}\\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\\${{APP_NAME}}"
    CreateShortcut "$SMPROGRAMS\\${{APP_NAME}}\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXE}}"
    CreateShortcut "$SMPROGRAMS\\${{APP_NAME}}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
    CreateShortcut "$DESKTOP\\${{APP_NAME}}.lnk" "$INSTDIR\\${{APP_EXE}}"
    
    ; Write registry keys
    WriteRegStr HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayName" "${{APP_NAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayIcon" "$INSTDIR\\${{APP_EXE}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "Publisher" "${{APP_PUBLISHER}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "DisplayVersion" "${{APP_VERSION}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}" "URLInfoAbout" "${{APP_URL}}"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    ; Remove files
    RMDir /r "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$SMPROGRAMS\\${{APP_NAME}}\\*.*"
    RMDir "$SMPROGRAMS\\${{APP_NAME}}"
    Delete "$DESKTOP\\${{APP_NAME}}.lnk"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APP_NAME}}"
    DeleteRegKey HKLM "Software\\${{APP_PUBLISHER}}\\${{APP_NAME}}"
SectionEnd
'''
        
        nsis_path = self.dist_dir / "installer.nsi"
        nsis_path.write_text(nsis_script)
        
        # Try to run NSIS if available
        try:
            subprocess.run(["makensis", str(nsis_path)], check=True)
            print("Installer created successfully!")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NSIS not found. Please install NSIS to create the installer.")
            print(f"NSIS script saved to: {nsis_path}")


def main():
    """Main entry point"""
    builder = AppBuilder()
    
    if "--clean" in sys.argv:
        builder.clean()
    else:
        builder.build()


if __name__ == "__main__":
    main()