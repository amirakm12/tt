#!/usr/bin/env python3
"""
Build script for AI-ARTWORKS Cyberpunk HUD
Creates Windows executable with Inno Setup
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Build configuration
APP_NAME = "AI-ARTWORKS Neural Interface"
APP_VERSION = "1.0.0"
APP_ID = "AIArtworks.NeuralInterface"
COMPANY = "Cyberpunk Systems"

def build_executable():
    """Build the executable using PyInstaller"""
    print("Building executable with PyInstaller...")
    
    # PyInstaller spec
    spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['ai_artworks/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ai_artworks/qml', 'qml'),
        ('ai_artworks/assets', 'assets'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtQml',
        'PySide6.QtQuick',
        'PySide6.QtQuick3D',
        'PySide6.QtQuickControls2',
        'whisper',
        'torch',
        'transformers',
        'sounddevice',
        'scipy.signal',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AIArtworks',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico',
)
'''
    
    # Write spec file
    with open('aiartworks.spec', 'w') as f:
        f.write(spec_content)
    
    # Run PyInstaller
    subprocess.run([
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        'aiartworks.spec'
    ], check=True)
    
    print("Executable built successfully!")

def create_installer():
    """Create Inno Setup installer"""
    print("Creating Inno Setup installer...")
    
    # Inno Setup script
    iss_content = f'''
[Setup]
AppId={{{APP_ID}}}
AppName={APP_NAME}
AppVersion={APP_VERSION}
AppPublisher={COMPANY}
AppPublisherURL=https://ai-artworks.com
AppSupportURL=https://ai-artworks.com/support
AppUpdatesURL=https://ai-artworks.com/updates
DefaultDirName={{autopf}}\\{APP_NAME}
DefaultGroupName={APP_NAME}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename=AIArtworks-Setup-{APP_VERSION}
SetupIconFile=assets\\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{{cm:CreateQuickLaunchIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "dist\\AIArtworks.exe"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "dist\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{{app}}"; Flags: ignoreversion

[Icons]
Name: "{{group}}\\{APP_NAME}"; Filename: "{{app}}\\AIArtworks.exe"
Name: "{{group}}\\{{cm:UninstallProgram,{APP_NAME}}}"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\\{APP_NAME}"; Filename: "{{app}}\\AIArtworks.exe"; Tasks: desktopicon
Name: "{{userappdata}}\\Microsoft\\Internet Explorer\\Quick Launch\\{APP_NAME}"; Filename: "{{app}}\\AIArtworks.exe"; Tasks: quicklaunchicon

[Run]
Filename: "{{app}}\\AIArtworks.exe"; Description: "{{cm:LaunchProgram,{APP_NAME}}}"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKLM; Subkey: "Software\\{COMPANY}\\{APP_NAME}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{{app}}"
Root: HKLM; Subkey: "Software\\{COMPANY}\\{APP_NAME}"; ValueType: string; ValueName: "Version"; ValueData: "{APP_VERSION}"

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  // Check for Visual C++ Redistributables
  if not FileExists(ExpandConstant('{{sys}}\\vcruntime140.dll')) then
  begin
    if MsgBox('Visual C++ Redistributables are required. Download and install now?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      ShellExec('open', 'https://aka.ms/vs/17/release/vc_redist.x64.exe', '', '', SW_SHOW, ewNoWait, ResultCode);
      Result := False;
      Exit;
    end;
  end;
  
  // Check for Vulkan Runtime
  if not FileExists(ExpandConstant('{{sys}}\\vulkan-1.dll')) then
  begin
    if MsgBox('Vulkan Runtime is recommended for best performance. Download and install now?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      ShellExec('open', 'https://vulkan.lunarg.com/sdk/home', '', '', SW_SHOW, ewNoWait, ResultCode);
    end;
  end;
  
  Result := True;
end;
'''
    
    # Write Inno Setup script
    with open('setup.iss', 'w') as f:
        f.write(iss_content)
    
    # Check if Inno Setup is installed
    inno_path = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if not os.path.exists(inno_path):
        print("Inno Setup not found. Please install from: https://jrsoftware.org/isdl.php")
        return False
    
    # Compile installer
    subprocess.run([inno_path, 'setup.iss'], check=True)
    
    print(f"Installer created: dist/AIArtworks-Setup-{APP_VERSION}.exe")
    return True

def create_assets():
    """Create necessary asset files"""
    print("Creating assets...")
    
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Create a simple icon if it doesn't exist
    icon_path = assets_dir / "icon.ico"
    if not icon_path.exists():
        print("Warning: icon.ico not found. Please add a proper icon.")
        # Create placeholder
        from PIL import Image, ImageDraw
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 206, 206], outline=(0, 255, 255, 255), width=5)
        draw.text((128, 128), "AI", fill=(0, 255, 255, 255), anchor="mm")
        img.save(icon_path, format='ICO')
    
    # Create particle texture
    particle_path = assets_dir / "particle_glow.png"
    if not particle_path.exists():
        from PIL import Image, ImageDraw, ImageFilter
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([16, 16, 48, 48], fill=(0, 255, 255, 128))
        img = img.filter(ImageFilter.GaussianBlur(radius=8))
        img.save(particle_path)
    
    print("Assets created.")

def main():
    """Main build process"""
    print(f"Building {APP_NAME} v{APP_VERSION}")
    
    # Create assets
    create_assets()
    
    # Build executable
    build_executable()
    
    # Create installer
    if sys.platform == "win32":
        create_installer()
    else:
        print("Inno Setup installer is only available on Windows.")
        print("For other platforms, use the executable in the dist/ folder.")
    
    print("\nBuild complete!")

if __name__ == "__main__":
    main()