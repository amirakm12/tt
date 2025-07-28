@echo off
echo AI System - Windows Installer Builder
echo =====================================

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

echo Python detected, proceeding with build...

:: Install build dependencies
echo Installing build dependencies...
python -m pip install --upgrade pip
python -m pip install pyinstaller pillow pywin32

:: Run the build script
echo Running build script...
python build_standalone_installer.py

if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Check the installer_output directory for the final installer.
pause