@echo off
title AI System - Advanced Multi-Agent AI Platform
color 0A

echo.
echo 🚀 AI System - Advanced Multi-Agent AI Platform
echo ================================================
echo.
echo 🔧 Starting AI System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    echo.
    echo 📥 Download Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "src\main.py" (
    echo ❌ AI System files not found.
    echo.
    echo 📁 Current directory: %CD%
    echo 📁 Available files:
    dir /b
    echo.
    echo 🔧 Please ensure you're running this from the AI System directory.
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
echo ✅ AI System files found
echo.
echo 🚀 Launching AI System...
echo ==============================
echo.

REM Run the AI System
python src\main.py

if errorlevel 1 (
    echo.
    echo ❌ AI System encountered an error.
    echo.
    echo 🔧 Troubleshooting:
    echo   1. Check that all dependencies are installed: pip install -r requirements.txt
    echo   2. Ensure you have sufficient permissions
    echo   3. Check the console output for specific errors
    echo.
    pause
) else (
    echo.
    echo ✅ AI System completed successfully.
    echo.
)

echo.
echo Press any key to exit...
pause >nul