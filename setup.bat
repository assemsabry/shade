@echo off
setlocal enabledelayedexpansion

title Shade AI - Installer
echo.
echo    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo       SHADE AI - AUTOMATIC INSTALLER
echo    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b
)

:: 2. Create Virtual Environment
if not exist .venv (
    echo [*] Creating virtual environment...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [OK] Virtual environment created.
) else (
    echo [!] .venv folder already exists. Skipping creation.
)

:: 3. Install dependencies
echo [*] Installing Shade and dependencies (this may take a few minutes)...
echo.
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed. Check your internet connection.
    pause
    exit /b
)

echo.
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo    INSTALLATION COMPLETE!
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo.
echo To run Shade, simply type:
echo    .venv\Scripts\shade
echo.
echo Or use the "shade" command in this folder after activating:
echo    call .venv\Scripts\activate
echo.
pause
