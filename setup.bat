@echo off
setlocal enabledelayedexpansion
title Hope-AD Setup

echo ============================================
echo   HOPE-AD Setup - AI Image Protection
echo ============================================
echo.

:: Check for admin rights (optional, for better PATH handling)
:: net session >nul 2>&1
:: if %errorLevel% == 0 (
::     echo Running as Administrator
:: ) else (
::     echo Note: Running without admin rights. Some features may be limited.
:: )

:: Step 1: Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo    Found Python %PYTHON_VER%

:: Step 2: Check .NET SDK
echo.
echo [2/5] Checking .NET SDK...
dotnet --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: .NET SDK not found!
    echo.
    echo Please install .NET 8.0 or later from https://dotnet.microsoft.com/download
    echo.
    pause
    exit /b 1
)

for /f %%v in ('dotnet --version') do set DOTNET_VER=%%v
echo    Found .NET SDK %DOTNET_VER%

:: Step 3: Check CUDA (optional)
echo.
echo [3/5] Checking NVIDIA GPU (optional)...
where nvcc >nul 2>&1
if %errorLevel% == 0 (
    for /f "tokens=5" %%v in ('nvcc --version ^| findstr release') do set CUDA_VER=%%v
    echo    Found CUDA !CUDA_VER!
) else (
    echo    CUDA not found - will use CPU mode (slower)
)

:: Step 4: Create virtual environment
echo.
echo [4/5] Setting up Python environment...
set VENV_DIR=%~dp0venv

if exist "%VENV_DIR%" (
    echo    Virtual environment already exists
) else (
    echo    Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

:: Step 5: Install dependencies
echo.
echo [5/5] Installing Python dependencies...
echo    This may take several minutes on first run...
echo.

:: Upgrade pip first
python -m pip install --upgrade pip --quiet

:: Install PyTorch with CUDA if available
where nvcc >nul 2>&1
if %errorLevel% == 0 (
    echo    Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    echo    Installing PyTorch (CPU version)...
    pip install torch torchvision --quiet
)

:: Install other dependencies
echo    Installing other dependencies...
pip install Pillow numpy scipy PyWavelets --quiet

:: Install CLIP
echo    Installing CLIP (may take a moment)...
pip install git+https://github.com/openai/CLIP.git --quiet

:: Step 6: Build the application
echo.
echo [6/6] Building Hope-AD application...
cd /d "%~dp0Hope\Hope"
dotnet build --configuration Release --verbosity quiet
if %errorLevel% neq 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SETUP COMPLETE!
echo ============================================
echo.
echo To run Hope-AD:
echo   1. Double-click run.bat
echo   OR
echo   2. Run: dotnet run --project Hope\Hope\Hope.csproj
echo.
echo For Python CLI:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run: python Hope\Hope\engine.py --help
echo.

pause
endlocal
