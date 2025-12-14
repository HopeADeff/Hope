@echo off
setlocal
title Hope-AD CLI

echo ============================================
echo   HOPE-AD CLI - Python Interface
echo ============================================
echo.

:: Check if venv exists
set VENV_DIR=%~dp0venv
if not exist "%VENV_DIR%" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

:: Go to Python module directory
cd /d "%~dp0Hope\Hope"

echo Virtual environment activated!
echo.
echo Available commands:
echo   python engine.py detect -i image.jpg
echo   python engine.py protect -i image.jpg -o output.jpg --target-style abstract
echo   python engine.py protect -i image.jpg -o output.jpg --nightshade
echo   python engine.py embed -i image.jpg -o output.jpg --owner "Artist Name"
echo   python engine.py extract -i image.jpg
echo.
echo Type 'exit' to leave the environment.
echo.

:: Open interactive prompt
cmd /k
