@echo off
setlocal
title Hope-AD

echo ============================================
echo   HOPE-AD - AI Image Protection System
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

:: Check if build exists
set BUILD_DIR=%~dp0Hope\Hope\bin\Release\net10.0-windows
if not exist "%BUILD_DIR%\Hope.exe" (
    set BUILD_DIR=%~dp0Hope\Hope\bin\Debug\net10.0-windows
)

if exist "%BUILD_DIR%\Hope.exe" (
    echo Starting Hope-AD GUI...
    echo.
    cd /d "%BUILD_DIR%"
    start "" Hope.exe
) else (
    echo Attempting to run via dotnet...
    cd /d "%~dp0Hope\Hope"
    dotnet run --configuration Release
)

endlocal
