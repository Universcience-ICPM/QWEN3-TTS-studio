@echo off
setlocal
cd /d "%~dp0"
title Voix (botqwenClaude)

if not exist ".venv-windows\Scripts\python.exe" (
    echo.
    echo [ERREUR] Voix n'est pas encore installe sur ce PC.
    echo Lancez une seule fois install_windows_nvidia.ps1 comme indique dans le guide.
    echo.
    pause
    exit /b 1
)

set BOTQWEN_BACKEND=cuda
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
".venv-windows\Scripts\python.exe" -u botqwenClaude.py
if errorlevel 1 pause
endlocal
