@echo off
setlocal
cd /d "%~dp0"
title Voix - NVIDIA

if not exist ".venv-windows\Scripts\python.exe" (
    echo.
    echo [ERREUR] Voix n'est pas encore installe sur ce PC.
    echo Lancez une seule fois install_windows_nvidia.ps1 comme indique dans le guide.
    echo.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows_nvidia.ps1"
set "BOTQWEN_EXIT=%ERRORLEVEL%"

if not "%BOTQWEN_EXIT%"=="0" (
    echo.
    echo [ERREUR] Voix s'est arrete avec le code %BOTQWEN_EXIT%.
    echo Consultez les messages ci-dessus avant de relancer l'installation.
    echo.
    pause
)

endlocal & exit /b %BOTQWEN_EXIT%
