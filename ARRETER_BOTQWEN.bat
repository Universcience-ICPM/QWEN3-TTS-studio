@echo off
setlocal
cd /d "%~dp0"
title Arreter Voix

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_windows_nvidia.ps1"
set "BOTQWEN_EXIT=%ERRORLEVEL%"

if not "%BOTQWEN_EXIT%"=="0" pause
endlocal & exit /b %BOTQWEN_EXIT%
