@echo off
setlocal
cd /d "%~dp0"
title Voix (botqwenClaude) - reseau local
rem Ouvre Voix et son API aux autres appareils du meme reseau (iPhone, autre ordinateur...).
rem Au premier lancement, Windows demande d autoriser Python dans le pare-feu : cochez "Reseaux prives".
rem Pour demander un mot de passe (identifiant : voix), enlevez "rem " devant la ligne suivante :
rem set BOTQWEN_MOT_DE_PASSE=change-moi

if not exist ".venv-windows\Scripts\python.exe" (
    echo.
    echo [ERREUR] Voix n'est pas encore installe sur ce PC.
    echo Lancez une seule fois install_windows_nvidia.ps1 comme indique dans le guide.
    echo.
    pause
    exit /b 1
)

set BOTQWEN_BACKEND=cuda
set BOTQWEN_RESEAU=1
set PYTHONWARNINGS=ignore
set PYTHONIOENCODING=utf-8
".venv-windows\Scripts\python.exe" -u botqwenClaude.py
if errorlevel 1 pause
endlocal
