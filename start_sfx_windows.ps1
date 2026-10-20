[CmdletBinding()]
param(
    [int]$Port = 7861
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EngineLauncher = Join-Path $ProjectDir ".stable-audio-3\optimized\tflite\sa3.bat"
$PythonExe = Join-Path $ProjectDir ".venv-windows\Scripts\python.exe"

if (-not (Test-Path $EngineLauncher)) {
    throw "Moteur SFX absent. Executez d'abord .\install_sfx_windows.ps1"
}
if (-not (Test-Path $PythonExe)) {
    throw "Environnement botQwen absent. Executez d'abord .\install_windows_nvidia.ps1"
}

Set-Location $ProjectDir
$env:BOTQWEN_SFX_BACKEND = "tflite"
$env:BOTQWEN_SFX_PORT = "$Port"

Write-Host "Studio SFX : http://127.0.0.1:$Port" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "sfx_app.py")
