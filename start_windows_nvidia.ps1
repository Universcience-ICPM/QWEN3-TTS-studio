[CmdletBinding()]
param(
    [ValidateSet("1.7B", "0.6B")]
    [string]$ModelSize = "1.7B",
    [string]$SttModel = "openai/whisper-small"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir ".venv-windows\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Environnement absent. Executez d'abord .\install_windows_nvidia.ps1"
}

Set-Location $ProjectDir
$env:BOTQWEN_BACKEND = "cuda"
$env:BOTQWEN_TTS_SIZE = $ModelSize
$env:BOTQWEN_STT_MODEL = $SttModel

Write-Host "botQwen2 - NVIDIA CUDA - Qwen $ModelSize" -ForegroundColor Cyan
Write-Host "Le premier lancement telecharge les modeles dans le cache Hugging Face."
& $PythonExe (Join-Path $ProjectDir "botQwen2.py")
