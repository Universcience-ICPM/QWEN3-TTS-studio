[CmdletBinding()]
param(
    [ValidateSet("1.7B", "0.6B")]
    [string]$ModelSize = "1.7B",
    [string]$SttModel = "openai/whisper-small"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir ".venv-windows\Scripts\python.exe"
$AppScript = Join-Path $ProjectDir "botQwen2.py"
$PidFile = Join-Path $ProjectDir ".botqwen-windows.pid"
$StopFile = Join-Path $ProjectDir ".botqwen-stop"

if (-not (Test-Path $PythonExe)) {
    throw "Environnement absent. Executez d'abord .\install_windows_nvidia.ps1"
}
$ExpectedPython = (Resolve-Path $PythonExe).Path

Set-Location $ProjectDir

if (Test-Path $PidFile) {
    $RecordedPid = 0
    [void][int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$RecordedPid)
    $ExistingProcess = Get-Process -Id $RecordedPid -ErrorAction SilentlyContinue
    if ($ExistingProcess -and $ExistingProcess.Path -eq $ExpectedPython) {
        Write-Host "Voix est deja actif (PID $RecordedPid)." -ForegroundColor Yellow
        Start-Process "http://127.0.0.1:7860"
        exit 0
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}
Remove-Item $StopFile -Force -ErrorAction SilentlyContinue

$env:BOTQWEN_BACKEND = "cuda"
$env:BOTQWEN_TTS_SIZE = $ModelSize
$env:BOTQWEN_STT_MODEL = $SttModel

Write-Host "Voix - NVIDIA CUDA - Qwen $ModelSize" -ForegroundColor Cyan
Write-Host "Verification de l'environnement Windows..." -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectDir "verify_runtime.py")
if ($LASTEXITCODE -ne 0) {
    throw "Environnement invalide. Consultez le diagnostic ci-dessus avant de reinstaller."
}
Write-Host "Le premier lancement telecharge les modeles dans le cache Hugging Face."
$AppProcess = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-u", "`"$AppScript`"") `
    -WorkingDirectory $ProjectDir `
    -NoNewWindow `
    -PassThru
Set-Content -Path $PidFile -Value $AppProcess.Id -Encoding Ascii

$AppExitCode = 0
try {
    $AppProcess.WaitForExit()
    $AppExitCode = $AppProcess.ExitCode
}
finally {
    if ($AppProcess -and -not $AppProcess.HasExited) {
        Stop-Process -Id $AppProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
}
exit $AppExitCode
