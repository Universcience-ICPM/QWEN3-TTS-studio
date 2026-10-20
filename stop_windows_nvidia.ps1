[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir ".venv-windows\Scripts\python.exe"
$PidFile = Join-Path $ProjectDir ".botqwen-windows.pid"
$StopFile = Join-Path $ProjectDir ".botqwen-stop"

if (-not (Test-Path $PidFile)) {
    Write-Host "Aucun serveur Voix lance par le raccourci." -ForegroundColor Yellow
    exit 0
}

$TargetProcessId = 0
if (-not [int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$TargetProcessId)) {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    throw "Fichier PID invalide. Il a ete supprime."
}

$AppProcess = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
if (-not $AppProcess) {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
    Write-Host "Le serveur etait deja arrete." -ForegroundColor Yellow
    exit 0
}
if (-not (Test-Path $PythonExe) -or $AppProcess.Path -ne (Resolve-Path $PythonExe).Path) {
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    throw "Le PID enregistre n'appartient pas a Voix. Aucun processus n'a ete arrete."
}

Write-Host "Arret propre de Voix (PID $TargetProcessId)..." -ForegroundColor Cyan
Set-Content -Path $StopFile -Value "stop" -Encoding Ascii

try {
    $AppProcess.WaitForExit(15000) | Out-Null
    $AppProcess.Refresh()
}
catch {
    # Le processus a pu disparaitre pendant l'attente.
}

$AppProcess = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
if ($AppProcess) {
    Write-Warning "L'arret propre a expire ; fermeture forcee du processus Voix."
    Stop-Process -Id $TargetProcessId -Force
}

Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
Write-Host "Voix est arrete. Le port 7860 est libre." -ForegroundColor Green
