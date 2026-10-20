[CmdletBinding()]
param(
    [ValidateSet("cu128", "cu126", "cu121")]
    [string]$CudaWheel = "cu128",
    [switch]$Update
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv-windows"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$VenvCreated = $false

Set-Location $ProjectDir

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi est introuvable. Installez ou mettez a jour le pilote NVIDIA."
}

Write-Host "GPU NVIDIA detecte :" -ForegroundColor Cyan
& nvidia-smi "--query-gpu=name,driver_version,memory.total" "--format=csv,noheader"
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi n'arrive pas a interroger le GPU."
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Le lanceur Python 'py' est absent. Installez Python 3.12 depuis python.org."
}

& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 est requis. Installez-le depuis python.org."
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creation de .venv-windows..." -ForegroundColor Cyan
    & py -3.12 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "La creation de l'environnement Python a echoue."
    }
    $VenvCreated = $true
}

if ($VenvCreated -or $Update) {
    & $PythonExe -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "La mise a jour de pip a echoue."
    }
}

$UpgradeArgs = @()
if ($Update) {
    $UpgradeArgs += "--upgrade"
    Write-Host "Mode mise a jour explicite active." -ForegroundColor Yellow
}

Write-Host "Installation de PyTorch CUDA ($CudaWheel)..." -ForegroundColor Cyan
& $PythonExe -m pip install @UpgradeArgs torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaWheel"
if ($LASTEXITCODE -ne 0) {
    throw "L'installation de PyTorch CUDA a echoue. Essayez un autre -CudaWheel."
}

Write-Host "Installation de botQwen et Qwen3-TTS..." -ForegroundColor Cyan
$Requirements = Join-Path $ProjectDir "requirements-windows-nvidia.txt"
& $PythonExe -m pip install @UpgradeArgs -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "L'installation des dependances botQwen a echoue."
}

$env:BOTQWEN_BACKEND = "cuda"
& $PythonExe (Join-Path $ProjectDir "verify_runtime.py")
if ($LASTEXITCODE -ne 0) {
    throw "La verification CUDA a echoue."
}

if (-not (Get-Command sox -ErrorAction SilentlyContinue)) {
    Write-Warning "SoX n'est pas installe. Si Qwen signale une erreur audio : winget install ChrisBagwell.SoX"
}

Write-Host ""
Write-Host "Installation terminee." -ForegroundColor Green
Write-Host "Lancez ensuite : .\start_windows_nvidia.ps1"
