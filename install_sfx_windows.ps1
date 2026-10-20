[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EngineRepo = Join-Path $ProjectDir ".stable-audio-3"
$EngineDir = Join-Path $EngineRepo "optimized\tflite"

Set-Location $ProjectDir

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Le lanceur Python 'py' est absent. Installez Python 3.12 en 64 bits."
}

& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 est requis."
}

if (Test-Path (Join-Path $EngineRepo ".git")) {
    Write-Host "Mise a jour du moteur Stable Audio 3..." -ForegroundColor Cyan
    & git -C $EngineRepo pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "La mise a jour du moteur a echoue." }
} elseif (Test-Path $EngineRepo) {
    throw "$EngineRepo existe mais n'est pas un depot Git valide."
} elseif (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Host "Telechargement du moteur Stable Audio 3 officiel..." -ForegroundColor Cyan
    & git clone --depth=1 https://github.com/Stability-AI/stable-audio-3.git $EngineRepo
    if ($LASTEXITCODE -ne 0) { throw "Le telechargement du moteur a echoue." }
} else {
    Write-Host "Git absent : telechargement de l'archive officielle..." -ForegroundColor Cyan
    $ZipPath = Join-Path $env:TEMP "stable-audio-3-main.zip"
    $ExtractDir = Join-Path $env:TEMP "stable-audio-3-extract"
    Invoke-WebRequest `
        "https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip" `
        -OutFile $ZipPath
    if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
    Expand-Archive $ZipPath -DestinationPath $ExtractDir -Force
    Move-Item (Join-Path $ExtractDir "stable-audio-3-main") $EngineRepo
    Remove-Item $ZipPath -Force
    Remove-Item $ExtractDir -Recurse -Force
}

if (-not (Test-Path (Join-Path $EngineDir "install.bat"))) {
    throw "Le runtime TFLite officiel est introuvable dans $EngineDir."
}

$PythonExe = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
$env:PATH = "$(Split-Path $PythonExe);$env:PATH"

Write-Host "Installation du runtime TFLite isole..." -ForegroundColor Cyan
Push-Location $EngineDir
try {
    & cmd.exe /d /c install.bat
    if ($LASTEXITCODE -ne 0) { throw "L'installation TFLite a echoue." }

    $EnginePython = Join-Path $EngineDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $EnginePython)) {
        throw "L'environnement Stable Audio n'a pas ete cree."
    }
    $env:INSTALL_SKIP_PIP = "1"
    & $EnginePython (Join-Path $EngineDir "scripts\install.py") --download sm-sfx
    if ($LASTEXITCODE -ne 0) { throw "Le telechargement Small-SFX a echoue." }
} finally {
    Remove-Item Env:INSTALL_SKIP_PIP -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host ""
Write-Host "Installation terminee. Lancez : .\start_sfx_windows.ps1" -ForegroundColor Green
