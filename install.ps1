param([switch]$VerifyOnly)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$modelDir = Join-Path $root 'models'
$modelPath = Join-Path $modelDir 'visionhid-val-yellow-v11s-640.onnx'
$modelUrl = 'https://github.com/anan1919bit/VisionHID/releases/download/v0.4.1/visionhid-val-yellow-v11s-640.onnx'
$modelSha256 = 'D3FCE9CE6D747787DEA397A5B24FEE00075EEE4DD4D20AA5FE4DC74A87C75E7D'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

if ($PSVersionTable.PSVersion.Major -le 5) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}

if ($env:OS -ne 'Windows_NT') { throw 'This installer supports Windows only.' }
$launcher = Get-Command py -ErrorAction SilentlyContinue
if (-not $launcher) { throw 'Install Python 3.12 with the Windows py launcher first.' }
& py -3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)' 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }

if (-not (Test-Path -LiteralPath $modelDir)) {
    New-Item -ItemType Directory -Path $modelDir | Out-Null
}

if (-not (Test-Path -LiteralPath $modelPath)) {
    if ($VerifyOnly) { throw "Model missing: $modelPath" }
    $temporary = "$modelPath.download"
    try {
        Write-Host 'Downloading the ONNX model from this repository release...'
        Invoke-WebRequest -Uri $modelUrl -OutFile $temporary -UseBasicParsing
        $downloadHash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
        if ($downloadHash -ne $modelSha256) { throw "Model SHA256 mismatch: $downloadHash" }
        Move-Item -LiteralPath $temporary -Destination $modelPath
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}
$actualHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
if ($actualHash -ne $modelSha256) { throw "Model SHA256 mismatch: $actualHash" }
Write-Host "Model verified: $actualHash"

if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($VerifyOnly) { throw "Virtual environment missing: $venvPython" }
    & py -3.12 -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
}

if (-not $VerifyOnly) {
    & $venvPython -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $venvPython -c 'import cv2, dxcam, numpy, onnxruntime; print("Python dependencies OK")'
if ($LASTEXITCODE -ne 0) { throw 'Python dependency verification failed.' }

$configPath = Join-Path $root 'config.ini'
if (-not (Test-Path -LiteralPath $configPath)) {
    if ($VerifyOnly) { throw "Configuration missing: $configPath" }
    Copy-Item -LiteralPath (Join-Path $root 'config.example.ini') -Destination $configPath
    Write-Host 'Created config.ini. Set device.port to your ESP32 COM port when ready.'
}

Write-Host 'Installation verified. Start with: .\启动 VisionHID.cmd'
Write-Host 'ESP32 firmware flashing and CUDA/cuDNN setup are device-specific and are not performed by this script.'
