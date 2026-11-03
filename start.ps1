$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$script = Join-Path $root 'src\visionhid.py'
$config = Join-Path $root 'config.ini'
if (-not (Test-Path -LiteralPath $config)) {
    Copy-Item -LiteralPath (Join-Path $root 'config.example.ini') -Destination $config
    Write-Host 'Created config.ini from config.example.ini. Set device.port to your COM port for HID output.'
}
$models = @(Get-ChildItem -LiteralPath (Join-Path $root 'models') -Filter '*.onnx' -File)
if ($models.Count -ne 1) { throw 'Place exactly one ONNX model in the models folder.' }
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = (& py -3.12 -c 'import sys; print(sys.executable)' 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $python) { throw 'Python 3.12 is required; run 一键安装.cmd first.' }
    $python = $python.Trim()
}
$pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw)) { throw "Missing pythonw.exe beside $python" }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'pythonw.exe') -and
    $_.CommandLine -match '[\\/](visionhid|v4_autoaim|v5_autoaim)\.py'
})
if ($active.Count -gt 0) { throw "Another aim process is running: PID=$($active[0].ProcessId)" }
$arguments = '-u -X utf8 "{0}" --config "{1}" --model "{2}" --headless' -f `
    $script, $config, $models[0].FullName
$child = Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root 'run.log') `
    -RedirectStandardError (Join-Path $root 'error.log') -PassThru
Write-Host "VisionHID started: PID=$($child.Id). See run.log and the status overlay."
