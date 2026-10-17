param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
Import-ProjectEnvironment
Initialize-LocalApiKey
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup.ps1 first.' }
Write-Host "Local service: http://127.0.0.1:$Port ; API key is stored in .env"
& $pythonPath -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
