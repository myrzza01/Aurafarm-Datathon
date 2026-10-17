param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if ($uvCommand) { & uv venv .venv --python 3.12 }
    else { & py -3.12 -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install Python 3.12 or uv, then rerun setup.' }
}
& $pythonPath -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for this locked PyTorch runtime'"
if ($LASTEXITCODE -ne 0) { throw 'Existing .venv uses an incompatible Python. Preserve it under another name and rerun setup.' }
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
if ($uvCommand) {
    & uv pip install --python $pythonPath torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & uv pip install --python $pythonPath -r requirements-dev.txt
} else {
    & $pythonPath -m pip install torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & $pythonPath -m pip install -r requirements-dev.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $pythonPath -c "import torch, av, timm, fastapi; print('Runtime ready:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Runtime verification failed' }
