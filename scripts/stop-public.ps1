[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$projectRoot = Get-ProjectRoot
$runtimeDirectory = Join-Path $projectRoot 'data/public'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$operationLock = [IO.File]::Open((Join-Path $runtimeDirectory 'operation.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $statePath = Join-Path $projectRoot 'data/public-run.json'
    if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No recorded public launch.'; return }
    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($state.projectRoot -ne $projectRoot) { throw 'Recorded launch belongs to another project path.' }
    # Validate both identities before stopping either process; never trust a PID alone.
    foreach ($tracked in @($state.tunnel, $state.backend)) {
        Get-OwnedPublicProcess $tracked | Out-Null
    }
    foreach ($tracked in @($state.tunnel, $state.backend)) {
        Stop-OwnedPublicProcessTree $tracked
    }
    Remove-Item -LiteralPath $statePath -Force
    $urlPath = Join-Path $projectRoot 'data/public-url.txt'
    if (Test-Path -LiteralPath $urlPath) { Remove-Item -LiteralPath $urlPath -Force }
    Write-Output 'Public tunnel and its local backend are stopped.'
} finally { $operationLock.Dispose() }
