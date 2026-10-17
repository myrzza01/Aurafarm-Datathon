param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
& (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\setup.ps1') -Cpu:$Cpu
exit $LASTEXITCODE
