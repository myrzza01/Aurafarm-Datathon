[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/common.ps1"
$projectRoot = Get-ProjectRoot
$runtimeDirectory = Join-Path $projectRoot 'data/public'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$operationLock = [IO.File]::Open((Join-Path $runtimeDirectory 'operation.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    Import-ProjectEnvironment
    Initialize-LocalApiKey
    $cloudflared = Install-Cloudflared
    if ($PrepareOnly) {
        & $cloudflared --version
        Write-Output 'Public launch files and the verified tunnel binary are prepared.'
        return
    }
    $pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts/setup.ps1 first to create .venv.' }
    $statePath = Join-Path $projectRoot 'data/public-run.json'
    if (Test-Path -LiteralPath $statePath) {
        $previous = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($previous.projectRoot -ne $projectRoot) { throw 'Recorded launch belongs to another project path.' }
        if ($null -ne (Get-OwnedPublicProcess $previous.backend) -or $null -ne (Get-OwnedPublicProcess $previous.tunnel)) {
            throw 'A public launch is already running. Use scripts/stop-public.ps1 first.'
        }
    }
    $listener = New-Object Net.Sockets.TcpClient
    try {
        if ($listener.ConnectAsync('127.0.0.1', $Port).Wait(500) -and $listener.Connected) {
            throw "Port $Port is already in use. Stop its current application first."
        }
    } catch [AggregateException] {
        # Connection refused means that the requested local port is available.
    } finally { $listener.Dispose() }
    $runId = [Guid]::NewGuid().ToString('N')
    $logPrefix = Join-Path $runtimeDirectory $runId
    $urlPath = Join-Path $projectRoot 'data/public-url.txt'
    if (Test-Path -LiteralPath $urlPath) { Remove-Item -LiteralPath $urlPath -Force }
    $state = [ordered]@{ schema = 1; projectRoot = $projectRoot; runId = $runId; port = $Port; createdUtc = [DateTime]::UtcNow.ToString('o'); backend = $null; tunnel = $null; url = $null }
    try {
        $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'backend.main:create_app', '--factory', '--host', '127.0.0.1', '--port', "$Port", '--proxy-headers', '--forwarded-allow-ips', '127.0.0.1', '--no-access-log') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput ($logPrefix + '.backend.out.log') -RedirectStandardError ($logPrefix + '.backend.err.log')
        $state.backend = Get-PublicProcessRecord $backendProcess 'backend'
        Save-PublicState $state
        $health = $null
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(45)
        while ([DateTime]::UtcNow -lt $readyDeadline) {
            if ($backendProcess.HasExited) { throw 'Backend exited. Inspect its local data/public logs.' }
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
                break
            } catch { Start-Sleep -Milliseconds 300 }
        }
        if ($null -eq $health) { throw 'Backend did not become ready within 45 seconds.' }
        if (-not $health.analyzer_configured -or $health.analyzer_kind -ne 'semantic') {
            throw 'The semantic analyzer must be configured before public launch.'
        }
        $tunnelLog = $logPrefix + '.tunnel.log'
        $tunnelConfiguration = $logPrefix + '.tunnel.yml'
        [IO.File]::WriteAllText($tunnelConfiguration, "{}`r`n", (New-Object Text.UTF8Encoding($false)))
        $tunnelProcess = Start-Process -FilePath $cloudflared -ArgumentList @('tunnel', '--config', ('"' + $tunnelConfiguration + '"'), '--no-autoupdate', '--url', "http://127.0.0.1:$Port", '--protocol', 'http2', '--metrics', '127.0.0.1:0', '--management-diagnostics=false', '--logfile', ('"' + $tunnelLog + '"'), '--loglevel', 'info') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput ($logPrefix + '.tunnel.out.log') -RedirectStandardError ($logPrefix + '.tunnel.err.log')
        $state.tunnel = Get-PublicProcessRecord $tunnelProcess 'tunnel'
        Save-PublicState $state
        $urlDeadline = [DateTime]::UtcNow.AddSeconds(75)
        while ([DateTime]::UtcNow -lt $urlDeadline) {
            if ($tunnelProcess.HasExited) { throw 'Tunnel exited. Inspect its local data/public logs.' }
            if (Test-Path -LiteralPath $tunnelLog) {
                $match = [regex]::Match((Read-SharedLog $tunnelLog), 'https://[a-z0-9-]+\.trycloudflare\.com')
                if ($match.Success) { $state.url = $match.Value; break }
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $state.url) { throw 'The tunnel did not supply a public URL within 75 seconds.' }
        Save-PublicState $state
        $publicReady = $false
        $publicDeadline = [DateTime]::UtcNow.AddSeconds(45)
        while ([DateTime]::UtcNow -lt $publicDeadline) {
            if ($tunnelProcess.HasExited -or $backendProcess.HasExited) { throw 'A public launch process exited during its readiness check.' }
            try {
                $publicHealth = Invoke-RestMethod -Uri ($state.url + '/health') -TimeoutSec 5
                if ($publicHealth.status -eq 'ok' -and $publicHealth.analyzer_kind -eq 'semantic') { $publicReady = $true; break }
            } catch { Start-Sleep -Milliseconds 500 }
        }
        if (-not $publicReady) { throw 'The public HTTPS health check did not become ready within 45 seconds.' }
        [IO.File]::WriteAllText($urlPath, $state.url + "`r`n", (New-Object Text.UTF8Encoding($false)))
        Write-Output "Public URL: $($state.url)"
        Write-Output 'The API key is stored in the local ignored .env file. Keep this computer awake.'
        Write-Output 'The temporary URL changes after tunnel restarts. Stop with scripts/stop-public.ps1.'
    } catch {
        foreach ($tracked in @($state.tunnel, $state.backend)) {
            Stop-OwnedPublicProcessTree $tracked
        }
        if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath -Force }
        throw
    }
} finally { $operationLock.Dispose() }
