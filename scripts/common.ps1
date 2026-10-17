$script:AurafarmProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Get-ProjectRoot {
    return $script:AurafarmProjectRoot
}

function ConvertFrom-EnvironmentLine {
    param([string]$Line)
    if ($Line -match '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
        $entryName = $Matches[1]
        $entryValue = $Matches[2].Trim()
        if ($entryValue.Length -ge 2 -and (
            ($entryValue.StartsWith('"') -and $entryValue.EndsWith('"')) -or
            ($entryValue.StartsWith("'") -and $entryValue.EndsWith("'"))
        )) { $entryValue = $entryValue.Substring(1, $entryValue.Length - 2) }
        return [pscustomobject]@{ Name = $entryName; Value = $entryValue }
    }
}

function Import-ProjectEnvironment {
    $environmentPath = Join-Path (Get-ProjectRoot) '.env'
    if (Test-Path -LiteralPath $environmentPath) {
        foreach ($line in [IO.File]::ReadAllLines($environmentPath, [Text.Encoding]::UTF8)) {
            $entry = ConvertFrom-EnvironmentLine $line
            if ($null -ne $entry) {
                [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value, 'Process')
            }
        }
    }
}

function Initialize-LocalApiKey {
    # Never print a credential or place it on a process command line.
    $environmentPath = Join-Path (Get-ProjectRoot) '.env'
    $stream = [IO.File]::Open($environmentPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8, $true, 1024, $true)
        try { $existingText = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $savedKey = $null
        foreach ($line in ($existingText -split "`r?`n")) {
            $entry = ConvertFrom-EnvironmentLine $line
            if ($null -ne $entry -and $entry.Name -eq 'API_KEY') { $savedKey = $entry.Value }
        }
        if ([string]::IsNullOrWhiteSpace($savedKey)) {
            $savedKey = [Environment]::GetEnvironmentVariable('API_KEY', 'Process')
            if ([string]::IsNullOrWhiteSpace($savedKey)) {
                $bytes = New-Object byte[] 32
                $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
                try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
                $savedKey = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
            }
            $prefix = if ($existingText.Length -gt 0 -and -not $existingText.EndsWith("`n")) { "`r`n" } else { '' }
            $addition = [Text.Encoding]::UTF8.GetBytes($prefix + 'API_KEY=' + $savedKey + "`r`n")
            $stream.Seek(0, [IO.SeekOrigin]::End) | Out-Null
            $stream.Write($addition, 0, $addition.Length)
            $stream.Flush()
        }
        [Environment]::SetEnvironmentVariable('API_KEY', $savedKey, 'Process')
    } finally { $stream.Dispose() }
}

function Get-StringSha256 {
    param([string]$Value)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value))).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
}

function Get-PublicProcessRecord {
    param([Diagnostics.Process]$Process, [string]$Role)
    $record = Get-CimInstance Win32_Process -Filter "ProcessId=$($Process.Id)"
    if ($null -eq $record) { throw "$Role exited before its ownership could be recorded." }
    return [pscustomobject]@{
        pid = $Process.Id
        executable = [IO.Path]::GetFullPath($record.ExecutablePath)
        startedUtc = $record.CreationDate.ToUniversalTime().ToString('o')
        commandHash = Get-StringSha256 $record.CommandLine
        role = $Role
    }
}

function Get-OwnedPublicProcess {
    param($Record)
    if ($null -eq $Record) { return $null }
    $expectedExecutable = switch ($Record.role) {
        'backend' { Join-Path (Get-ProjectRoot) '.venv/Scripts/python.exe' }
        'tunnel' { Join-Path (Get-ProjectRoot) 'data/tools/cloudflared.exe' }
        default { throw 'Unknown recorded process role.' }
    }
    if (-not [string]::Equals([IO.Path]::GetFullPath($expectedExecutable), [string]$Record.executable, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Recorded executable is outside this project launch.'
    }
    $processId = [int]$Record.pid
    if ($processId -le 0) { throw 'Invalid recorded process ID.' }
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$processId"
    if ($null -eq $current) { return $null }
    $sameExecutable = [string]::Equals([IO.Path]::GetFullPath($current.ExecutablePath), [string]$Record.executable, [StringComparison]::OrdinalIgnoreCase)
    $sameStart = $current.CreationDate.ToUniversalTime().ToString('o') -eq $Record.startedUtc
    $sameCommand = (Get-StringSha256 $current.CommandLine) -eq $Record.commandHash
    if (-not ($sameExecutable -and $sameStart -and $sameCommand)) {
        throw "Refusing to stop PID $processId because its recorded ownership does not match."
    }
    return $current
}

function Stop-OwnedPublicProcessTree {
    param($Record)
    $owned = Get-OwnedPublicProcess $Record
    if ($null -ne $owned) {
        # Windows venv python.exe can be a launcher with a separate Python child.
        # Stop the verified root and its descendants together.
        & (Join-Path $env:SystemRoot 'System32/taskkill.exe') /PID $owned.ProcessId /T /F 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0 -and $null -ne (Get-OwnedPublicProcess $Record)) {
            throw "Could not stop the owned $($Record.role) process tree."
        }
        Wait-Process -Id $owned.ProcessId -Timeout 10 -ErrorAction SilentlyContinue
    }
}

function Save-PublicState {
    param($State)
    $statePath = Join-Path (Get-ProjectRoot) 'data/public-run.json'
    $temporaryPath = $statePath + '.new'
    [IO.File]::WriteAllText($temporaryPath, ($State | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporaryPath -Destination $statePath -Force
}

function Read-SharedLog {
    param([string]$Path)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
        try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally { $stream.Dispose() }
}

function Install-Cloudflared {
    $toolDirectory = Join-Path (Get-ProjectRoot) 'data/tools'
    New-Item -ItemType Directory -Path $toolDirectory -Force | Out-Null
    $binaryPath = Join-Path $toolDirectory 'cloudflared.exe'
    $version = '2026.10.1'
    $expectedHash = '4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6'
    $downloadUrl = "https://github.com/cloudflare/cloudflared/releases/download/$version/cloudflared-windows-amd64.exe"
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'This local tunnel package requires 64-bit Windows.' }
    if (-not (Test-Path -LiteralPath $binaryPath)) {
        $downloadPath = $binaryPath + '.download'
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $downloadPath
            if ((Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
                throw 'Cloudflared download failed its official SHA-256 verification.'
            }
            Move-Item -LiteralPath $downloadPath -Destination $binaryPath
        } finally {
            if (Test-Path -LiteralPath $downloadPath) { Remove-Item -LiteralPath $downloadPath -Force }
        }
    }
    if ((Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw 'Existing cloudflared binary does not match the official pinned release.'
    }
    $source = [ordered]@{ version = $version; url = $downloadUrl; sha256 = $expectedHash; publisher = 'Cloudflare'; release = "https://github.com/cloudflare/cloudflared/releases/tag/$version" }
    [IO.File]::WriteAllText((Join-Path $toolDirectory 'cloudflared.source.json'), ($source | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    return $binaryPath
}
