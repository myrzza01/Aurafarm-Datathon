param(
    [string]$PythonPath = "",
    [string]$Bundle = "artifacts/handoff_selected",
    [string]$Video = "",
    [string]$Output = "artifacts/os_offline/checks.json",
    [string]$ControlIp = "1.1.1.1",
    [int]$ControlPort = 443
)
# Run from an ALREADY elevated PowerShell when available. This script never elevates
# itself and never changes global firewall settings. Its temporary rules apply only
# to the Python executables used for the experiment; every rule is removed in finally.
$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $taskRoot ".venv\Scripts\python.exe" }
$taskOutput = [IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory((Split-Path -Parent $taskOutput)) | Out-Null
$taskWorker = Join-Path $PSScriptRoot "os_offline_worker.py"
$taskRules = New-Object 'System.Collections.Generic.List[string]'
$taskReport = [ordered]@{
    status = "running"
    os_firewall_test_executed = $false
    python_socket_guard_used = $false
    controls_send_media = $false
    rule_cleanup_verified = $null
    inference = $null
    note = "TCP control sends no application payload; inference reports aggregates only."
}

function Invoke-TaskWorker([string[]]$WorkerArguments) {
    $taskText = & $PythonPath -I $taskWorker @WorkerArguments
    if ($LASTEXITCODE -ne 0) { throw "PythonWorkerFailed" }
    return ($taskText | ConvertFrom-Json)
}

try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $taskPrincipal = New-Object Security.Principal.WindowsPrincipal($taskIdentity)
    $taskAdmin = $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    $taskReport.is_administrator = $taskAdmin
    $taskReport.services = @(Get-Service BFE,MpsSvc | ForEach-Object {
        [ordered]@{ name = $_.Name; status = $_.Status.ToString() }
    })
    $taskReport.firewall_profiles = @(Get-NetFirewallProfile | ForEach-Object {
        [ordered]@{ name = $_.Name; enabled = ($_.Enabled.ToString() -eq "True") }
    })
    if (-not $taskAdmin) {
        $taskReport.status = "blocked"
        $taskReport.reason_code = "ADMINISTRATOR_TOKEN_REQUIRED"
        $taskReport.reason = "Current token cannot create the temporary Windows Firewall rules; no elevation attempted."
    } elseif (($taskReport.services | Where-Object status -ne "Running").Count -gt 0) {
        $taskReport.status = "blocked"
        $taskReport.reason_code = "FIREWALL_SERVICE_NOT_RUNNING"
    } elseif (($taskReport.firewall_profiles | Where-Object enabled -eq $false).Count -gt 0) {
        $taskReport.status = "blocked"
        $taskReport.reason_code = "FIREWALL_PROFILE_DISABLED"
    } else {
        if (-not $Video) { throw "VideoPathRequired" }
        $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
        $taskBundle = (Resolve-Path -LiteralPath $Bundle).Path
        $taskVideo = (Resolve-Path -LiteralPath $Video).Path
        if ((Test-Path -LiteralPath (Join-Path $taskBundle "cashwatch\train.py")) -or
            (Test-Path -LiteralPath (Join-Path $taskBundle "cashwatch\data.py"))) {
            throw "ExpectedRuntimeOnlyBundle"
        }
        $taskExecutables = Invoke-TaskWorker -WorkerArguments @("executables")
        $taskReport.pre_block_control = Invoke-TaskWorker -WorkerArguments @("probe", "--ip", $ControlIp, "--port", "$ControlPort")
        if (-not $taskReport.pre_block_control.connected) {
            throw "ExternalControlUnreachableBeforeBlock"
        }
        foreach ($taskExecutable in $taskExecutables.paths) {
            $taskRuleName = "CashwatchOffline-" + [guid]::NewGuid().ToString("N")
            # Record the unique name before creation so cleanup also covers partial failure.
            $taskRules.Add($taskRuleName)
            New-NetFirewallRule -Name $taskRuleName -DisplayName $taskRuleName -Direction Outbound `
                -Action Block -Enabled True -Profile Any -Program $taskExecutable -Protocol Any | Out-Null
            $taskRule = Get-NetFirewallRule -PolicyStore ActiveStore -Name $taskRuleName
            $taskApp = $taskRule | Get-NetFirewallApplicationFilter
            if ($taskRule.Enabled.ToString() -ne "True" -or $taskRule.Direction.ToString() -ne "Outbound" -or
                $taskRule.Action.ToString() -ne "Block" -or $taskApp.Program -ne $taskExecutable) {
                throw "ActiveRuleVerificationFailed"
            }
        }
        $taskReport.active_rule_count = $taskRules.Count
        $taskReport.os_firewall_test_executed = $true
        $taskReport.blocked_control = Invoke-TaskWorker -WorkerArguments @("probe", "--ip", $ControlIp, "--port", "$ControlPort")
        if ($taskReport.blocked_control.connected) { throw "FirewallDidNotBlockControl" }
        if ($taskReport.blocked_control.winerror -ne 10013 -and $taskReport.blocked_control.errno -ne 10013) {
            throw "ControlDeniedButNotWindowsAccessDenied_Inconclusive"
        }
        $taskReport.inference = Invoke-TaskWorker -WorkerArguments @("infer", "--bundle", $taskBundle, "--video", $taskVideo)
        $taskReport.status = "passed_pending_cleanup_control"
    }
} catch {
    $taskReport.status = "failed"
    # Exception type plus a controlled reason; do not log command lines or video data.
    $taskReport.error_type = $_.Exception.GetType().Name
    $taskSafeReason = $_.Exception.Message
    if ($taskSafeReason -match '^[A-Za-z_]+$') { $taskReport.reason_code = $taskSafeReason }
} finally {
    $taskCleanupOk = $true
    foreach ($taskRuleName in $taskRules) {
        try {
            Get-NetFirewallRule -Name $taskRuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
            if (Get-NetFirewallRule -Name $taskRuleName -ErrorAction SilentlyContinue) { $taskCleanupOk = $false }
        } catch { $taskCleanupOk = $false }
    }
    if ($taskRules.Count -gt 0) {
        $taskReport.rule_cleanup_verified = $taskCleanupOk
        if (-not $taskCleanupOk) {
            $taskReport.status = "failed"
            $taskReport.reason_code = "TEMPORARY_RULE_CLEANUP_FAILED"
            $taskReport.remaining_rule_names = @($taskRules)
        } elseif ($taskReport.os_firewall_test_executed) {
            try {
                $taskReport.post_cleanup_control = Invoke-TaskWorker -WorkerArguments @("probe", "--ip", $ControlIp, "--port", "$ControlPort")
                if (-not $taskReport.post_cleanup_control.connected) {
                    $taskReport.status = "failed"
                    $taskReport.reason_code = "CONNECTIVITY_NOT_RESTORED_INCONCLUSIVE"
                } elseif ($taskReport.status -eq "passed_pending_cleanup_control") {
                    $taskReport.status = "passed"
                }
            } catch { $taskReport.status = "failed"; $taskReport.reason_code = "POST_CLEANUP_CONTROL_FAILED" }
        }
    }
    [IO.File]::WriteAllText($taskOutput, ($taskReport | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
}
[ordered]@{ status = $taskReport.status; os_firewall_test_executed = $taskReport.os_firewall_test_executed;
    reason_code = $taskReport.reason_code } | ConvertTo-Json
if ($taskReport.status -eq "passed") { exit 0 }
if ($taskReport.status -eq "blocked") { exit 2 }
exit 1
