$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$runner = Join-Path $PSScriptRoot "run_local_s10r_queue_9h_force_20260227.ps1"
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner) `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$pidFile = Join-Path "logs" "s10r_local_force9h_queue_launcher_${ts}.pid"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
"pid=$($proc.Id)`nstart_time=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`nrunner=$runner" | Set-Content -Path $pidFile -Encoding UTF8

Write-Output ("LAUNCHED pid={0} pid_file={1}" -f $proc.Id, $pidFile)
