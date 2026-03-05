param(
    [string]$ProjectRoot = 'C:\AERIS-WSN-Protocol',
    [string]$RunTagWait = 'local_fix550_run4_20260211_210500',
    [int]$PollSeconds = 20
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

$queueTs = Get-Date -Format 'yyyyMMdd_HHmmss'
$queueLog = Join-Path $ProjectRoot ("results\\mega_experiments\\queue_local_9h_" + $queueTs + ".log")

function LogLine([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    $line | Tee-Object -FilePath $queueLog -Append
}

LogLine "Queue watcher started. Waiting for running tag: $RunTagWait"

while ($true) {
    $running = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like '*run_scalability_experiment.py*' -and $_.CommandLine -like "*$RunTagWait*" }

    if (-not $running) {
        break
    }

    $pids = ($running | Select-Object -ExpandProperty ProcessId) -join ','
    LogLine "Current run still active. PIDs=$pids. Sleep ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}

LogLine "Current run finished. Launching queued 9h experiment..."

$cmd = @(
    '-File', 'scripts\\run_overnight_scalability_10h.ps1',
    '-Replicates', '1000',
    '-Workers', '14',
    '-Nodes', '100,200,300,500,800,1000',
    '-Rounds', '300',
    '-MaxCpuPercent', '80',
    '-MaxMemPercent', '70',
    '-Environments', 'indoor_factory,outdoor_urban',
    '-GenerateProvenance'
)

LogLine ("Launch command: powershell " + ($cmd -join ' '))

# Run in this watcher process so queue semantics are deterministic.
& powershell @cmd 2>&1 | Tee-Object -FilePath $queueLog -Append
$exitCode = $LASTEXITCODE

LogLine "Queued experiment finished. ExitCode=$exitCode"
exit $exitCode
