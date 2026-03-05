# S10R Server Queue - outdoor_urban + outdoor_suburban (6 tasks serial)
# Launched by Claude session, 2026-02-27
# CPU budget: 90%, 9h window

$ErrorActionPreference = "Continue"
$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$WORKDIR = "C:\Users\sshuser\AERIS-WSN-Protocol"
$LOGDIR = "$WORKDIR\logs"
$OUTDIR = "$WORKDIR\results\mega_experiments"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOGFILE = "$LOGDIR\s10r_server_queue_${TS}.log"

if (-not (Test-Path $LOGDIR)) { New-Item -ItemType Directory -Path $LOGDIR -Force | Out-Null }
if (-not (Test-Path $OUTDIR)) { New-Item -ItemType Directory -Path $OUTDIR -Force | Out-Null }

Set-Location $WORKDIR

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LOGFILE -Value $line
}

Log "=== S10R Server Queue START ==="
Log "git_commit=$(git rev-parse --short HEAD)"
Log "workers=20, max_cpu=90, max_mem=96"

$tasks = @(
    @{env="outdoor_urban";    tx=5},
    @{env="outdoor_urban";    tx=10},
    @{env="outdoor_urban";    tx=15},
    @{env="outdoor_suburban"; tx=5},
    @{env="outdoor_suburban"; tx=10},
    @{env="outdoor_suburban"; tx=15}
)

$total = $tasks.Count
$done = 0
$failed_tasks = @()

foreach ($t in $tasks) {
    $env_name = $t.env
    $tx = $t.tx
    $outfile = "$OUTDIR\scalability_${env_name}_server_s10r_tx${tx}_20260227.json"
    $tasknum = $done + 1

    # Skip if already completed
    if (Test-Path $outfile) {
        Log "SKIP $tasknum/$total env=$env_name tx=$tx (output exists)"
        $done++
        continue
    }

    Log "TASK $tasknum/$total START env=$env_name tx=$tx output=$outfile"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & $PYTHON scripts/run_scalability_experiment.py `
        --env $env_name --tx-power $tx `
        --replicates 1000 --nodes 100,200,300,500,800,1000 `
        --rounds 300 --workers 20 --run-tier publication `
        --mac-collision --multihop-relay `
        --max-cpu-percent 90 --max-mem-percent 96 --allow-partial `
        --output $outfile 2>&1 | Tee-Object -Append -FilePath $LOGFILE

    $exit_code = $LASTEXITCODE
    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalMinutes, 1)

    if ($exit_code -eq 0 -and (Test-Path $outfile)) {
        Log "TASK $tasknum/$total DONE env=$env_name tx=$tx exit=0 elapsed=${elapsed}min"
        $done++
    } else {
        Log "TASK $tasknum/$total FAIL env=$env_name tx=$tx exit=$exit_code elapsed=${elapsed}min"
        $failed_tasks += "$env_name tx$tx"
    }
}

Log "=== S10R Server Queue END ==="
Log "completed=$done/$total failed=$($failed_tasks.Count)"
if ($failed_tasks.Count -gt 0) {
    Log "failed_tasks: $($failed_tasks -join ', ')"
}
