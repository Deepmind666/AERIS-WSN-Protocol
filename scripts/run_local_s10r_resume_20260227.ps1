$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = "python"
$runner = "scripts/run_scalability_experiment.py"
$resultDir = "results/mega_experiments"
$logDir = "logs"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "s10r_local_resume_${ts}.log"
$workers = 18
$maxCpu = 90
$maxMem = 96
$retryLimit = 1

New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-Quality {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    try {
        $obj = Get-Content -Raw -Path $Path | ConvertFrom-Json
        $raw = @($obj.raw_results)
        $bad = @($raw | Where-Object { -not $_.success }).Count
        return [pscustomobject]@{
            raw_results = $raw.Count
            error_runs = [int]$obj.error_runs
            bad_runs = [int]$bad
            run_tier = [string]$obj.run_tier
            primary_metric = [string]$obj.primary_metric
        }
    }
    catch {
        return $null
    }
}

function Is-Accepted {
    param([string]$Path)
    $q = Get-Quality -Path $Path
    if ($null -eq $q) { return $false }
    return (
        $q.raw_results -eq 30000 -and
        $q.error_runs -eq 0 -and
        $q.bad_runs -eq 0 -and
        $q.run_tier -eq "publication" -and
        $q.primary_metric -eq "pdr_expected"
    )
}

function Run-Task {
    param(
        [string]$EnvName,
        [double]$TxPower,
        [string]$OutPath
    )

    $args = @(
        $runner,
        "--env", $EnvName,
        "--tx-power", ("{0:N1}" -f $TxPower),
        "--replicates", "1000",
        "--seed", "42001",
        "--nodes", "100,200,300,500,800,1000",
        "--rounds", "300",
        "--workers", "$workers",
        "--run-tier", "publication",
        "--mac-collision",
        "--multihop-relay",
        "--max-cpu-percent", "$maxCpu",
        "--max-mem-percent", "$maxMem",
        "--allow-partial",
        "--output", $OutPath
    )

    for ($attempt = 0; $attempt -le $retryLimit; $attempt++) {
        if ($attempt -gt 0) {
            Write-Log ("RETRY attempt={0} env={1} tx={2}" -f $attempt, $EnvName, $TxPower)
        }
        Write-Log ("START env={0} tx={1} workers={2} max_cpu={3}% max_mem={4}% output={5}" -f $EnvName, $TxPower, $workers, $maxCpu, $maxMem, $OutPath)
        & $pythonExe @args 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        $accepted = Is-Accepted -Path $OutPath
        if ($exitCode -eq 0 -and $accepted) {
            $q = Get-Quality -Path $OutPath
            Write-Log ("DONE env={0} tx={1} raw={2} error_runs={3} bad_runs={4}" -f $EnvName, $TxPower, $q.raw_results, $q.error_runs, $q.bad_runs)
            return $true
        }
        Write-Log ("FAIL env={0} tx={1} exit={2} accepted={3}" -f $EnvName, $TxPower, $exitCode, $accepted)
    }
    return $false
}

$tasks = @(
    @{ env = "indoor_office"; tx = 10.0; out = "scalability_indoor_office_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_office"; tx = 15.0; out = "scalability_indoor_office_local_s10r_tx15_20260226.json" },
    @{ env = "indoor_factory"; tx = 5.0;  out = "scalability_indoor_factory_local_s10r_tx5_20260226.json" },
    @{ env = "indoor_factory"; tx = 10.0; out = "scalability_indoor_factory_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_factory"; tx = 15.0; out = "scalability_indoor_factory_local_s10r_tx15_20260226.json" }
)

$todo = @()
foreach ($t in $tasks) {
    $outPath = Join-Path $resultDir $t.out
    if (Is-Accepted -Path $outPath) {
        Write-Log ("SKIP env={0} tx={1} reason=accepted_output_exists" -f $t.env, $t.tx)
    }
    else {
        $todo += $t
    }
}

Write-Log ("QUEUE_START todo_tasks={0} eta_basis=~1.8h_per_task_from_latest_s10r_tail_rate" -f $todo.Count)

$index = 0
foreach ($t in $todo) {
    $index += 1
    $remain = $todo.Count - $index + 1
    $etaMin = [math]::Round($remain * 108, 0)
    Write-Log ("ETA task={0}/{1} env={2} tx={3} eta_global~{4}min basis=108min_per_task" -f $index, $todo.Count, $t.env, $t.tx, $etaMin)
    $outPath = Join-Path $resultDir $t.out
    $ok = Run-Task -EnvName $t.env -TxPower $t.tx -OutPath $outPath
    if (-not $ok) {
        Write-Log ("ABORT env={0} tx={1} reason=max_retry_exceeded" -f $t.env, $t.tx)
        exit 1
    }
}

Write-Log "QUEUE_END all_todo_done=true"
