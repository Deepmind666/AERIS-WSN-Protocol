$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = "python"
$scriptPath = "scripts/run_scalability_experiment.py"
$logDir = "logs"
$resultDir = "results/mega_experiments"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "s10r_local_force9h_queue_${ts}.log"
$deadlineHours = 9
$maxRetriesPerTask = 1

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-RunMeta {
    param([string]$JsonPath)
    if (-not (Test-Path $JsonPath)) {
        return $null
    }
    try {
        $obj = Get-Content -Raw -Path $JsonPath | ConvertFrom-Json
        [pscustomobject]@{
            raw_results = [int]$obj.raw_results
            error_runs = [int]$obj.error_runs
            run_tier = [string]$obj.run_tier
            primary_metric = [string]$obj.primary_metric
        }
    }
    catch {
        return $null
    }
}

function Is-AcceptedOutput {
    param([string]$JsonPath)
    $meta = Get-RunMeta -JsonPath $JsonPath
    if ($null -eq $meta) { return $false }
    if ($meta.raw_results -ne 30000) { return $false }
    if ($meta.error_runs -ne 0) { return $false }
    if ($meta.run_tier -ne "publication") { return $false }
    if ($meta.primary_metric -ne "pdr_expected") { return $false }
    return $true
}

function Run-Step {
    param(
        [string]$EnvName,
        [double]$TxPower,
        [string]$OutputFile
    )

    $cmd = @(
        $scriptPath,
        "--env", $EnvName,
        "--tx-power", ("{0:N1}" -f $TxPower),
        "--replicates", "1000",
        "--seed", "42001",
        "--nodes", "100,200,300,500,800,1000",
        "--rounds", "300",
        "--workers", "18",
        "--run-tier", "publication",
        "--mac-collision",
        "--multihop-relay",
        "--max-cpu-percent", "90",
        "--max-mem-percent", "96",
        "--allow-partial",
        "--output", $OutputFile
    )

    for ($attempt = 0; $attempt -le $maxRetriesPerTask; $attempt++) {
        if ($attempt -gt 0) {
            Write-Log ("RETRY attempt={0} env={1} tx={2}" -f $attempt, $EnvName, $TxPower)
        }
        Write-Log ("START env={0} tx={1} output={2}" -f $EnvName, $TxPower, $OutputFile)
        & $pythonExe @cmd 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0 -and (Is-AcceptedOutput -JsonPath $OutputFile)) {
            Write-Log ("DONE env={0} tx={1} accepted=true" -f $EnvName, $TxPower)
            return $true
        }
        Write-Log ("FAIL env={0} tx={1} exit={2} accepted={3}" -f $EnvName, $TxPower, $exitCode, (Is-AcceptedOutput -JsonPath $OutputFile))
    }
    return $false
}

$tasks = @(
    @{ env = "indoor_office";  tx = 5.0;  out = "scalability_indoor_office_local_s10r_tx5_20260226.json"  },
    @{ env = "indoor_office";  tx = 10.0; out = "scalability_indoor_office_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_office";  tx = 15.0; out = "scalability_indoor_office_local_s10r_tx15_20260226.json" },
    @{ env = "indoor_factory"; tx = 5.0;  out = "scalability_indoor_factory_local_s10r_tx5_20260226.json"  },
    @{ env = "indoor_factory"; tx = 10.0; out = "scalability_indoor_factory_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_factory"; tx = 15.0; out = "scalability_indoor_factory_local_s10r_tx15_20260226.json" }
)

$t0 = Get-Date
Write-Log ("QUEUE_START deadline_hours={0} stop_rule=do_not_interrupt_current_task" -f $deadlineHours)
Write-Log ("T0={0}" -f $t0.ToString("yyyy-MM-dd HH:mm:ss"))

$completedDurations = @()
$taskIndex = 0
$totalTasks = $tasks.Count

foreach ($t in $tasks) {
    $taskIndex += 1
    $now = Get-Date
    $elapsedHours = ($now - $t0).TotalHours
    if ($elapsedHours -ge $deadlineHours) {
        Write-Log ("CUTOFF_REACHED elapsed_hours={0:N2}. no_new_task_after_cutoff=true" -f $elapsedHours)
        break
    }

    $outPath = Join-Path $resultDir $t.out
    if (Is-AcceptedOutput -JsonPath $outPath) {
        Write-Log ("SKIP task={0}/{1} env={2} tx={3} reason=accepted_output_exists" -f $taskIndex, $totalTasks, $t.env, $t.tx)
        continue
    }

    $remainingTasks = ($tasks.Count - $taskIndex + 1)
    if ($completedDurations.Count -gt 0) {
        $avgMinutes = (($completedDurations | Measure-Object -Average).Average)
        $etaCurrent = [TimeSpan]::FromMinutes($avgMinutes)
        $etaGlobal = [TimeSpan]::FromMinutes($avgMinutes * $remainingTasks)
        Write-Log ("ETA task={0}/{1} env={2} tx={3} eta_current~{4} eta_global~{5} basis=avg_of_{6}_completed_tasks" -f `
            $taskIndex, $totalTasks, $t.env, $t.tx, $etaCurrent.ToString("hh\:mm"), $etaGlobal.ToString("hh\:mm"), $completedDurations.Count)
    }
    else {
        Write-Log ("ETA task={0}/{1} env={2} tx={3} eta_current~01:50 eta_global~11:00 basis=bootstrap_from_recent_local_s10r_log_rate_3.7_per_sec" -f `
            $taskIndex, $totalTasks, $t.env, $t.tx)
    }

    $taskStart = Get-Date
    $ok = Run-Step -EnvName $t.env -TxPower $t.tx -OutputFile $outPath
    $taskMinutes = (Get-Date - $taskStart).TotalMinutes
    $completedDurations += $taskMinutes

    if (-not $ok) {
        Write-Log ("TASK_ABORT task={0}/{1} env={2} tx={3} reason=max_retry_exceeded" -f $taskIndex, $totalTasks, $t.env, $t.tx)
        break
    }

    Write-Log ("TASK_DONE task={0}/{1} env={2} tx={3} duration_min={4:N1}" -f $taskIndex, $totalTasks, $t.env, $t.tx, $taskMinutes)
}

$elapsed = (Get-Date - $t0)
Write-Log ("QUEUE_END elapsed={0:hh\\:mm\\:ss}" -f $elapsed)
Write-Log ("LOG_PATH={0}" -f $logPath)
