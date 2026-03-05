$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonExe = "python"
$scriptPath = "scripts/run_scalability_experiment.py"
$logDir = "logs"
$resultDir = "results/mega_experiments"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "s10r_local_queue_${ts}.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Test-LocalGuard {
    $samples = @()
    for ($i = 0; $i -lt 6; $i++) {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        $samples += $cpu
        Start-Sleep -Seconds 10
    }
    $avgCpu = (($samples | Measure-Object -Average).Average)
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGb = $os.FreePhysicalMemory / 1MB
    $chipWorkers = @(Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "run_chip_worker\.py" })

    Write-Log ("Resource guard: avg_cpu_60s={0:N2} free_mem_gb={1:N2} chip_workers={2}" -f $avgCpu, $freeGb, $chipWorkers.Count)

    if ($avgCpu -ge 45) { return $false }
    if ($freeGb -le 35) { return $false }
    if ($chipWorkers.Count -gt 0) { return $false }
    return $true
}

function Run-Step {
    param([string]$EnvName, [double]$TxPower, [string]$OutputFile)
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
    Write-Log ("START env={0} tx={1} output={2}" -f $EnvName, $TxPower, $OutputFile)
    & $pythonExe @cmd 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "run_scalability_experiment failed: env=$EnvName tx=$TxPower exit=$LASTEXITCODE"
    }
    Write-Log ("DONE env={0} tx={1}" -f $EnvName, $TxPower)
}

if (-not (Test-LocalGuard)) {
    Write-Log "Local guard not satisfied. Queue aborted to protect concurrent workloads."
    exit 2
}

$tasks = @(
    @{ env = "indoor_office"; tx = 5.0;  out = "scalability_indoor_office_local_s10r_tx5_20260226.json"  },
    @{ env = "indoor_office"; tx = 10.0; out = "scalability_indoor_office_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_office"; tx = 15.0; out = "scalability_indoor_office_local_s10r_tx15_20260226.json" },
    @{ env = "indoor_factory"; tx = 5.0;  out = "scalability_indoor_factory_local_s10r_tx5_20260226.json"  },
    @{ env = "indoor_factory"; tx = 10.0; out = "scalability_indoor_factory_local_s10r_tx10_20260226.json" },
    @{ env = "indoor_factory"; tx = 15.0; out = "scalability_indoor_factory_local_s10r_tx15_20260226.json" }
)

foreach ($t in $tasks) {
    $outPath = Join-Path $resultDir $t.out
    Run-Step -EnvName $t.env -TxPower $t.tx -OutputFile $outPath
}

Write-Log "All local S10R tasks completed."
