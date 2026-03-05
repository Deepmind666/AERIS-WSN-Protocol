param(
    [string]$PythonExe = "python",
    [int]$Replicates = 3200,
    [string]$Nodes = "100,200,300,500,800,1000",
    [int]$Rounds = 300,
    [int]$Workers = 12,
    [int]$MaxCpuPercent = 88,
    [int]$MaxMemPercent = 88
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "results/mega_experiments/queue_v50rigor_14h_${ts}.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $logFile -Append
}

function Get-V50Running {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*run_scalability_experiment.py*" -and
        $_.CommandLine -like "*v50rigor*"
    }
    return @($procs)
}

function Run-Env {
    param([string]$EnvName)
    $runTs = Get-Date -Format "yyyyMMdd_HHmmss"
    $outJson = "results/mega_experiments/scalability_${EnvName}_v50rigor_${runTs}.json"
    $args = @(
        "scripts/run_scalability_experiment.py",
        "--env", $EnvName,
        "--replicates", "$Replicates",
        "--seed", "42001",
        "--nodes", $Nodes,
        "--rounds", "$Rounds",
        "--workers", "$Workers",
        "--run-tier", "publication",
        "--tx-power", "10.0",
        "--max-cpu-percent", "$MaxCpuPercent",
        "--max-mem-percent", "$MaxMemPercent",
        "--resource-check-sec", "1",
        "--mac-collision",
        "--multihop-relay",
        "--allow-partial",
        "--output", $outJson
    )

    Write-Log "START env=$EnvName output=$outJson workers=$Workers cpu_cap=$MaxCpuPercent mem_cap=$MaxMemPercent"
    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        throw "run_scalability_experiment failed for env=$EnvName exit=$LASTEXITCODE"
    }
    Write-Log "DONE env=$EnvName output=$outJson"
}

Write-Log "Queue script started"
Write-Log "Waiting for active v50rigor run to finish (if any)..."

while ($true) {
    $running = Get-V50Running
    if ($running.Count -eq 0) {
        break
    }
    $pidList = ($running | ForEach-Object { "$($_.ProcessId)" } | Sort-Object) -join ","
    Write-Log ("Active v50rigor python processes: " + $pidList)
    Start-Sleep -Seconds 60
}

Write-Log "No active v50rigor run detected. Starting queued environments."

Run-Env -EnvName "indoor_office"
Run-Env -EnvName "outdoor_suburban"
Run-Env -EnvName "outdoor_urban"

Write-Log "Queue completed successfully"
