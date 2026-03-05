$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$queueScript = Join-Path $PSScriptRoot "run_local_s10r_queue_20260226.ps1"
$logDir = Join-Path $projectRoot "logs"
$launcherLog = Join-Path $logDir "s10r_local_guarded_launcher_20260226.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-LaunchLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $launcherLog -Append
}

function Test-LaunchGuard {
    $samples = @()
    for ($i = 0; $i -lt 6; $i++) {
        $samples += (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        Start-Sleep -Seconds 10
    }

    $avgCpu = (($samples | Measure-Object -Average).Average)
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGb = $os.FreePhysicalMemory / 1MB

    $chipWorkers = @(
        Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "run_chip_worker\.py" }
    )

    $otherHeavyPython = @(
        Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -and
            $_.CommandLine -notmatch "run_scalability_experiment\.py" -and
            $_.CommandLine -match "(train|fine[- ]?tune|llama|pytorch|tensorflow|deepspeed|accelerate)"
        }
    )

    $pass = $true
    if ($avgCpu -ge 45) { $pass = $false }
    if ($freeGb -le 35) { $pass = $false }
    if ($chipWorkers.Count -gt 0) { $pass = $false }
    if ($otherHeavyPython.Count -gt 0) { $pass = $false }

    [pscustomobject]@{
        Pass = $pass
        AvgCpu60s = [math]::Round($avgCpu, 2)
        FreeMemGb = [math]::Round($freeGb, 2)
        ChipWorkers = $chipWorkers.Count
        OtherHeavyPython = $otherHeavyPython.Count
    }
}

Write-LaunchLog "Guarded launcher started. check_interval=30min"

while ($true) {
    $g = Test-LaunchGuard
    Write-LaunchLog (
        "Guard check: pass={0} avg_cpu_60s={1:N2} free_mem_gb={2:N2} chip_workers={3} heavy_py={4}" -f
        $g.Pass, $g.AvgCpu60s, $g.FreeMemGb, $g.ChipWorkers, $g.OtherHeavyPython
    )

    if ($g.Pass) {
        $args = "-NoProfile -ExecutionPolicy Bypass -File `"$queueScript`""
        $proc = Start-Process -FilePath "powershell" -ArgumentList $args -WorkingDirectory $projectRoot -PassThru
        Write-LaunchLog ("Queue launched: pid={0} script={1}" -f $proc.Id, $queueScript)
        break
    }

    Write-LaunchLog "Guard not satisfied. Next check in 30 minutes."
    Start-Sleep -Seconds 1800
}
