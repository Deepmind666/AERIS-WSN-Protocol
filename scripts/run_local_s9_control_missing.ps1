$ErrorActionPreference = "Stop"

Set-Location "C:\AERIS-WSN-Protocol"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "results/mega_experiments/local_s9_control_missing_${ts}.log"

function Run-Step {
    param([string]$Cmd)
    "[RUN] $Cmd" | Tee-Object -FilePath $log -Append
    cmd /c $Cmd 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

"[START] local_s9_control_missing ts=$ts" | Tee-Object -FilePath $log -Append

Run-Step "python scripts/run_scalability_experiment.py --env indoor_factory --replicates 600 --seed 132001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 16 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 85 --output results/mega_experiments/scalability_indoor_factory_local_s9_control_${ts}.json"

Run-Step "python scripts/run_scalability_experiment.py --env outdoor_urban --replicates 600 --seed 142001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 16 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 85 --output results/mega_experiments/scalability_outdoor_urban_local_s9_control_${ts}.json"

"[DONE] local_s9_control_missing ts=$ts" | Tee-Object -FilePath $log -Append
