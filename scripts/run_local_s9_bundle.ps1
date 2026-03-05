$ErrorActionPreference = "Stop"

Set-Location "C:\AERIS-WSN-Protocol"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "results/mega_experiments/local_s9_rigor_bundle_${ts}.log"

function Run-Step {
    param(
        [string]$Cmd
    )
    "[RUN] $Cmd" | Tee-Object -FilePath $log -Append
    cmd /c $Cmd 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

"[START] local_s9_rigor_bundle ts=$ts" | Tee-Object -FilePath $log -Append

Run-Step "python scripts/run_scalability_experiment.py --env indoor_factory --replicates 1000 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 85 --mac-collision --multihop-relay --output results/mega_experiments/scalability_indoor_factory_local_s9_${ts}.json"

Run-Step "python scripts/run_scalability_experiment.py --env outdoor_urban --replicates 1000 --seed 52001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 85 --mac-collision --multihop-relay --output results/mega_experiments/scalability_outdoor_urban_local_s9_${ts}.json"

"[DONE] local_s9_rigor_bundle ts=$ts" | Tee-Object -FilePath $log -Append
