$ErrorActionPreference = 'Stop'
Set-Location 'c:\AERIS-WSN-Protocol'
$envs = @('indoor_office','indoor_factory','outdoor_urban','outdoor_suburban')
foreach ($e in $envs) {
  Write-Host ("=== START PUB PILOT R2 " + $e + " ===")
  $start = Get-Date
  python scripts/run_scalability_experiment.py --env $e --replicates 60 --seed 42001 --nodes 100,500,1000 --rounds 300 --workers 10 --run-tier publication --tx-power 10.0 --max-cpu-percent 80 --max-mem-percent 80 --resource-check-sec 1 --allow-partial --output ("results/mega_experiments/pilot_rigor_pub_r2_" + $e + "_20260215_181356.json")
  $elapsed = [math]::Round(((Get-Date)-$start).TotalSeconds,1)
  Write-Host ("ELAPSED_SEC_" + $e + "=" + $elapsed)
}
