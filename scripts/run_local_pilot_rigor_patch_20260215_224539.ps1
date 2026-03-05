$ErrorActionPreference = 'Stop'
Set-Location 'c:\AERIS-WSN-Protocol'
$envs = @('indoor_office','indoor_factory','outdoor_urban','outdoor_suburban')
foreach ($e in $envs) {
  Write-Host ("=== START RIGOR-PATCH PILOT " + $e + " ===")
  $start = Get-Date
  python scripts/run_scalability_experiment.py --env $e --replicates 60 --seed 42001 --nodes 100,500,1000 --rounds 300 --workers 12 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 85 --resource-check-sec 1 --allow-partial --mac-collision --multihop-relay --output ("results/mega_experiments/pilot_rigor_patch_" + $e + "_20260215_224539.json")
  $elapsed = [math]::Round(((Get-Date)-$start).TotalSeconds,1)
  Write-Host ("ELAPSED_SEC_" + $e + "=" + $elapsed)
}
