$wd = "C:\Users\sshuser\AERIS-WSN"
$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$out = "results\mega_experiments\scalability_indoor_factory_v50rigor_20260222_server.json"
$logOut = Join-Path $wd "results\mega_experiments\scalability_indoor_factory_v50rigor_20260222_server.out.log"
$logErr = Join-Path $wd "results\mega_experiments\scalability_indoor_factory_v50rigor_20260222_server.err.log"

Set-Location $wd

Write-Host "[START] indoor_factory replicates=3200 at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

& $py scripts/run_scalability_experiment.py `
    --env indoor_factory `
    --replicates 3200 `
    --seed 42001 `
    --nodes 100,200,300,500,800,1000 `
    --rounds 300 `
    --workers 20 `
    --run-tier publication `
    --tx-power 10.0 `
    --max-cpu-percent 90 `
    --max-mem-percent 90 `
    --resource-check-sec 1 `
    --mac-collision `
    --multihop-relay `
    --allow-partial `
    --output $out `
    > $logOut 2> $logErr

Write-Host "[DONE] indoor_factory exit=$LASTEXITCODE at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
