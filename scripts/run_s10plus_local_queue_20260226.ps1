$ErrorActionPreference = 'Stop'
$python = 'C:\Users\admin\anaconda3\envs\aether-wsn\python.exe'
$script = 'scripts/run_scalability_experiment.py'
$base = 'results/mega_experiments'
$jobs = @(
    @{ env='indoor_office'; tx='5.0'; out='scalability_indoor_office_local_s10plus_tx5_20260226.json' },
    @{ env='indoor_office'; tx='15.0'; out='scalability_indoor_office_local_s10plus_tx15_20260226.json' },
    @{ env='outdoor_suburban'; tx='5.0'; out='scalability_outdoor_suburban_local_s10plus_tx5_20260226.json' },
    @{ env='outdoor_suburban'; tx='15.0'; out='scalability_outdoor_suburban_local_s10plus_tx15_20260226.json' }
)

foreach ($j in $jobs) {
    $outPath = Join-Path $base $j.out
    if (Test-Path $outPath) {
        Write-Host "[SKIP] exists: $outPath"
        continue
    }
    Write-Host "[RUN] env=$($j.env) tx=$($j.tx) -> $outPath"
    & $python $script `
        --env $j.env `
        --tx-power $j.tx `
        --replicates 1000 `
        --nodes 100,200,300,500,800,1000 `
        --rounds 300 `
        --workers 18 `
        --run-tier publication `
        --mac-collision `
        --multihop-relay `
        --max-cpu-percent 88 `
        --max-mem-percent 96 `
        --allow-partial `
        --output $outPath
    if ($LASTEXITCODE -ne 0) {
        throw "run failed env=$($j.env) tx=$($j.tx), exit=$LASTEXITCODE"
    }
}
Write-Host '[DONE] Local S10+ queue completed.'
