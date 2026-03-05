$wd = "C:\Users\sshuser\AERIS-WSN"
$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"

$envs = @("indoor_office", "outdoor_suburban", "outdoor_urban")

Set-Location $wd

foreach ($envName in $envs) {
    $out = "results\mega_experiments\scalability_${envName}_v50rigor_20260222.json"
    $logOut = Join-Path $wd "results\mega_experiments\scalability_${envName}_v50rigor_20260222.out.log"
    $logErr = Join-Path $wd "results\mega_experiments\scalability_${envName}_v50rigor_20260222.err.log"

    # 跳过已完成的环境（检查JSON是否存在且非空）
    if (Test-Path $out) {
        $sz = (Get-Item $out).Length
        if ($sz -gt 1000) {
            Write-Host "[SKIP] $envName already done ($sz bytes)"
            continue
        }
    }

    Write-Host "[START] $envName at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    & $py scripts/run_scalability_experiment.py `
        --env $envName `
        --replicates 1000 `
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

    $exit = $LASTEXITCODE
    Write-Host "[DONE] $envName exit=$exit at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Host "[ALL DONE] 3-env queue finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
