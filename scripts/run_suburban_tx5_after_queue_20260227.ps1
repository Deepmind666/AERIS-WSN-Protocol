$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$log="logs/s10r_suburban_tx5_afterqueue_20260227.log"
"[{0}] START watcher" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $log -Encoding utf8 -Append
while($true){
  $active=@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_scalability_experiment.py' }).Count
  "[{0}] active_runner={1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$active | Out-File -FilePath $log -Encoding utf8 -Append
  if($active -eq 0){break}
  Start-Sleep -Seconds 120
}
"[{0}] START suburban_tx5 repair ETA~110min" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $log -Encoding utf8 -Append
python scripts/run_scalability_experiment.py --env outdoor_suburban --tx-power 5.0 --replicates 1000 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx5_20260226.json 2>&1 | Tee-Object -FilePath $log -Append
"[{0}] END suburban_tx5 repair exit=$LASTEXITCODE" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $log -Encoding utf8 -Append
