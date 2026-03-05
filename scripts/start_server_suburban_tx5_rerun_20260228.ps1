$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$remoteCmd = @"
Set-Location 'C:\Users\sshuser\AERIS-WSN-Protocol'
& 'C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe' scripts/run_scalability_experiment.py `
  --env outdoor_suburban --tx-power 5 --replicates 1000 --seed 42001 `
  --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 `
  --run-tier publication --mac-collision --multihop-relay `
  --max-cpu-percent 90 --max-mem-percent 96 --allow-partial `
  --output results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx5_rerun_20260227.json
"@

ssh FatMachine powershell -NoProfile -Command $remoteCmd
