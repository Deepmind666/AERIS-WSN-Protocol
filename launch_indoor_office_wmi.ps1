$wd = "C:\Users\sshuser\AERIS-WSN"
$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$out = "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.json"
$logOut = Join-Path $wd "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.out.log"
$logErr = Join-Path $wd "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.err.log"

$cmd = "$py scripts/run_scalability_experiment.py --env indoor_office --replicates 1000 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --tx-power 10.0 --max-cpu-percent 90 --max-mem-percent 90 --resource-check-sec 1 --mac-collision --multihop-relay --allow-partial --output $out > $logOut 2> $logErr"

$proc = ([wmiclass]"Win32_Process").Create("cmd /c cd /d $wd && $cmd")
Write-Host "PID=$($proc.ProcessId) ReturnValue=$($proc.ReturnValue)"
