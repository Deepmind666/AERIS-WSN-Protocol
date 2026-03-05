$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\sshuser\AERIS-WSN-Protocol'

$PYTHON = 'C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe'
$SCRIPT = 'C:\Users\sshuser\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py'
$OUTDIR = 'C:\Users\sshuser\AERIS-WSN-Protocol\results\mega_experiments'
$LOG    = "$OUTDIR\server_s9_bundle_20260216.log"

function Log($msg) { "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $msg" | Out-File -Append -FilePath $LOG -Encoding UTF8 }

Log 'S9 BUNDLE START'

# S9-A-1: indoor_office PATCH
Log 'S9-A-1: indoor_office PATCH start'
& $PYTHON $SCRIPT --env indoor_office --replicates 1000 --seed 62001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_indoor_office_server_s9_patch_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S9-A-1: indoor_office PATCH done (exit=$LASTEXITCODE)"

# S9-A-2: outdoor_suburban PATCH
Log 'S9-A-2: outdoor_suburban PATCH start'
& $PYTHON $SCRIPT --env outdoor_suburban --replicates 1000 --seed 72001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_outdoor_suburban_server_s9_patch_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S9-A-2: outdoor_suburban PATCH done (exit=$LASTEXITCODE)"

# S9-B-1: indoor_office CONTROL
Log 'S9-B-1: indoor_office CONTROL start'
& $PYTHON $SCRIPT --env indoor_office --replicates 600 --seed 82001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output "$OUTDIR\scalability_indoor_office_server_s9_control_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S9-B-1: indoor_office CONTROL done (exit=$LASTEXITCODE)"

# S9-B-2: outdoor_suburban CONTROL
Log 'S9-B-2: outdoor_suburban CONTROL start'
& $PYTHON $SCRIPT --env outdoor_suburban --replicates 600 --seed 92001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output "$OUTDIR\scalability_outdoor_suburban_server_s9_control_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S9-B-2: outdoor_suburban CONTROL done (exit=$LASTEXITCODE)"

Log 'S9 BUNDLE ALL DONE'
