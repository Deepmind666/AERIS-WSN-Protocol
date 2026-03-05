$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\sshuser\AERIS-WSN-Protocol'
$PYTHON = 'C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe'
$SCRIPT = 'C:\Users\sshuser\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py'
$OUTDIR = 'C:\Users\sshuser\AERIS-WSN-Protocol\results\mega_experiments'
$LOG    = "$OUTDIR\server_s10_bundle_20260216.log"

function Log($msg) {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $msg" | Out-File -Append -FilePath $LOG -Encoding UTF8
}

Log 'S10 BUNDLE START'

# S10-1: indoor_factory tx5
Log 'S10-1: indoor_factory tx5 start'
& $PYTHON $SCRIPT --env indoor_factory --replicates 600 --seed 152001 --nodes 100,500,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 5.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_indoor_factory_server_s10_tx5_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S10-1: indoor_factory tx5 done (exit=$LASTEXITCODE)"

# S10-2: indoor_factory tx15
Log 'S10-2: indoor_factory tx15 start'
& $PYTHON $SCRIPT --env indoor_factory --replicates 600 --seed 162001 --nodes 100,500,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 15.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_indoor_factory_server_s10_tx15_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S10-2: indoor_factory tx15 done (exit=$LASTEXITCODE)"

# S10-3: outdoor_urban tx5
Log 'S10-3: outdoor_urban tx5 start'
& $PYTHON $SCRIPT --env outdoor_urban --replicates 600 --seed 172001 --nodes 100,500,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 5.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_outdoor_urban_server_s10_tx5_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S10-3: outdoor_urban tx5 done (exit=$LASTEXITCODE)"

# S10-4: outdoor_urban tx15
Log 'S10-4: outdoor_urban tx15 start'
& $PYTHON $SCRIPT --env outdoor_urban --replicates 600 --seed 182001 --nodes 100,500,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 15.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output "$OUTDIR\scalability_outdoor_urban_server_s10_tx15_20260216.json" 2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "S10-4: outdoor_urban tx15 done (exit=$LASTEXITCODE)"

Log 'S10 BUNDLE ALL DONE'
