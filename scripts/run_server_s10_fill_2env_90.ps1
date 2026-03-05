$ErrorActionPreference = "Continue"

# S10 fill run: complete missing two environments (indoor_office, outdoor_suburban)
# under tx=5/15 dBm with high CPU target.
#
# Design notes:
# - Keep sequential execution to avoid memory spikes and host reboot risk.
# - Use high cpu cap (90) as requested, while keeping mem cap below cpu cap.
# - Output filenames are explicit to avoid overwriting existing S10 files.

Set-Location "C:\Users\sshuser\AERIS-WSN-Protocol"

$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$SCRIPT = "C:\Users\sshuser\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py"
$OUTDIR = "C:\Users\sshuser\AERIS-WSN-Protocol\results\mega_experiments"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = "$OUTDIR\server_s10_fill_2env_90_$TS.log"

function Log($msg) {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $msg" |
        Out-File -Append -FilePath $LOG -Encoding UTF8
}

Log "S10 FILL START ts=$TS"
Log "Config: replicates=600, nodes=100,500,1000, rounds=300, workers=20, cpu=80, mem=78"

# FILL-1: indoor_office tx5
Log "FILL-1 start: indoor_office tx5"
& $PYTHON $SCRIPT `
    --env indoor_office --replicates 600 --seed 192001 `
    --nodes 100,500,1000 --rounds 300 --workers 20 `
    --run-tier publication --tx-power 5.0 `
    --max-cpu-percent 80 --max-mem-percent 78 `
    --mac-collision --multihop-relay `
    --output "$OUTDIR\scalability_indoor_office_server_s10_tx5_fill_20260216.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "FILL-1 done: exit=$LASTEXITCODE"

# FILL-2: indoor_office tx15
Log "FILL-2 start: indoor_office tx15"
& $PYTHON $SCRIPT `
    --env indoor_office --replicates 600 --seed 202001 `
    --nodes 100,500,1000 --rounds 300 --workers 20 `
    --run-tier publication --tx-power 15.0 `
    --max-cpu-percent 80 --max-mem-percent 78 `
    --mac-collision --multihop-relay `
    --output "$OUTDIR\scalability_indoor_office_server_s10_tx15_fill_20260216.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "FILL-2 done: exit=$LASTEXITCODE"

# FILL-3: outdoor_suburban tx5
Log "FILL-3 start: outdoor_suburban tx5"
& $PYTHON $SCRIPT `
    --env outdoor_suburban --replicates 600 --seed 212001 `
    --nodes 100,500,1000 --rounds 300 --workers 20 `
    --run-tier publication --tx-power 5.0 `
    --max-cpu-percent 80 --max-mem-percent 78 `
    --mac-collision --multihop-relay `
    --output "$OUTDIR\scalability_outdoor_suburban_server_s10_tx5_fill_20260216.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "FILL-3 done: exit=$LASTEXITCODE"

# FILL-4: outdoor_suburban tx15
Log "FILL-4 start: outdoor_suburban tx15"
& $PYTHON $SCRIPT `
    --env outdoor_suburban --replicates 600 --seed 222001 `
    --nodes 100,500,1000 --rounds 300 --workers 20 `
    --run-tier publication --tx-power 15.0 `
    --max-cpu-percent 80 --max-mem-percent 78 `
    --mac-collision --multihop-relay `
    --output "$OUTDIR\scalability_outdoor_suburban_server_s10_tx15_fill_20260216.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "FILL-4 done: exit=$LASTEXITCODE"

Log "S10 FILL ALL DONE"
Write-Host "DONE. Log: $LOG"
