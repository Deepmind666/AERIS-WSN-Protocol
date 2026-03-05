$ErrorActionPreference = "Continue"

# S11 control补洞: 4环境 control n=1000, tx=10.0, 无mac-collision/multihop-relay
# 目的: 与S9 patch n=1000形成对称对照

Set-Location "C:\Users\sshuser\AERIS-WSN-Protocol"

$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$SCRIPT = "C:\Users\sshuser\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py"
$OUTDIR = "C:\Users\sshuser\AERIS-WSN-Protocol\results\mega_experiments"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = "$OUTDIR\server_s11_control_$TS.log"

function Log($msg) {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $msg" |
        Out-File -Append -FilePath $LOG -Encoding UTF8
}

Log "S11 CONTROL START ts=$TS"
Log "Config: replicates=1000, nodes=100,200,300,500,800,1000, rounds=300, workers=18, cpu=90, mem=85, tx=10.0"

# ENV-1: indoor_office
Log "ENV-1 start: indoor_office"
& $PYTHON $SCRIPT `
    --env indoor_office --replicates 1000 --seed 42001 `
    --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 `
    --run-tier publication --tx-power 10.0 `
    --max-cpu-percent 90 --max-mem-percent 85 `
    --output "$OUTDIR\scalability_indoor_office_server_s11_control_20260217.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "ENV-1 done: exit=$LASTEXITCODE"

# ENV-2: indoor_factory
Log "ENV-2 start: indoor_factory"
& $PYTHON $SCRIPT `
    --env indoor_factory --replicates 1000 --seed 42001 `
    --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 `
    --run-tier publication --tx-power 10.0 `
    --max-cpu-percent 90 --max-mem-percent 85 `
    --output "$OUTDIR\scalability_indoor_factory_server_s11_control_20260217.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "ENV-2 done: exit=$LASTEXITCODE"

# ENV-3: outdoor_urban
Log "ENV-3 start: outdoor_urban"
& $PYTHON $SCRIPT `
    --env outdoor_urban --replicates 1000 --seed 42001 `
    --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 `
    --run-tier publication --tx-power 10.0 `
    --max-cpu-percent 90 --max-mem-percent 85 `
    --output "$OUTDIR\scalability_outdoor_urban_server_s11_control_20260217.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "ENV-3 done: exit=$LASTEXITCODE"

# ENV-4: outdoor_suburban
Log "ENV-4 start: outdoor_suburban"
& $PYTHON $SCRIPT `
    --env outdoor_suburban --replicates 1000 --seed 42001 `
    --nodes 100,200,300,500,800,1000 --rounds 300 --workers 18 `
    --run-tier publication --tx-power 10.0 `
    --max-cpu-percent 90 --max-mem-percent 85 `
    --output "$OUTDIR\scalability_outdoor_suburban_server_s11_control_20260217.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "ENV-4 done: exit=$LASTEXITCODE"

Log "S11 CONTROL ALL DONE"
Write-Host "DONE. Log: $LOG"
