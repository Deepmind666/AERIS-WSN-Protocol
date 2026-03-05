$ErrorActionPreference = "Continue"

# S10 FILL-1 retry: indoor_office tx5 (was killed by taskkill accident)
# This script waits for the main bundle to finish, then re-runs FILL-1,
# then runs postprocess_s10_full4env.py.

Set-Location "C:\Users\sshuser\AERIS-WSN-Protocol"

$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$SCRIPT = "C:\Users\sshuser\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py"
$OUTDIR = "C:\Users\sshuser\AERIS-WSN-Protocol\results\mega_experiments"
$LOG    = "$OUTDIR\server_s10_fill1_retry_20260217.log"

function Log($msg) {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $msg" |
        Out-File -Append -FilePath $LOG -Encoding UTF8
}

Log "FILL-1 RETRY START"

# Step 0: Wait for main bundle to finish (poll every 60s)
$bundleLog = "$OUTDIR\server_s10_fill_2env_90_20260217_004638.log"
Log "Waiting for main bundle to finish..."
while ($true) {
    $content = Get-Content $bundleLog -Raw -ErrorAction SilentlyContinue
    if ($content -match "S10 FILL ALL DONE") {
        Log "Main bundle finished, proceeding with FILL-1 retry"
        break
    }
    Start-Sleep -Seconds 60
}

# Step 1: Re-run FILL-1 indoor_office tx5
Log "FILL-1 retry: indoor_office tx5"
& $PYTHON $SCRIPT `
    --env indoor_office --replicates 600 --seed 192001 `
    --nodes 100,500,1000 --rounds 300 --workers 20 `
    --run-tier publication --tx-power 5.0 `
    --max-cpu-percent 80 --max-mem-percent 78 `
    --mac-collision --multihop-relay `
    --output "$OUTDIR\scalability_indoor_office_server_s10_tx5_fill_20260216.json" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "FILL-1 retry done: exit=$LASTEXITCODE"

# Step 2: Validate all 4 fill JSONs exist
$fillFiles = @(
    "scalability_indoor_office_server_s10_tx5_fill_20260216.json",
    "scalability_indoor_office_server_s10_tx15_fill_20260216.json",
    "scalability_outdoor_suburban_server_s10_tx5_fill_20260216.json",
    "scalability_outdoor_suburban_server_s10_tx15_fill_20260216.json"
)
$allExist = $true
foreach ($f in $fillFiles) {
    $fp = "$OUTDIR\$f"
    if (Test-Path $fp) {
        Log "OK: $f exists"
    } else {
        Log "MISSING: $f"
        $allExist = $false
    }
}

if (-not $allExist) {
    Log "ERROR: Not all fill JSONs present, skipping postprocess"
    exit 1
}

# Step 3: Run postprocess
Log "Running postprocess_s10_full4env.py"
& $PYTHON "C:\Users\sshuser\AERIS-WSN-Protocol\scripts\postprocess_s10_full4env.py" `
    2>&1 | Out-File -Append -FilePath $LOG -Encoding UTF8
Log "Postprocess done: exit=$LASTEXITCODE"

Log "ALL DONE"
