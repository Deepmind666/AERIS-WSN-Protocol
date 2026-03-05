$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$WORKDIR = "C:\Users\sshuser\AERIS-WSN-Protocol"
$OUTDIR = "$WORKDIR\results\mega_experiments"
$LOGDIR = "$WORKDIR\logs"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = "$LOGDIR\s10r_suburban_repair_${TS}.log"

New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOGDIR | Out-Null
Set-Location $WORKDIR

function Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $line
    Add-Content -Path $LOG -Value $line
}

function Check-Accept([string]$JsonPath) {
    if (-not (Test-Path $JsonPath)) { return $false }
    try {
        $d = Get-Content -Raw -Path $JsonPath | ConvertFrom-Json
        $raw = @($d.raw_results)
        $bad = @($raw | Where-Object { -not $_.success }).Count
        return (
            $raw.Count -eq 30000 -and
            [int]$d.error_runs -eq 0 -and
            $bad -eq 0 -and
            [string]$d.run_tier -eq "publication" -and
            [string]$d.primary_metric -eq "pdr_expected"
        )
    }
    catch {
        return $false
    }
}

function Run-One([double]$Tx, [bool]$ForceRerun) {
    $out = "$OUTDIR\scalability_outdoor_suburban_server_s10r_tx$([int]$Tx)_20260226.json"
    if ((-not $ForceRerun) -and (Check-Accept $out)) {
        Log ("SKIP tx={0} reason=accepted_output_exists file={1}" -f $Tx, $out)
        return
    }
    Log ("START tx={0} output={1}" -f $Tx, $out)
    & $PYTHON scripts/run_scalability_experiment.py `
        --env outdoor_suburban --tx-power $Tx `
        --replicates 1000 --seed 42001 `
        --nodes 100,200,300,500,800,1000 `
        --rounds 300 --workers 20 `
        --run-tier publication `
        --mac-collision --multihop-relay `
        --max-cpu-percent 90 --max-mem-percent 96 `
        --allow-partial `
        --output $out 2>&1 | Tee-Object -FilePath $LOG -Append
    if ($LASTEXITCODE -ne 0) {
        throw "run failed tx=$Tx exit=$LASTEXITCODE"
    }
    if (-not (Check-Accept $out)) {
        throw "acceptance failed tx=$Tx file=$out"
    }
    Log ("DONE tx={0} file={1}" -f $Tx, $out)
}

Log "QUEUE_START suburban_repair tx15_then_tx5"
Run-One -Tx 15.0 -ForceRerun:$false
Run-One -Tx 5.0  -ForceRerun:$true
Log "QUEUE_END suburban_repair completed"
Log ("LOG_PATH={0}" -f $LOG)
