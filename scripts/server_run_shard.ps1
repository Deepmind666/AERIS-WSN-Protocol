$ErrorActionPreference = "Continue"
$PYTHON = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$PROJECT = "C:\Users\sshuser\AERIS-WSN-Protocol"
$OUTDIR = "$PROJECT\results\mega_experiments\overnight_scalability_20260209_233631"
$TS = "20260209_233631"

Set-Location $PROJECT

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path "$OUTDIR\run.log" -Value $line
}

Log "Starting server shard experiment"
Log ("git commit: " + (git rev-parse --short=8 HEAD))

$envs = @("indoor_office", "outdoor_suburban")

for ($i = 0; $i -lt $envs.Count; $i++) {
    $env = $envs[$i]
    $outFile = "$OUTDIR\scalability_${env}_${TS}.json"
    $stdoutLog = "$OUTDIR\stdout_${env}.log"
    $stderrLog = "$OUTDIR\stderr_${env}.log"

    Log ("=== ENV {0}/{1}: {2} ===" -f ($i+1), $envs.Count, $env)

    $args = @(
        "$PROJECT\scripts\run_scalability_experiment.py",
        "--nodes", "100,200,300,500,800,1000",
        "--replicates", "550",
        "--workers", "12",
        "--rounds", "300",
        "--env", $env,
        "--max-cpu-percent", "65",
        "--max-mem-percent", "65",
        "--run-tier", "publication",
        "--output", $outFile
    )

    $proc = Start-Process -FilePath $PYTHON -ArgumentList $args `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog

    Log ("{0} exit code: {1}" -f $env, $proc.ExitCode)
}

Log "Server shard experiment finished."
