$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "s10r_postqueue_repairs_${ts}.log"

function Log([string]$m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  $line | Tee-Object -FilePath $logPath -Append
}

function Has-ActiveRunner {
  $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_scalability_experiment.py' }
  return (@($procs).Count -gt 0)
}

function Get-Quality([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  try {
    $d = Get-Content -Raw -Path $path | ConvertFrom-Json
    $raw = @($d.raw_results)
    $bad = @($raw | Where-Object { -not $_.success }).Count
    [pscustomobject]@{
      raw_results = $raw.Count
      error_runs = [int]$d.error_runs
      bad_runs = [int]$bad
      run_tier = [string]$d.run_tier
      primary_metric = [string]$d.primary_metric
    }
  } catch { $null }
}

function Accepted([string]$path) {
  $q = Get-Quality $path
  if ($null -eq $q) { return $false }
  return ($q.raw_results -eq 30000 -and $q.error_runs -eq 0 -and $q.bad_runs -eq 0 -and $q.run_tier -eq 'publication' -and $q.primary_metric -eq 'pdr_expected')
}

$target = "results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx5_20260226.json"
$args = @(
  "scripts/run_scalability_experiment.py",
  "--env","outdoor_suburban",
  "--tx-power","5.0",
  "--replicates","1000",
  "--seed","42001",
  "--nodes","100,200,300,500,800,1000",
  "--rounds","300",
  "--workers","18",
  "--run-tier","publication",
  "--mac-collision","--multihop-relay",
  "--max-cpu-percent","90",
  "--max-mem-percent","96",
  "--allow-partial",
  "--output",$target
)

Log "WAIT_START reason=run_scalability_active"
while (Has-ActiveRunner) {
  Start-Sleep -Seconds 60
}
Log "WAIT_END no_active_runner"

if (Accepted $target) {
  Log "SKIP suburban_tx5 reason=already_accepted"
} else {
  Log "START suburban_tx5_repair eta~110min"
  & python @args 2>&1 | Tee-Object -FilePath $logPath -Append
  $exitCode = $LASTEXITCODE
  $ok = Accepted $target
  if ($exitCode -ne 0 -or -not $ok) {
    Log "FAIL suburban_tx5_repair exit=$exitCode accepted=$ok"
    exit 1
  }
  Log "DONE suburban_tx5_repair"
}

$tx15a = "results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx15_20260226.json"
$tx15b = "results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx15_20260227.json"
if ((-not (Test-Path $tx15a)) -and (Test-Path $tx15b)) {
  Copy-Item $tx15b $tx15a -Force
  Log "COPIED tx15 20260227 -> 20260226 for naming consistency"
}

Log "POSTQUEUE_REPAIRS_DONE"
Log ("LOG_PATH={0}" -f $logPath)
