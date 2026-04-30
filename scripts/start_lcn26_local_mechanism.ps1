$ErrorActionPreference = 'Stop'

$python = 'C:\Users\admin\anaconda3\python.exe'
$project = 'C:\AERIS-WSN-Protocol'
$wave = 'lcn26_targeted_20260420'
$root = Join-Path $project "results\$wave\mechanism_grid"
$logDir = Join-Path $root 'logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

$envs = @('indoor_office', 'indoor_factory', 'outdoor_urban', 'outdoor_suburban')
$nodes = @(100, 500, 1000)

New-Item -ItemType Directory -Force $root | Out-Null
New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $project

$manifest = @()

foreach ($envName in $envs) {
  foreach ($nodeCount in $nodes) {
    $jobTag = "${envName}_${nodeCount}"
    $jobRoot = Join-Path $root $jobTag
    New-Item -ItemType Directory -Force $jobRoot | Out-Null

    $stdoutLog = Join-Path $logDir "lcn26_local_mechanism_${jobTag}_$stamp.stdout.log"
    $stderrLog = Join-Path $logDir "lcn26_local_mechanism_${jobTag}_$stamp.stderr.log"

    $args = @(
      "$project\scripts\run_lcn26_aeris_mechanism_matrix.py",
      "--envs", $envName,
      "--nodes", "$nodeCount",
      "--replicates", "400",
      "--workers", "1",
      "--batch-size", "1",
      "--max-cpu-percent", "95",
      "--max-mem-percent", "90",
      "--resource-check-sec", "2",
      "--mac-collision",
      "--output-root", $jobRoot
    )

    $proc = Start-Process -FilePath $python -ArgumentList $args `
      -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $stdoutLog `
      -RedirectStandardError $stderrLog

    $manifest += [pscustomobject]@{
      job = $jobTag
      env = $envName
      nodes = $nodeCount
      pid = $proc.Id
      stdout = $stdoutLog
      stderr = $stderrLog
      output_root = $jobRoot
    }

    Write-Host "[STARTED] $jobTag PID=$($proc.Id)"
  }
}

$manifestPath = Join-Path $root "launch_manifest_$stamp.json"
$manifest | ConvertTo-Json -Depth 3 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "[MANIFEST] $manifestPath"
