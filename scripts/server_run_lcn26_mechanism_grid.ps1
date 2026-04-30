$ErrorActionPreference = 'Stop'

$python = 'C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe'
$project = 'C:\Users\sshuser\AERIS-WSN-Protocol'
$wave = 'lcn26_targeted_20260420'
$root = Join-Path $project "results\$wave\mechanism_grid_fat"
$logDir = Join-Path $root 'logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$maxParallel = 8

$envs = @('indoor_office', 'indoor_factory', 'outdoor_urban', 'outdoor_suburban')
$nodes = @(100, 500, 1000)

New-Item -ItemType Directory -Force $root | Out-Null
New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $project

$manifest = @()
$running = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

foreach ($envName in $envs) {
  foreach ($nodeCount in $nodes) {
    while (($running | Where-Object { -not $_.HasExited }).Count -ge $maxParallel) {
      Start-Sleep -Seconds 5
    }

    $jobTag = "${envName}_${nodeCount}"
    $jobRoot = Join-Path $root $jobTag
    New-Item -ItemType Directory -Force $jobRoot | Out-Null

    $stdoutLog = Join-Path $logDir "lcn26_fat_mechanism_${jobTag}_$stamp.stdout.log"
    $stderrLog = Join-Path $logDir "lcn26_fat_mechanism_${jobTag}_$stamp.stderr.log"

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
      -PassThru `
      -RedirectStandardOutput $stdoutLog `
      -RedirectStandardError $stderrLog

    $running.Add($proc)
    $manifest += [pscustomobject]@{
      job = $jobTag
      env = $envName
      nodes = $nodeCount
      pid = $proc.Id
      stdout = $stdoutLog
      stderr = $stderrLog
      output_root = $jobRoot
    }
    Write-Host "[FAT-STARTED] $jobTag PID=$($proc.Id)"
  }
}

while (($running | Where-Object { -not $_.HasExited }).Count -gt 0) {
  $alive = ($running | Where-Object { -not $_.HasExited }).Count
  Write-Host "[FAT-WAIT] alive=$alive"
  Start-Sleep -Seconds 10
}

$manifestPath = Join-Path $root "launch_manifest_$stamp.json"
$manifest | ConvertTo-Json -Depth 3 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host "[FAT-MANIFEST] $manifestPath"

$mergeOut = Join-Path $root "merged_$stamp"
& $python "$project\scripts\merge_lcn26_aeris_mechanism_cells.py" --input-root $root --output-dir $mergeOut
& $python "$project\scripts\summarize_lcn26_aeris_mechanism.py" --input (Join-Path $mergeOut 'mechanism_raw_merged.json')
Write-Host "[FAT-DONE] merged outputs in $mergeOut"
