param(
  [string]$Cells = 'outdoor_urban:1000,indoor_factory:1000,outdoor_suburban:1000',
  [int]$Replicates = 400,
  [int]$Workers = 1,
  [int]$BatchSize = 1,
  [int]$MaxParallel = 2,
  [double]$LaunchCpuPercent = 70.0,
  [double]$LaunchMemPercent = 88.0,
  [int]$MinAvailMemMB = 12000,
  [double]$MaxCpuPercent = 90.0,
  [double]$MaxMemPercent = 90.0,
  [double]$ResourceCheckSec = 2.0,
  [string]$Wave = 'lcn26_targeted_20260421_followup'
)

$ErrorActionPreference = 'Stop'

$python = 'C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe'
$project = 'C:\Users\sshuser\AERIS-WSN-Protocol'
$root = Join-Path $project "results\$Wave\mechanism_grid_fat_targeted"
$logDir = Join-Path $root 'logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

function Get-HostSnapshot {
  $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
  $os = Get-CimInstance Win32_OperatingSystem
  $freeMb = [math]::Round($os.FreePhysicalMemory / 1024.0, 1)
  $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1024.0, 1)
  $usedPct = if ($totalMb -gt 0) { 100.0 - ($freeMb / $totalMb * 100.0) } else { 100.0 }
  return [pscustomobject]@{
    cpu = [math]::Round($cpu, 1)
    freeMb = $freeMb
    totalMb = $totalMb
    memPct = [math]::Round($usedPct, 1)
  }
}

function Wait-ForLaunchHeadroom {
  while ($true) {
    $snap = Get-HostSnapshot
    if ($snap.cpu -le $LaunchCpuPercent -and $snap.memPct -le $LaunchMemPercent -and $snap.freeMb -ge $MinAvailMemMB) {
      Write-Host "[FAT-HEADROOM] cpu=$($snap.cpu)% mem=$($snap.memPct)% freeMB=$($snap.freeMb)"
      return
    }
    Write-Host "[FAT-WAIT-HEADROOM] cpu=$($snap.cpu)% mem=$($snap.memPct)% freeMB=$($snap.freeMb) thresholdCpu=$LaunchCpuPercent thresholdMem=$LaunchMemPercent thresholdFreeMB=$MinAvailMemMB"
    Start-Sleep -Seconds 20
  }
}

function Parse-Cells([string]$CellSpec) {
  $items = @()
  foreach ($token in ($CellSpec -split ',')) {
    if ([string]::IsNullOrWhiteSpace($token)) { continue }
    $parts = $token.Trim() -split ':'
    if ($parts.Count -ne 2) {
      throw "Invalid cell token '$token'. Expected env:nodes"
    }
    $items += [pscustomobject]@{
      env = $parts[0].Trim()
      nodes = [int]$parts[1].Trim()
    }
  }
  return $items
}

$cellsParsed = Parse-Cells $Cells
New-Item -ItemType Directory -Force $root | Out-Null
New-Item -ItemType Directory -Force $logDir | Out-Null
Set-Location $project

$running = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
$manifest = @()

Write-Host "[FAT-TARGETED] cells=$Cells replicates=$Replicates maxParallel=$MaxParallel"
Wait-ForLaunchHeadroom

foreach ($cell in $cellsParsed) {
  while (($running | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) {
    $alive = ($running | Where-Object { -not $_.HasExited }).Count
    Write-Host "[FAT-WAIT-SLOTS] alive=$alive maxParallel=$MaxParallel"
    Start-Sleep -Seconds 10
  }

  Wait-ForLaunchHeadroom

  $jobTag = "$($cell.env)_$($cell.nodes)"
  $jobRoot = Join-Path $root $jobTag
  New-Item -ItemType Directory -Force $jobRoot | Out-Null

  $stdoutLog = Join-Path $logDir "lcn26_fat_targeted_${jobTag}_$stamp.stdout.log"
  $stderrLog = Join-Path $logDir "lcn26_fat_targeted_${jobTag}_$stamp.stderr.log"

  $args = @(
    "$project\scripts\run_lcn26_aeris_mechanism_matrix.py",
    "--envs", $cell.env,
    "--nodes", "$($cell.nodes)",
    "--replicates", "$Replicates",
    "--workers", "$Workers",
    "--batch-size", "$BatchSize",
    "--max-cpu-percent", "$MaxCpuPercent",
    "--max-mem-percent", "$MaxMemPercent",
    "--resource-check-sec", "$ResourceCheckSec",
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
    env = $cell.env
    nodes = $cell.nodes
    pid = $proc.Id
    stdout = $stdoutLog
    stderr = $stderrLog
    output_root = $jobRoot
  }
  Write-Host "[FAT-STARTED] $jobTag PID=$($proc.Id)"
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
