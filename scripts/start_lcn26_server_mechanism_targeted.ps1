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

$remote = 'FatMachine'
$remoteScript = 'C:\Users\sshuser\AERIS-WSN-Protocol\scripts\server_run_lcn26_mechanism_targeted.ps1'

$argList = @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', $remoteScript,
  '-Cells', $Cells,
  '-Replicates', "$Replicates",
  '-Workers', "$Workers",
  '-BatchSize', "$BatchSize",
  '-MaxParallel', "$MaxParallel",
  '-LaunchCpuPercent', "$LaunchCpuPercent",
  '-LaunchMemPercent', "$LaunchMemPercent",
  '-MinAvailMemMB', "$MinAvailMemMB",
  '-MaxCpuPercent', "$MaxCpuPercent",
  '-MaxMemPercent', "$MaxMemPercent",
  '-ResourceCheckSec', "$ResourceCheckSec",
  '-Wave', $Wave
)

Write-Host "[REMOTE] powershell $($argList -join ' ')"
ssh $remote powershell @argList
