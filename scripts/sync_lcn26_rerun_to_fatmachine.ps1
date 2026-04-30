$ErrorActionPreference = 'Stop'

$remote = 'FatMachine'
$remoteRoot = 'C:/Users/sshuser/AERIS-WSN-Protocol'

$remoteDirs = @(
  'C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation',
  'C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\results',
  'C:\Users\sshuser\AERIS-WSN-Protocol\scripts',
  'C:\Users\sshuser\AERIS-WSN-Protocol\docs'
)

$mkScript = ($remoteDirs | ForEach-Object { "New-Item -ItemType Directory -Force -Path '$_' | Out-Null" }) -join '; '
$bytes = [System.Text.Encoding]::Unicode.GetBytes($mkScript)
$enc = [Convert]::ToBase64String($bytes)
ssh $remote powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $enc

$files = @(
  'ns3_validation/aeris-validation-standalone.cc',
  'ns3_validation/run_lcn26_focused_matrix.sh',
  'ns3_validation/merge_lcn26_focused_results.py',
  'scripts/server_build_lcn26_ns3.ps1',
  'scripts/server_build_lcn26_ns3.sh',
  'scripts/server_run_lcn26_ns3_audit.ps1',
  'scripts/server_run_lcn26_ns3_expanded.ps1',
  'scripts/server_smoke_lcn26_ns3_expanded.ps1',
  'scripts/start_lcn26_server_ns3_expanded.ps1',
  'scripts/start_lcn26_server_ns3_expanded_smoke.ps1',
  'scripts/server_run_lcn26_mechanism_grid.ps1',
  'scripts/server_run_lcn26_mechanism_targeted.ps1',
  'scripts/run_lcn26_aeris_mechanism_matrix.py',
  'scripts/merge_lcn26_aeris_mechanism_cells.py',
  'scripts/summarize_lcn26_aeris_mechanism.py',
  'docs/20260419_LCN26_dual_machine_execution_card.md'
)

foreach ($rel in $files) {
  $src = Join-Path (Get-Location) $rel
  $dst = "$remote`:$remoteRoot/$rel"
  Write-Host "[SYNC] $rel"
  scp $src $dst
}

Write-Host "[SYNC] done"
