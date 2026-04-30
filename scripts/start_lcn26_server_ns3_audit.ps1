$ErrorActionPreference = 'Stop'

$remote = 'FatMachine'
$remoteScript = 'C:\Users\sshuser\AERIS-WSN-Protocol\scripts\server_run_lcn26_ns3_audit.ps1'
Write-Host "[REMOTE] powershell -File $remoteScript"
ssh $remote powershell -NoProfile -ExecutionPolicy Bypass -File $remoteScript
