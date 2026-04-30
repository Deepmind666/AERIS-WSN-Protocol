$ErrorActionPreference = 'Stop'

$wslScript = '/mnt/c/Users/sshuser/AERIS-WSN-Protocol/scripts/server_build_lcn26_ns3.sh'

Write-Host "[SERVER-BUILD] Rebuilding NS-3 binary in WSL"
wsl -u ns3user -- bash $wslScript
Write-Host "[SERVER-BUILD] Done"
