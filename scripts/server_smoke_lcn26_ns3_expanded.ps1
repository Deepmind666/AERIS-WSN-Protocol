$ErrorActionPreference = 'Stop'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$project = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol"
$outDir = "$project/ns3_validation/results/lcn26_ns3_expanded_smoke_$stamp"
$outDirWin = "C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\results\lcn26_ns3_expanded_smoke_$stamp"
$bin = "/home/ns3user/ns-allinone-3.40/ns-3.40/build/scratch/ns3.40-aeris-validation-standalone-default"
$lib = "/home/ns3user/ns-allinone-3.40/ns-3.40/build/lib"
$merge = "$project/ns3_validation/merge_lcn26_focused_results.py"

$bashCmd = @'
set -euo pipefail
mkdir -p '__OUTDIR__/raw' '__OUTDIR__/logs' '__OUTDIR__/summary'
export LD_LIBRARY_PATH='__LIB__':${LD_LIBRARY_PATH:-}
for proto in AERIS RPL-MRHOF CTP; do
  echo "[SMOKE] $proto"
  '__BIN__' --runShard --protocol="$proto" --env=indoor_factory --nodes=100 --numRounds=30 --output="__OUTDIR__/raw/shard_${proto}_indoor_factory.json" > "__OUTDIR__/logs/${proto}_indoor_factory.log" 2>&1
done
python3 '__MERGE__' --input-dir '__OUTDIR__/raw' --output-dir '__OUTDIR__/summary'
echo "[OUTDIR] __OUTDIR__"
'@

$bashCmd = $bashCmd.Replace('__OUTDIR__', $outDir).Replace('__LIB__', $lib).Replace('__BIN__', $bin).Replace('__MERGE__', $merge)

New-Item -ItemType Directory -Force $outDirWin | Out-Null
$runScriptWin = Join-Path $outDirWin 'run_smoke.sh'
$runScriptWsl = "$outDir/run_smoke.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runScriptWin, $bashCmd, $utf8NoBom)

Write-Host "[SERVER-SMOKE] expanded NS-3 smoke"
Write-Host "[SERVER-SMOKE] script: $runScriptWsl"
wsl -u ns3user -- bash $runScriptWsl
