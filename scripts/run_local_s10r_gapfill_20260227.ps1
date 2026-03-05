$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location (Split-Path -Parent $PSScriptRoot)

# Use the project environment that contains numpy/pandas/matplotlib.
$python = 'C:\Users\admin\anaconda3\envs\aether-wsn\python.exe'
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

$tasks = @(
    @{
        Env = 'indoor_factory'
        Tx = '5'
        Output = 'results\mega_experiments\scalability_indoor_factory_local_s10r_tx5_20260227.json'
    },
    @{
        Env = 'indoor_factory'
        Tx = '10'
        Output = 'results\mega_experiments\scalability_indoor_factory_local_s10r_tx10_20260227.json'
    },
    @{
        Env = 'indoor_factory'
        Tx = '15'
        Output = 'results\mega_experiments\scalability_indoor_factory_local_s10r_tx15_20260227.json'
    },
    @{
        Env = 'outdoor_suburban'
        Tx = '5'
        Output = 'results\mega_experiments\scalability_outdoor_suburban_local_s10r_tx5_rerun_20260227.json'
    }
)

$globalStart = Get-Date
Write-Output ("[{0}] QUEUE_START tasks={1} max_cpu=88 max_mem=92 workers=16" -f ($globalStart.ToString('yyyy-MM-dd HH:mm:ss')), $tasks.Count)

for ($i = 0; $i -lt $tasks.Count; $i++) {
    $t = $tasks[$i]
    $taskIdx = $i + 1

    $etaCurrent = '~110-140min'
    $etaGlobal = "~{0}-{1}min" -f (110 * ($tasks.Count - $i)), (140 * ($tasks.Count - $i))
    Write-Output ("[{0}] ETA task={1}/{2} env={3} tx={4} eta_current={5} eta_global={6} basis=recent_s10r_rate_window" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $taskIdx, $tasks.Count, $t.Env, $t.Tx, $etaCurrent, $etaGlobal)
    Write-Output ("[{0}] START env={1} tx={2} output={3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $t.Env, $t.Tx, $t.Output)

    $args = @(
        'scripts/run_scalability_experiment.py',
        '--env', $t.Env,
        '--replicates', '1000',
        '--nodes', '100,200,300,500,800,1000',
        '--rounds', '300',
        '--workers', '16',
        '--run-tier', 'publication',
        '--tx-power', $t.Tx,
        '--max-cpu-percent', '88',
        '--max-mem-percent', '92',
        '--allow-partial',
        '--mac-collision',
        '--multihop-relay',
        '--seed', '42001',
        '--output', $t.Output
    )

    & $python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Output ("[{0}] FAIL env={1} tx={2} exit={3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $t.Env, $t.Tx, $LASTEXITCODE)
        exit $LASTEXITCODE
    }

    Write-Output ("[{0}] DONE env={1} tx={2} output={3}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $t.Env, $t.Tx, $t.Output)
}

$elapsed = (Get-Date) - $globalStart
Write-Output ("[{0}] QUEUE_DONE elapsed={1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $elapsed)
