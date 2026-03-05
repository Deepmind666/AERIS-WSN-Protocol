param(
    [int]$Replicates = 60,
    [int]$Workers = 14,
    [string]$Nodes = "100,200,300,500,800,1000",
    [int]$Rounds = 300,
    [double]$MaxCpuPercent = 80.0,
    [double]$MaxMemPercent = 80.0
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ProjectRoot = "C:\AERIS-WSN-Protocol"
$Script = Join-Path $ProjectRoot "scripts\run_scalability_experiment.py"
$OutDir = Join-Path $ProjectRoot "results\mega_experiments"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestPath = Join-Path $OutDir ("scalability_fix2_local_2env_" + $stamp + "_manifest.json")

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

# Local host runs two environments; server runs the other two.
$envs = @("indoor_factory", "outdoor_urban")
$records = @()
$startAll = Get-Date

foreach ($envName in $envs) {
    $start = Get-Date
    $outFile = Join-Path $OutDir ("scalability_fix2_" + $envName + "_" + $stamp + ".json")
    $cmd = @(
        "python", $Script,
        "--replicates", $Replicates,
        "--workers", $Workers,
        "--seed", 42001,
        "--nodes", $Nodes,
        "--rounds", $Rounds,
        "--env", $envName,
        "--tx-power", 10.0,
        "--run-tier", "publication",
        "--max-cpu-percent", $MaxCpuPercent,
        "--max-mem-percent", $MaxMemPercent,
        "--resource-check-sec", 2.0,
        "--output", $outFile
    )

    Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] START " + $envName)
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $start).TotalSeconds
    Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] END   " + $envName + " exit=" + $exitCode + " elapsed=" + [int]$elapsed + "s")

    $records += [ordered]@{
        environment = $envName
        output = $outFile
        exit_code = $exitCode
        elapsed_seconds = [int]$elapsed
    }
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    total_elapsed_seconds = [int](((Get-Date) - $startAll).TotalSeconds)
    project_root = $ProjectRoot
    script = $Script
    settings = [ordered]@{
        replicates = $Replicates
        workers = $Workers
        nodes = $Nodes
        rounds = $Rounds
        max_cpu_percent = $MaxCpuPercent
        max_mem_percent = $MaxMemPercent
    }
    environments = $records
}

$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8
Write-Host ("Manifest: " + $manifestPath)
