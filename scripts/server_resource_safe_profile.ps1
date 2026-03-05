#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Server resource safe profile for AERIS experiment execution.
    Enforces CPU<=75%, MEM<=80%, auto-throttles workers when limits exceeded.

.DESCRIPTION
    Source this script before launching any experiment batch.
    It exports resource-guard environment variables and provides
    a Watch-Resources function for continuous monitoring.

.NOTES
    Date: 2026-02-15
    Policy: No unattended heavy-load tasks. No n=1000 long runs without explicit approval.
#>

# ── Resource Limits ──────────────────────────────────────────────────
$env:AERIS_MAX_CPU_PERCENT = "75"
$env:AERIS_MAX_MEM_PERCENT = "80"
$env:AERIS_MIN_WORKERS = "1"
$env:AERIS_MAX_WORKERS = "10"
$env:AERIS_POLL_INTERVAL_SEC = "30"
$env:AERIS_GUARD_ENABLED = "1"

# ── Blocked Operations ───────────────────────────────────────────────
# These patterns are checked by the guard before launching experiments
$env:AERIS_BLOCKED_PATTERNS = "n=1000;overnight;unattended;10h"

Write-Host "[ResourceGuard] Loaded safe profile: CPU<=$($env:AERIS_MAX_CPU_PERCENT)%, MEM<=$($env:AERIS_MAX_MEM_PERCENT)%" -ForegroundColor Cyan

function Get-SystemLoad {
    <#
    .SYNOPSIS
        Returns current CPU and memory usage percentages.
    #>
    $cpu = (Get-CimInstance -ClassName Win32_Processor |
            Measure-Object -Property LoadPercentage -Average).Average
    $mem = Get-CimInstance -ClassName Win32_OperatingSystem
    $memUsed = ($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / $mem.TotalVisibleMemorySize * 100
    return @{
        CpuPercent = [math]::Round($cpu, 1)
        MemPercent = [math]::Round($memUsed, 1)
    }
}

function Get-RecommendedWorkers {
    <#
    .SYNOPSIS
        Returns recommended worker count based on current system load.
    #>
    param(
        [int]$RequestedWorkers = [int]$env:AERIS_MAX_WORKERS
    )

    $load = Get-SystemLoad
    $maxCpu = [double]$env:AERIS_MAX_CPU_PERCENT
    $maxMem = [double]$env:AERIS_MAX_MEM_PERCENT
    $minW = [int]$env:AERIS_MIN_WORKERS
    $maxW = [int]$env:AERIS_MAX_WORKERS

    $workers = $RequestedWorkers

    # Throttle down if CPU or MEM exceeds limits
    if ($load.CpuPercent -gt $maxCpu) {
        $ratio = $maxCpu / [math]::Max($load.CpuPercent, 1)
        $workers = [math]::Max($minW, [math]::Floor($workers * $ratio))
        Write-Warning "[ResourceGuard] CPU at $($load.CpuPercent)% > $($maxCpu)% limit. Throttling to $workers workers."
    }

    if ($load.MemPercent -gt $maxMem) {
        $ratio = $maxMem / [math]::Max($load.MemPercent, 1)
        $workers = [math]::Max($minW, [math]::Floor($workers * $ratio))
        Write-Warning "[ResourceGuard] MEM at $($load.MemPercent)% > $($maxMem)% limit. Throttling to $workers workers."
    }

    $workers = [math]::Min($workers, $maxW)
    $workers = [math]::Max($workers, $minW)

    return $workers
}

function Test-BlockedOperation {
    <#
    .SYNOPSIS
        Checks if a command string matches any blocked pattern.
    #>
    param(
        [string]$CommandString
    )

    $blocked = $env:AERIS_BLOCKED_PATTERNS -split ";"
    foreach ($pattern in $blocked) {
        if ($CommandString -match [regex]::Escape($pattern)) {
            Write-Error "[ResourceGuard] BLOCKED: Command matches prohibited pattern '$pattern'. This operation requires explicit human approval."
            return $true
        }
    }
    return $false
}

function Watch-Resources {
    <#
    .SYNOPSIS
        Continuous resource monitor. Prints warnings when limits approached.
        Press Ctrl+C to stop.
    #>
    param(
        [int]$IntervalSec = [int]$env:AERIS_POLL_INTERVAL_SEC
    )

    Write-Host "[ResourceGuard] Monitoring started (interval=${IntervalSec}s, Ctrl+C to stop)" -ForegroundColor Green

    while ($true) {
        $load = Get-SystemLoad
        $cpuColor = if ($load.CpuPercent -gt [double]$env:AERIS_MAX_CPU_PERCENT) { "Red" }
                    elseif ($load.CpuPercent -gt ([double]$env:AERIS_MAX_CPU_PERCENT * 0.85)) { "Yellow" }
                    else { "Green" }
        $memColor = if ($load.MemPercent -gt [double]$env:AERIS_MAX_MEM_PERCENT) { "Red" }
                    elseif ($load.MemPercent -gt ([double]$env:AERIS_MAX_MEM_PERCENT * 0.85)) { "Yellow" }
                    else { "Green" }

        $ts = Get-Date -Format "HH:mm:ss"
        Write-Host "[$ts] CPU: " -NoNewline
        Write-Host "$($load.CpuPercent)%" -ForegroundColor $cpuColor -NoNewline
        Write-Host " | MEM: " -NoNewline
        Write-Host "$($load.MemPercent)%" -ForegroundColor $memColor -NoNewline
        Write-Host " | Workers: $(Get-RecommendedWorkers)"

        Start-Sleep -Seconds $IntervalSec
    }
}

# ── Export functions ─────────────────────────────────────────────────
Export-ModuleMember -Function Get-SystemLoad, Get-RecommendedWorkers, Test-BlockedOperation, Watch-Resources -ErrorAction SilentlyContinue

Write-Host "[ResourceGuard] Functions available: Get-SystemLoad, Get-RecommendedWorkers, Test-BlockedOperation, Watch-Resources" -ForegroundColor Cyan
Write-Host "[ResourceGuard] Blocked patterns: $($env:AERIS_BLOCKED_PATTERNS)" -ForegroundColor Yellow
