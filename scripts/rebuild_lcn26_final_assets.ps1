$ErrorActionPreference = 'Stop'

$project = 'C:\AERIS-WSN-Protocol'
$python = 'C:\Users\admin\anaconda3\python.exe'
$workflowSource = 'C:\AERIS-WSN-Protocol\_LCN26_AERIS\AERIS流程图.pdf'
$workflowTarget = 'C:\AERIS-WSN-Protocol\_LCN26_AERIS\generated\fig0_aeris_workflow_temp_20260420.pdf'

Set-Location $project

Write-Host '[LCN26] Refreshing workflow figure from _LCN26_AERIS\\AERIS流程图.pdf'
Copy-Item $workflowSource $workflowTarget -Force

Write-Host '[LCN26] Rebuilding base figure set'
& $python 'scripts\build_lcn26_cvstyle_figures.py'

Write-Host '[LCN26] Refreshing corrected canonical NS-3 figure'
& $python 'scripts\build_lcn26_ns3_canonical_refresh.py'

Write-Host '[LCN26] Refreshing compact canonical NS-3 figure'
& $python 'scripts\build_lcn26_canonical_compact.py'

Write-Host '[LCN26] Refreshing compact strict-physics figure'
& $python 'scripts\build_lcn26_strict_compact.py'

Write-Host '[LCN26] Refreshing corrected tradeoff/mechanism figure'
& $python 'scripts\build_lcn26_tradeoff_refresh.py'

Write-Host '[LCN26] Refreshing compact ablation/mechanism tail figures'
& $python 'scripts\build_lcn26_compact_tail_figures.py'

Write-Host '[LCN26] Refreshing final tail composite figure'
& $python 'scripts\build_lcn26_tail_composite.py'

Write-Host '[LCN26] Compiling IEEE conference PDF'
Push-Location '_LCN26_AERIS'
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
bibtex aeris_lcn2026 | Out-Host
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
Pop-Location

Write-Host '[LCN26] Done'
