$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dataDir = $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

Set-Location $repoRoot

Write-Host '[AERIS] Rebuilding Fig. 2 from packed canonical NS-3 data'
& $python (Join-Path $dataDir 'scripts\build_lcn26_ns3_canonical_margin.py')

Write-Host '[AERIS] Rebuilding Fig. 3 from packed strict-physics data'
& $python (Join-Path $dataDir 'scripts\build_lcn26_strict_compact.py')

Write-Host '[AERIS] Rebuilding Fig. 4 from packed ablation data'
& $python (Join-Path $dataDir 'scripts\build_lcn26_ns3_ablation_figure.py')

Write-Host '[AERIS] Rebuilding Fig. 5 from packed mechanism data'
& $python (Join-Path $dataDir 'scripts\build_lcn26_compact_tail_figures.py')

Write-Host '[AERIS] Mirroring generated figures into Overleaf package and data-pack exports'
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.pdf' 'paper\LCN26_AERIS_overleaf\figures\fig2_classical_margin.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.pdf' 'paper\LCN26_AERIS_overleaf\figures\fig3_stress.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.pdf' 'paper\LCN26_AERIS_overleaf\figures\fig4_ablation.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.pdf' 'paper\LCN26_AERIS_overleaf\figures\fig5_mechanism.pdf' -Force

Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.pdf' (Join-Path $dataDir '00_final_outputs\fig2_classical_margin.pdf') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.png' (Join-Path $dataDir '00_final_outputs\fig2_classical_margin.png') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.pdf' (Join-Path $dataDir '00_final_outputs\fig3_stress.pdf') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.png' (Join-Path $dataDir '00_final_outputs\fig3_stress.png') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.pdf' (Join-Path $dataDir '00_final_outputs\fig4_ablation.pdf') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.png' (Join-Path $dataDir '00_final_outputs\fig4_ablation.png') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.pdf' (Join-Path $dataDir '00_final_outputs\fig5_mechanism.pdf') -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.png' (Join-Path $dataDir '00_final_outputs\fig5_mechanism.png') -Force

Write-Host '[AERIS] Done'
