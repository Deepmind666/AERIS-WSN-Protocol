$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\admin\anaconda3\python.exe'

Set-Location $repoRoot

Write-Host '[AERIS] Rebuilding Fig. 2 from packed canonical NS-3 data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_ns3_canonical_margin.py'

Write-Host '[AERIS] Rebuilding Fig. 3 from packed strict-physics data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_strict_compact.py'

Write-Host '[AERIS] Rebuilding Fig. 4 from packed ablation data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_ns3_ablation_figure.py'

Write-Host '[AERIS] Rebuilding Fig. 5 from packed mechanism data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_compact_tail_figures.py'

Write-Host '[AERIS] Mirroring generated figures into Overleaf package and data-pack exports'
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.pdf' 'LCN26_AERIS_overleaf\figures\fig2_classical_margin.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.pdf' 'LCN26_AERIS_overleaf\figures\fig3_stress.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.pdf' 'LCN26_AERIS_overleaf\figures\fig4_ablation.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.pdf' 'LCN26_AERIS_overleaf\figures\fig5_mechanism.pdf' -Force

Copy-Item 'LCN26_AERIS_overleaf\figures\fig1_workflow.png' 'fig2_fig5_data\exported_figures\fig1_workflow.png' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.pdf' 'fig2_fig5_data\exported_figures\fig2_classical_margin.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.png' 'fig2_fig5_data\exported_figures\fig2_classical_margin.png' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.pdf' 'fig2_fig5_data\exported_figures\fig3_stress.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.png' 'fig2_fig5_data\exported_figures\fig3_stress.png' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.pdf' 'fig2_fig5_data\exported_figures\fig4_ablation.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.png' 'fig2_fig5_data\exported_figures\fig4_ablation.png' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.pdf' 'fig2_fig5_data\exported_figures\fig5_mechanism.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.png' 'fig2_fig5_data\exported_figures\fig5_mechanism.png' -Force

Write-Host '[AERIS] Done'
