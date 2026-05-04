$ErrorActionPreference = 'Stop'

$project = 'C:\AERIS-WSN-Protocol'
$python = 'C:\Users\admin\anaconda3\python.exe'

Set-Location $project

Write-Host '[LCN26] Refreshing Fig. 1 workflow raster from packed source'
Copy-Item 'fig2_fig5_data\fig1_workflow\source\fig1_workflow.png' 'overleaf_upload_ready_20260503\figures\fig1_workflow.png' -Force

Write-Host '[LCN26] Rebuilding Fig. 2 from packed data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_ns3_canonical_margin.py'

Write-Host '[LCN26] Rebuilding Fig. 3 from packed data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_strict_compact.py'

Write-Host '[LCN26] Rebuilding Fig. 4 from packed data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_ns3_ablation_figure.py'

Write-Host '[LCN26] Rebuilding Fig. 5 from packed data'
& $python 'fig2_fig5_data\plot_scripts\build_lcn26_compact_tail_figures.py'

Write-Host '[LCN26] Mirroring regenerated figure PDFs into the Overleaf package'
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_canonical_margin.pdf' 'overleaf_upload_ready_20260503\figures\fig2_classical_margin.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_strict_compact.pdf' 'overleaf_upload_ready_20260503\figures\fig3_stress.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_ns3_ablation_expanded.pdf' 'overleaf_upload_ready_20260503\figures\fig4_ablation.pdf' -Force
Copy-Item '_LCN26_AERIS\generated\fig_lcn26_mechanism_compact.pdf' 'overleaf_upload_ready_20260503\figures\fig5_mechanism.pdf' -Force

Write-Host '[LCN26] Compiling IEEE conference PDF'
Push-Location 'overleaf_upload_ready_20260503'
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
bibtex aeris_lcn2026 | Out-Host
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
pdflatex -interaction=nonstopmode aeris_lcn2026.tex | Out-Host
Pop-Location

Write-Host '[LCN26] Done'
