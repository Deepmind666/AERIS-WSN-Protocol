# 2026-02-16 S9 Result Synthesis and Draft Update

## 1) Scope of this update

This note summarizes the completed S9 patch/control bundle and the manuscript updates applied in `v22`.

## 2) Data used

- `results/mega_experiments/s9_merged_comparison_20260216.csv`
- `results/mega_experiments/s9_delta_patch_vs_control_20260216.csv`
- `results/mega_experiments/s9_significance_patch_vs_control_20260216.csv`

## 3) Key numerical findings (S9)

### AERIS patch vs control (selected cells)

| Environment | Nodes | Patch | Control | Delta |
|---|---:|---:|---:|---:|
| indoor_office | 100 | 0.9714 | 0.9948 | -0.0235 |
| indoor_office | 500 | 0.8679 | 0.9902 | -0.1223 |
| indoor_office | 1000 | 0.6779 | 0.9899 | -0.3120 |
| outdoor_suburban | 100 | 0.9556 | 0.9744 | -0.0188 |
| outdoor_suburban | 500 | 0.9198 | 0.9859 | -0.0661 |
| outdoor_suburban | 1000 | 0.7280 | 0.9897 | -0.2617 |

### Statistical summary

- Significant cells after Holm correction: `48/60`
- Non-significant cells: `12/60`
- PEGASIS patch/control: `0/12` significant (all non-significant; near-zero delta)

## 4) Interpretation used in v22 draft

1. S8 frozen matrix remains the core reporting matrix for the current draft version.
2. S9 is presented as a matched stress-test block, not as a full replacement matrix.
3. The paper now explicitly states that final camera-ready scalability claims require full rerun in the upgraded simulator regime.
4. Remaining gap: matched patch/control rerun for `indoor_factory` and `outdoor_urban`.

## 5) Manuscript artifacts updated

- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260216_v22.tex`
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260216_v22.pdf`

