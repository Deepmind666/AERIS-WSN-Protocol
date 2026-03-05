# S8 Server Experiment Review (Codex)

Date: 2026-02-13

## Input Files
- results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.json
- results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.provenance.json
- results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.json
- results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.provenance.json

## Integrity Check
- File existence: PASS (4/4).
- raw_results: 30000 for each environment, error_runs=0.
- Per-cell sample count: PASS (6 node counts x 5 protocols, each cell n=1000).
- Publication metadata: run_tier=publication, primary_metric=pdr_expected.

## Critical Findings
1. Config state is not frozen: both S8 JSON files are generated under git_dirty=True and include unstaged diff stats.
2. S8 differs strongly from prior S7 in outdoor_suburban baselines at node=1000 (example deltas): LEACH -0.2904, PEGASIS -0.4991, HEED -0.2312, TEEN -0.3229; AERIS delta is small (-0.0004).
3. run_scalability_experiment.py uses profile="energy" for AERIS in scalability runs. This must be disclosed and treated as a regime-specific setting, not directly pooled with 100-node baseline matrix.

## Publication Gate Decision
- Gate status for direct manuscript replacement: HOLD (not approved yet).
- Reason: mixed-regime risk (S7/S8 inconsistency + dirty-state provenance).

## Release Conditions (must all pass)
1. Re-run indoor_office and indoor_factory with the same S8 pipeline (n=1000 per cell) and generate matching sidecar provenance.
2. Use one frozen commit/tag and clean working tree for all four environments, or explicitly archive full patch hash and lock script SHA in all sidecars.
3. Rebuild scalability descriptive/significance tables from the four S8 files only, then regenerate manuscript Figure 3 and Table 1000-node ranking from this single regime.
4. Add one sentence in Experimental Setup clarifying scalability regime and AERIS profile setting (energy profile) to avoid cross-regime misinterpretation.

## Current Usable Conclusion (safe)
- S8 outdoor_urban and outdoor_suburban runs are internally complete and statistically usable as standalone environment-specific evidence.
- They are not yet safe for merged 4-environment headline claims until the above release conditions are satisfied.
