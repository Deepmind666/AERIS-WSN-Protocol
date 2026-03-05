## S8 Final Merge Checklist (Codex -> Claude)

Date: 2026-02-14
Owner split:
- Local paper lead: Codex
- Server/remote run lead: Claude

### Objective
Close S8 scalability as a single publication-tier matrix for four environments at n=1000 per cell (or explicitly audited partial-cell exception), then hand over merge-ready artifacts for v16 manuscript update.

### Mandatory execution order (Claude)
1) Finish indoor_factory run
- Target output:
  - results/mega_experiments/scalability_indoor_factory_server_s8_20260214.json
- Acceptance:
  - raw_results = 30000
  - run_tier = publication
  - primary_metric = pdr_expected
  - error_runs reported and explained

2) Generate missing provenance sidecars
- Required sidecars:
  - results/mega_experiments/scalability_indoor_office_server_s8_20260213.provenance.json
  - results/mega_experiments/scalability_indoor_factory_server_s8_20260214.provenance.json
- Provenance fields must include:
  - git_commit, git_dirty, git_diff_stat
  - script_sha256
  - config_hash
  - platform/python metadata

3) Rebuild unified 4-environment tables
- Run:
  - scripts/rebuild_scalability_from_s8.py
- Required outputs:
  - results/mega_experiments/s8_unified_20260214_descriptive.csv
  - results/mega_experiments/s8_unified_20260214_significance.csv

4) Run integrity gate
- Run:
  - scripts/check_scalability_regime_integrity.py --expected-cell-n 1000
- Required output:
  - explicit PASS/FAIL report
- If one cell is 999 (known indoor_office HEED@800), label as:
  - PASS_WITH_KNOWN_EXCEPTION
  - include exact cell key and impact note

5) Update NS-3 and claim boundary docs only after unified table is final
- Update:
  - ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md
  - ns3_validation/results/NS3_CLAIM_GATE.md
- Constraint:
  - trend-level wording only
  - no numerical-equivalence claims

### Claude response template (must follow)
1. File list (full paths)
2. Completed items
3. Remaining checks (if any)
4. Metadata summary:
   - git_commit
   - git_dirty
   - script_sha256
   - config_hash
5. Key statistical summary:
   - 1000-node ranking by environment
   - significant / non-significant cells (Holm-adjusted)

### Hard constraints
- No core algorithm edits under src/ in this S8 closeout.
- No extra experiments beyond the listed S8 closeout steps.
- Report in Chinese only.
- Every progress update must include ETA.
