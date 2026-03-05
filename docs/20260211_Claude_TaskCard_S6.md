# Claude Task Card S6 (NS-3 Evidence Normalization)

Date: 2026-02-11  
Owner: Claude (server/doc side)  
Constraint: Do not start any new large experiment in this task card.

## Scope

This task card focuses on evidence normalization and cross-check only.

1. Normalize NS-3 evidence packaging for `ns3_scale_ext_1000_20260211`.
2. Verify consistency among:
   - `ns3_scale_ext_1000_significance.csv`
   - `NS3_Section8_附录表.md`
   - `NS3_vs_Python_对照表.md`
3. Prepare manuscript-safe statements (trend-level only) for Section 8.

## Inputs

- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_1000_20260211.json`
- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_1000_stats.csv`
- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_1000_significance.csv`
- `C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_Section8_附录表.md`
- `C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_vs_Python_对照表.md`
- `C:\AERIS-WSN-Protocol\docs\20260207_Claim_Gating_List.md`

## Task S6-A: Manifest + Provenance Unification

Create:

- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_1000_manifest.json`
- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_1000_provenance_audit_20260211.csv`

Requirements:

- Manifest must include:
  - `timestamp`, `experiment_type`, `run_tier`, `primary_metric`
  - `git_commit`, `git_dirty`, `git_diff_stat`
  - `config.seeds`, `config.node_counts`, `config.environments`
  - `counts.total_records` and per `(environment, node_count, protocol)` counts
- Use `primary_metric = "pdr_expected"` (mapping NS-3 `pdr` to this metric name in manifest note).
- Do not overwrite original JSON.

## Task S6-B: Table Consistency Check

Create:

- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_table_consistency_20260211.md`

Check items:

1. `ns3_scale_ext_1000_significance.csv` row count should be 40 (excluding header).
2. Every row cited in `NS3_Section8_附录表.md` Table A1/A2 must map to a CSV row.
3. `NS3_vs_Python_对照表.md` must explicitly state:
   - trend-level validation only
   - no numerical equivalence claim
4. If any mismatch exists, list exact file path + line + expected value + observed value.

## Task S6-C: Section 8 Safe Claim Patch Draft (no direct edit)

Create:

- `C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_section8_safe_claim_patch_20260211.md`

Content:

- Provide a patch-ready text block for Section 8:
  - allowed statements (`CAN_WRITE`)
  - blocked statements (`MUST_NOT_WRITE`)
- Each statement must cite exact source file path and metric row.

## Report Format (must follow)

Return exactly:

1. File list (absolute paths)
2. What was done
3. What still needs verification
4. If any script/check ran longer than 60s, include ETA basis

## Forbidden

- No new NS-3 runs
- No source code changes under `src/`
- No manuscript edits in this task card
