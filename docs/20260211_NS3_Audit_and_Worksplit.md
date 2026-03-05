# 20260211 NS-3 Audit and Codex-Claude Worksplit

## 1) Audit Scope
Reviewed files:
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_aligned_publication_20260211.json
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_aligned_stats.csv
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_aligned_significance.csv
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_ALIGNMENT_EVIDENCE.md
- C:\AERIS-WSN-Protocol\results\mega_experiments\env_sensitivity_20260207_205317.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\pre_ns3_scalability_summary_20260210_231438.csv

## 2) Findings (severity ordered)

### HIGH-1: NS-3 publication JSON does not meet project metadata schema
Evidence:
- ns3_aligned_publication_20260211.json only has top-level keys `channel_model`, `experiments`.
- Missing required fields: `run_tier`, `primary_metric`, `git_commit`, `git_dirty`, `config`.
Impact:
- Current NS-3 result cannot serve as final publication evidence under project RULES.

### HIGH-2: NS-3 coverage is insufficient for numeric-level validation
Evidence:
- NS3_ALIGNMENT_EVIDENCE.md line 237: "Remaining gap: NS-3 only tests INDOOR_LOS".
- Same file line 307: "Overall Gate: TREND-LEVEL ONLY".
Impact:
- Manuscript can only claim trend-level NS-3 support, not numeric-level cross-platform validation.

### MEDIUM-1: AERIS advantage in NS-3 is not stable across tested scales
Evidence from ns3_aligned_significance.csv:
- line 2: n=50 significant (Holm p=9.08e-12, diff=+1.29%)
- line 3: n=100 not significant (Holm p=0.114)
- line 4: n=200 not significant (Holm p=0.924)
Impact:
- Text must avoid broad "NS-3 confirms superiority" claims.

### MEDIUM-2: CAS is negative in aligned NS-3 good-channel condition
Evidence from ns3_aligned_significance.csv line 5:
- FULL vs noCAS diff=-1.6927%, Holm p=2.77e-34 (significant), noCAS higher.
Impact:
- CAS claims must be explicitly scenario-scoped.

### MEDIUM-3: Python-vs-NS3 LEACH gap is structurally large
Evidence:
- NS3_ALIGNMENT_EVIDENCE.md line 295 states root cause around MAC/implementation differences.
- Python indoor_office (n=30): AERIS 0.973886, LEACH 0.554277 (env_sensitivity file)
- NS-3 indoor_los (n=30): AERIS 0.920240, LEACH 0.917490 (ns3 stats)
Impact:
- Numeric alignment cannot be asserted before implementation harmonization and multi-environment NS-3 runs.

### LOW-1: Existing NS-3 set includes only AERIS/LEACH + AERIS ablations
Evidence:
- ns3_aligned_publication_20260211.json protocols: AERIS, LEACH, AERIS-FULL/noCAS/noFair/noGW.
Impact:
- NS-3 cannot currently support five-protocol parity statements.

## 3) Combined Readiness Assessment
- Python evidence chain: strong for manuscript writing (multi-env n=30 + scalability n=550).
- NS-3 evidence chain: trend-level only, not yet publication gate pass.
- Decision: continue paper drafting now, but keep NS-3 claim strictly scoped until gate is closed.

## 4) Worksplit (next)

### Codex (local) - owner
P0-L1 (ETA 00:30-00:45)
- Build `NS3_CLAIM_SAFE_TEXT_20260211.md` with allowed/forbidden NS-3 statements and direct file evidence.

P0-L2 (ETA 00:45-01:15)
- Patch manuscript wording (Section 1/6/8) to use safe NS-3 language only.
- No algorithm claims beyond current NS-3 support.

P1-L3 (ETA 00:40-01:00)
- Add schema validator for NS-3 publication JSON to enforce project metadata fields.

### Claude (server) - owner
P0-S1 (ETA 00:50-01:30)
- Re-export aligned NS-3 result into project-compliant schema with full metadata and sidecar.
- Output: `ns3_aligned_publication_20260211_fixed.json` + `.provenance.json`.

P1-S2 (ETA 02:00-04:00)
- Extend NS-3 run matrix from INDOOR_LOS to 4 environment mappings (or closest supported mapping with explicit table).
- Run AERIS vs LEACH at n=30 for nodes 100/200.

P1-S3 (ETA 01:30-02:30)
- Produce comparison CSV: Python vs NS-3 per environment/node with abs_diff and trend_match.

## 5) Gate to start large NS-3 batch
Required before large batch starts:
1. fixed schema JSON exists and passes validator
2. 4-env mapping file exists with explicit unsupported cases
3. CPU/MEM limits logged in run header
4. ETA included in every progress update
