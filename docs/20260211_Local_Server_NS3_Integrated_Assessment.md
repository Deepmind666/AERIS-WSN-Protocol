# 20260211 Integrated Assessment (Local + Server + NS-3)

## 1) Scope

This note integrates three evidence streams:

1. Python scalability (n=550) from local + server runs
2. NS-3 multi-environment aligned results (n=30)
3. Manuscript claim gate implications for Sensors submission

Primary metric is `pdr_expected`.

## 2) Python Scalability (n=550) Summary

Unified source package:

- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_descriptive.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_significance.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738.md

Key outcomes:

- AERIS ranks first in indoor_factory, outdoor_urban, outdoor_suburban across all tested scales (100-1000 nodes), 18/24 cells.
- In indoor_office, AERIS is not rank-1 at any tested scale; PEGASIS is higher at 100-1000 nodes with Holm-corrected significance.

## 3) NS-3 Multi-Environment Summary (Trend-Level)

Source files:

- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_multienv_publication_v2_20260211.json
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_multienv_stats.csv
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_multienv_significance.csv

AERIS vs LEACH at n=100:

- indoor_office: diff +0.0017, not significant (Holm p=1.0)
- indoor_factory: diff +0.0661, significant
- outdoor_urban: diff +0.0169, significant
- outdoor_suburban: diff +0.0838, significant

Interpretation:

- NS-3 supports trend-level consistency: AERIS advantage is clearer in harsher environments.
- Current NS-3 evidence should not be used as full numeric-level cross-platform equivalence.

## 4) Claim Gate (Can/Cannot)

Can write:

- At 100 nodes (n=30 sensitivity matrix), AERIS ranks first in all four tested environments.
- In scalability matrix (n=550), AERIS ranks first in 3/4 environments and all scales for those three environments.
- In indoor_office scalability, PEGASIS is higher than AERIS at all tested scales.
- NS-3 provides trend-level support, strongest in harsher environments.

Cannot write:

- "AERIS is always rank-1 across all environments and scales."
- "NS-3 fully numerically validates Python results."
- Any absolute latency milliseconds claim without dedicated latency-time evidence.

## 5) Remaining Risks

1. Mixed-commit evidence window for the 4-environment scalability package:
   - server files from commit bf59e4a8
   - local files from commit b6b2e5e
   - `git diff bf59e4a8..b6b2e5e` shows only docs change, but this should still be disclosed.
2. Server provenance schema issue:
   - two server sidecars use short `script_sha256` (16 chars), not full 64 chars.
3. Missing alignment document path used in prior reports:
   - C:\AERIS-WSN-Protocol\ns3_validation\NS3_ALIGNMENT_EVIDENCE.md

## 6) Immediate Actions

1. Keep manuscript scope conservative (already updated in Section 1/6/8).
2. Ask server side to re-export provenance with full 64-char script hash.
3. Add/restore NS-3 alignment evidence document at the canonical path.
4. If final camera-ready requires single-commit freeze, rerun 4 environments under one commit.

