# v32 Strict Review Report (2026-02-19)

> Reviewer: Claude 4.6 (Opus)
> Reviewed file: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.tex`
> Baseline: v31 review report (`docs/20260219_v31_Strict_Review_Report.md`)

---

## Section 0: Forced Checks (v31 → v32 Fix Verification)

### Fix 1: NS3_CLAIM_GATE.md line 16 (26→25)
- **Status: FIXED**
- `ns3_validation/results/NS3_CLAIM_GATE.md` line 16 now reads "25个统计显著"
- Consistent with line 18 "25/28" — no internal contradiction remains

### Fix 2: Hedges' g methodological footnote
- **Status: FIXED**
- v32 line 219 contains footnote: "For large-n comparisons with very small within-group variance, standardized effect sizes can become extremely large..."
- Correctly advises joint interpretation with ΔPDR and CI

### Fix 3: S9→S11 bridging sentence
- **Status: FIXED**
- v32 line 276 ends with: "...we then run S11 as a matched confirmation block (patch n=1000, control n=1000 per cell) and report that matrix separately below."
- Clear motivation for why S11 exists after S9

**All 3 conditional fixes from v31 review: CLOSED.**

---

## Section A: Data Cross-Validation

### S9 Table (tab:s9_patch_control, lines 254–274)
- 24 AERIS values (patch + control, 4 env × 3 node counts × 2 arms)
- Cross-validated against `results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_merged.csv`
- **Result: 24/24 exact match (diff=0.0000)**

### S11 Table (tab:s11_aeris_delta, lines 302–316)
- 8 AERIS delta values (4 env × 2 node counts)
- Cross-validated against `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv`
- **Result: 8/8 exact match (diff=0.0000)**

### NS3 CLAIM_GATE Consistency
- Line 16: "25个统计显著" ✓
- Line 18: "25/28" ✓
- Tex line 338: "25/28 comparisons are significant" ✓
- Tex line 356: "25/28 AERIS-versus-LEACH comparisons are significant" ✓
- **Result: All four references consistent**

### Bibliography
- 17 cite keys used in tex; all 17 present in `bibliography.bib`
- 0 missing keys
- 73 unused entries (acceptable — bib is a shared pool)

---

## Section B: v31→v32 Diff Analysis (Defensive Phrase Compression)

v32 makes 10 targeted edits to reduce defensive repetition. Key changes:

| v31 wording | v32 replacement | Line |
|---|---|---|
| "does not claim universal protocol superiority, five-protocol cross-platform ranking, or cross-platform numerical equivalence" | "focuses on reproducible conclusions under explicit settings...and does not claim cross-platform numerical equivalence" | 49 |
| "Claims are constrained to this evidence scope. Cross-platform numerical equivalence is intentionally not claimed" | "Cross-regime numerical pooling is avoided." | 92 |
| "bounded to the tested environment taxonomy, protocol set, and simulator assumptions" | "under the reported simulator assumptions" | 196 |
| "scoped positioning rather than universal superiority...bounded to the tested baselines" | "environment-scoped positioning rather than universal superiority" | 359 |
| "intentionally separates" | "separates" | 361 |
| "intentionally avoided" | "avoided" | 367 |
| "bounded to the tested channel taxonomy...intentionally not used for" | "under the tested channel taxonomy...trend-level only" | 399 |
| "intentionally not claimed because of platform-depth differences" | "outside the current validation scope" | 404 |
| "bounded to the tested channel taxonomy and original simulator assumptions" | "should be interpreted under the original simulator assumptions" | 406 |
| "bounded to the tested baseline set" | "limited to the tested baseline set" | 411 |

**Phrase count comparison (v31 → v32):**
- "bounded to": 5 → 0
- "intentionally not": 3 → 0
- "treated as": 9 → 9 (unchanged, acceptable — these are regime labels)
- "rather than": 7 → 7 (unchanged, acceptable — these are scope qualifiers)
- "does not claim": 3 → 2
- Total defensive-hedging phrases reduced from ~11 to ~2 explicit hedges

**Assessment: P1 (defensive repetition) is RESOLVED.** The text now reads more naturally without losing scientific caution.

---

## Section C: Remaining Findings

### P1 Issues (from v31 review)

| # | v31 Issue | v32 Status |
|---|---|---|
| P1-1 | CLAIM_GATE line 16: 26→25 | CLOSED (Fix 1) |
| P1-2 | Defensive repetition (~10 occurrences) | CLOSED (Section B above) |
| P1-3 | S9 table only shows AERIS but text discusses PEGASIS | OPEN — see below |
| P1-4 | Hedges' g footnote missing | CLOSED (Fix 2) |
| P1-5 | S9→S11 bridging sentence missing | CLOSED (Fix 3) |

### P1-3 Detail (OPEN)
- v32 line 276 states: "PEGASIS is minimally affected by the patch, with near-zero deltas and non-significant tests in all 24 PEGASIS cells"
- But tab:s9_patch_control (lines 254–274) only shows AERIS values
- A reviewer could ask: "Where is the PEGASIS evidence for this claim?"
- **Minimal fix**: Add a one-line footnote referencing the CSV file, or add a compact PEGASIS row to the S9 table

### New P2 Issues (v32-specific)

**P2-1: Abstract word count (172 words) — unchanged from v31**
- Within the 200-word MDPI target. No action needed.

**P2-2: Chen2023Survey bib entry mislabeled**
- `bibliography.bib` line 427: cite key is `Chen2023Survey` but the actual paper is Praveen Kumar et al. 2019 (Information Fusion, vol 49). The year field says 2019, not 2023.
- Risk: A reviewer checking references may flag the date inconsistency.
- **Minimal fix**: Rename key to match actual publication year, or update the entry to a genuine 2023 survey.

**P2-3: Conclusion is dense (single paragraph, ~120 words)**
- Not a blocker for Sensors format, but readability could improve with one line break after the S8 sentence.
- **Minimal fix**: Optional paragraph break at line 417 after "...three environments."

---

## Section D: Four-Role Assessment

### R1 (Methodology)
- Evidence regime separation (S8/S9/S10/S11/NS3) is clearly documented in tab:regime_map
- No cross-regime pooling
- Metric definition explicit (pdr_expected)
- **Verdict: PASS**

### R2 (Statistics)
- Welch t-test + Holm correction consistently applied
- Hedges' g footnote now present (v32 line 219)
- Large effect sizes properly contextualized
- **Verdict: PASS**

### R3 (Reproducibility)
- Seed lists, sample sizes, and force_ctp_reliable=False documented
- Data Availability Statement present (line 422–423)
- Provenance sidecar records mentioned
- **Verdict: PASS**

### R4 (Application)
- Deployment guidance table (tab:deployment_summary) is environment-scoped
- Validity notes section explicitly lists three boundaries
- **Verdict: PASS**

---

## Section E: Gate Decision

### v31 → v32 Progress
- v31: 0 P0, 5 P1, 3 P2 → Minor Revision
- v32: 0 P0, 1 P1 (open), 2 P2 (new minor) → **Conditional Accept**

### Open Items for v33
1. **P1-3**: Add PEGASIS evidence reference (footnote or table row) to support the claim at line 276
2. **P2-2**: Fix `Chen2023Survey` bib key/year mismatch (optional but recommended)
3. **P2-3**: Optional paragraph break in Conclusion (cosmetic)

### Overall Verdict: **Conditional Accept**
v32 can be submitted to Sensors after fixing P1-3 (one footnote). The remaining P2 items are cosmetic and can be addressed in camera-ready.

### Cross-Validation Summary
| Check | Result |
|---|---|
| S9 table vs CSV | 24/24 OK |
| S11 table vs CSV | 8/8 OK |
| NS3 25/28 consistency | 4/4 references aligned |
| Bibliography completeness | 17/17 OK |
| Gate script | PASS (78/0/2) |
| Defensive phrase compression | Resolved |
| Abstract word count | 172 (≤200) |

---

*Report generated by Claude 4.6 (Opus), 2026-02-19.*
