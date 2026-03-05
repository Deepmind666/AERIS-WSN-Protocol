# 20260223 v61 Strict Review Response (Sensors)

## Scope
Response to the latest third-party/Claude strict review comments. This note records which comments are accepted, which are outdated for `v61`, and what was changed.

## Accepted and Fixed (this round)

1. **Potentially non-resolvable/fabricated DOI risk** (accepted)
   - Action: removed non-resolving cited entries and replaced in-text citations with Crossref-resolving alternatives.
   - Updated file: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260223_v61.tex`
     - Related Work citation set updated at line 54.
     - Discussion citation set updated at line 509.
   - Updated file: `for_submission/bibliography.bib`
     - Removed entries: `Wang2024EECR`, `Zhang2024DRL`, `Liu2024LSTM`, `Sharma2024QoS`, `Li2023Reinforcement`, `Lin2024CoopARQ`, duplicate `GreyWolf2024Sensors`.
   - Evidence: `docs/20260223_v61_cited_doi_validation.csv` (30/30 cited entries resolve on Crossref API; bad=0).

2. **Regime boundary wording for Table 1** (accepted)
   - Action: tightened caption wording to explicitly mark legacy-comparability regime and replicate-level origin of SD.
   - Updated file: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260223_v61.tex:200`.

3. **Recent-reference ratio concern** (accepted)
   - Action: kept cited set at 30 keys with 18 keys from 2022+ (exactly 60%).
   - Evidence computed from current `v61` cites + `bibliography.bib` years.

## Reviewed but Not Adopted as Blocking (current v61)

1. **"Table 1 vs v50-rigor mismatch" as hard error**
   - Current status: not a hidden mismatch in `v61`; it is explicitly declared as a separate legacy regime in caption + text (`v61`: lines 200 and 219).
   - Decision: retain with explicit boundary language (already present).

2. **"Near-5-year references only 37.5%"**
   - Current status: outdated for `v61`; current cited set is 18/30 = 60% (2022+).
   - Decision: no extra forced citation inflation this round.

## Build Check
- Recompiled successfully:
  - `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260223_v61.pdf`
- No unresolved citation keys in final build state.

## Remaining Open Risks (not edited this round)
1. PEGASIS zero-delta anomaly remains an interpretation risk; still needs code-path audit in methods/supplement.
2. Figure workflow quality remains pending user’s manual redraw (icon/text legibility addressed separately).

