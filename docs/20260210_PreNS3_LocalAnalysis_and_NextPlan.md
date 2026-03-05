# Pre-NS3 Local Scalability Analysis and Next-Step Plan (2026-02-10)

## 1) Scope

This report uses currently available 550-replicate scalability results from local + S1-S4 server outputs, before NS-3 completion.

Input files:
- results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json
- results/mega_experiments/overnight_scalability_20260209_163524/scalability_indoor_factory_20260209_163524.json
- results/mega_experiments/scalability_outdoor_urban_fix550_20260210_102734.json
- results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json

Generated outputs:
- results/mega_experiments/pre_ns3_scalability_summary_20260210_231438.csv
- results/mega_experiments/pre_ns3_scalability_aeris_vs_baselines_20260210_231438.csv
- results/mega_experiments/pre_ns3_scalability_aeris_vs_best_20260210_231438.csv
- results/mega_experiments/pre_ns3_scalability_analysis_20260210_231438.md

## 2) Data quality gate

- All 4 files are publication tier and complete for intended cardinality:
  - raw_results = 16500 per environment
  - error_runs = 0 per environment
- Commit IDs differ across files:
  - indoor_office: bf59e4a8
  - indoor_factory: 8d76e47
  - outdoor_urban: b6b2e5e
  - outdoor_suburban: bf59e4a8
- Diff check 8d76e47..b6b2e5e shows no core protocol changes (rules/docs/extract script only), so combined analysis is acceptable for pre-NS3 manuscript work.

## 3) Core findings (550 replicates, nodes 100-1000)

AERIS rank-1 count:
- 18 / 24 cells overall
- indoor_factory: 6 / 6
- outdoor_urban: 6 / 6
- outdoor_suburban: 6 / 6
- indoor_office: 0 / 6

Key contradiction to avoid in manuscript:
- indoor_office: PEGASIS > AERIS at all tested scales (100-1000), all significant after Holm correction.
- Therefore, any claim implying "universal superiority across all environments/scales" is invalid.

Representative 1000-node comparisons (AERIS vs best baseline):
- indoor_office: 0.9899 vs 0.9992 (PEGASIS), diff = -0.0094, p_holm ~ 0, g = -6.18
- indoor_factory: 0.9900 vs 0.6102 (PEGASIS), diff = +0.3798, p_holm << 0.001, g = +4.60
- outdoor_urban: 0.9899 vs 0.2617 (PEGASIS), diff = +0.7282, p_holm << 0.001, g = +10.29
- outdoor_suburban: 0.9900 vs 0.7871 (PEGASIS), diff = +0.2029, p_holm << 0.001, g = +3.09

Additional note:
- AERIS vs LEACH in indoor_office at some scales is not significant after Holm correction (3 rows). This does not affect the primary contradiction (PEGASIS > AERIS in indoor_office).

## 4) What is publication-safe now

Safe to write now:
1. At 100 nodes (n=30), AERIS leads in all four environments (existing env_sensitivity evidence).
2. At larger scales (100-1000, n=550), AERIS leads in 3/4 environments (indoor_factory, outdoor_urban, outdoor_suburban).
3. At larger scales in indoor_office, PEGASIS is significantly higher than AERIS.
4. Claims must remain environment-scoped and scale-scoped.

Not safe to write now:
1. Any universal claim across all environments and all scales.
2. Any NS-3 numerical-validation-complete claim (NS-3 gate still pending).

## 5) Next-step plan (before NS-3 completion)

### Local (Codex) - execute immediately

P0:
- Update manuscript result wording and tables to match pre-NS3 550 summary.
- Use only generated files above as source of truth.

P1:
- Generate publication-quality scalability figures from pre-NS3 tables:
  - Environment-wise rank heatmap (nodes x environments)
  - AERIS-vs-best delta plot (with confidence intervals)
  - Ensure no overlap, no garbled text, consistent color scheme

P2:
- Add a strict evidence map section in manuscript linking each claim to exact file path.

### Server (Claude) - after NS-3 environment ready

S0:
- Run NS-3 alignment smoke with parameter parity checklist first.
- Output: ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md updated with explicit parameter map.

S1:
- Run NS-3 n=30 core scenarios, export JSON + significance table.
- Gate wording remains trend-level unless all NS-3 gate conditions are met.

## 6) Gate for starting final writing

Can continue writing now (recommended): YES, with pre-NS3 scoped conclusions.
Can finalize submission claims now: NO, wait for NS-3 gate and final cross-platform consistency check.
