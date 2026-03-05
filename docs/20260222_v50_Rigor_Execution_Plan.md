# v50-Rigor Execution Plan

Date: 2026-02-22
Branch: v50-rigor (from sensors-v49-freeze-20260221)
Goal: address fatal reviewer risks (R1 physics realism, R2 baseline fairness) without touching frozen v49 submission assets.

## Completed in this turn

- Created branch `v50-rigor` from tag `sensors-v49-freeze-20260221`.
- Applied fairness fixes in local code:
  - `src/benchmark_protocols.py`: random round-skip removed (`data_transmission_probability=1.0` and unconditional steady-state communication).
  - `src/benchmark_protocols.py`: PEGASIS leader->BS uplink now uses `self.tx_power_dbm` (removed hardcoded +5 dBm path).
  - `src/baseline_protocols/leach_protocol.py`: removed extra closer-than-BS gating in CH assignment; nearest-CH assignment kept.
- Sanity check: `python -m py_compile src/benchmark_protocols.py src/baseline_protocols/leach_protocol.py` passed.

## Next experiments (required)

### Local (Codex) - smoke fairness check (fast)
- Script: `scripts/run_fair_5protocol.py`
- Conditions: 4 environments, nodes=100, seeds=30, run_tier=publication
- Purpose: verify ranking shifts after fairness fixes before long runs.
- ETA: 25-40 min.

### Server (Claude) - full rigor matrix (long)
- Script: `scripts/run_scalability_experiment.py`
- Conditions: 4 environments, nodes=100,200,300,500,800,1000, replicates=1000, run_tier=publication
- Requirements: same tx_power and profile across all protocols; output provenance sidecars for every JSON.
- Purpose: rebuild S8-like matrix under corrected fairness assumptions.
- ETA: 6-10 h (depends on worker stability and memory).

## Acceptance gates

1. No hardcoded protocol-specific tx-power override in baseline wrappers.
2. No probabilistic skip of communication rounds in publication runs.
3. All 120 env-node-protocol cells complete with explicit `n` and `error_runs`.
4. Regenerated significance CSV and manuscript delta statements updated only from new matrix.

## Deliverables

- `*_descriptive.csv` and `*_significance.csv` for corrected matrix
- Updated claim-gate memo with allowed/forbidden statements based on corrected matrix
- Draft delta note: v49 (frozen) vs v50-rigor (fairness-corrected)
