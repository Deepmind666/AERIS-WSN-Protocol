#!/usr/bin/env python3
"""
Validate whether scalability JSON files belong to the same publication regime.

This script is a hard gate before merging scalability results into manuscript:
1) each file must pass basic publication constraints;
2) cross-file configuration must match on critical fields;
3) each (node_count, protocol) cell must have a uniform sample size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


CRITICAL_TOP_FIELDS = [
    "run_tier",
    "primary_metric",
    "tx_power_dbm",
    "environment",
]

CRITICAL_CONFIG_FIELDS = [
    "node_counts",
    "round_counts",
    "dropout_rates",
    "protocols",
    "packet_size",
    "initial_energy",
    "area_size",
    "base_station",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_counts(raw_results: List[dict]) -> Dict[Tuple[int, str], int]:
    out: Dict[Tuple[int, str], int] = {}
    for r in raw_results:
        if r.get("error"):
            continue
        key = (int(r["num_nodes"]), str(r["protocol"]))
        out[key] = out.get(key, 0) + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check scalability regime integrity")
    parser.add_argument("--files", nargs="+", required=True, help="Scalability JSON files")
    parser.add_argument("--expected-cell-n", type=int, default=1000, help="Expected samples per (node, protocol) cell")
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    data = [load_json(p) for p in paths]
    ref = data[0]

    failures: List[str] = []
    warnings: List[str] = []

    # Per-file checks
    for p, d in zip(paths, data):
        rr = d.get("raw_results", [])
        if d.get("run_tier") != "publication":
            failures.append(f"{p}: run_tier != publication")
        if d.get("primary_metric") != "pdr_expected":
            failures.append(f"{p}: primary_metric != pdr_expected")
        if int(d.get("error_runs", 0)) != 0:
            failures.append(f"{p}: error_runs != 0")
        if not rr:
            failures.append(f"{p}: raw_results is empty")
            continue

        counts = cell_counts(rr)
        bad_cells = [f"{k[0]}-{k[1]}:{v}" for k, v in counts.items() if v != args.expected_cell_n]
        if bad_cells:
            failures.append(f"{p}: cell sample mismatch ({', '.join(bad_cells[:8])})")
        if str(d.get("git_dirty")).lower() in {"true", "1"}:
            warnings.append(f"{p}: git_dirty=True (usable but not frozen)")

    # Cross-file config consistency checks
    for p, d in zip(paths[1:], data[1:]):
        # Top fields
        for k in CRITICAL_TOP_FIELDS:
            if k == "environment":
                continue
            if d.get(k) != ref.get(k):
                failures.append(f"{p}: top field mismatch {k} ({d.get(k)} != {ref.get(k)})")

        # Config fields
        rcfg = ref.get("config", {})
        cfg = d.get("config", {})
        for k in CRITICAL_CONFIG_FIELDS:
            if cfg.get(k) != rcfg.get(k):
                failures.append(f"{p}: config mismatch {k} ({cfg.get(k)} != {rcfg.get(k)})")


    print("== Scalability Regime Integrity Check ==")
    print(f"files={len(paths)}")
    print(f"expected_cell_n={args.expected_cell_n}")
    print(f"failures={len(failures)} warnings={len(warnings)}")
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        raise SystemExit(2)
    print("[PASS] All files satisfy integrity constraints")


if __name__ == "__main__":
    main()
