#!/usr/bin/env python3
"""
Analyze local pilot-rigor scalability runs and export concise summary artifacts.

Inputs:
  - pilot_rigor_pub_*_20260215_local.json

Outputs:
  - pilot_rigor_pub_20260215_descriptive.csv
  - pilot_rigor_pub_20260215_significance.csv
  - pilot_rigor_pub_20260215_summary.md

This script keeps analysis deterministic and audit-friendly for handoff.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
NODE_ORDER = [100, 500, 1000]
PROTO_ORDER = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]


def load_rows() -> List[dict]:
    rows: List[dict] = []
    for env in ENV_ORDER:
        fp = RESULTS_DIR / f"pilot_rigor_pub_{env}_20260215_local.json"
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data["raw_results"]:
            if r.get("error"):
                continue
            m = r.get("metrics", {})
            rows.append(
                {
                    "environment": env,
                    "num_nodes": int(r["num_nodes"]),
                    "protocol": r["protocol"],
                    "pdr_expected": float(m.get("pdr_expected", r.get("pdr_expected", 0.0))),
                    "total_energy_consumed": float(
                        m.get("total_energy_consumed", r.get("total_energy_consumed", 0.0))
                    ),
                    "total_rounds": int(m.get("total_rounds", r.get("total_rounds", 0))),
                }
            )
    return rows


def aggregate(rows: List[dict]) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    out: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            for proto in PROTO_ORDER:
                vals = [
                    r["pdr_expected"]
                    for r in rows
                    if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == proto
                ]
                arr = np.asarray(vals, dtype=float)
                std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                ci95 = 1.96 * std / math.sqrt(max(len(arr), 1))
                out[(env, node, proto)] = {
                    "n": len(arr),
                    "pdr_mean": float(arr.mean()),
                    "pdr_std": std,
                    "ci95_half_width": ci95,
                }
    return out


def export_descriptive(agg: Dict[Tuple[str, int, str], Dict[str, float]]) -> Path:
    out = RESULTS_DIR / "pilot_rigor_pub_20260215_descriptive.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["environment", "num_nodes", "protocol", "n", "pdr_mean", "pdr_std", "ci95_half_width"])
        for env in ENV_ORDER:
            for node in NODE_ORDER:
                for proto in PROTO_ORDER:
                    x = agg[(env, node, proto)]
                    w.writerow(
                        [
                            env,
                            node,
                            proto,
                            x["n"],
                            f"{x['pdr_mean']:.6f}",
                            f"{x['pdr_std']:.6f}",
                            f"{x['ci95_half_width']:.6f}",
                        ]
                    )
    return out


def export_significance(rows: List[dict]) -> Path:
    out = RESULTS_DIR / "pilot_rigor_pub_20260215_significance.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "environment",
                "num_nodes",
                "comparison",
                "aeris_mean",
                "baseline_mean",
                "diff",
                "p_value_raw",
                "sig_0_05",
            ]
        )
        for env in ENV_ORDER:
            for node in NODE_ORDER:
                aeris = np.asarray(
                    [
                        r["pdr_expected"]
                        for r in rows
                        if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == "AERIS"
                    ],
                    dtype=float,
                )
                for b in BASELINES:
                    base = np.asarray(
                        [
                            r["pdr_expected"]
                            for r in rows
                            if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == b
                        ],
                        dtype=float,
                    )
                    _, p = stats.ttest_ind(aeris, base, equal_var=False)
                    diff = float(aeris.mean() - base.mean())
                    w.writerow(
                        [
                            env,
                            node,
                            f"AERIS vs {b}",
                            f"{aeris.mean():.6f}",
                            f"{base.mean():.6f}",
                            f"{diff:.6f}",
                            f"{p:.6e}",
                            "yes" if p < 0.05 else "no",
                        ]
                    )
    return out


def export_summary(agg: Dict[Tuple[str, int, str], Dict[str, float]]) -> Path:
    out = RESULTS_DIR / "pilot_rigor_pub_20260215_summary.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Local Pilot-Rigor Summary (Publication Tier)\n\n")
        f.write("- Matrix: 4 env x 3 node counts (100/500/1000) x 5 protocols x n=60\n")
        f.write("- Metric: pdr_expected\n\n")
        f.write("## AERIS monotonicity check (expected non-increasing with larger scale)\n\n")
        for env in ENV_ORDER:
            vals = [agg[(env, node, "AERIS")]["pdr_mean"] for node in NODE_ORDER]
            is_non_inc = vals[1] <= vals[0] and vals[2] <= vals[1]
            f.write(f"- {env}: {[round(v, 4) for v in vals]} -> non_increasing={is_non_inc}\n")
        f.write("\n## Top protocol by environment and node count\n\n")
        for env in ENV_ORDER:
            for node in NODE_ORDER:
                rank = sorted(
                    ((proto, agg[(env, node, proto)]["pdr_mean"]) for proto in PROTO_ORDER),
                    key=lambda x: x[1],
                    reverse=True,
                )
                f.write(f"- {env}, {node}: {rank[0][0]} ({rank[0][1]:.4f})\n")
    return out


def main() -> None:
    rows = load_rows()
    agg = aggregate(rows)
    desc = export_descriptive(agg)
    sig = export_significance(rows)
    summary = export_summary(agg)
    print(desc)
    print(sig)
    print(summary)


if __name__ == "__main__":
    main()

