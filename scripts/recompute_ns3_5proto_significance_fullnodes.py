"""
Recompute NS-3 five-protocol statistics on full node set (50..1000) from merged data.

Outputs:
  - ns3_validation/results/ns3_5proto_fullnodes_descriptive_20260226.csv
  - ns3_validation/results/ns3_5proto_fullnodes_significance_20260226.csv
  - ns3_validation/results/ns3_5proto_fullnodes_recalc_report_20260226.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


BASE = Path("ns3_validation/results")
MERGED = BASE / "ns3_5proto_merged.json"
SUMMARY = BASE / "ns3_5proto_summary.json"
PROTO_MAIN = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]


def hedges_g(a: List[float], b: List[float]) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / np.sqrt(pooled)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(d * correction)


def holm(rows: List[dict], p_key: str = "p_raw", out_key: str = "p_holm") -> None:
    pvals = np.array([r[p_key] for r in rows], dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    for i, row in enumerate(rows):
        row[out_key] = float(adjusted[i])
        row["significant_005"] = "YES" if row[out_key] < 0.05 else "NO"


def main() -> None:
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    exps = merged["experiments"]

    # Normalize protocol key to main-paper naming.
    for e in exps:
        if e["protocol"] == "AERIS-FULL":
            e["protocol"] = "AERIS"

    envs = sorted({e["environment"] for e in exps})
    nodes = sorted({int(e["num_nodes"]) for e in exps})

    # descriptive from summary json (prefer pre-aggregated n/mean/std).
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    desc_rows = []
    for s in summary:
        p = s["protocol"]
        if p == "AERIS-FULL":
            p = "AERIS"
        if p not in PROTO_MAIN:
            continue
        if int(s["num_nodes"]) not in nodes:
            continue
        desc_rows.append(
            {
                "protocol": p,
                "environment": s["environment"],
                "num_nodes": int(s["num_nodes"]),
                "n": int(s["n"]),
                "pdr_mean": float(s["pdr_mean"]),
                "pdr_std": float(s["pdr_std"]),
            }
        )

    desc_out = BASE / "ns3_5proto_fullnodes_descriptive_20260226.csv"
    with desc_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "environment", "num_nodes", "n", "pdr_mean", "pdr_std"])
        w.writeheader()
        w.writerows(sorted(desc_rows, key=lambda r: (r["environment"], r["num_nodes"], r["protocol"])))

    # significance AERIS vs each baseline.
    sig_rows = []
    for env in envs:
        for nn in nodes:
            aeris = [
                float(e["pdr"])
                for e in exps
                if e["environment"] == env and int(e["num_nodes"]) == nn and e["protocol"] == "AERIS"
            ]
            for base in BASELINES:
                b = [
                    float(e["pdr"])
                    for e in exps
                    if e["environment"] == env and int(e["num_nodes"]) == nn and e["protocol"] == base
                ]
                if len(aeris) < 2 or len(b) < 2:
                    continue
                t_stat, p_raw = stats.ttest_ind(aeris, b, equal_var=False)
                sig_rows.append(
                    {
                        "environment": env,
                        "num_nodes": nn,
                        "comparison": f"AERIS_vs_{base}",
                        "n_aeris": len(aeris),
                        "n_baseline": len(b),
                        "aeris_mean": round(float(np.mean(aeris)), 6),
                        "baseline_mean": round(float(np.mean(b)), 6),
                        "delta": round(float(np.mean(aeris) - np.mean(b)), 6),
                        "t_stat": round(float(t_stat), 6),
                        "p_raw": float(p_raw),
                        "hedges_g": round(hedges_g(aeris, b), 6),
                    }
                )

    holm(sig_rows)
    sig_out = BASE / "ns3_5proto_fullnodes_significance_20260226.csv"
    with sig_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "environment",
                "num_nodes",
                "comparison",
                "n_aeris",
                "n_baseline",
                "aeris_mean",
                "baseline_mean",
                "delta",
                "t_stat",
                "p_raw",
                "p_holm",
                "hedges_g",
                "significant_005",
            ],
        )
        w.writeheader()
        w.writerows(sorted(sig_rows, key=lambda r: (r["environment"], r["num_nodes"], r["comparison"])))

    report = BASE / "ns3_5proto_fullnodes_recalc_report_20260226.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# NS-3 5-Protocol Full-Node Recalculation (20260226)\n\n")
        f.write(f"- merged experiments: {len(exps)}\n")
        f.write(f"- environments: {', '.join(envs)}\n")
        f.write(f"- nodes: {nodes}\n")
        f.write(f"- descriptive rows: {len(desc_rows)}\n")
        f.write(f"- significance rows: {len(sig_rows)}\n")

    print(f"[OK] wrote {desc_out}")
    print(f"[OK] wrote {sig_out}")
    print(f"[OK] wrote {report}")


if __name__ == "__main__":
    main()
