#!/usr/bin/env python3
"""
Rebuild scalability tables from a single S8 regime.

Why this script exists:
- Previous scalability summaries were assembled from mixed sources.
- This tool enforces one-file-per-environment inputs and produces
  a unified descriptive + significance package for manuscript use.

Usage:
  python scripts/rebuild_scalability_from_s8.py \
    --indoor-office results/mega_experiments/scalability_indoor_office_server_s8_20260213.json \
    --indoor-factory results/mega_experiments/scalability_indoor_factory_server_s8_20260213.json \
    --outdoor-urban results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.json \
    --outdoor-suburban results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.json \
    --out-prefix s8_unified_20260213
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_COUNTS = [100, 200, 300, 500, 800, 1000]
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    pooled = ((nx - 1) * sx2 + (ny - 1) * sy2) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return 0.0
    d = (x.mean() - y.mean()) / math.sqrt(pooled)
    correction = 1.0 - 3.0 / max(4.0 * (nx + ny) - 9.0, 1.0)
    return float(d * correction)


def load_raw(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in data["raw_results"] if not r.get("error")]


def build_descriptive(rows: List[dict], out_csv: Path) -> None:
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in rows:
        key = (r["environment"], int(r["num_nodes"]), r["protocol"])
        grouped[key].append(float(r["metrics"]["pdr_expected"]))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["environment", "num_nodes", "protocol", "n", "pdr_mean", "pdr_std", "ci95_half_width"])
        for env in ENV_ORDER:
            for node in NODE_COUNTS:
                for proto in PROTOCOLS:
                    vals = np.asarray(grouped[(env, node, proto)], dtype=float)
                    n = len(vals)
                    mean = float(vals.mean()) if n else float("nan")
                    std = float(vals.std(ddof=1)) if n > 1 else 0.0
                    ci95 = 1.96 * std / math.sqrt(max(n, 1))
                    w.writerow([env, node, proto, n, f"{mean:.6f}", f"{std:.6f}", f"{ci95:.6f}"])


def build_significance(rows: List[dict], out_csv: Path) -> None:
    grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in rows:
        key = (r["environment"], int(r["num_nodes"]), r["protocol"])
        grouped[key].append(float(r["metrics"]["pdr_expected"]))

    out_rows: List[dict] = []
    for env in ENV_ORDER:
        for node in NODE_COUNTS:
            aeris = np.asarray(grouped[(env, node, "AERIS")], dtype=float)
            pvals: List[float] = []
            tmp: List[dict] = []
            for baseline in BASELINES:
                b = np.asarray(grouped[(env, node, baseline)], dtype=float)
                t_stat, p_raw = stats.ttest_ind(aeris, b, equal_var=False)
                tmp.append(
                    {
                        "environment": env,
                        "num_nodes": node,
                        "comparison": f"AERIS vs {baseline}",
                        "baseline": baseline,
                        "metric": "pdr_expected",
                        "aeris_mean": float(aeris.mean()),
                        "baseline_mean": float(b.mean()),
                        "diff": float(aeris.mean() - b.mean()),
                        "hedges_g": float(hedges_g(aeris, b)),
                        "t_stat": float(t_stat),
                        "p_value_raw": float(p_raw),
                    }
                )
                pvals.append(float(p_raw))
            p_holm = holm_bonferroni(pvals)
            for i, row in enumerate(tmp):
                row["p_value_holm"] = float(p_holm[i])
                row["sig_holm_0_05"] = "yes" if p_holm[i] < 0.05 else "no"
                out_rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "environment",
            "num_nodes",
            "comparison",
            "baseline",
            "metric",
            "aeris_mean",
            "baseline_mean",
            "diff",
            "hedges_g",
            "t_stat",
            "p_value_raw",
            "p_value_holm",
            "sig_holm_0_05",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(
                {
                    k: (f"{row[k]:.6e}" if "p_value" in k else f"{row[k]:.6f}")
                    if isinstance(row[k], float)
                    else row[k]
                    for k in fieldnames
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild unified S8 scalability tables.")
    parser.add_argument("--indoor-office", required=True)
    parser.add_argument("--indoor-factory", required=True)
    parser.add_argument("--outdoor-urban", required=True)
    parser.add_argument("--outdoor-suburban", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    in_map = {
        "indoor_office": Path(args.indoor_office),
        "indoor_factory": Path(args.indoor_factory),
        "outdoor_urban": Path(args.outdoor_urban),
        "outdoor_suburban": Path(args.outdoor_suburban),
    }
    for env, p in in_map.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input for {env}: {p}")

    rows: List[dict] = []
    for env, p in in_map.items():
        raw = load_raw(p)
        for r in raw:
            r = dict(r)
            r["environment"] = env
            rows.append(r)

    out_dir = Path("results/mega_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    desc_csv = out_dir / f"{args.out_prefix}_descriptive.csv"
    sig_csv = out_dir / f"{args.out_prefix}_significance.csv"
    build_descriptive(rows, desc_csv)
    build_significance(rows, sig_csv)
    print(f"descriptive={desc_csv}")
    print(f"significance={sig_csv}")


if __name__ == "__main__":
    main()

