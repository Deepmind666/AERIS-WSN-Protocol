#!/usr/bin/env python3
"""Analyze expanded NS-3 AERIS ablation shards for the LCN26 draft."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


VARIANTS = ["AERIS-noGW", "AERIS-noCAS", "AERIS-noFair"]
REFERENCE = "AERIS-FULL"
FIELDS = [
    "environment",
    "num_nodes",
    "variant",
    "full_mean",
    "variant_mean",
    "delta_points",
    "full_std",
    "variant_std",
    "n",
    "hedges_g",
    "t_stat",
    "p_raw",
    "p_holm",
    "significant_005",
]


def welch_t(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> tuple[float, float]:
    se1 = s1**2 / max(n1, 1)
    se2 = s2**2 / max(n2, 1)
    se = se1 + se2
    if se < 1e-15:
        return 0.0, float(max(n1 + n2 - 2, 1))
    t_stat = (m1 - m2) / math.sqrt(se)
    df = se**2 / (se1**2 / max(n1 - 1, 1) + se2**2 / max(n2 - 1, 1))
    return t_stat, max(df, 1.0)


def t_to_p_twosided(t_stat: float) -> float:
    # Normal approximation is sufficient here for ranking/annotation; n=30 per cell.
    return min(2.0 * (0.5 * math.erfc(abs(t_stat) / math.sqrt(2))), 1.0)


def hedges_g(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1)
    sp = math.sqrt(sp2) if sp2 > 0 else 1e-15
    d = (m1 - m2) / sp
    df = max(n1 + n2 - 2, 1)
    j = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * j


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_val) in enumerate(indexed):
        adj = min(p_val * (m - rank), 1.0)
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted


def load_experiments(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["experiments"]


def summarize(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var), n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ns3_focused_merged.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(input_path)
    buckets: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for exp in experiments:
        proto = str(exp["protocol"])
        if proto not in {REFERENCE, *VARIANTS}:
            continue
        buckets[(str(exp["environment"]), int(exp["num_nodes"]), proto)].append(float(exp["pdr"]))

    rows: list[dict[str, object]] = []
    for env, nodes, proto in sorted(buckets):
        if proto != REFERENCE:
            continue
        full = buckets[(env, nodes, REFERENCE)]
        m_full, s_full, n_full = summarize(full)
        for variant in VARIANTS:
            vals = buckets.get((env, nodes, variant), [])
            if not vals:
                continue
            m_var, s_var, n_var = summarize(vals)
            t_stat, _ = welch_t(m_full, s_full, n_full, m_var, s_var, n_var)
            rows.append(
                {
                    "environment": env,
                    "num_nodes": nodes,
                    "variant": variant,
                    "full_mean": round(m_full, 6),
                    "variant_mean": round(m_var, 6),
                    "delta_points": round((m_var - m_full) * 100.0, 4),
                    "full_std": round(s_full, 6),
                    "variant_std": round(s_var, 6),
                    "n": min(n_full, n_var),
                    "hedges_g": round(hedges_g(m_full, s_full, n_full, m_var, s_var, n_var), 6),
                    "t_stat": round(t_stat, 6),
                    "p_raw": t_to_p_twosided(t_stat),
                }
            )

    adjusted = holm_bonferroni([float(row["p_raw"]) for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
        row["significant_005"] = "YES" if p_adj < 0.05 else "NO"

    delta_csv = output_dir / "ns3_ablation_delta.csv"
    with delta_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    env_rows = []
    for env in sorted({str(row["environment"]) for row in rows}):
        for variant in VARIANTS:
            vals = [float(row["delta_points"]) for row in rows if row["environment"] == env and row["variant"] == variant]
            if not vals:
                continue
            env_rows.append(
                {
                    "environment": env,
                    "variant": variant,
                    "mean_delta_points": round(sum(vals) / len(vals), 4),
                    "min_delta_points": round(min(vals), 4),
                    "max_delta_points": round(max(vals), 4),
                    "significant_cells": sum(
                        1
                        for row in rows
                        if row["environment"] == env and row["variant"] == variant and row["significant_005"] == "YES"
                    ),
                    "cells": len(vals),
                }
            )

    env_csv = output_dir / "ns3_ablation_environment_summary.csv"
    with env_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "environment",
                "variant",
                "mean_delta_points",
                "min_delta_points",
                "max_delta_points",
                "significant_cells",
                "cells",
            ],
        )
        writer.writeheader()
        writer.writerows(env_rows)

    lines = [
        "# LCN26 NS-3 AERIS Ablation Summary",
        "",
        f"- input: `{input_path}`",
        f"- experiments: `{len(experiments)}`",
        f"- reference: `{REFERENCE}`",
        "",
        "Delta is variant minus full AERIS in percentage points; negative values mean the removed module hurts delivery.",
        "",
        "| Environment | Variant | Mean delta pts | Range pts | Significant cells |",
        "|---|---|---:|---:|---:|",
    ]
    for row in env_rows:
        lines.append(
            f"| {row['environment']} | {row['variant']} | {row['mean_delta_points']:.2f} | "
            f"{row['min_delta_points']:.2f} to {row['max_delta_points']:.2f} | "
            f"{row['significant_cells']}/{row['cells']} |"
        )

    md_path = output_dir / "ns3_ablation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[LCN26-ABLATION] wrote {delta_csv}")
    print(f"[LCN26-ABLATION] wrote {env_csv}")
    print(f"[LCN26-ABLATION] wrote {md_path}")


if __name__ == "__main__":
    main()
