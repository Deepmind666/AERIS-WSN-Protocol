#!/usr/bin/env python3
"""Merge focused NS-3 shard results and compute descriptive/significance tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def welch_t(m1, s1, n1, m2, s2, n2):
    se1, se2 = s1**2 / max(n1, 1), s2**2 / max(n2, 1)
    se = se1 + se2
    if se < 1e-15:
        return 0.0, max(n1 + n2 - 2, 1)
    t = (m1 - m2) / math.sqrt(se)
    df = se**2 / (se1**2 / max(n1 - 1, 1) + se2**2 / max(n2 - 1, 1))
    return t, max(df, 1.0)


def t_to_p_twosided(t_stat):
    z = abs(t_stat)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p


def hedges_g(m1, s1, n1, m2, s2, n2):
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1)
    sp = math.sqrt(sp2) if sp2 > 0 else 1e-15
    d = (m1 - m2) / sp
    df = max(n1 + n2 - 2, 1)
    j = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * j


def holm_bonferroni(p_values):
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (m - rank), 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("shard_*.json"))
    if not files:
        raise SystemExit(f"No shard JSON found in {input_dir}")

    all_exp = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_exp.extend(data["experiments"])

    merged_path = output_dir / "ns3_focused_merged.json"
    merged_path.write_text(json.dumps({"experiments": all_exp}, indent=2), encoding="utf-8")

    buckets = defaultdict(list)
    for e in all_exp:
        buckets[(e["protocol"], e["environment"], int(e["num_nodes"]))].append(float(e["pdr"]))

    desc_rows = []
    for (proto, env, nodes), pdrs in sorted(buckets.items()):
        n = len(pdrs)
        mean = sum(pdrs) / n
        var = sum((x - mean) ** 2 for x in pdrs) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        desc_rows.append({
            "protocol": proto,
            "environment": env,
            "num_nodes": nodes,
            "n": n,
            "pdr_mean": round(mean, 6),
            "pdr_std": round(std, 6),
        })

    desc_csv = output_dir / "ns3_focused_descriptive.csv"
    with desc_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protocol", "environment", "num_nodes", "n", "pdr_mean", "pdr_std"])
        writer.writeheader()
        writer.writerows(desc_rows)

    comparisons = []
    envs = sorted({row["environment"] for row in desc_rows})
    nodes_set = sorted({row["num_nodes"] for row in desc_rows})
    for env in envs:
        for nodes in nodes_set:
            aeris = buckets.get(("AERIS", env, nodes), [])
            if not aeris:
                continue
            n_a = len(aeris)
            m_a = sum(aeris) / n_a
            s_a = math.sqrt(sum((x - m_a) ** 2 for x in aeris) / (n_a - 1)) if n_a > 1 else 0.0
            baselines = sorted({proto for proto, e, n in buckets if e == env and n == nodes and proto != "AERIS"})
            for baseline in baselines:
                base = buckets.get((baseline, env, nodes), [])
                if not base:
                    continue
                n_b = len(base)
                m_b = sum(base) / n_b
                s_b = math.sqrt(sum((x - m_b) ** 2 for x in base) / (n_b - 1)) if n_b > 1 else 0.0
                t_stat, _ = welch_t(m_a, s_a, n_a, m_b, s_b, n_b)
                p_raw = t_to_p_twosided(t_stat)
                g = hedges_g(m_a, s_a, n_a, m_b, s_b, n_b)
                comparisons.append({
                    "environment": env,
                    "num_nodes": nodes,
                    "baseline": baseline,
                    "aeris_mean": round(m_a, 6),
                    "baseline_mean": round(m_b, 6),
                    "diff": round(m_a - m_b, 6),
                    "hedges_g": round(g, 6),
                    "t_stat": round(t_stat, 6),
                    "p_raw": p_raw,
                })

    adjusted = holm_bonferroni([c["p_raw"] for c in comparisons])
    for c, p_adj in zip(comparisons, adjusted):
        c["p_holm"] = p_adj
        c["significant_005"] = "YES" if p_adj < 0.05 else "NO"

    sig_csv = output_dir / "ns3_focused_significance.csv"
    with sig_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "environment", "num_nodes", "baseline", "aeris_mean", "baseline_mean",
            "diff", "hedges_g", "t_stat", "p_raw", "p_holm", "significant_005"
        ])
        writer.writeheader()
        writer.writerows(comparisons)

    md_path = output_dir / "ns3_focused_summary.md"
    lines = [
        "# LCN26 Focused NS-3 Audit Summary",
        "",
        f"- input_dir: `{input_dir}`",
        f"- merged experiments: `{len(all_exp)}`",
        f"- shards: `{len(files)}`",
        "",
        "## Winner by environment-node cell",
        "",
        "| Environment | Nodes | Winner | PDR |",
        "|---|---:|---|---:|",
    ]

    for env in envs:
        for nodes in nodes_set:
            rows = [r for r in desc_rows if r["environment"] == env and r["num_nodes"] == nodes]
            rows = sorted(rows, key=lambda r: r["pdr_mean"], reverse=True)
            if rows:
                lines.append(f"| {env} | {nodes} | {rows[0]['protocol']} | {rows[0]['pdr_mean']:.3f} |")

    office_pegasis = [r for r in desc_rows if r["environment"] == "indoor_office" and r["protocol"] == "PEGASIS"]
    if office_pegasis:
        vals = [r["pdr_mean"] for r in sorted(office_pegasis, key=lambda x: x["num_nodes"])]
        lines += [
            "",
            "## PEGASIS office trend check",
            "",
            f"- indoor_office PEGASIS PDR values: `{vals}`",
            f"- range across tested scales: `{max(vals) - min(vals):.6f}`",
        ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LCN26-NS3] wrote {merged_path}")
    print(f"[LCN26-NS3] wrote {desc_csv}")
    print(f"[LCN26-NS3] wrote {sig_csv}")
    print(f"[LCN26-NS3] wrote {md_path}")


if __name__ == "__main__":
    main()
