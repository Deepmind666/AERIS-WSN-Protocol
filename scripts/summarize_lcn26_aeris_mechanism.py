#!/usr/bin/env python3
"""Summarize focused AERIS mechanism runs into CSV/Markdown files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCALAR_FIELDS = [
    "pdr_expected",
    "energy",
    "lifetime",
    "first_node_death_round",
    "half_nodes_death_round",
    "avg_hop_count",
    "cluster_to_ch_pdr_total",
    "ch_to_bs_pdr_total",
    "gateway_link_pdr_total",
    "gateway_uplink_attempts_total",
    "gateway_uplink_success_total",
    "gateway_uplink_pdr_total",
    "gateway_uplink_suppressed_total",
    "gateway_concurrency_usage_avg",
    "skeleton_backbone_size",
    "skeleton_assignments",
    "cas_total_decisions",
]

CAS_FIELDS = ["DIRECT", "CHAIN", "TWO_HOP", "safety_override"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    summary = data["summary"]
    out_dir = in_path.parent

    csv_path = out_dir / "mechanism_summary.csv"
    md_path = out_dir / "mechanism_summary.md"

    fieldnames = ["environment", "num_nodes", "n"]
    for field in SCALAR_FIELDS:
        fieldnames += [f"{field}_mean", f"{field}_std"]
    for field in CAS_FIELDS:
        fieldnames += [f"cas_{field}_mean", f"cas_{field}_std"]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in sorted(summary.items(), key=lambda item: (item[1]["environment"], item[1]["num_nodes"])):
            out = {
                "environment": row["environment"],
                "num_nodes": row["num_nodes"],
                "n": row["n"],
            }
            for field in SCALAR_FIELDS:
                out[f"{field}_mean"] = row[field]["mean"]
                out[f"{field}_std"] = row[field]["std"]
            for field in CAS_FIELDS:
                out[f"cas_{field}_mean"] = row["cas_mode_usage_stats"][field]["mean"]
                out[f"cas_{field}_std"] = row["cas_mode_usage_stats"][field]["std"]
            writer.writerow(out)

    lines = [
        "# LCN26 AERIS Mechanism Summary",
        "",
        f"- Source: `{in_path}`",
        f"- git_commit: `{data.get('git_commit', 'unknown')}`",
        f"- run_tier: `{data.get('run_tier', 'unknown')}`",
        f"- environments: `{', '.join(data['config'].get('environments', []))}`",
        f"- node_counts: `{', '.join(str(x) for x in data['config'].get('node_counts', []))}`",
        "",
        "## Per-cell means",
        "",
        "| Environment | Nodes | PDR | FND | Half-death | GW uplink PDR | Skeleton assign | CAS Direct | CAS Chain | CAS Two-hop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in sorted(summary.items(), key=lambda item: (item[1]["environment"], item[1]["num_nodes"])):
        lines.append(
            f"| {row['environment']} | {row['num_nodes']} | "
            f"{row['pdr_expected']['mean']:.3f} | {row['first_node_death_round']['mean']:.1f} | "
            f"{row['half_nodes_death_round']['mean']:.1f} | {row['gateway_uplink_pdr_total']['mean']:.3f} | "
            f"{row['skeleton_assignments']['mean']:.1f} | "
            f"{row['cas_mode_usage_stats']['DIRECT']['mean']:.1f} | "
            f"{row['cas_mode_usage_stats']['CHAIN']['mean']:.1f} | "
            f"{row['cas_mode_usage_stats']['TWO_HOP']['mean']:.1f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LCN26-MECH] wrote {csv_path}")
    print(f"[LCN26-MECH] wrote {md_path}")


if __name__ == "__main__":
    main()
