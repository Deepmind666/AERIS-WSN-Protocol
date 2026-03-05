#!/usr/bin/env python3
"""
Pre-NS3 scalability figure generator for manuscript Section 6.4.

Input:
- pre_ns3_scalability_summary_20260210_231438.csv

Outputs:
- fig4_scalability_multienv_lines_<suffix>.pdf/png/svg
- fig4b_indoor_office_gap_<suffix>.pdf/png/svg
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
})

PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
ENV_LABELS = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_urban": "Outdoor Urban",
    "outdoor_suburban": "Outdoor Suburban",
}
COLORS = {
    "AERIS": "#1f77b4",
    "LEACH": "#d62728",
    "PEGASIS": "#ff7f0e",
    "HEED": "#2ca02c",
    "TEEN": "#9467bd",
}
MARKERS = {
    "AERIS": "o",
    "LEACH": "s",
    "PEGASIS": "^",
    "HEED": "D",
    "TEEN": "v",
}


def load_summary(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "environment": r["environment"],
                    "num_nodes": int(r["num_nodes"]),
                    "protocol": r["protocol"],
                    "pdr_mean": float(r["pdr_mean"]),
                    "pdr_std": float(r["pdr_std"]),
                    "rank": int(r["rank"]),
                }
            )
    return rows


def plot_multienv_lines(rows, out_dir: Path, suffix: str):
    data = defaultdict(dict)
    for r in rows:
        data[(r["environment"], r["protocol"])][r["num_nodes"]] = (r["pdr_mean"], r["pdr_std"])

    node_counts = sorted({r["num_nodes"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        for proto in PROTOCOLS:
            ys = []
            es = []
            for n in node_counts:
                m, s = data[(env, proto)][n]
                ys.append(m)
                es.append(s)
            ax.errorbar(
                node_counts,
                ys,
                yerr=es,
                color=COLORS[proto],
                marker=MARKERS[proto],
                markersize=4,
                linewidth=1.5,
                elinewidth=0.7,
                capsize=2,
                label=proto,
            )
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Node Count")
        ax.set_ylabel("PDR (pdr_expected)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))

    for fmt in ["pdf", "png", "svg"]:
        fig.savefig(out_dir / f"fig4_scalability_multienv_lines_{suffix}.{fmt}")

    plt.close(fig)


def plot_indoor_gap(rows, out_dir: Path, suffix: str):
    indoor = [r for r in rows if r["environment"] == "indoor_office" and r["protocol"] in ("AERIS", "PEGASIS")]
    node_counts = sorted({r["num_nodes"] for r in indoor})

    by_proto = defaultdict(dict)
    for r in indoor:
        by_proto[r["protocol"]][r["num_nodes"]] = (r["pdr_mean"], r["pdr_std"])

    aeris = np.array([by_proto["AERIS"][n][0] for n in node_counts])
    peg = np.array([by_proto["PEGASIS"][n][0] for n in node_counts])
    gap = peg - aeris

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)

    ax1.plot(node_counts, aeris, marker="o", color=COLORS["AERIS"], linewidth=1.8, label="AERIS")
    ax1.plot(node_counts, peg, marker="^", color=COLORS["PEGASIS"], linewidth=1.8, label="PEGASIS")
    ax1.set_ylabel("PDR")
    ax1.set_title("Indoor Office Scalability: AERIS vs PEGASIS")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend(loc="lower right")

    ax2.bar(node_counts, gap * 100.0, width=55, color="#555555", edgecolor="black", linewidth=0.5)
    for x, g in zip(node_counts, gap):
        ax2.text(x, g * 100.0 + 0.03, f"{g*100:.2f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("Node Count")
    ax2.set_ylabel("PEGASIS - AERIS (PDR %)" )
    ax2.grid(True, linestyle="--", alpha=0.25)

    for fmt in ["pdf", "png", "svg"]:
        fig.savefig(out_dir / f"fig4b_indoor_office_gap_{suffix}.{fmt}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate pre-NS3 scalability figures.")
    parser.add_argument(
        "--summary-csv",
        default="results/mega_experiments/pre_ns3_scalability_summary_20260210_231438.csv",
        help="Input summary CSV path",
    )
    parser.add_argument(
        "--out-dir",
        default="for_submission/figures",
        help="Output figure directory",
    )
    parser.add_argument(
        "--suffix",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Suffix for output file names",
    )
    args = parser.parse_args()

    csv_path = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(csv_path)
    plot_multienv_lines(rows, out_dir, args.suffix)
    plot_indoor_gap(rows, out_dir, args.suffix)

    print("Generated scalability figures:")
    print(f"  fig4_scalability_multienv_lines_{args.suffix}.(pdf/png/svg)")
    print(f"  fig4b_indoor_office_gap_{args.suffix}.(pdf/png/svg)")


if __name__ == "__main__":
    main()
