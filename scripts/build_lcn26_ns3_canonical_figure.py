#!/usr/bin/env python3
"""Build the canonical NS-3 five-protocol figure for the LCN 2026 draft."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "ns3_validation" / "results" / "ns3_5proto_fullnodes_descriptive_20260226.csv"
OUTPUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
OUTPUT_CSV = OUTPUT_DIR / "ns3_canonical_5proto_filtered.csv"
OUTPUT_PDF = OUTPUT_DIR / "fig_ns3_canonical_5proto.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig_ns3_canonical_5proto.png"

ENV_ORDER = [
    "indoor_office",
    "indoor_factory",
    "outdoor_suburban",
    "outdoor_urban",
]
ENV_TITLES = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_suburban": "Outdoor Suburban",
    "outdoor_urban": "Outdoor Urban",
}
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
PROTO_LABELS = {
    "AERIS": "AERIS",
    "PEGASIS": "PEGASIS",
    "LEACH": "LEACH",
    "HEED": "HEED",
    "TEEN": "TEEN",
}
PROTO_COLORS = {
    "AERIS": "#0b4f6c",
    "PEGASIS": "#f18f01",
    "LEACH": "#c73e1d",
    "HEED": "#3a7d44",
    "TEEN": "#6f2dbd",
}
PROTO_MARKERS = {
    "AERIS": "o",
    "PEGASIS": "s",
    "LEACH": "^",
    "HEED": "D",
    "TEEN": "P",
}


def load_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protocol = row["protocol"]
            if protocol not in PROTO_ORDER:
                continue
            rows.append(
                {
                    "environment": row["environment"],
                    "num_nodes": int(row["num_nodes"]),
                    "protocol": protocol,
                    "n": int(row["n"]),
                    "pdr_mean": float(row["pdr_mean"]),
                    "pdr_std": float(row["pdr_std"]),
                }
            )
    return rows


def write_filtered_csv(rows: list[dict[str, float | int | str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["environment", "num_nodes", "protocol", "n", "pdr_mean", "pdr_std"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["environment"], item["protocol"], item["num_nodes"])):
            writer.writerow(row)


def plot(rows: list[dict[str, float | int | str]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.95), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, environment in zip(axes, ENV_ORDER):
        env_rows = [row for row in rows if row["environment"] == environment]
        for protocol in PROTO_ORDER:
            proto_rows = sorted(
                [row for row in env_rows if row["protocol"] == protocol],
                key=lambda item: int(item["num_nodes"]),
            )
            x_vals = [int(row["num_nodes"]) for row in proto_rows]
            y_vals = [float(row["pdr_mean"]) for row in proto_rows]
            line_width = 2.4 if protocol == "AERIS" else 1.6
            zorder = 4 if protocol == "AERIS" else 3
            ax.plot(
                x_vals,
                y_vals,
                label=PROTO_LABELS[protocol],
                color=PROTO_COLORS[protocol],
                marker=PROTO_MARKERS[protocol],
                linewidth=line_width,
                markersize=4.2,
                markeredgewidth=0.4,
                zorder=zorder,
            )

        ax.set_title(ENV_TITLES[environment])
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_xticks([50, 100, 200, 300, 500, 800, 1000])
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(top=0.82, left=0.07, right=0.995, bottom=0.12, wspace=0.12, hspace=0.26)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_filtered_csv(rows)
    plot(rows)
    print(f"[LCN26] Wrote {OUTPUT_PDF}")
    print(f"[LCN26] Wrote {OUTPUT_PNG}")
    print(f"[LCN26] Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
