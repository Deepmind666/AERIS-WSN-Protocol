#!/usr/bin/env python3
"""Build a compact canonical NS-3 margin figure for the LCN26 draft.

This draft figure is intended as a candidate for Fig. 2. It visualizes the
margin of AERIS over the strongest classical baseline within the five-protocol
canonical audit, keeping the main result readable without line overlap.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "figure_reproduction").exists():
            return parent
        if (parent / "fig2_fig5_data").exists() and (parent / "scripts").exists():
            return parent
    return current.parents[1]


ROOT = find_repo_root()
DATA_DIR = ROOT / "data" / "figure_reproduction"
if not DATA_DIR.exists():
    DATA_DIR = ROOT / "fig2_fig5_data"
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
PACKED_INPUT_CSV = (
    DATA_DIR
    / "01_fig2_classical_margin"
    / "source"
    / "ns3_5proto_fullnodes_descriptive_20260226.csv"
)
LEGACY_INPUT_CSV = (
    ROOT
    / "ns3_validation"
    / "results"
    / "ns3_5proto_fullnodes_descriptive_20260226.csv"
)
INPUT_CSV = PACKED_INPUT_CSV if PACKED_INPUT_CSV.exists() else LEGACY_INPUT_CSV
OUTPUT_PDF = OUT_DIR / "fig_lcn26_ns3_canonical_margin.pdf"
OUTPUT_PNG = OUT_DIR / "fig_lcn26_ns3_canonical_margin.png"
DERIVED_CSV = (
    DATA_DIR
    / "01_fig2_classical_margin"
    / "derived"
    / "ns3_classical_margin_summary.csv"
)

ENV_ORDER = [
    "indoor_office",
    "indoor_factory",
    "outdoor_suburban",
    "outdoor_urban",
]
ENV_TITLES = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
NODE_LABELS = ["50", "100", "200", "300", "500", "800", "1k"]
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
BASELINE_PROTOCOLS = [p for p in PROTO_ORDER if p != "AERIS"]

COLORS = {
    "AERIS": "#C13136",
    "baseline": "#6D6D6D",
    "near_tie": "#F29440",
    "band": "#F1D5B3",
    "axis": "#111111",
    "grid": "#D0D0D0",
}


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.4,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["axis"],
            "axes.labelcolor": COLORS["axis"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "grid.alpha": 0.9,
            "grid.linestyle": "--",
            "axes.grid.axis": "y",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.7,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.1,
            "legend.handletextpad": 0.28,
            "legend.columnspacing": 0.65,
            "legend.borderaxespad": 0.05,
        }
    )


def load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], tuple[float, float, int]]:
    grouped: dict[tuple[str, int, str], list[tuple[float, float, int]]] = defaultdict(list)
    for row in rows:
        key = (row["environment"], int(row["num_nodes"]), row["protocol"])
        grouped[key].append((float(row["pdr_mean"]), float(row["pdr_std"]), int(row["n"])))

    stats: dict[tuple[str, int, str], tuple[float, float, int]] = {}
    for key, values in grouped.items():
        means = np.asarray([item[0] for item in values], dtype=float)
        stds = np.asarray([item[1] for item in values], dtype=float)
        ns = [item[2] for item in values]
        mean = float(means.mean())
        std = float(stds.mean())
        stats[key] = (mean, std, int(np.median(ns)))
    return stats


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def write_margin_summary(rows: list[dict[str, object]]) -> None:
    DERIVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "num_nodes",
        "aeris_pdr_mean",
        "best_classical_protocol",
        "best_classical_pdr_mean",
        "aeris_minus_best_classical_pp",
        "margin_ci95_pp",
        "aeris_rank",
        "aeris_top2",
        "near_tie_abs_margin_le_0p1pp",
    ]
    with DERIVED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_figure(stats: dict[tuple[str, int, str], tuple[float, float, int]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.52, 2.92), sharex=True, sharey=True)
    axes = axes.flatten()
    x = np.arange(len(NODE_ORDER), dtype=float)
    derived_rows: list[dict[str, object]] = []
    any_near_tie = False

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        margins: list[float] = []
        errors: list[float] = []
        classical_wins = 0
        top2 = 0
        for node in NODE_ORDER:
            means = {proto: stats[(env, node, proto)][0] for proto in PROTO_ORDER}
            stds = {proto: stats[(env, node, proto)][1] for proto in PROTO_ORDER}
            ns = {proto: stats[(env, node, proto)][2] for proto in PROTO_ORDER}
            best_baseline = max(BASELINE_PROTOCOLS, key=lambda proto: means[proto])
            margin = (means["AERIS"] - means[best_baseline]) * 100.0
            margin_err = 1.96 * math.sqrt(
                (stds["AERIS"] ** 2) / max(ns["AERIS"], 1)
                + (stds[best_baseline] ** 2) / max(ns[best_baseline], 1)
            ) * 100.0
            margins.append(margin)
            errors.append(margin_err)
            ordered = sorted(PROTO_ORDER, key=lambda proto: means[proto], reverse=True)
            aeris_rank = ordered.index("AERIS") + 1
            classical_wins += int(ordered[0] == "AERIS")
            top2 += int(aeris_rank <= 2)
            derived_rows.append(
                {
                    "environment": env,
                    "num_nodes": node,
                    "aeris_pdr_mean": f"{means['AERIS']:.6f}",
                    "best_classical_protocol": best_baseline,
                    "best_classical_pdr_mean": f"{means[best_baseline]:.6f}",
                    "aeris_minus_best_classical_pp": f"{margin:.4f}",
                    "margin_ci95_pp": f"{margin_err:.4f}",
                    "aeris_rank": aeris_rank,
                    "aeris_top2": int(aeris_rank <= 2),
                    "near_tie_abs_margin_le_0p1pp": int(abs(margin) <= 0.1),
                }
            )

        margins_arr = np.asarray(margins, dtype=float)
        err_arr = np.asarray(errors, dtype=float)
        near_tie_mask = np.abs(margins_arr) <= 0.1
        any_near_tie = any_near_tie or bool(np.any(near_tie_mask))
        colors = np.where(
            near_tie_mask,
            COLORS["near_tie"],
            np.where(margins_arr > 0, COLORS["AERIS"], COLORS["baseline"]),
        )

        ax.axhspan(-0.1, 0.1, color=COLORS["band"], alpha=0.35, zorder=0)
        ax.axhline(0, color=COLORS["axis"], linewidth=0.7, zorder=1)
        for xpos, margin, color in zip(x, margins_arr, colors):
            ax.vlines(xpos, 0, margin, color=color, linewidth=1.1, alpha=0.92, zorder=2)
        ax.errorbar(
            x,
            margins_arr,
            yerr=err_arr,
            fmt="none",
            ecolor="#303030",
            elinewidth=0.55,
            capsize=1.6,
            capthick=0.55,
            zorder=3,
        )
        ax.scatter(
            x,
            margins_arr,
            s=14.0,
            facecolors=colors,
            edgecolors="#202020",
            linewidths=0.35,
            zorder=4,
        )

        mean_margin = float(np.mean(margins_arr))
        ax.set_title(
            f"{ENV_TITLES[env]}\nClass. {classical_wins}/7, top-2 {top2}/7; mean {mean_margin:+.1f} pp",
            pad=1.1,
            fontweight="bold",
        )
        ax.set_xlim(-0.10, len(NODE_ORDER) - 0.65)
        ax.set_ylim(-10.5, 8.8)
        ax.set_xticks(x)
        ax.set_xticklabels(NODE_LABELS)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, color=COLORS["grid"])
        ax.grid(axis="x", linestyle=":", linewidth=0.28, color="#E4E4E4")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.2, pad=1.4)

    axes[0].set_ylabel("AERIS margin (pp)")
    axes[2].set_ylabel("AERIS margin (pp)")
    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLORS["AERIS"], markerfacecolor=COLORS["AERIS"], markeredgecolor="#202020", linewidth=1.0, markersize=4.0, label="AERIS lead"),
        Line2D([0], [0], marker="o", color=COLORS["baseline"], markerfacecolor=COLORS["baseline"], markeredgecolor="#202020", linewidth=1.0, markersize=4.0, label="baseline lead"),
    ]
    if any_near_tie:
        legend_handles.append(
            Line2D([0], [0], marker="o", color=COLORS["near_tie"], markerfacecolor=COLORS["near_tie"], markeredgecolor="#202020", linewidth=1.0, markersize=4.0, label="near-tie")
        )
    fig.legend(
        handles=legend_handles,
        ncol=len(legend_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        columnspacing=0.72,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.82, bottom=0.15, wspace=0.20, hspace=0.38)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=320)
    plt.close(fig)
    write_margin_summary(derived_rows)


def main() -> None:
    rows = load_rows()
    stats = group_rows(rows)
    build_figure(stats)
    print(f"[LCN26] Wrote {OUTPUT_PDF}")
    print(f"[LCN26] Wrote {OUTPUT_PNG}")
    print(f"[LCN26] Wrote {DERIVED_CSV}")


if __name__ == "__main__":
    apply_style()
    main()
