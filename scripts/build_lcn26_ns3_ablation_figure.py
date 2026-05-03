#!/usr/bin/env python3
"""Build a compact NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_DIR = ROOT / "ns3_validation" / "results" / "lcn26_ns3_ablation_combined_20260501_010355_011001" / "summary"
SUMMARY_FILE = ABLATION_DIR / "ns3_ablation_environment_summary.csv"
DELTA_FILE = ABLATION_DIR / "ns3_ablation_delta.csv"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABEL = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
VARIANTS = [
    ("AERIS-noGW", "Gateway"),
    ("AERIS-noCAS", "CAS"),
    ("AERIS-noFair", "CH score"),
]
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
NODE_LABELS = ["50", "100", "200", "300", "500", "800", "1k"]
COLORS = {
    "Gateway": "#1F77B4",
    "CAS": "#FF7F0E",
    "CH score": "#9E9E9E",
    "axis": "#111111",
    "grid": "#CFCFCF",
    "text": "#111111",
    "muted": "#555555",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.95,
            "grid.linestyle": "--",
        }
    )


def load_summary() -> dict[tuple[str, str], tuple[float, float, float, int, int]]:
    summary: dict[tuple[str, str], tuple[float, float, float, int, int]] = {}
    for row in load_rows(SUMMARY_FILE):
        # Stored delta is ablated-full. Contribution is full-ablated.
        contribution = -float(row["mean_delta_points"])
        min_contribution = -float(row["max_delta_points"])
        max_contribution = -float(row["min_delta_points"])
        summary[(row["environment"], row["variant"])] = (
            contribution,
            min_contribution,
            max_contribution,
            int(row["significant_cells"]),
            int(row["cells"]),
        )
    return summary


def load_cell_deltas() -> dict[tuple[str, str, int], tuple[float, bool]]:
    cells: dict[tuple[str, str, int], tuple[float, bool]] = {}
    for row in load_rows(DELTA_FILE):
        # Stored delta is ablated-full. Contribution is full-ablated.
        contribution = -float(row["delta_points"])
        sig_flag = row["significant_005"].strip().lower()
        cells[(row["environment"], row["variant"], int(row["num_nodes"]))] = (
            contribution,
            sig_flag in {"true", "yes", "1"},
        )
    return cells


def build() -> None:
    apply_style()
    cells = load_cell_deltas()
    fig, axes = plt.subplots(3, 1, figsize=(3.50, 2.52), sharex=True)
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=8.0)
    last_heat = None

    panel_tags = ["(a)", "(b)", "(c)"]
    for panel, ax, (variant, label) in zip(panel_tags, axes, VARIANTS):
        matrix = np.asarray(
            [
                [cells[(env, variant, node)][0] for node in NODE_ORDER]
                for env in ENV_ORDER
            ],
            dtype=float,
        )
        sig_matrix = np.asarray(
            [
                [cells[(env, variant, node)][1] for node in NODE_ORDER]
                for env in ENV_ORDER
            ],
            dtype=bool,
        )
        last_heat = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=norm)
        ax.set_yticks(np.arange(len(ENV_ORDER)))
        ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
        ax.set_title(f"{panel} {label}", loc="left", pad=1.5, fontsize=6.8, fontweight="bold")
        sig_y, sig_x = np.where(sig_matrix)
        ax.set_xticks(np.arange(len(NODE_ORDER)))
        ax.set_xticklabels(NODE_LABELS)
        ax.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ENV_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", length=2.2, pad=1.5)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.scatter(sig_x, sig_y, marker="o", s=15, facecolor="white", edgecolor="black", linewidth=0.55, alpha=0.98, zorder=5)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("")
    fig.text(0.025, 0.56, "Environment", rotation=90, ha="center", va="center", fontsize=6.6)
    if last_heat is not None:
        cax = fig.add_axes([0.18, 0.075, 0.77, 0.035])
        cbar = fig.colorbar(last_heat, cax=cax, orientation="horizontal")
        cbar.set_label("Contribution: full minus ablated PDR (pp); dots = significant", fontsize=5.6, labelpad=1)
        cbar.ax.tick_params(labelsize=5.3, length=2.0, pad=1)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.965, bottom=0.18, hspace=0.20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
