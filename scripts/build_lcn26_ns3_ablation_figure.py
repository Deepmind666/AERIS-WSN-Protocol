#!/usr/bin/env python3
"""Build a compact NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

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
    "Gateway": "#1C7ABA",
    "CAS": "#FF7F0E",
    "CH score": "#9E9E9E",
    "AERIS": "#C13136",
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
    summary = load_summary()
    fig, axes = plt.subplots(3, 1, figsize=(3.50, 2.62), sharex=True)
    y = np.arange(len(ENV_ORDER), dtype=float)
    x_min, x_max = -2.1, 8.4
    panel_tags = ["(a)", "(b)", "(c)"]

    def pretty_value(value: float) -> str:
        if abs(value) < 0.05:
            return "+0.0"
        return f"{value:+.1f}"

    for panel, ax, (variant, label) in zip(panel_tags, axes, VARIANTS):
        vals = np.asarray([summary[(env, variant)][0] for env in ENV_ORDER], dtype=float)
        lo = np.asarray([summary[(env, variant)][1] for env in ENV_ORDER], dtype=float)
        hi = np.asarray([summary[(env, variant)][2] for env in ENV_ORDER], dtype=float)
        sig = [summary[(env, variant)][3] for env in ENV_ORDER]
        total = [summary[(env, variant)][4] for env in ENV_ORDER]
        color = COLORS[label]

        ax.axvspan(-0.1, 0.1, color="#EFEFEF", zorder=0)
        ax.axvline(0.0, color=COLORS["axis"], linewidth=0.70, zorder=1)
        ax.barh(y, vals, height=0.50, color=color, alpha=0.86, edgecolor="white", linewidth=0.45, zorder=3)
        ax.errorbar(
            vals,
            y,
            xerr=[np.maximum(vals - lo, 0.0), np.maximum(hi - vals, 0.0)],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.65,
            capsize=1.8,
            zorder=4,
        )

        for yi, val, sc, tc in zip(y, vals, sig, total):
            ha = "left"
            offset = 0.14
            ax.text(
                val + offset,
                yi,
                f"{pretty_value(val)} ({sc}/{tc})",
                ha=ha,
                va="center",
                fontsize=5.5,
                color=COLORS["text"],
            )

        ax.set_yticks(y)
        ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
        ax.set_xlim(x_min, x_max)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.50, color=COLORS["grid"])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{panel} remove {label}", loc="left", pad=1.4, fontsize=6.8, fontweight="bold")
        ax.tick_params(length=2.0, pad=1.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["axis"])

    axes[-1].set_xlabel("Full AERIS minus ablated PDR (percentage points)")
    axes[-1].text(
        1.0,
        -0.46,
        "parentheses = significant node scales after Holm correction",
        transform=axes[-1].transAxes,
        ha="right",
        va="top",
        fontsize=5.1,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.20, right=0.965, top=0.965, bottom=0.19, hspace=0.31)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
