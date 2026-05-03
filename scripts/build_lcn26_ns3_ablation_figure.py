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
    "Office": "#FF7F0E",
    "Factory": "#F2A65A",
    "Suburban": "#1F77B4",
    "Urban": "#A9C8E8",
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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
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
    fig, axes = plt.subplots(3, 1, figsize=(3.50, 2.82), sharex=True)
    x = np.arange(len(ENV_ORDER), dtype=float)
    env_labels = [ENV_LABEL[e] for e in ENV_ORDER]
    env_colors = [COLORS[ENV_LABEL[e]] for e in ENV_ORDER]

    def pretty_value(value: float) -> str:
        return "+0.0" if abs(value) < 0.05 else f"{value:+.1f}"

    for ax, (variant, label), tag in zip(axes, VARIANTS, ["(a)", "(b)", "(c)"]):
        vals = np.asarray([summary[(env, variant)][0] for env in ENV_ORDER], dtype=float)
        lo = np.asarray([summary[(env, variant)][1] for env in ENV_ORDER], dtype=float)
        hi = np.asarray([summary[(env, variant)][2] for env in ENV_ORDER], dtype=float)
        sig = [summary[(env, variant)][3] for env in ENV_ORDER]
        total = [summary[(env, variant)][4] for env in ENV_ORDER]

        ax.axhline(0.0, color=COLORS["axis"], linewidth=0.70, zorder=1)
        ax.bar(x, vals, width=0.58, color=env_colors, alpha=0.92, edgecolor="white", linewidth=0.45, zorder=3)
        err_low = np.maximum(vals - lo, 0.0)
        err_high = np.maximum(hi - vals, 0.0)
        ax.errorbar(
            x,
            vals,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.70,
            capsize=2.0,
            zorder=4,
        )

        low = float(min(np.min(lo), 0.0))
        high = float(max(np.max(hi), 0.0))
        span = max(high - low, 1.0)
        pad = 0.28 * span
        ax.set_ylim(low - pad, high + pad)

        for xi, val, sc, tc in zip(x, vals, sig, total):
            offset = 0.075 * span
            ax.text(
                xi,
                val + (offset if val >= 0 else -offset),
                f"{pretty_value(val)} ({sc}/{tc})",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=5.2,
                color=COLORS["text"],
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(env_labels)
        ax.grid(axis="y", linestyle="--", linewidth=0.50, color=COLORS["grid"])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{tag} remove {label}", loc="left", pad=1.3, fontsize=6.8, fontweight="bold")
        ax.tick_params(length=2.0, pad=1.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_color(COLORS["axis"])

    fig.text(0.025, 0.54, "Full minus ablated PDR (pp)", rotation=90, ha="center", va="center", fontsize=6.6)
    axes[-1].set_xlabel("Environment")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.965, bottom=0.14, hspace=0.33)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
