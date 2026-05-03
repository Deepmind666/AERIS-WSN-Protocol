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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.6,
            "axes.labelsize": 6.8,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.2,
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


def build() -> None:
    apply_style()
    summary = load_summary()
    fig, ax = plt.subplots(figsize=(3.52, 2.58))
    y_base = np.arange(len(ENV_ORDER), dtype=float)
    offsets = [-0.26, 0.0, 0.26]
    bar_h = 0.16

    ax.axvline(0, color=COLORS["axis"], linewidth=0.8, linestyle="--", zorder=1)
    for offset, (variant, label) in zip(offsets, VARIANTS):
        vals = np.asarray([summary[(env, variant)][0] for env in ENV_ORDER], dtype=float)
        mins = np.asarray([summary[(env, variant)][1] for env in ENV_ORDER], dtype=float)
        maxs = np.asarray([summary[(env, variant)][2] for env in ENV_ORDER], dtype=float)
        sig = [summary[(env, variant)][3] for env in ENV_ORDER]
        y = y_base + offset
        xerr = np.vstack((np.maximum(vals - mins, 0.0), np.maximum(maxs - vals, 0.0)))
        ax.barh(y, vals, height=bar_h, color=COLORS[label], edgecolor="black", linewidth=0.35, label=label, zorder=3)
        ax.errorbar(vals, y, xerr=xerr, fmt="none", ecolor=COLORS["axis"], elinewidth=0.55, capsize=1.8, zorder=4)
        for yi, val, sig_count in zip(y, vals, sig):
            if abs(val) < 0.45 and sig_count < 5:
                continue
            dx = 0.12
            ax.text(
                val + dx,
                yi,
                f"{val:+.1f} ({sig_count}/7)",
                ha="left",
                va="center",
                fontsize=5.8,
                color=COLORS["muted"],
            )

    ax.set_yticks(y_base)
    ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-2.8, 8.95)
    ax.set_xlabel("Mean PDR contribution (percentage points)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color=COLORS["grid"])
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.legend(
        loc="upper center",
        frameon=True,
        facecolor="white",
        edgecolor=COLORS["grid"],
        framealpha=0.95,
        handletextpad=0.25,
        borderaxespad=0.05,
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.27, right=0.985, top=0.82, bottom=0.17)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
