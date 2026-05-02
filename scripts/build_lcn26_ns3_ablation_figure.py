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
    "Gateway": "#36A657",
    "CAS": "#2D83BD",
    "CH score": "#D15B9A",
    "axis": "#5C6670",
    "grid": "#D8DDE3",
    "text": "#222A33",
    "muted": "#77818B",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.0,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.8,
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
            "grid.alpha": 0.80,
        }
    )


def load_summary() -> dict[tuple[str, str], tuple[float, int, int]]:
    summary: dict[tuple[str, str], tuple[float, int, int]] = {}
    for row in load_rows(SUMMARY_FILE):
        # Stored delta is ablated-full. Contribution is full-ablated.
        contribution = -float(row["mean_delta_points"])
        summary[(row["environment"], row["variant"])] = (
            contribution,
            int(row["significant_cells"]),
            int(row["cells"]),
        )
    return summary


def build() -> None:
    apply_style()
    summary = load_summary()
    fig, ax = plt.subplots(figsize=(3.52, 2.45))
    y_base = np.arange(len(ENV_ORDER), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    markers = {"Gateway": "o", "CAS": "s", "CH score": "D"}

    ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    for offset, (variant, label) in zip(offsets, VARIANTS):
        vals = np.asarray([summary[(env, variant)][0] for env in ENV_ORDER], dtype=float)
        sig = [summary[(env, variant)][1] for env in ENV_ORDER]
        y = y_base + offset
        for yi, val in zip(y, vals):
            ax.plot([0, val], [yi, yi], color=COLORS[label], linewidth=1.55, alpha=0.82, solid_capstyle="round")
        ax.scatter(vals, y, s=28, marker=markers[label], color=COLORS[label], edgecolor="white", linewidth=0.55, zorder=3, label=label)
        for yi, val, sig_count in zip(y, vals, sig):
            if label == "Gateway" or (label == "CAS" and val > 0.5):
                ha = "left"
                dx = 0.12
                ax.text(val + dx, yi, f"{val:+.1f} ({sig_count}/7)", ha=ha, va="center", fontsize=6.2, color=COLORS["muted"])

    ax.set_yticks(y_base)
    ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-2.6, 8.8)
    ax.set_xlabel("Mean PDR contribution (points)")
    ax.grid(axis="x")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.35, borderaxespad=0.1)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
