#!/usr/bin/env python3
"""Build a compact NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_DIR = (
    ROOT
    / "ns3_validation"
    / "results"
    / "lcn26_ns3_ablation_combined_20260501_010355_011001"
    / "summary"
)
SUMMARY_FILE = ABLATION_DIR / "ns3_ablation_environment_summary.csv"
OUTPUT_PDF = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf"
OUTPUT_PNG = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABEL = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
MODULES = [
    ("AERIS-noGW", "Gateway", "#C13136"),
    ("AERIS-noCAS", "CAS", "#1C7ABA"),
    ("AERIS-noFair", "CH score", "#9E9E9E"),
]
COLORS = {
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


def build() -> None:
    apply_style()
    summary = load_summary()
    fig, ax = plt.subplots(figsize=(3.50, 2.56))
    x = np.arange(len(ENV_ORDER), dtype=float)
    group_width = 0.72
    bar_width = group_width / len(MODULES)
    offsets = (np.arange(len(MODULES)) - (len(MODULES) - 1) / 2.0) * bar_width
    module_handles = []

    def label_value(value: float) -> str:
        return "+0.0" if abs(value) < 0.05 else f"{value:+.1f}"

    ax.axhline(0.0, color=COLORS["axis"], linewidth=0.75, zorder=1)
    ax.axhspan(-0.12, 0.12, color="#F3F3F3", zorder=0)

    all_lows: list[float] = []
    all_highs: list[float] = []

    for offset, (variant, label, color) in zip(offsets, MODULES):
        vals = np.asarray([summary[(env, variant)][0] for env in ENV_ORDER], dtype=float)
        lo = np.asarray([summary[(env, variant)][1] for env in ENV_ORDER], dtype=float)
        hi = np.asarray([summary[(env, variant)][2] for env in ENV_ORDER], dtype=float)
        all_lows.append(float(np.min(lo)))
        all_highs.append(float(np.max(hi)))
        module_handles.append(
            Patch(facecolor=color, edgecolor="#666666", label=label)
        )

        ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.90,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        err_low = np.maximum(vals - lo, 0.0)
        err_high = np.maximum(hi - vals, 0.0)
        ax.errorbar(
            x + offset,
            vals,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.70,
            capsize=2.0,
            zorder=4,
        )

        for xi, val in zip(x + offset, vals):
            text_y = val + (0.20 if val >= 0 else -0.20)
            ax.text(
                xi,
                text_y,
                label_value(val),
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=5.35,
                color=COLORS["text"],
                bbox={
                    "boxstyle": "round,pad=0.05",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.74,
                },
                zorder=5,
            )

    low = float(min(min(all_lows), 0.0))
    high = float(max(max(all_highs), 0.0))
    span = max(high - low, 1.0)
    pad = 0.17 * span
    ax.set_ylim(low - pad, high + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.set_xlabel("Environment")
    ax.set_ylabel("Full minus ablated PDR (pp)")
    ax.grid(axis="y", linestyle="--", linewidth=0.50, color=COLORS["grid"])
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=2.0, pad=1.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])

    fig.legend(
        handles=module_handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        columnspacing=0.90,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.15, right=0.985, top=0.82, bottom=0.20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=320)
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUTPUT_PDF}")
    print(f"[LCN26-FIG] wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    build()
