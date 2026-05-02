#!/usr/bin/env python3
"""Build a compact NS-3 boundary figure for the LCN26 draft."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
DUAL_FILE = (
    ROOT
    / "ns3_validation"
    / "results"
    / "lcn26_ns3_dual_combined_20260430_191527_191528"
    / "summary"
    / "ns3_focused_descriptive.csv"
)

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABELS = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
PLOT_PROTOCOLS = ["AERIS", "RPL-MRHOF", "CTP", "PEGASIS"]

COLORS = {
    "AERIS": "#5A5A5A",
    "RPL-MRHOF": "#2D83BD",
    "CTP": "#36A657",
    "PEGASIS": "#C6373D",
    "grid": "#D8DDE3",
    "axis": "#5C6670",
    "text": "#222A33",
}
MARKERS = {"AERIS": "o", "RPL-MRHOF": "s", "CTP": "^", "PEGASIS": "D"}
LABELS = {"AERIS": "AERIS", "RPL-MRHOF": "RPL", "CTP": "CTP", "PEGASIS": "PEG"}


def load_rows() -> dict[tuple[str, int, str], tuple[float, float, int]]:
    data: dict[tuple[str, int, str], tuple[float, float, int]] = {}
    with DUAL_FILE.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data[(row["environment"], int(row["num_nodes"]), row["protocol"])] = (
                float(row["pdr_mean"]),
                float(row["pdr_std"]),
                int(row["n"]),
            )
    return data


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
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
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
            "grid.alpha": 0.82,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.grid(axis="y")


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def build() -> None:
    data = load_rows()
    fig, axes = plt.subplots(2, 2, figsize=(3.52, 3.36), sharex=True, sharey=True)
    axes = axes.flatten()
    x = np.arange(len(NODE_ORDER), dtype=float)
    ticks = ["50", "100", "200", "300", "500", "800", "1k"]

    for ax, env in zip(axes, ENV_ORDER):
        for proto in PLOT_PROTOCOLS:
            y = np.asarray([data[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
            std = np.asarray([data[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
            reps = np.asarray([data[(env, n, proto)][2] for n in NODE_ORDER], dtype=float)
            band = np.asarray([ci95(s, int(r)) for s, r in zip(std, reps)], dtype=float)
            ax.fill_between(x, y - band, y + band, color=COLORS[proto], alpha=0.055, linewidth=0)
            ax.plot(
                x,
                y,
                color=COLORS[proto],
                marker=MARKERS[proto],
                markersize=2.4,
                linewidth=1.55 if proto == "AERIS" else 1.25,
                alpha=0.98,
            )

        style_axis(ax)
        ax.set_title(ENV_LABELS[env], pad=3)
        ax.set_xlim(-0.1, len(NODE_ORDER) - 0.55)
        ax.set_ylim(0.0, 1.04)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks)
        values_1000 = {proto: data[(env, 1000, proto)][0] for proto in PLOT_PROTOCOLS}
        top_proto = max(values_1000, key=values_1000.get)
        ax.text(
            x[-1] + 0.08,
            min(values_1000[top_proto], 0.99),
            f"{LABELS[top_proto]} {values_1000[top_proto]:.2f}",
            ha="left",
            va="center",
            fontsize=5.7,
            color=COLORS[top_proto],
            fontweight="semibold",
            clip_on=False,
        )

    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linewidth=1.45 if p == "AERIS" else 1.2, label=LABELS[p])
        for p in PLOT_PROTOCOLS
    ]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.52, 1.01), frameon=False, columnspacing=0.72, handletextpad=0.28)
    fig.subplots_adjust(top=0.82, left=0.12, right=0.98, bottom=0.14, wspace=0.18, hspace=0.32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_expanded_boundary.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_expanded_boundary.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
