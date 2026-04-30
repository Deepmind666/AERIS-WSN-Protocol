#!/usr/bin/env python3
"""Build a compact single-column strict-physics figure for the LCN draft."""

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
STRICT_FILE = ROOT / "results" / "mega_experiments" / "scalability_4env_v50rigor_20260222_descriptive.csv"
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABELS = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburb",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [100, 200, 300, 500, 800, 1000]
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
DRAW_ORDER = ["LEACH", "HEED", "TEEN", "PEGASIS", "AERIS"]
COLORS = {
    "AERIS": "#2F5D7C",
    "PEGASIS": "#B07A8F",
    "LEACH": "#D58A5B",
    "HEED": "#7AA08D",
    "TEEN": "#C7A74D",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "muted": "#7A8794",
}
MARKERS = {"AERIS": "o", "PEGASIS": "s", "LEACH": "^", "HEED": "D", "TEEN": "P"}
LINESTYLES = {
    "AERIS": "-",
    "PEGASIS": "--",
    "LEACH": (0, (4, 2)),
    "HEED": (0, (2, 2)),
    "TEEN": (0, (1.5, 1.5)),
}


def load_csv(path: Path) -> list[dict[str, str]]:
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
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["text"],
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.grid(axis="y")


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def load_data() -> dict[tuple[str, int, str], tuple[float, float, int]]:
    data = {}
    for row in load_csv(STRICT_FILE):
        data[(row["environment"], int(row["num_nodes"]), row["protocol"])] = (
            float(row["pdr_mean"]),
            float(row["pdr_std"]),
            int(row["n"]),
        )
    return data


def build() -> None:
    data = load_data()
    fig, axes = plt.subplots(2, 2, figsize=(3.52, 3.62), sharex=True, sharey=True)
    axes = axes.flatten()
    x = np.arange(len(NODE_ORDER), dtype=float)
    compact_ticks = ["100", "", "300", "500", "", "1000"]
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for proto in DRAW_ORDER:
            y = np.asarray([data[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
            s = np.asarray([data[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
            nrep = np.asarray([data[(env, n, proto)][2] for n in NODE_ORDER], dtype=float)
            band = np.asarray([ci95(si, int(ni)) for si, ni in zip(s, nrep)], dtype=float)
            if proto in {"AERIS", "PEGASIS"}:
                ax.fill_between(x, y - band, y + band, color=COLORS[proto], alpha=0.14 if proto == "AERIS" else 0.10, linewidth=0)
            ax.plot(
                x,
                y,
                color=COLORS[proto],
                marker=MARKERS[proto],
                markersize=2.6 if proto in {"AERIS", "PEGASIS"} else 2.2,
                linewidth=1.4 if proto == "AERIS" else 1.2 if proto == "PEGASIS" else 0.95,
                linestyle=LINESTYLES[proto],
                alpha=1.0 if proto in {"AERIS", "PEGASIS"} else 0.72,
            )
        style_axes(ax)
        ax.set_title(ENV_LABELS[env], pad=3)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(-0.12, len(NODE_ORDER) - 0.42)
        ax.set_xticks(x)
        ax.set_xticklabels(compact_ticks, rotation=0, ha="center")
        last_vals = {proto: data[(env, NODE_ORDER[-1], proto)][0] for proto in ["AERIS", "PEGASIS"]}
        offsets = {"AERIS": 0.0, "PEGASIS": 0.0}
        if abs(last_vals["AERIS"] - last_vals["PEGASIS"]) < 0.08:
            offsets = {"AERIS": 0.035, "PEGASIS": -0.025}
        for proto in ["AERIS", "PEGASIS"]:
            y_last = data[(env, NODE_ORDER[-1], proto)][0]
            ax.text(
                x[-1] + 0.08,
                min(max(y_last + offsets[proto], 0.03), 0.99),
                f"{y_last:.2f}",
                ha="left",
                va="center",
                fontsize=5.7,
                color=COLORS[proto],
                fontweight="semibold",
                clip_on=False,
            )
    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p], linewidth=1.4 if p == "AERIS" else 1.2 if p == "PEGASIS" else 0.95, label=p)
        for p in PROTO_ORDER
    ]
    fig.legend(handles=handles, labels=PROTO_ORDER, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False, columnspacing=0.55, handletextpad=0.22)
    fig.subplots_adjust(top=0.79, left=0.12, right=0.98, bottom=0.14, wspace=0.16, hspace=0.30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_strict_compact.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_strict_compact.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
