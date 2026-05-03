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
    "AERIS": "#1F77B4",
    "PEGASIS": "#FF7F0E",
    "LEACH": "#BDBDBD",
    "HEED": "#8C8C8C",
    "TEEN": "#D9D9D9",
    "grid": "#CFCFCF",
    "axis": "#111111",
    "text": "#111111",
    "muted": "#555555",
}
MARKERS = {"AERIS": "o", "PEGASIS": "s", "LEACH": "^", "HEED": "D", "TEEN": "v"}
LINESTYLES = {
    "AERIS": "-",
    "PEGASIS": "-",
    "LEACH": "--",
    "HEED": "-.",
    "TEEN": ":",
}


def load_csv(path: Path) -> list[dict[str, str]]:
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
            "axes.labelsize": 6.9,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.95,
            "grid.linestyle": "--",
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
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color=COLORS["grid"])


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
    fig, axes = plt.subplots(2, 2, figsize=(3.52, 3.00), sharex=True, sharey=True)
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
            ax.plot(
                x,
                y,
                color=COLORS[proto],
                marker=MARKERS[proto],
                markersize=2.8 if proto in {"AERIS", "PEGASIS"} else 2.2,
                linewidth=1.55 if proto in {"AERIS", "PEGASIS"} else 0.9,
                linestyle=LINESTYLES[proto],
                alpha=1.0 if proto in {"AERIS", "PEGASIS"} else 0.82,
                markeredgecolor="black" if proto in {"AERIS", "PEGASIS"} else COLORS[proto],
                markeredgewidth=0.25 if proto in {"AERIS", "PEGASIS"} else 0.0,
            )
        style_axes(ax)
        ax.set_title(ENV_LABELS[env], pad=2)
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
                fontsize=5.3,
                color=COLORS[proto],
                fontweight="bold" if proto == "AERIS" else "regular",
                clip_on=False,
            )
    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p], linewidth=1.5 if p in {"AERIS", "PEGASIS"} else 0.9, label=p)
        for p in PROTO_ORDER
    ]
    fig.legend(handles=handles, labels=PROTO_ORDER, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.985), frameon=True, facecolor="white", edgecolor=COLORS["grid"], framealpha=0.95, columnspacing=0.45, handletextpad=0.18)
    fig.subplots_adjust(top=0.79, left=0.13, right=0.975, bottom=0.15, wspace=0.18, hspace=0.28)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_strict_compact.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_strict_compact.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
