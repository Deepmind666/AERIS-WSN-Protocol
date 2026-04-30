#!/usr/bin/env python3
"""Refresh the corrected canonical NS-3 figure from the valid 2026-04-20 rerun."""

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
NS3_FILE = ROOT / "ns3_validation" / "results" / "lcn26_ns3_audit_20260420_012811" / "summary" / "ns3_focused_descriptive.csv"
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABELS = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_suburban": "Outdoor Suburban",
    "outdoor_urban": "Outdoor Urban",
}
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
PROTO_DRAW_ORDER = ["LEACH", "HEED", "TEEN", "PEGASIS", "AERIS"]
NODE_ORDER = [100, 500, 1000]
COLORS = {
    "AERIS": "#2F5D7C",
    "PEGASIS": "#B07A8F",
    "LEACH": "#D58A5B",
    "HEED": "#7AA08D",
    "TEEN": "#C7A74D",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "benign_bg": "#FAF3F0",
    "harsh_bg": "#F2F7F5",
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
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.2,
            "axes.labelsize": 10.4,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
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
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y")


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def load_ns3() -> dict[tuple[str, int, str], tuple[float, float, int]]:
    data = {}
    for row in load_csv(NS3_FILE):
        data[(row["environment"], int(row["num_nodes"]), row["protocol"])] = (
            float(row["pdr_mean"]),
            float(row["pdr_std"]),
            int(row["n"]),
        )
    return data


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D7DEE7", alpha=0.95),
    )


def draw_panel(ax: plt.Axes, env: str, data: dict[tuple[str, int, str], tuple[float, float, int]]) -> None:
    ax.set_facecolor(COLORS["benign_bg"] if env == "indoor_office" else COLORS["harsh_bg"])
    x = np.arange(len(NODE_ORDER), dtype=float)
    for proto in PROTO_DRAW_ORDER:
        y = np.asarray([data[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
        s = np.asarray([data[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
        nrep = np.asarray([data[(env, n, proto)][2] for n in NODE_ORDER], dtype=float)
        band = np.asarray([ci95(si, int(ni)) for si, ni in zip(s, nrep)], dtype=float)
        width = 2.5 if proto == "AERIS" else 2.1 if proto == "PEGASIS" else 1.4
        alpha = 1.0 if proto in {"AERIS", "PEGASIS"} else 0.72
        z = 4 if proto == "AERIS" else 3 if proto == "PEGASIS" else 2
        if proto in {"AERIS", "PEGASIS"}:
            ax.fill_between(x, y - band, y + band, color=COLORS[proto], alpha=0.16 if proto == "AERIS" else 0.12, linewidth=0)
        ax.plot(
            x,
            y,
            color=COLORS[proto],
            marker=MARKERS[proto],
            markersize=4.2 if proto in {"AERIS", "PEGASIS"} else 3.4,
            linewidth=width,
            linestyle=LINESTYLES[proto],
            alpha=alpha,
            zorder=z,
        )
    style_axes(ax)
    ax.set_title(ENV_LABELS[env], pad=7)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.05, len(NODE_ORDER) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NODE_ORDER])
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    last_node = NODE_ORDER[-1]
    for proto in ["AERIS", "PEGASIS"]:
        y_end = data[(env, last_node, proto)][0]
        ax.text(x[-1] + 0.06, y_end, f"{y_end:.2f}", va="center", ha="left", fontsize=8.2, color=COLORS[proto], fontweight="semibold")


def build() -> None:
    data = load_ns3()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.05), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, env in enumerate(ENV_ORDER):
        draw_panel(axes[idx], env, data)
        panel_label(axes[idx], f"({chr(97 + idx)})")
    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Number of nodes")
    axes[3].set_xlabel("Number of nodes")
    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p], linewidth=2.3 if p == "AERIS" else 2.0 if p == "PEGASIS" else 1.4, label=p)
        for p in PROTO_ORDER
    ]
    fig.legend(handles=handles, labels=PROTO_ORDER, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, handletextpad=0.5, columnspacing=1.0)
    fig.subplots_adjust(top=0.83, left=0.08, right=0.99, bottom=0.11, wspace=0.12, hspace=0.24)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_canonical.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_canonical.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
