#!/usr/bin/env python3
"""Build the compact NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_DIR = (
    ROOT
    / "ns3_validation"
    / "results"
    / "lcn26_ns3_ablation_combined_20260501_010355_011001"
    / "summary"
)
DELTA_FILE = ABLATION_DIR / "ns3_ablation_delta.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from lcn26_style import COLUMN_WIDTH_IN, PALETTE, apply_lcn26_style  # noqa: E402

ENV_ORDER = [
    "indoor_office",
    "indoor_factory",
    "outdoor_suburban",
    "outdoor_urban",
]
ENV_LABEL = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
VARIANTS = [
    ("AERIS-noGW", "-GW"),
    ("AERIS-noCAS", "-CAS"),
    ("AERIS-noFair", "-CH score"),
]
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
NODE_LABELS = ["50", "100", "200", "300", "500", "800", "1k"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def load_cell_deltas() -> dict[tuple[str, str, int], tuple[float, bool]]:
    cells: dict[tuple[str, str, int], tuple[float, bool]] = {}
    for row in load_rows(DELTA_FILE):
        contribution = -float(row["delta_points"])
        sig_flag = row["significant_005"].strip().lower()
        cells[(row["environment"], row["variant"], int(row["num_nodes"]))] = (
            contribution,
            sig_flag in {"true", "yes", "1"},
        )
    return cells


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)


def build() -> None:
    apply_lcn26_style()
    plt.rcParams.update(
        {
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.9,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.5,
        }
    )

    cells = load_cell_deltas()
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH_IN, 2.62), sharex=True)
    norm = TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0)
    last_heat = None

    for panel, ax, (variant, label) in zip(["(a)", "(b)", "(c)"], axes, VARIANTS):
        matrix = np.asarray(
            [[cells[(env, variant, node)][0] for node in NODE_ORDER] for env in ENV_ORDER],
            dtype=float,
        )
        sig_matrix = np.asarray(
            [[cells[(env, variant, node)][1] for node in NODE_ORDER] for env in ENV_ORDER],
            dtype=bool,
        )
        last_heat = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_yticks(np.arange(len(ENV_ORDER)))
        ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
        ax.set_title(f"{panel} {label}", loc="left", pad=1.0, fontsize=6.9, fontweight="bold")
        ax.set_xticks(np.arange(len(NODE_ORDER)))
        ax.set_xticklabels(NODE_LABELS)
        ax.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ENV_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", pad=1.2)
        ax.tick_params(axis="x", pad=1.0)
        sig_y, sig_x = np.where(sig_matrix)
        ax.scatter(
            sig_x,
            sig_y,
            marker="o",
            s=15,
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            zorder=5,
        )
        style_axes(ax)

    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    if last_heat is not None:
        cax = fig.add_axes([0.18, 0.075, 0.76, 0.04])
        cbar = fig.colorbar(last_heat, cax=cax, orientation="horizontal")
        cbar.set_label("Full minus ablated PDR (pp)", fontsize=5.4, labelpad=1)
        cbar.ax.tick_params(labelsize=5.2, length=1.8, pad=1)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.965, bottom=0.18, hspace=0.20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf"
    out_png = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {out_pdf}")
    print(f"[LCN26-FIG] wrote {out_png}")


if __name__ == "__main__":
    build()
