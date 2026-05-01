#!/usr/bin/env python3
"""Build the expanded NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_DIR = ROOT / "ns3_validation" / "results" / "lcn26_ns3_ablation_combined_20260501_010355_011001" / "summary"
DELTA_FILE = ABLATION_DIR / "ns3_ablation_delta.csv"
SUMMARY_FILE = ABLATION_DIR / "ns3_ablation_environment_summary.csv"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABEL = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
VARIANTS = [
    ("AERIS-noGW", "Gateway"),
    ("AERIS-noCAS", "CAS"),
    ("AERIS-noFair", "CH score"),
]

COLORS = {
    "text": "#24323F",
    "axis": "#52616E",
    "muted": "#7A8794",
    "grid": "#D7DEE7",
    "pos": "#2F6F7E",
    "neg": "#B65F6B",
    "zero": "#F5F7FA",
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
            "font.size": 8.4,
            "axes.labelsize": 8.7,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
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
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["text"],
    )


def contribution_tables() -> tuple[dict[tuple[str, int, str], dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    cell: dict[tuple[str, int, str], dict[str, object]] = {}
    for row in load_rows(DELTA_FILE):
        env = row["environment"]
        nodes = int(row["num_nodes"])
        variant = row["variant"]
        # Stored delta is variant-full. Contribution is full-variant.
        contribution = -float(row["delta_points"])
        cell[(env, nodes, variant)] = {
            "contribution": contribution,
            "significant": row["significant_005"] == "YES",
        }

    summary: dict[tuple[str, str], dict[str, object]] = {}
    for row in load_rows(SUMMARY_FILE):
        env = row["environment"]
        variant = row["variant"]
        summary[(env, variant)] = {
            "mean_contribution": -float(row["mean_delta_points"]),
            "significant_cells": int(row["significant_cells"]),
            "cells": int(row["cells"]),
        }
    return cell, summary


def draw_heatmap(ax: plt.Axes, cell: dict[tuple[str, int, str], dict[str, object]], variant: str, title: str, cmap, norm) -> None:
    for y, env in enumerate(ENV_ORDER):
        for x, nodes in enumerate(NODE_ORDER):
            item = cell[(env, nodes, variant)]
            val = float(item["contribution"])
            sig = bool(item["significant"])
            rect = Rectangle((x, y), 1, 1, facecolor=cmap(norm(val)), edgecolor="white", linewidth=0.9)
            ax.add_patch(rect)
            ax.text(
                x + 0.5,
                y + 0.52,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=6.4,
                fontweight="bold" if sig else "normal",
                color=COLORS["text"],
            )
            if sig:
                ax.plot(x + 0.83, y + 0.17, marker="o", markersize=2.0, color=COLORS["text"], markeredgewidth=0)

    ax.set_xlim(0, len(NODE_ORDER))
    ax.set_ylim(0, len(ENV_ORDER))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(NODE_ORDER)) + 0.5)
    ax.set_xticklabels([str(n) for n in NODE_ORDER], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(ENV_ORDER)) + 0.5)
    ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, pad=6)


def draw_environment_summary(ax: plt.Axes, summary: dict[tuple[str, str], dict[str, object]]) -> None:
    y = np.arange(len(ENV_ORDER), dtype=float)
    offsets = [-0.22, 0.0, 0.22]
    colors = ["#2F6F7E", "#7CA6B2", "#B7C0CA"]

    ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    for idx, (variant, label) in enumerate(VARIANTS):
        vals = [float(summary[(env, variant)]["mean_contribution"]) for env in ENV_ORDER]
        sig = [f"{summary[(env, variant)]['significant_cells']}/7" for env in ENV_ORDER]
        ax.scatter(vals, y + offsets[idx], s=32, color=colors[idx], edgecolor="white", linewidth=0.5, zorder=3, label=label)
        for val, yy, sig_label in zip(vals, y + offsets[idx], sig):
            ax.plot([0, val], [yy, yy], color=colors[idx], linewidth=1.5, alpha=0.75, zorder=2)
            if abs(val) >= 0.55:
                ha = "left" if val >= 0 else "right"
                dx = 0.12 if val >= 0 else -0.12
                ax.text(val + dx, yy, sig_label, ha=ha, va="center", fontsize=6.8, color=COLORS["muted"])

    ax.set_yticks(y)
    ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Mean contribution (PDR points)")
    ax.set_xlim(-2.2, 8.8)
    ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="y", length=0)
    ax.set_title("Environment-level effect", pad=6)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.2)


def build() -> None:
    apply_style()
    cell, summary = contribution_tables()
    cmap = LinearSegmentedColormap.from_list("contribution", [COLORS["neg"], COLORS["zero"], COLORS["pos"]])
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=8.0)

    fig = plt.figure(figsize=(7.15, 4.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.02, 1.0], hspace=0.43, wspace=0.35)
    ax_gateway = fig.add_subplot(gs[0, 0])
    ax_cas = fig.add_subplot(gs[0, 1])
    ax_ch = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    draw_heatmap(ax_gateway, cell, "AERIS-noGW", "Gateway contribution", cmap, norm)
    draw_heatmap(ax_cas, cell, "AERIS-noCAS", "CAS contribution", cmap, norm)
    draw_heatmap(ax_ch, cell, "AERIS-noFair", "CH-score contribution", cmap, norm)
    draw_environment_summary(ax_summary, summary)

    ax_gateway.set_xlabel("")
    ax_cas.set_xlabel("")
    ax_gateway.set_xticklabels([])
    ax_cas.set_xticklabels([])
    ax_cas.set_yticklabels([])
    ax_ch.set_xlabel("")

    panel_label(ax_gateway, "(a)", x=-0.13)
    panel_label(ax_cas, "(b)", x=-0.11)
    panel_label(ax_ch, "(c)", x=-0.13)
    panel_label(ax_summary, "(d)", x=-0.12)

    cax = fig.add_axes([0.14, 0.035, 0.38, 0.018])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Full AERIS minus ablated variant (PDR points); dot = Holm p < 0.05", fontsize=7.0, labelpad=2)
    cbar.ax.tick_params(labelsize=6.6, length=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png")
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.pdf'}")
    print(f"[LCN26-FIG] wrote {OUT_DIR / 'fig_lcn26_ns3_ablation_expanded.png'}")


if __name__ == "__main__":
    build()
