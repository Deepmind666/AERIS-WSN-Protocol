#!/usr/bin/env python3
"""Build the compact mechanism figure for the LCN draft."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
MECH_FILE = ROOT / "results" / "lcn26_targeted_20260420" / "mechanism_grid_fat" / "mechanism_summary.csv"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_SHORT = {"indoor_office": "Office", "indoor_factory": "Factory", "outdoor_suburban": "Suburb", "outdoor_urban": "Urban"}
ENV_TAG = {"indoor_office": "O", "indoor_factory": "F", "outdoor_suburban": "S", "outdoor_urban": "U"}
NODE_ORDER = [100, 500, 1000]
COLORS = {
    "AERIS": "#C13136",
    "PEGASIS": "#1C7ABA",
    "GW": "#1C7ABA",
    "CAS": "#FF7F0E",
    "Office": "#FF7F0E",
    "Factory": "#F2A65A",
    "Suburb": "#1F77B4",
    "Urban": "#A9C8E8",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "muted": "#7A8794",
    "best": "#FFF2A8",
    "second": "#F6DFC2",
    "third": "#F4CED6",
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
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)


def build_mechanism_compact() -> None:
    mech_rows = {(row["environment"], int(row["num_nodes"])): row for row in load_csv(MECH_FILE)}
    pdr = np.asarray([[float(mech_rows[(env, n)]["pdr_expected_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    pdr_std = np.asarray([[float(mech_rows[(env, n)]["pdr_expected_std"]) for n in NODE_ORDER] for env in ENV_ORDER])
    gw = np.asarray([[float(mech_rows[(env, n)]["gateway_uplink_pdr_total_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    gw_std = np.asarray([[float(mech_rows[(env, n)]["gateway_uplink_pdr_total_std"]) for n in NODE_ORDER] for env in ENV_ORDER])
    fnd = np.asarray([[float(mech_rows[(env, n)]["first_node_death_round_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    fnd_std = np.asarray([[float(mech_rows[(env, n)]["first_node_death_round_std"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_direct = np.asarray([[float(mech_rows[(env, n)]["cas_DIRECT_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_twohop = np.asarray([[float(mech_rows[(env, n)]["cas_TWO_HOP_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_chain = np.asarray([[float(mech_rows[(env, n)]["cas_CHAIN_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_total = np.maximum(cas_direct + cas_twohop + cas_chain, 1e-9)
    twohop_share = cas_twohop / cas_total * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(3.50, 2.76), sharex=True)
    x = np.arange(len(NODE_ORDER), dtype=float)
    xticklabels = ["100", "500", "1k"]
    env_labels = [ENV_SHORT[e] for e in ENV_ORDER]
    env_colors = [COLORS[name] for name in env_labels]
    env_markers = {"Office": "o", "Factory": "s", "Suburb": "^", "Urban": "D"}
    offsets = np.linspace(-1.5, 1.5, len(ENV_ORDER)) * 0.17

    def ci95(std: np.ndarray, n: int = 400) -> np.ndarray:
        return 1.96 * std / np.sqrt(n)

    panels = [
        (axes[0, 0], "(a) End-to-end PDR", pdr, ci95(pdr_std), "Mean PDR", (0.0, 1.03)),
        (axes[0, 1], "(b) Gateway uplink PDR", gw, ci95(gw_std), "Mean PDR", (0.0, 1.03)),
        (axes[1, 0], "(c) First-node death", fnd, ci95(fnd_std), "Round", (0.0, max(15.0, float(np.max(fnd)) + 1.0))),
        (axes[1, 1], "(d) CAS two-hop share", twohop_share, None, "Share (%)", (0.0, 80.0)),
    ]

    for ax, title, matrix, err_matrix, ylabel, ylim in panels:
        for env_idx, env_name in enumerate(env_labels):
            ax.bar(
                x + offsets[env_idx],
                matrix[env_idx],
                width=0.17,
                color=env_colors[env_idx],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.90,
                label=env_name if ax is axes[0, 0] else None,
                zorder=3,
            )
            if err_matrix is not None:
                ax.errorbar(
                    x + offsets[env_idx],
                    matrix[env_idx],
                    yerr=err_matrix[env_idx],
                    fmt="none",
                    ecolor="#3A4756",
                    elinewidth=0.65,
                    capsize=1.8,
                    zorder=4,
                )
        ax.text(
            0.5,
            1.01,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.40, len(NODE_ORDER) - 0.12)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.grid(axis="y", linestyle="--", linewidth=0.50, color=COLORS["grid"])
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_color(COLORS["axis"])
        ax.tick_params(length=2.1, pad=1.4, labelsize=5.8)

    axes[0, 1].annotate(
        "urban-1k\nbottleneck",
        xy=(x[-1] + offsets[3], gw[ENV_ORDER.index("outdoor_urban"), -1]),
        xytext=(x[-1] + 0.48, 0.50),
        arrowprops={"arrowstyle": "-", "color": COLORS["Urban"], "linewidth": 0.60},
        fontsize=5.1,
        color=COLORS["axis"],
        ha="left",
        va="center",
    )
    axes[1, 0].annotate(
        "early FND",
        xy=(x[1] + offsets[3], fnd[ENV_ORDER.index("outdoor_urban"), 1]),
        xytext=(x[1] + 0.22, 7.1),
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.55},
        fontsize=5.1,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.52, 0.998), frameon=False, columnspacing=0.85, handletextpad=0.25)
    axes[1, 0].set_xlabel("Nodes")
    axes[1, 1].set_xlabel("Nodes")
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.15, wspace=0.28, hspace=0.38)
    save(fig, "fig_lcn26_mechanism_compact")


if __name__ == "__main__":
    apply_style()
    build_mechanism_compact()
