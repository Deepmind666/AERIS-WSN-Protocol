#!/usr/bin/env python3
"""Refresh the LCN26 tradeoff/mechanism figure from the corrected result bundle."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ENERGY_FILE = ROOT / "results" / "mega_experiments" / "energy_lifetime_stats.csv"
LATENCY_FILE = ROOT / "results" / "mega_experiments" / "latency_hop_v3_20260211_stats.csv"
MECH_FILE = ROOT / "results" / "lcn26_targeted_20260420" / "mechanism_grid_fat" / "mechanism_summary.csv"

PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_SHORT = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [100, 500, 1000]
COLORS = {
    "AERIS": "#2F5D7C",
    "PEGASIS": "#B07A8F",
    "HND": "#6C7A99",
    "LEACH": "#D58A5B",
    "HEED": "#7AA08D",
    "TEEN": "#C7A74D",
    "CAS": "#3E8E9B",
    "GW": "#D56C5B",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "muted": "#7A8794",
    "benign_bg": "#FAF3F0",
    "harsh_bg": "#F2F7F5",
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
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 9.2,
            "axes.labelsize": 9.6,
            "axes.titlesize": 10.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.3,
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


def save(fig: plt.Figure, stem: str, close: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png")
    if close:
        plt.close(fig)


def build() -> None:
    energy_rows = load_csv(ENERGY_FILE)
    latency_rows = load_csv(LATENCY_FILE)
    mech_rows = {(row["environment"], int(row["num_nodes"])): row for row in load_csv(MECH_FILE)}

    avg_metrics: dict[str, dict[str, float]] = {}
    for proto in PROTO_ORDER:
        subset = [r for r in energy_rows if r["protocol"] == proto]
        lat_subset = [r for r in latency_rows if r["protocol"] == proto]
        avg_metrics[proto] = {
            "pdr": float(np.mean([float(r["pdr_mean"]) for r in subset])),
            "energy": float(np.mean([float(r["energy_mean"]) for r in subset])),
            "life": float(np.mean([float(r["lifetime_mean"]) for r in subset])),
            "fnd": float(np.mean([float(r["fnd_mean"]) for r in subset])),
            "hops": float(np.mean([float(r["hops_mean"]) for r in lat_subset])),
        }

    metric_order = ["pdr", "energy", "life", "fnd", "hops"]
    metric_labels = ["PDR ↑", "Energy ↓", "Lifetime ↑", "FND ↑", "Hops ↓"]
    better = {"pdr": "high", "energy": "low", "life": "high", "fnd": "high", "hops": "low"}

    fig = plt.figure(figsize=(7.1, 5.95))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.90, 0.10, 1.50], hspace=0.08)
    ax_table = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    bottom = gs[2].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.10)
    ax_plot = fig.add_subplot(bottom[0])
    ax_bottom = fig.add_subplot(bottom[1], sharex=ax_plot)

    ax_table.axis("off")
    table_data = []
    for proto in PROTO_ORDER:
        table_data.append(
            [
                proto,
                f"{avg_metrics[proto]['pdr']:.3f}",
                f"{avg_metrics[proto]['energy']:.1f}",
                f"{avg_metrics[proto]['life']:.1f}",
                f"{avg_metrics[proto]['fnd']:.1f}",
                f"{avg_metrics[proto]['hops']:.2f}",
            ]
        )
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Methods"] + metric_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.18, 1.0, 0.76],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D6DCE4")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#F4F7FA")
            cell.set_text_props(fontweight="semibold", color=COLORS["text"])
        elif c == 0:
            cell.set_text_props(fontweight="semibold", color=COLORS["text"])

    rank_colors = [COLORS["best"], COLORS["second"], COLORS["third"]]
    for j, metric in enumerate(metric_order, start=1):
        vals = [(proto, avg_metrics[proto][metric]) for proto in PROTO_ORDER]
        vals = sorted(vals, key=lambda item: item[1], reverse=(better[metric] == "high"))
        for rank_idx, (proto, _) in enumerate(vals[:3]):
            row_idx = PROTO_ORDER.index(proto) + 1
            tbl[(row_idx, j)].set_facecolor(rank_colors[rank_idx])
            if rank_idx == 0:
                tbl[(row_idx, j)].set_text_props(fontweight="bold")

    ax_table.text(
        0.0,
        0.98,
        "(a) Pooled 100-node protocol summary",
        transform=ax_table.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.9,
        fontweight="semibold",
    )
    ax_leg.axis("off")

    cells = [(env, nodes) for env in ENV_ORDER for nodes in NODE_ORDER]
    x = np.arange(len(cells), dtype=float)
    pdr = np.asarray([float(mech_rows[(env, nodes)]["pdr_expected_mean"]) for env, nodes in cells], dtype=float)
    gw = np.asarray([float(mech_rows[(env, nodes)]["gateway_uplink_pdr_total_mean"]) for env, nodes in cells], dtype=float)
    fnd = np.asarray([float(mech_rows[(env, nodes)]["first_node_death_round_mean"]) for env, nodes in cells], dtype=float)
    hnd = np.asarray([float(mech_rows[(env, nodes)]["half_nodes_death_round_mean"]) for env, nodes in cells], dtype=float)
    cas_direct = np.asarray([float(mech_rows[(env, nodes)]["cas_DIRECT_mean"]) for env, nodes in cells], dtype=float)
    cas_twohop = np.asarray([float(mech_rows[(env, nodes)]["cas_TWO_HOP_mean"]) for env, nodes in cells], dtype=float)
    cas_chain = np.asarray([float(mech_rows[(env, nodes)]["cas_CHAIN_mean"]) for env, nodes in cells], dtype=float)
    cas_total = np.maximum(cas_direct + cas_twohop + cas_chain, 1e-9)
    twohop_share = cas_twohop / cas_total

    group_bg = [COLORS["benign_bg"], "#F7F8FA", COLORS["harsh_bg"], "#F7F8FA"]
    for group_idx, env in enumerate(ENV_ORDER):
        start = group_idx * len(NODE_ORDER) - 0.5
        stop = start + len(NODE_ORDER)
        ax_plot.axvspan(start, stop, color=group_bg[group_idx], alpha=0.95, zorder=0)
        ax_plot.text(
            group_idx * len(NODE_ORDER) + 1.0,
            0.93,
            ENV_SHORT[env],
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="semibold",
            color=COLORS["muted"],
            transform=ax_plot.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        )
        if group_idx > 0:
            ax_plot.axvline(start, color="#C7D0DA", linestyle="--", linewidth=0.8, zorder=1)

    ax_plot.bar(x, pdr, width=0.54, color=COLORS["AERIS"], alpha=0.72, edgecolor="white", linewidth=0.7, zorder=3)
    ax_plot.plot(x, gw, color=COLORS["GW"], marker="o", markersize=4.0, linewidth=1.8, zorder=4)
    style_axes(ax_plot)
    ax_plot.set_ylabel("PDR / gateway uplink")
    ax_plot.set_ylim(0.0, 1.02)
    ax_plot.set_xlim(-0.6, len(cells) - 0.4)
    ax_plot.set_xticks(x)
    ax_plot.tick_params(axis="x", labelbottom=False)

    highlight_bar_idx = {0, 2, 3, 5, 6, 8, 9, 11}
    for idx, (xi, yi) in enumerate(zip(x, pdr)):
        if idx not in highlight_bar_idx:
            continue
        if yi >= 0.86:
            ax_plot.text(xi, yi - 0.035, f"{yi:.2f}", ha="center", va="top", fontsize=7.0, color="white", fontweight="semibold")
        else:
            ax_plot.text(xi, yi + 0.018, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.0, color=COLORS["AERIS"])

    style_axes(ax_bottom)
    ax_bottom.plot(x, fnd, color=COLORS["PEGASIS"], marker="s", markersize=4.0, linewidth=1.7, linestyle="--", zorder=5)
    ax_bottom.plot(x, hnd, color=COLORS["HND"], marker="^", markersize=4.0, linewidth=1.5, linestyle=":", zorder=5)
    ax_bottom.set_ylabel("FND / HND")
    ax_bottom.set_ylim(0, max(float(np.max(hnd)) * 1.08, 16.0))
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(nodes) for _, nodes in cells], fontsize=8.3)
    ax_bottom.set_xlabel("Nodes within each environment group")

    ax_bottom_r = ax_bottom.twinx()
    ax_bottom_r.plot(x, twohop_share * 100.0, color=COLORS["CAS"], marker="D", markersize=3.8, linewidth=1.6, linestyle="-.", zorder=5)
    ax_bottom_r.set_ylabel("Two-hop share (%)")
    ax_bottom_r.set_ylim(0, 100)
    ax_bottom_r.spines["top"].set_visible(False)
    ax_bottom_r.spines["left"].set_visible(False)
    ax_bottom_r.spines["right"].set_color(COLORS["axis"])
    ax_bottom_r.tick_params(axis="y", colors=COLORS["axis"])

    for idx, (xi, yi) in enumerate(zip(x, fnd)):
        if idx in {0, 3, 6, 9, 11}:
            ax_bottom.text(xi, yi + 0.7, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.7, color=COLORS["PEGASIS"])
    for idx, (xi, yi) in enumerate(zip(x, hnd)):
        if idx in {0, 4, 7, 10, 11}:
            ax_bottom.text(xi, yi + 2.2, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.5, color=COLORS["HND"])

    urban_1000_idx = cells.index(("outdoor_urban", 1000))
    ax_plot.annotate(
        "GW bottleneck",
        xy=(urban_1000_idx, gw[urban_1000_idx]),
        xytext=(urban_1000_idx - 1.9, 0.43),
        textcoords="data",
        fontsize=7.5,
        color=COLORS["GW"],
        arrowprops=dict(arrowstyle="->", color=COLORS["GW"], linewidth=0.9),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#D7DEE7", alpha=0.94),
    )

    handles = [
        Line2D([0], [0], color=COLORS["AERIS"], linewidth=6, alpha=0.72, label="AERIS PDR"),
        Line2D([0], [0], color=COLORS["GW"], marker="o", linewidth=1.8, label="GW uplink"),
        Line2D([0], [0], color=COLORS["PEGASIS"], marker="s", linewidth=1.7, linestyle="--", label="FND"),
        Line2D([0], [0], color=COLORS["HND"], marker="^", linewidth=1.5, linestyle=":", label="HND"),
        Line2D([0], [0], color=COLORS["CAS"], marker="D", linewidth=1.6, linestyle="-.", label="Two-hop share"),
    ]
    ax_leg.legend(handles=handles, ncol=5, loc="center", frameon=False, columnspacing=0.9, handletextpad=0.5)
    save(fig, "fig_lcn26_tradeoff_cv")


if __name__ == "__main__":
    apply_style()
    build()
