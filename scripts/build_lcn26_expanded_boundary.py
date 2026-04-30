#!/usr/bin/env python3
"""Build the seven-protocol NS-3 boundary figure for the LCN26 draft."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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
PROTO_ORDER = ["AERIS", "CTP", "RPL-MRHOF", "PEGASIS", "LEACH", "HEED", "TEEN"]
CLASSICAL = {"AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"}

COLORS = {
    "AERIS": "#2F6F7E",
    "RPL-MRHOF": "#5D638E",
    "CTP": "#778A9C",
    "PEGASIS": "#B65F6B",
    "LEACH": "#D59A61",
    "HEED": "#7FA58B",
    "TEEN": "#C5A447",
    "grid": "#D7DEE7",
    "axis": "#52616E",
    "text": "#24323F",
    "muted": "#7C8792",
    "panel": "#F7F9FB",
}

PROTO_SHORT = {
    "AERIS": "AERIS",
    "RPL-MRHOF": "RPL",
    "CTP": "CTP",
    "PEGASIS": "PEG",
    "LEACH": "LEA",
    "HEED": "HEE",
    "TEEN": "TEE",
}


def load_rows() -> list[dict[str, object]]:
    with DUAL_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "protocol": row["protocol"],
                    "environment": row["environment"],
                    "num_nodes": int(row["num_nodes"]),
                    "n": int(row["n"]),
                    "pdr_mean": float(row["pdr_mean"]),
                    "pdr_std": float(row["pdr_std"]),
                }
            )
        return rows


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 8.6,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 7.4,
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.015,
        1.12,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["text"],
    )


def grouped(rows: list[dict[str, object]], protocols: set[str] | None = None) -> dict[tuple[str, int], list[dict[str, object]]]:
    out: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if protocols is not None and row["protocol"] not in protocols:
            continue
        key = (str(row["environment"]), int(row["num_nodes"]))
        out.setdefault(key, []).append(row)
    return out


def best_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: float(row["pdr_mean"]))


def aeris_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return next(row for row in rows if row["protocol"] == "AERIS")


def compute_rank_stats(rows: list[dict[str, object]], protocols: set[str]) -> tuple[int, int, int]:
    wins = 0
    top2 = 0
    total = 0
    for cell_rows in grouped(rows, protocols).values():
        ordered = sorted(cell_rows, key=lambda row: float(row["pdr_mean"]), reverse=True)
        rank = 1 + next(idx for idx, row in enumerate(ordered) if row["protocol"] == "AERIS")
        wins += int(rank == 1)
        top2 += int(rank <= 2)
        total += 1
    return wins, top2, total


def style_matrix_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, len(NODE_ORDER))
    ax.set_ylim(0, len(ENV_ORDER))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(NODE_ORDER)) + 0.5)
    ax.set_xticklabels([str(n) for n in NODE_ORDER])
    ax.set_yticks(np.arange(len(ENV_ORDER)) + 0.5)
    ax.set_yticklabels([ENV_LABELS[e] for e in ENV_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_winner_map(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    data = grouped(rows)
    for y, env in enumerate(ENV_ORDER):
        for x, nodes in enumerate(NODE_ORDER):
            cell = data[(env, nodes)]
            best = best_row(cell)
            proto = str(best["protocol"])
            pdr = float(best["pdr_mean"])
            rect = Rectangle((x, y), 1, 1, facecolor=COLORS[proto], edgecolor="white", linewidth=1.0)
            ax.add_patch(rect)
            txt_color = "white" if proto in {"AERIS", "RPL-MRHOF", "CTP", "PEGASIS"} else COLORS["text"]
            ax.text(
                x + 0.5,
                y + 0.43,
                PROTO_SHORT[proto],
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color=txt_color,
            )
            ax.text(
                x + 0.5,
                y + 0.68,
                f"{pdr:.3f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=txt_color,
            )
    style_matrix_axis(ax)
    ax.set_xlabel("Nodes")
    ax.set_title("Cell winner and mean PDR")
    panel_label(ax, "(a)")


def draw_gap_map(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    data = grouped(rows)
    gap_cmap = LinearSegmentedColormap.from_list("aeris_gap", ["#B65F6B", "#F6F8FA", COLORS["AERIS"]])
    norm = Normalize(vmin=-9.0, vmax=1.0)
    for y, env in enumerate(ENV_ORDER):
        for x, nodes in enumerate(NODE_ORDER):
            cell = data[(env, nodes)]
            best = best_row(cell)
            aeris = aeris_row(cell)
            gap = (float(aeris["pdr_mean"]) - float(best["pdr_mean"])) * 100.0
            rect = Rectangle((x, y), 1, 1, facecolor=gap_cmap(norm(gap)), edgecolor="white", linewidth=1.0)
            ax.add_patch(rect)
            label = f"{gap:+.1f}"
            ax.text(
                x + 0.5,
                y + 0.55,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold" if gap >= -0.2 else "normal",
                color=COLORS["text"],
            )
    style_matrix_axis(ax)
    ax.set_xlabel("Nodes")
    ax.set_title("AERIS gap to cell winner (percentage points)")
    panel_label(ax, "(b)")


def draw_rank_summary(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    all_protocols = set(PROTO_ORDER)
    regimes = [
        ("Classical only", CLASSICAL),
        ("AERIS+LLN", {"AERIS", "CTP", "RPL-MRHOF"}),
        ("All 7 protocols", all_protocols),
    ]
    y = np.arange(len(regimes), dtype=float)
    wins = []
    top2 = []
    totals = []
    for _, protos in regimes:
        w, t2, total = compute_rank_stats(rows, protos)
        wins.append(w)
        top2.append(t2)
        totals.append(total)

    ax.barh(y, totals, height=0.56, color="#EEF2F5", edgecolor="none", label="Cells")
    ax.barh(y, top2, height=0.56, color="#BDD2D8", edgecolor="white", linewidth=0.8, label="AERIS top-2")
    ax.barh(y, wins, height=0.56, color=COLORS["AERIS"], edgecolor="white", linewidth=0.8, label="AERIS rank-1")
    for yi, w, t2, total in zip(y, wins, top2, totals):
        ax.text(w + 0.35, yi - 0.12, f"{w}/{total}", ha="left", va="center", fontsize=7.4, color=COLORS["AERIS"], fontweight="bold")
        ax.text(t2 + 0.35, yi + 0.14, f"top-2 {t2}/{total}", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])
    ax.set_xlim(0, 30.5)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in regimes])
    ax.invert_yaxis()
    ax.set_xlabel("Environment-node cells")
    ax.set_title("AERIS rank sensitivity")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "(c)")


def draw_env_gap_summary(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    data = grouped(rows)
    env_gap = []
    for env in ENV_ORDER:
        gaps = []
        for nodes in NODE_ORDER:
            cell = data[(env, nodes)]
            gaps.append((float(aeris_row(cell)["pdr_mean"]) - float(best_row(cell)["pdr_mean"])) * 100.0)
        env_gap.append(float(np.mean(gaps)))
    y = np.arange(len(ENV_ORDER), dtype=float)
    colors = [COLORS["AERIS"] if gap > -0.2 else "#B65F6B" for gap in env_gap]
    ax.axvline(0.0, color=COLORS["axis"], linewidth=0.8, linestyle="--")
    ax.barh(y, env_gap, color=colors, alpha=0.86, height=0.55)
    for yi, gap in zip(y, env_gap):
        if gap < -1.0:
            ax.text(gap + 0.18, yi, f"{gap:+.2f}", ha="left", va="center", fontsize=7.5, color="white", fontweight="bold")
        else:
            ax.text(gap - 0.12, yi, f"{gap:+.2f}", ha="right", va="center", fontsize=7.5, color=COLORS["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([ENV_LABELS[e] for e in ENV_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-9.2, 1.0)
    ax.set_xlabel("Mean gap to winner (points)")
    ax.set_title("Environment-level boundary")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "(d)")


def build() -> None:
    rows = load_rows()
    fig = plt.figure(figsize=(7.25, 4.28))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.50, 0.98], wspace=0.36)
    left = outer[0].subgridspec(2, 1, hspace=0.44)
    right = outer[1].subgridspec(2, 1, hspace=0.42)

    ax_winner = fig.add_subplot(left[0])
    ax_gap = fig.add_subplot(left[1])
    ax_rank = fig.add_subplot(right[0])
    ax_env = fig.add_subplot(right[1])

    draw_winner_map(ax_winner, rows)
    draw_gap_map(ax_gap, rows)
    draw_rank_summary(ax_rank, rows)
    draw_env_gap_summary(ax_env, rows)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=7, markerfacecolor=COLORS[p], markeredgecolor="none", label=label)
        for p, label in [("AERIS", "AERIS"), ("RPL-MRHOF", "RPL-MRHOF"), ("CTP", "CTP"), ("PEGASIS", "PEGASIS")]
    ]
    fig.legend(
        handles=legend_handles,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.08, right=0.985)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_expanded_boundary.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_ns3_expanded_boundary.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
