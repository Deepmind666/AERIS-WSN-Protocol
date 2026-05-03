#!/usr/bin/env python3
"""Build compact single-column ablation and mechanism figures for the LCN draft."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_FILE = ROOT / "results" / "mega_experiments" / "ablation_diag_multi_20260207_205448.json"
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
    "Office": "#6D6D6D",
    "Factory": "#1C7ABA",
    "Suburb": "#32A344",
    "Urban": "#C13136",
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


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def ci95(std: float, n: int) -> float:
    return 1.96 * std / max(n, 1) ** 0.5


def build_ablation_compact() -> None:
    raw = json.loads(ABLATION_FILE.read_text(encoding="utf-8"))["raw_results"]
    rows = [r for r in raw if r["protocol"] == "AERIS" and r["ablation_config"] in {"full", "no_gateway", "no_cas"}]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["ablation_config"])].append(float(row["pdr_expected"]))

    fig, (ax_bar, ax_delta) = plt.subplots(2, 1, figsize=(3.42, 4.18), gridspec_kw={"height_ratios": [1.22, 1.0]})

    cfgs = ["full", "no_gateway", "no_cas"]
    colors = [COLORS["AERIS"], COLORS["GW"], COLORS["CAS"]]
    labels = ["Full", "-GW", "-CAS"]
    x = np.arange(len(ENV_ORDER), dtype=float)
    width = 0.22
    for idx, (cfg, color, label) in enumerate(zip(cfgs, colors, labels)):
        means, errs = [], []
        for env in ENV_ORDER:
            m, s = mean_std(grouped[(env, cfg)])
            means.append(m)
            errs.append(ci95(s, len(grouped[(env, cfg)])))
        ax_bar.bar(x + (idx - 1) * width, means, width=width, color=color, alpha=0.78 if cfg == "full" else 0.68, edgecolor="white", linewidth=0.6, label=label, zorder=3)
        ax_bar.errorbar(x + (idx - 1) * width, means, yerr=errs, fmt="none", ecolor="#394552", elinewidth=0.8, capsize=2.0, zorder=4)
    style_axes(ax_bar)
    ax_bar.set_ylim(0.0, 1.02)
    ax_bar.set_ylabel("Mean PDR")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ENV_SHORT[e] for e in ENV_ORDER])
    ax_bar.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.15, columnspacing=0.7)

    gw_delta, cas_delta = [], []
    for env in ENV_ORDER:
        full_m, _ = mean_std(grouped[(env, "full")])
        nogw_m, _ = mean_std(grouped[(env, "no_gateway")])
        nocas_m, _ = mean_std(grouped[(env, "no_cas")])
        gw_delta.append((nogw_m - full_m) * 100.0)
        cas_delta.append((nocas_m - full_m) * 100.0)
    y = np.arange(len(ENV_ORDER), dtype=float)
    ax_delta.axvline(0.0, color="#394552", linewidth=0.9, linestyle="--", zorder=1)
    ax_delta.hlines(y + 0.12, 0, gw_delta, color=COLORS["GW"], linewidth=1.6)
    ax_delta.hlines(y - 0.12, 0, cas_delta, color=COLORS["CAS"], linewidth=1.6)
    ax_delta.scatter(gw_delta, y + 0.12, color=COLORS["GW"], s=22, zorder=3)
    ax_delta.scatter(cas_delta, y - 0.12, color=COLORS["CAS"], s=22, marker="s", zorder=3)
    style_axes(ax_delta)
    ax_delta.grid(axis="x")
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([ENV_SHORT[e] for e in ENV_ORDER])
    ax_delta.set_xlabel("Delta vs. full (pts)")
    lim = max(abs(min(gw_delta + cas_delta)), abs(max(gw_delta + cas_delta))) + 0.5
    ax_delta.set_xlim(-lim, lim)
    ax_delta.invert_yaxis()
    fig.subplots_adjust(hspace=0.34)
    save(fig, "fig_lcn26_ablation_compact")


def build_mechanism_compact() -> None:
    mech_rows = {(row["environment"], int(row["num_nodes"])): row for row in load_csv(MECH_FILE)}
    pdr = np.asarray([[float(mech_rows[(env, n)]["pdr_expected_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    gw = np.asarray([[float(mech_rows[(env, n)]["gateway_uplink_pdr_total_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    fnd = np.asarray([[float(mech_rows[(env, n)]["first_node_death_round_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_direct = np.asarray([[float(mech_rows[(env, n)]["cas_DIRECT_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_twohop = np.asarray([[float(mech_rows[(env, n)]["cas_TWO_HOP_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_chain = np.asarray([[float(mech_rows[(env, n)]["cas_CHAIN_mean"]) for n in NODE_ORDER] for env in ENV_ORDER])
    cas_total = np.maximum(cas_direct + cas_twohop + cas_chain, 1e-9)
    twohop_share = cas_twohop / cas_total * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(3.50, 2.62), sharex=True)
    x = np.arange(len(NODE_ORDER), dtype=float)
    xticklabels = ["100", "500", "1k"]
    env_labels = [ENV_SHORT[e] for e in ENV_ORDER]
    markers = {"Office": "o", "Factory": "s", "Suburb": "^", "Urban": "D"}
    panels = [
        (axes[0, 0], "(a) End-to-end PDR", pdr, "Mean PDR", (0.0, 1.03)),
        (axes[0, 1], "(b) Gateway uplink PDR", gw, "Mean PDR", (0.0, 1.03)),
        (axes[1, 0], "(c) First-node death", fnd, "Round", (0.0, max(15.0, float(np.max(fnd)) + 1.0))),
        (axes[1, 1], "(d) CAS two-hop share", twohop_share, "Share (%)", (0.0, 80.0)),
    ]

    for ax, title, matrix, ylabel, ylim in panels:
        for env_idx, env_name in enumerate(env_labels):
            ax.plot(
                x,
                matrix[env_idx],
                color=COLORS[env_name],
                marker=markers[env_name],
                linewidth=1.18 if env_name != "Urban" else 1.34,
                markersize=2.7,
                alpha=0.96,
                label=env_name,
            )
        ax.set_title(title, loc="left", pad=1.4, fontsize=6.8, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.08, len(NODE_ORDER) - 0.70)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.grid(axis="y", linestyle="--", linewidth=0.50, color=COLORS["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_color(COLORS["axis"])
        ax.tick_params(length=2.1, pad=1.4, labelsize=5.8)

    axes[0, 1].annotate(
        "urban-1k\nbottleneck",
        xy=(x[-1], gw[ENV_ORDER.index("outdoor_urban"), -1]),
        xytext=(x[-1] - 0.45, 0.50),
        arrowprops={"arrowstyle": "-", "color": COLORS["Urban"], "linewidth": 0.55},
        fontsize=5.1,
        color=COLORS["Urban"],
        ha="right",
        va="center",
    )
    axes[1, 0].annotate(
        "early FND",
        xy=(x[1], fnd[ENV_ORDER.index("outdoor_suburban"), 1]),
        xytext=(x[1] + 0.16, 7.2),
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.55},
        fontsize=5.1,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.52, 0.995), frameon=False, columnspacing=0.85, handletextpad=0.25)
    axes[1, 0].set_xlabel("Nodes")
    axes[1, 1].set_xlabel("Nodes")
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.15, wspace=0.28, hspace=0.38)
    save(fig, "fig_lcn26_mechanism_compact")


if __name__ == "__main__":
    apply_style()
    build_ablation_compact()
    build_mechanism_compact()
