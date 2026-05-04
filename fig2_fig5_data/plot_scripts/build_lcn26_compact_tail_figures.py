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
    "AERIS": "#5A5A5A",
    "PEGASIS": "#C6373D",
    "GW": "#36A657",
    "CAS": "#2D83BD",
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

    fig, axes = plt.subplots(2, 2, figsize=(3.50, 2.52))
    panels = [
        (axes[0, 0], "(a) End-to-end PDR", pdr, "Blues", 0.0, 1.0, "{:.2f}"),
        (axes[0, 1], "(b) Gateway PDR", gw, "Greens", 0.0, 1.0, "{:.2f}"),
        (axes[1, 0], "(c) FND rounds", fnd, "Reds", 0.0, max(15.0, float(np.max(fnd))), "{:.1f}"),
        (axes[1, 1], "(d) CAS two-hop %", twohop_share, "Oranges", 0.0, 80.0, "{:.0f}"),
    ]

    for ax, title, matrix, cmap, vmin, vmax, fmt in panels:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        del image
        ax.set_title(title, loc="left", pad=1.5, fontsize=6.8, fontweight="bold")
        ax.set_xticks(np.arange(len(NODE_ORDER)))
        ax.set_xticklabels(["100", "500", "1k"])
        ax.set_yticks(np.arange(len(ENV_ORDER)))
        ax.set_yticklabels([ENV_SHORT[e] for e in ENV_ORDER])
        ax.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ENV_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0, pad=1.4, labelsize=5.8)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                val = float(matrix[row_idx, col_idx])
                normed = (val - vmin) / max(vmax - vmin, 1e-9)
                txt_color = "white" if normed > 0.58 else "#222222"
                ax.text(col_idx, row_idx, fmt.format(val), ha="center", va="center", fontsize=5.4, color=txt_color)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[0, :]:
        ax.tick_params(axis="x", labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", labelleft=False)
    fig.text(0.50, 0.03, "Node scale", ha="center", va="center", fontsize=6.5, color=COLORS["text"])
    fig.subplots_adjust(left=0.14, right=0.985, top=0.93, bottom=0.12, wspace=0.12, hspace=0.25)
    save(fig, "fig_lcn26_mechanism_compact")


if __name__ == "__main__":
    apply_style()
    build_ablation_compact()
    build_mechanism_compact()
