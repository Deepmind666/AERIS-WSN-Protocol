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
    cells = [(env, n) for env in ENV_ORDER for n in NODE_ORDER]
    x = np.arange(len(cells), dtype=float)
    pdr = np.asarray([float(mech_rows[(env, n)]["pdr_expected_mean"]) for env, n in cells], dtype=float)
    gw = np.asarray([float(mech_rows[(env, n)]["gateway_uplink_pdr_total_mean"]) for env, n in cells], dtype=float)
    fnd = np.asarray([float(mech_rows[(env, n)]["first_node_death_round_mean"]) for env, n in cells], dtype=float)
    cas_direct = np.asarray([float(mech_rows[(env, n)]["cas_DIRECT_mean"]) for env, n in cells], dtype=float)
    cas_twohop = np.asarray([float(mech_rows[(env, n)]["cas_TWO_HOP_mean"]) for env, n in cells], dtype=float)
    cas_chain = np.asarray([float(mech_rows[(env, n)]["cas_CHAIN_mean"]) for env, n in cells], dtype=float)
    cas_total = np.maximum(cas_direct + cas_twohop + cas_chain, 1e-9)
    twohop_share = cas_twohop / cas_total * 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(3.42, 3.82), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})

    for group_idx, env in enumerate(ENV_ORDER):
        start = group_idx * len(NODE_ORDER) - 0.5
        stop = start + len(NODE_ORDER)
        bg = "#F6F7F8" if env == "indoor_office" else "#F9FAFB"
        ax_top.axvspan(start, stop, color=bg, alpha=0.95, zorder=0)
        ax_bottom.axvspan(start, stop, color=bg, alpha=0.95, zorder=0)
        if group_idx > 0:
            ax_top.axvline(start, color="#C7D0DA", linestyle="--", linewidth=0.7)
            ax_bottom.axvline(start, color="#C7D0DA", linestyle="--", linewidth=0.7)

    ax_top.bar(x, pdr, width=0.56, color=COLORS["AERIS"], alpha=0.76, edgecolor="white", linewidth=0.5, zorder=3)
    ax_top.plot(x, gw, color=COLORS["GW"], marker="o", markersize=3.2, linewidth=1.55, zorder=4)
    style_axes(ax_top)
    ax_top.set_ylabel("PDR / GW")
    ax_top.set_ylim(0.0, 1.02)
    for idx in [0, 2, 3, 5, 6, 8, 9, 11]:
        ax_top.text(x[idx], pdr[idx] + 0.02, f"{pdr[idx]:.2f}", ha="center", va="bottom", fontsize=6.2, color=COLORS["AERIS"])
    urban_1000 = cells.index(("outdoor_urban", 1000))
    ax_top.annotate("GW bottleneck", xy=(urban_1000, gw[urban_1000]), xytext=(urban_1000 - 2.0, 0.42), fontsize=6.2, color=COLORS["GW"], arrowprops=dict(arrowstyle="->", color=COLORS["GW"], linewidth=0.8), bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#D7DEE7", alpha=0.95))

    fnd_handle, = ax_bottom.plot(x, fnd, color=COLORS["PEGASIS"], marker="s", markersize=3.0, linewidth=1.4, linestyle="--", label="FND")
    style_axes(ax_bottom)
    ax_bottom.set_ylabel("FND (rounds)")
    ax_bottom.set_ylim(0, max(float(np.max(fnd)) * 1.20, 16.0))
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([f"{ENV_TAG[e]}{('1k' if n == 1000 else n)}" for e, n in cells], fontsize=6.4, rotation=35, ha="right")

    ax_share = ax_bottom.twinx()
    twohop_handle, = ax_share.plot(x, twohop_share, color=COLORS["CAS"], marker="D", markersize=3.0, linewidth=1.4, linestyle="-.", label="Two-hop %")
    ax_share.set_ylabel("Two-hop share (%)")
    ax_share.set_ylim(0, 100)
    ax_share.spines["top"].set_visible(False)
    ax_share.spines["left"].set_visible(False)
    ax_share.spines["right"].set_color(COLORS["axis"])
    ax_share.tick_params(axis="y", colors=COLORS["axis"])
    ax_bottom.legend(handles=[fnd_handle, twohop_handle], frameon=False, loc="upper right", ncol=1, handlelength=1.4, columnspacing=0.8)

    for idx in [0, 3, 6, 9, 11]:
        ax_bottom.text(x[idx], fnd[idx] + 1.2, f"{fnd[idx]:.1f}", ha="center", va="bottom", fontsize=6.1, color=COLORS["PEGASIS"])

    fig.subplots_adjust(hspace=0.14)
    save(fig, "fig_lcn26_mechanism_compact")


if __name__ == "__main__":
    apply_style()
    build_ablation_compact()
    build_mechanism_compact()
