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
ENV_LABEL = {"indoor_office": "Office", "indoor_factory": "Factory", "outdoor_suburban": "Suburban", "outdoor_urban": "Urban"}
ENV_TAG = {"indoor_office": "O", "indoor_factory": "F", "outdoor_suburban": "S", "outdoor_urban": "U"}
NODE_ORDER = [100, 500, 1000]
COLORS = {
    "AERIS": "#1F77B4",
    "PEGASIS": "#D62728",
    "GW": "#2CA02C",
    "CAS": "#FF7F0E",
    "grid": "#CFCFCF",
    "axis": "#111111",
    "text": "#111111",
    "muted": "#555555",
    "best": "#E6E6E6",
    "second": "#F2F2F2",
    "third": "#F7F7F7",
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
            "axes.labelsize": 6.8,
            "axes.titlesize": 6.4,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
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

    fig, axes = plt.subplots(2, 2, figsize=(3.52, 3.02), sharex=True)
    ax_pdr, ax_gw, ax_fnd, ax_twohop = axes.flatten()
    plot_axes = (ax_pdr, ax_gw, ax_fnd, ax_twohop)
    xlabels = [f"{ENV_TAG[e]}{('1k' if n == 1000 else n)}" for e, n in cells]

    for ax in plot_axes:
        style_axes(ax)
        ax.set_xlim(-0.6, len(cells) - 0.4)
        for group_idx, _env in enumerate(ENV_ORDER):
            start = group_idx * len(NODE_ORDER) - 0.5
            if group_idx > 0:
                ax.axvline(start, color=COLORS["grid"], linestyle="--", linewidth=0.7, zorder=1)
        ax.tick_params(axis="x", length=2.0, pad=1.2)

    ax_pdr.bar(x, pdr, width=0.58, color=COLORS["AERIS"], edgecolor="black", linewidth=0.35, zorder=3)
    ax_pdr.set_ylim(0.0, 1.05)
    ax_pdr.set_ylabel("PDR")
    ax_pdr.set_title("(a) End-to-end", pad=1.5)

    ax_gw.plot(x, gw, color=COLORS["GW"], marker="o", markersize=3.0, linewidth=1.35, zorder=3)
    ax_gw.set_ylim(0.0, 1.05)
    ax_gw.set_ylabel("PDR")
    ax_gw.set_title("(b) Gateway uplink", pad=1.5)

    ax_fnd.bar(x, fnd, width=0.58, color=COLORS["PEGASIS"], edgecolor="black", linewidth=0.35, zorder=3)
    ax_fnd.set_ylim(0, max(float(np.max(fnd)) * 1.18, 15.0))
    ax_fnd.set_ylabel("FND")
    ax_fnd.set_title("(c) First-node death", pad=1.5)

    ax_twohop.plot(x, twohop_share, color=COLORS["CAS"], marker="s", markersize=2.8, linewidth=1.25, zorder=3)
    ax_twohop.set_ylim(0, 100)
    ax_twohop.set_ylabel("Two-hop (%)")
    ax_twohop.set_title("(d) CAS two-hop", pad=1.5)

    urban_1000 = cells.index(("outdoor_urban", 1000))
    ax_pdr.text(urban_1000, pdr[urban_1000] + 0.035, f"{pdr[urban_1000]:.2f}", ha="center", va="bottom", fontsize=5.0, color=COLORS["AERIS"])
    for idx in [0, 3, 6, 9, 11]:
            ax_fnd.text(idx, fnd[idx] + 0.35, f"{fnd[idx]:.1f}", ha="center", va="bottom", fontsize=4.9, color=COLORS["PEGASIS"])

    for ax in (ax_pdr, ax_gw):
        ax.tick_params(axis="x", labelbottom=False)
    for ax in (ax_fnd, ax_twohop):
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=5.4, rotation=38, ha="right")

    fig.subplots_adjust(left=0.13, right=0.985, top=0.85, bottom=0.16, wspace=0.34, hspace=0.44)
    save(fig, "fig_lcn26_mechanism_compact")


if __name__ == "__main__":
    apply_style()
    if ABLATION_FILE.exists():
        build_ablation_compact()
    else:
        print(f"[LCN26] Skipping compact ablation figure; missing {ABLATION_FILE}")
    build_mechanism_compact()
