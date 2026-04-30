#!/usr/bin/env python3
"""Build a compact composite tail figure for the LCN draft."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
NS3_STRICT = ROOT / "results" / "mega_experiments" / "scalability_4env_v50rigor_20260222_descriptive.csv"
ABLATION_FILE = ROOT / "results" / "mega_experiments" / "ablation_diag_multi_20260207_205448.json"
MECH_FILE = ROOT / "results" / "lcn26_targeted_20260420" / "mechanism_grid_fat" / "mechanism_summary.csv"
ENERGY_FILE = ROOT / "results" / "mega_experiments" / "energy_lifetime_stats.csv"
LATENCY_FILE = ROOT / "results" / "mega_experiments" / "latency_hop_v3_20260211_stats.csv"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABELS = {"indoor_office": "Office", "indoor_factory": "Factory", "outdoor_suburban": "Suburb", "outdoor_urban": "Urban"}
ENV_TAG = {"indoor_office": "O", "indoor_factory": "F", "outdoor_suburban": "S", "outdoor_urban": "U"}
NODE_ORDER_STRICT = [100, 200, 300, 500, 800, 1000]
NODE_ORDER_MECH = [100, 500, 1000]
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
DRAW_ORDER = ["LEACH", "HEED", "TEEN", "PEGASIS", "AERIS"]
COLORS = {
    "AERIS": "#2F5D7C",
    "PEGASIS": "#B07A8F",
    "LEACH": "#D58A5B",
    "HEED": "#7AA08D",
    "TEEN": "#C7A74D",
    "GW": "#D56C5B",
    "CAS": "#3E8E9B",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "muted": "#7A8794",
    "best": "#FFF2A8",
    "second": "#F6DFC2",
    "third": "#F4CED6",
}
MARKERS = {"AERIS": "o", "PEGASIS": "s", "LEACH": "^", "HEED": "D", "TEEN": "P"}
LINESTYLES = {"AERIS": "-", "PEGASIS": "--", "LEACH": (0, (4, 2)), "HEED": (0, (2, 2)), "TEEN": (0, (1.5, 1.5))}


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
            "font.size": 8.4,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
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


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def build() -> None:
    strict_rows = load_csv(NS3_STRICT)
    strict = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (float(r["pdr_mean"]), float(r["pdr_std"]), int(r["n"])) for r in strict_rows}
    raw = json.loads(ABLATION_FILE.read_text(encoding="utf-8"))["raw_results"]
    ab_rows = [r for r in raw if r["protocol"] == "AERIS" and r["ablation_config"] in {"full", "no_gateway", "no_cas"}]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in ab_rows:
        grouped[(row["environment"], row["ablation_config"])].append(float(row["pdr_expected"]))
    mech_rows = {(row["environment"], int(row["num_nodes"])): row for row in load_csv(MECH_FILE)}
    energy_rows = load_csv(ENERGY_FILE)
    latency_rows = load_csv(LATENCY_FILE)

    fig = plt.figure(figsize=(7.1, 5.45))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.10, 1.0], wspace=0.22, hspace=0.30)
    ax_strict = fig.add_subplot(gs[0, 0])
    ax_ab = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_mech = fig.add_subplot(gs[1, 1])

    # (a) strict compact in one panel as small multiples within axis
    ax_strict.axis("off")
    sub = gs[0, 0].subgridspec(2, 2, wspace=0.18, hspace=0.32)
    strict_axes = [fig.add_subplot(sub[i, j]) for i in range(2) for j in range(2)]
    x = np.arange(len(NODE_ORDER_STRICT), dtype=float)
    for idx, env in enumerate(ENV_ORDER):
        ax = strict_axes[idx]
        for proto in DRAW_ORDER:
            y = np.asarray([strict[(env, n, proto)][0] for n in NODE_ORDER_STRICT], dtype=float)
            ax.plot(
                x, y, color=COLORS[proto], marker=MARKERS[proto],
                markersize=2.5 if proto in {"AERIS", "PEGASIS"} else 2.1,
                linewidth=1.3 if proto == "AERIS" else 1.1 if proto == "PEGASIS" else 0.9,
                linestyle=LINESTYLES[proto], alpha=1.0 if proto in {"AERIS", "PEGASIS"} else 0.72
            )
        style_axes(ax)
        ax.set_title(ENV_LABELS[env], pad=2)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in NODE_ORDER_STRICT], rotation=25, ha="right")
        last = NODE_ORDER_STRICT[-1]
        ax.text(x[-1] + 0.04, strict[(env, last, "AERIS")][0], f"{strict[(env, last, 'AERIS')][0]:.2f}", fontsize=5.7, color=COLORS["AERIS"], va="center")
        ax.text(x[-1] + 0.04, strict[(env, last, "PEGASIS")][0], f"{strict[(env, last, 'PEGASIS')][0]:.2f}", fontsize=5.7, color=COLORS["PEGASIS"], va="center")
        if idx in (0, 2):
            ax.set_ylabel("Mean PDR", labelpad=1)
        else:
            ax.set_yticklabels([])
        if idx not in (2, 3):
            ax.set_xticklabels([])
    handles = [Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p], linewidth=1.2, label=p) for p in PROTO_ORDER]
    fig.legend(handles=handles, labels=PROTO_ORDER, ncol=5, loc="upper left", bbox_to_anchor=(0.05, 1.01), frameon=False, columnspacing=0.45, handletextpad=0.22)

    # (b) ablation compact
    sub2 = gs[0, 1].subgridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.33)
    ab_top = fig.add_subplot(sub2[0, 0])
    ab_bot = fig.add_subplot(sub2[1, 0])
    cfgs = ["full", "no_gateway", "no_cas"]
    ab_colors = [COLORS["AERIS"], COLORS["GW"], COLORS["CAS"]]
    labels = ["Full", "-GW", "-CAS"]
    x2 = np.arange(len(ENV_ORDER), dtype=float)
    width = 0.22
    for idx, (cfg, color, label) in enumerate(zip(cfgs, ab_colors, labels)):
        means, errs = [], []
        for env in ENV_ORDER:
            m, s = mean_std(grouped[(env, cfg)])
            means.append(m); errs.append(ci95(s, len(grouped[(env, cfg)])))
        ab_top.bar(x2 + (idx - 1) * width, means, width=width, color=color, alpha=0.78 if cfg == "full" else 0.68, edgecolor="white", linewidth=0.5, label=label, zorder=3)
        ab_top.errorbar(x2 + (idx - 1) * width, means, yerr=errs, fmt="none", ecolor="#394552", elinewidth=0.7, capsize=1.6, zorder=4)
    style_axes(ab_top)
    ab_top.set_ylim(0.25, 1.02)
    ab_top.set_ylabel("Mean PDR")
    ab_top.set_xticks(x2)
    ab_top.set_xticklabels([ENV_LABELS[e] for e in ENV_ORDER])
    ab_top.set_title("")
    ab_top.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.2, columnspacing=0.7)
    gw_delta, cas_delta = [], []
    for env in ENV_ORDER:
        full_m, _ = mean_std(grouped[(env, "full")]); nogw_m, _ = mean_std(grouped[(env, "no_gateway")]); nocas_m, _ = mean_std(grouped[(env, "no_cas")])
        gw_delta.append((nogw_m - full_m) * 100.0); cas_delta.append((nocas_m - full_m) * 100.0)
    y = np.arange(len(ENV_ORDER), dtype=float)
    ab_bot.axvline(0.0, color="#394552", linewidth=0.8, linestyle="--", zorder=1)
    ab_bot.hlines(y + 0.12, 0, gw_delta, color=COLORS["GW"], linewidth=1.4)
    ab_bot.hlines(y - 0.12, 0, cas_delta, color=COLORS["CAS"], linewidth=1.4)
    ab_bot.scatter(gw_delta, y + 0.12, color=COLORS["GW"], s=16, zorder=3)
    ab_bot.scatter(cas_delta, y - 0.12, color=COLORS["CAS"], s=16, marker="s", zorder=3)
    style_axes(ab_bot)
    ab_bot.grid(axis="x")
    ab_bot.set_yticks(y)
    ab_bot.set_yticklabels([ENV_LABELS[e] for e in ENV_ORDER])
    ab_bot.set_xlabel("Delta vs. full (pts)")
    ax_ab.axis("off")

    # (c) table
    ax_tbl.axis("off")
    pooled = {}
    for proto in PROTO_ORDER:
        e = [r for r in energy_rows if r["protocol"] == proto]
        l = [r for r in latency_rows if r["protocol"] == proto]
        pooled[proto] = {
            "pdr": round(float(np.mean([float(r["pdr_mean"]) for r in e])), 3),
            "energy": round(float(np.mean([float(r["energy_mean"]) for r in e])), 1),
            "life": round(float(np.mean([float(r["lifetime_mean"]) for r in e])), 1),
            "fnd": round(float(np.mean([float(r["fnd_mean"]) for r in e])), 1),
            "hops": round(float(np.mean([float(r["hops_mean"]) for r in l])), 2),
        }
    table_data = [[p, f"{pooled[p]['pdr']:.3f}", f"{pooled[p]['energy']:.1f}", f"{pooled[p]['life']:.1f}", f"{pooled[p]['fnd']:.1f}", f"{pooled[p]['hops']:.2f}"] for p in PROTO_ORDER]
    tbl = ax_tbl.table(cellText=table_data, colLabels=["Method", "PDR↑", "Energy↓", "Life↑", "FND↑", "Hops↓"], cellLoc="center", colLoc="center", loc="center", bbox=[0.0, 0.18, 1.0, 0.72])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D6DCE4"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#F4F7FA"); cell.set_text_props(fontweight="semibold", color=COLORS["text"])

    # (d) mechanism compact
    subm = gs[1, 1].subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.28)
    mtop = fig.add_subplot(subm[0, 0]); mbot = fig.add_subplot(subm[1, 0])
    cells = [(env, n) for env in ENV_ORDER for n in NODE_ORDER_MECH]
    x3 = np.arange(len(cells), dtype=float)
    pdr = np.asarray([float(mech_rows[(env, n)]["pdr_expected_mean"]) for env, n in cells], dtype=float)
    gw = np.asarray([float(mech_rows[(env, n)]["gateway_uplink_pdr_total_mean"]) for env, n in cells], dtype=float)
    fnd = np.asarray([float(mech_rows[(env, n)]["first_node_death_round_mean"]) for env, n in cells], dtype=float)
    cas_direct = np.asarray([float(mech_rows[(env, n)]["cas_DIRECT_mean"]) for env, n in cells], dtype=float)
    cas_twohop = np.asarray([float(mech_rows[(env, n)]["cas_TWO_HOP_mean"]) for env, n in cells], dtype=float)
    cas_chain = np.asarray([float(mech_rows[(env, n)]["cas_CHAIN_mean"]) for env, n in cells], dtype=float)
    cas_total = np.maximum(cas_direct + cas_twohop + cas_chain, 1e-9); twohop_share = cas_twohop / cas_total * 100.0
    for group_idx, env in enumerate(ENV_ORDER):
        start = group_idx * len(NODE_ORDER_MECH) - 0.5; stop = start + len(NODE_ORDER_MECH)
        bg = "#FAF3F0" if env == "indoor_office" else "#F5F8FA"
        mtop.axvspan(start, stop, color=bg, alpha=0.95, zorder=0); mbot.axvspan(start, stop, color=bg, alpha=0.95, zorder=0)
        if group_idx > 0:
            mtop.axvline(start, color="#C7D0DA", linestyle="--", linewidth=0.7); mbot.axvline(start, color="#C7D0DA", linestyle="--", linewidth=0.7)
    mtop.bar(x3, pdr, width=0.56, color=COLORS["AERIS"], alpha=0.72, edgecolor="white", linewidth=0.5, zorder=3)
    mtop.plot(x3, gw, color=COLORS["GW"], marker="o", markersize=3.0, linewidth=1.4, zorder=4)
    style_axes(mtop); mtop.set_ylabel("PDR / GW"); mtop.set_ylim(0.0, 1.02)
    urban_1000 = cells.index(("outdoor_urban", 1000))
    mtop.annotate("GW bottleneck", xy=(urban_1000, gw[urban_1000]), xytext=(urban_1000 - 2.3, 0.42), fontsize=6.5, color=COLORS["GW"], arrowprops=dict(arrowstyle="->", color=COLORS["GW"], linewidth=0.7), bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#D7DEE7", alpha=0.95))
    mbot.plot(x3, fnd, color=COLORS["PEGASIS"], marker="s", markersize=2.8, linewidth=1.2, linestyle="--", label="FND")
    mbot.plot(x3, twohop_share, color=COLORS["CAS"], marker="D", markersize=2.8, linewidth=1.2, linestyle="-.", label="Two-hop %")
    style_axes(mbot); mbot.set_ylabel("FND / 2-hop %")
    mbot.set_xticks(x3)
    mbot.set_xticklabels([f"{ENV_TAG[e]}{('1k' if n == 1000 else n)}" for e, n in cells], fontsize=6.7, rotation=35, ha="right")
    mbot.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.6)
    ax_mech.axis("off")

    fig.text(0.02, 0.955, "(a)", fontsize=8.3, fontweight="bold")
    fig.text(0.54, 0.955, "(b)", fontsize=8.3, fontweight="bold")
    fig.text(0.02, 0.485, "(c)", fontsize=8.3, fontweight="bold")
    fig.text(0.54, 0.485, "(d)", fontsize=8.3, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_lcn26_tail_composite.pdf")
    fig.savefig(OUT_DIR / "fig_lcn26_tail_composite.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    build()
