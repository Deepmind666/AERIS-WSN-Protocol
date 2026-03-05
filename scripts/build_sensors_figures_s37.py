#!/usr/bin/env python3
"""
Sensors submission figures (S37): high-legibility, low-saturation scientific style.

Outputs:
- fig1_env_pdr_panel_20260219_s37.{pdf,svg,png}
- fig2_ablation_panel_20260219_s37.{pdf,svg,png}
- fig3_scalability_panel_20260219_s37.{pdf,svg,png}
- fig4_tradeoff_panel_20260219_s37.{pdf,svg,png}
- fig5_s11_patch_control_delta_20260219_s37.{pdf,svg,png}
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260219_s37"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
PROTOCOL_ORDER = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]

ENV_LABEL = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_urban": "Outdoor Urban",
    "outdoor_suburban": "Outdoor Suburban",
}

# Publication-friendly muted palette (white background safe + grayscale robust).
PROTO_COLORS = {
    "AERIS": "#3F78A8",   # muted blue
    "LEACH": "#D18B61",   # muted orange
    "PEGASIS": "#5EA38E", # muted green
    "HEED": "#B189B3",    # muted purple
    "TEEN": "#C2A24F",    # muted gold
}

PROTO_MARKERS = {
    "AERIS": "o",
    "LEACH": "s",
    "PEGASIS": "^",
    "HEED": "D",
    "TEEN": "P",
}

PROTO_LINESTYLES = {
    "AERIS": "-",
    "LEACH": "--",
    "PEGASIS": "-.",
    "HEED": (0, (5, 1.6)),
    "TEEN": (0, (2.2, 1.4)),
}

ENV_FILE = RESULTS_DIR / "env_sensitivity_20260207_205317.json"
ABLATION_FILE = RESULTS_DIR / "ablation_diag_multi_20260207_205448.json"
ENERGY_FILE = RESULTS_DIR / "energy_lifetime_stats.csv"
LATENCY_FILE = RESULTS_DIR / "latency_hop_v3_20260211_stats.csv"
DESC_FILE = RESULTS_DIR / "scalability_4env_s8_unified_20260215_descriptive.csv"
S11_DELTA_FILE = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_delta.csv"
S11_SIG_FILE = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_significance.csv"


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 10.5,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.4,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "lines.markersize": 5.2,
            "grid.color": "#DCE4ED",
            "grid.alpha": 0.26,
            "grid.linewidth": 0.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)


def panel_label(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.02,
        0.96,
        tag,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.9},
    )


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")


def group_mean_std(rows: Iterable[dict], value_fn, key_fields: Sequence[str]) -> Dict[Tuple, Tuple[float, float, int]]:
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        val = value_fn(row)
        if val is None:
            continue
        groups[key].append(float(val))

    out: Dict[Tuple, Tuple[float, float, int]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[key] = (float(arr.mean()), std, int(len(arr)))
    return out


def draw_rounded_bars(ax: plt.Axes, x: np.ndarray, heights: Sequence[float], errors: Sequence[float], colors: Sequence[str]) -> None:
    width = 0.76
    for xi, h, c in zip(x, heights, colors):
        ax.add_patch(
            FancyBboxPatch(
                (xi - width / 2, 0.0),
                width,
                float(max(0.0, h)),
                boxstyle="round,pad=0.0,rounding_size=0.02",
                linewidth=0.7,
                edgecolor="#617385",
                facecolor=c,
                zorder=2,
            )
        )
    ax.errorbar(
        x,
        heights,
        yerr=errors,
        fmt="none",
        ecolor="#2A2A2A",
        elinewidth=1.0,
        capsize=3.0,
        zorder=3,
    )
    ax.set_xlim(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5)


def plot_fig1() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(PROTOCOL_ORDER))

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        means = [stats_map[(env, p)][0] for p in PROTOCOL_ORDER]
        stds = [stats_map[(env, p)][1] for p in PROTOCOL_ORDER]
        draw_rounded_bars(ax, x, means, stds, [PROTO_COLORS[p] for p in PROTOCOL_ORDER])
        for xi, val in zip(x, means):
            ax.text(xi, val + 0.013, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#3D4B5A")
        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PDR")
        ax.grid(axis="y")

    stem = f"fig1_env_pdr_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig2() -> str:
    rows = [r for r in load_json(ABLATION_FILE)["raw_results"] if not r.get("error")]
    pdr = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "ablation_config"))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr[(env, cfg)][0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "soft_yellow_blue",
        ["#F8F9FB", "#F3F5F7", "#E8EDF2", "#D7E4EE", "#BFD6E7"],
    )
    im = axes[0].imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=18, ha="right")
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in configs])
    panel_label(axes[0], "(a)")
    style_axes(axes[0])
    axes[0].set_title("Ablation PDR heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="#202020")
    cb = fig.colorbar(im, ax=axes[0], shrink=0.84, pad=0.02)
    cb.set_label("PDR")

    full = np.array([pdr[(e, "full")][0] for e in ENV_ORDER])
    no_gw = np.array([pdr[(e, "no_gateway")][0] for e in ENV_ORDER])
    no_cas = np.array([pdr[(e, "no_cas")][0] for e in ENV_ORDER])
    y = np.arange(len(ENV_ORDER))
    gw_delta = (no_gw - full) * 100
    cas_delta = (no_cas - full) * 100

    axes[1].hlines(y + 0.13, 0, gw_delta, color="#C58D66", linewidth=2.2, label="no_gateway - full")
    axes[1].hlines(y - 0.13, 0, cas_delta, color="#3F78A8", linewidth=2.2, label="no_cas - full")
    axes[1].plot(gw_delta, y + 0.13, marker="o", linestyle="none", color="#C58D66", markersize=5.1)
    axes[1].plot(cas_delta, y - 0.13, marker="s", linestyle="none", color="#3F78A8", markersize=5.1)
    axes[1].axvline(0, color="#303030", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    axes[1].set_xlabel("Delta PDR (percentage points)")
    axes[1].set_title("Marginal effects")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#C5D0DB")
    axes[1].grid(axis="x")
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    all_delta = np.concatenate([gw_delta, cas_delta])
    axes[1].set_xlim(min(-3.0, float(all_delta.min()) - 0.35), max(2.8, float(all_delta.max()) + 0.35))

    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig3() -> str:
    rows = load_csv(DESC_FILE)
    stats_map = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (float(r["pdr_mean"]), float(r["ci95_half_width"])) for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.3), constrained_layout=True)
    axes = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        for proto in PROTOCOL_ORDER:
            means = np.array([stats_map[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
            ci = np.array([stats_map[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
            ax.plot(
                NODE_ORDER,
                means,
                linestyle=PROTO_LINESTYLES[proto],
                marker=PROTO_MARKERS[proto],
                color=PROTO_COLORS[proto],
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=2.5,
                label=proto,
            )
            ax.fill_between(NODE_ORDER, means - ci, means + ci, color=PROTO_COLORS[proto], alpha=0.13, linewidth=0)

        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        ax.set_xlim(90, 1010)
        ax.set_xticks(NODE_ORDER)
        if env == "indoor_office":
            ax.set_ylim(0.988, 1.0002)
            ax.set_yticks([0.988, 0.991, 0.994, 0.997, 1.000])
        elif env == "indoor_factory":
            ax.set_ylim(0.14, 1.02)
        elif env == "outdoor_suburban":
            ax.set_ylim(0.52, 1.02)
        else:
            ax.set_ylim(0.03, 0.93)
        ax.grid(axis="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        framealpha=0.93,
        edgecolor="#C5D0DB",
        handlelength=2.6,
        columnspacing=1.1,
    )

    stem = f"fig3_scalability_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig4() -> str:
    energy_rows = load_csv(ENERGY_FILE)
    latency_rows = load_csv(LATENCY_FILE)

    e_map = {(r["environment"], r["protocol"]): (float(r["pdr_mean"]), float(r["energy_mean"]), float(r["lifetime_mean"])) for r in energy_rows}
    l_map = {(r["environment"], r["protocol"]): float(r["hops_mean"]) for r in latency_rows}

    avg = {}
    for p in PROTOCOL_ORDER:
        avg[p] = {
            "pdr": float(np.mean([e_map[(e, p)][0] for e in ENV_ORDER])),
            "energy": float(np.mean([e_map[(e, p)][1] for e in ENV_ORDER])),
            "hops": float(np.mean([l_map[(e, p)] for e in ENV_ORDER])),
            "life": float(np.mean([e_map[(e, p)][2] for e in ENV_ORDER])),
        }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    metrics = [
        ("pdr", "Average PDR", True, "Reliability ranking"),
        ("energy", "Average total energy (J)", False, "Energy ranking"),
        ("hops", "Average hops to BS", False, "Hop-latency ranking"),
        ("life", "Average lifetime (rounds)", False, "Lifetime ranking"),
    ]

    for i, (ax, (metric, xlabel, desc, title)) in enumerate(zip(axes.flatten(), metrics)):
        vals = [avg[p][metric] for p in PROTOCOL_ORDER]
        order = np.argsort(vals)[::-1] if desc else np.argsort(vals)
        protos = [PROTOCOL_ORDER[j] for j in order]
        ranked = [vals[j] for j in order]
        y = np.arange(len(protos))

        bars = ax.barh(
            y,
            ranked,
            color=[PROTO_COLORS[p] for p in protos],
            edgecolor="#607283",
            linewidth=0.7,
            alpha=0.94,
            zorder=2,
        )
        for b in bars:
            b.set_height(0.72)
        style_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(protos)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x")
        panel_label(ax, f"({chr(97 + i)})")
        vmax = max(ranked)
        ax.set_xlim(0, vmax * 1.08)
        for yi, v in zip(y, ranked):
            label = f"{v:.3f}" if metric == "pdr" else (f"{v:.2f}" if metric == "hops" else f"{v:.1f}")
            ax.text(v + vmax * 0.009, yi, label, va="center", ha="left", fontsize=8)

    stem = f"fig4_tradeoff_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig5() -> str:
    rows = load_csv(S11_DELTA_FILE)
    sig_rows = load_csv(S11_SIG_FILE)

    delta = {(r["environment"], int(r["num_nodes"]), r["protocol"]): float(r["delta"]) for r in rows}
    sig = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (r["significant_005"] == "yes") for r in sig_rows}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    ax = axes[0]
    for env in ENV_ORDER:
        vals = np.array([delta[(env, n, "AERIS")] for n in NODE_ORDER], dtype=float)
        ax.plot(NODE_ORDER, vals, marker="o", linewidth=2.6, markersize=5.2, color=PROTO_COLORS["AERIS"], label=ENV_LABEL[env])
    ax.axhline(0.0, color="#404040", linewidth=0.8)
    ax.set_title("AERIS delta by scale (patch - control)")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Delta PDR")
    ax.set_xticks(NODE_ORDER)
    ax.set_ylim(-0.82, 0.03)
    ax.grid(axis="both")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#C5D0DB")
    panel_label(ax, "(a)")
    style_axes(ax)

    ax = axes[1]
    env_idx = np.arange(len(ENV_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(PROTOCOL_ORDER)) * width
    for i, proto in enumerate(PROTOCOL_ORDER):
        vals = [delta[(env, 1000, proto)] for env in ENV_ORDER]
        bars = ax.bar(env_idx + offsets[i], vals, width=width, color=PROTO_COLORS[proto], edgecolor="#607283", linewidth=0.7, label=proto)
        for j, b in enumerate(bars):
            if not sig[(ENV_ORDER[j], 1000, proto)]:
                ax.plot(b.get_x() + b.get_width() / 2, vals[j], marker="o", markersize=4.1, markerfacecolor="white", markeredgecolor="#333333", zorder=4)
    ax.axhline(0.0, color="#404040", linewidth=0.8)
    ax.set_title("Protocol delta at 1000 nodes")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Delta PDR")
    ax.set_xticks(env_idx)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=15, ha="right")
    ax.set_ylim(-0.82, 0.03)
    ax.grid(axis="y")
    ax.legend(loc="lower left", ncol=3, frameon=True, framealpha=0.92, edgecolor="#C5D0DB")
    panel_label(ax, "(b)")
    style_axes(ax)

    stem = f"fig5_s11_patch_control_delta_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    fig1 = plot_fig1()
    fig2 = plot_fig2()
    fig3 = plot_fig3()
    fig4 = plot_fig4()
    fig5 = plot_fig5()
    print("Generated figures:")
    for stem in [fig1, fig2, fig3, fig4, fig5]:
        print(" ", FIG_DIR / f"{stem}.pdf")


if __name__ == "__main__":
    main()
