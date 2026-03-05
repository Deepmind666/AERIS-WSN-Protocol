#!/usr/bin/env python3
"""
Sensors submission figures (S42): deep style upgrade with explicit protocol
workflow diagram for manuscript-level method clarity.

Outputs:
- fig0_aeris_workflow_20260220_s42.{pdf,svg,png}
- fig1_env_pdr_panel_20260220_s42.{pdf,svg,png}
- fig2_ablation_panel_20260220_s42.{pdf,svg,png}
- fig3_scalability_panel_20260220_s42.{pdf,svg,png}
- fig4_tradeoff_panel_20260220_s42.{pdf,svg,png}
- fig5_s11_patch_control_delta_20260220_s42.{pdf,svg,png}
- fig6_s10_power_sensitivity_20260220_s42.{pdf,svg,png}
- fig7_ns3_trend_panel_20260220_s42.{pdf,svg,png}
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260220_s42"

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

# Soft scientific palette, white-background safe and grayscale-distinguishable.
PROTO_COLORS = {
    "AERIS": "#4F7EA8",
    "LEACH": "#CB8A58",
    "PEGASIS": "#63A891",
    "HEED": "#B58AB8",
    "TEEN": "#BFA356",
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
S10_DESC_FILE = RESULTS_DIR / "s10_4env_merged_descriptive_20260216.csv"
S10_SIG_FILE = RESULTS_DIR / "s10_4env_significance_tx5_vs_tx15_20260216.csv"
NS3_SIG_FILE = PROJECT_ROOT / "ns3_validation" / "results" / "ns3_scale_ext_1000_significance.csv"


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 11.0,
            "axes.labelsize": 11.2,
            "axes.titlesize": 11.6,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 8.8,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.9,
            "lines.markersize": 6.0,
            "grid.color": "#DCE4EB",
            "grid.alpha": 0.18,
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
                linewidth=0.55,
                edgecolor="#7F8FA0",
                facecolor=c,
                zorder=2,
            )
        )
    ax.errorbar(
        x,
        heights,
        yerr=errors,
        fmt="none",
        ecolor="#3A3A3A",
        elinewidth=0.9,
        capsize=3.0,
        zorder=3,
    )
    ax.set_xlim(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5)


def plot_fig0_workflow() -> str:
    fig, ax = plt.subplots(figsize=(12.4, 5.8), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def node(x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> Tuple[float, float]:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.85,
            edgecolor="#5A6A7A",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color="#1F2A35")
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=8.4, color="#334455")
        return (x + w, y + h / 2)

    def arrow(p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#6C7F92",
                zorder=1,
            )
        )

    c_a = "#E9F1F8"
    c_b = "#EEF5F0"
    c_c = "#F5EEF7"
    c_d = "#F8F2E9"

    p1 = node(0.5, 4.5, 2.6, 1.6, "Sensing Nodes", "Packet generation\n+ energy update", c_a)
    p2 = node(3.7, 4.5, 2.7, 1.6, "CH Election", "Fuzzy score + fairness\ncluster formation", c_b)
    p3 = node(7.1, 4.5, 2.7, 1.6, "CAS Switching", "direct / chain / twohop\nmode selection", c_c)
    p4 = node(10.5, 4.5, 2.9, 1.6, "Uplink Routing", "Gateway/Skeleton\nCH-to-BS forwarding", c_d)

    q1 = node(3.7, 1.1, 2.7, 1.6, "Safety Layer", "retry + power adaptation\nconstraint handling", c_a)
    q2 = node(7.1, 1.1, 2.7, 1.6, "Metric Logger", "pdr_expected, energy,\nhops, lifetime", c_b)
    q3 = node(10.5, 1.1, 2.9, 1.6, "Output Bundle", "publication CSV/JSON\n+ provenance sidecar", c_c)

    arrow((p1[0], p1[1]), (3.7, 5.3))
    arrow((p2[0], p2[1]), (7.1, 5.3))
    arrow((p3[0], p3[1]), (10.5, 5.3))
    arrow((11.9, 4.5), (11.9, 2.7))
    arrow((6.4, 1.9), (7.1, 1.9))
    arrow((9.8, 1.9), (10.5, 1.9))
    arrow((8.45, 4.5), (5.05, 2.7))

    ax.text(0.5, 6.55, "AERIS round-level workflow and evidence pipeline", fontsize=12.2, fontweight="bold", color="#1F2A35")
    ax.text(0.5, 6.15, "The flow separates routing logic, safety control, and publication-tier evidence recording.", fontsize=9.2, color="#4D5E70")

    panel_label(ax, "(a)")
    stem = f"fig0_aeris_workflow_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig1() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.1), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(PROTOCOL_ORDER))

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        means = [stats_map[(env, p)][0] for p in PROTOCOL_ORDER]
        stds = [stats_map[(env, p)][1] for p in PROTOCOL_ORDER]
        draw_rounded_bars(ax, x, means, stds, [PROTO_COLORS[p] for p in PROTOCOL_ORDER])
        for xi, val in zip(x, means):
            ax.text(xi, val + 0.013, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#3D4B5A")

        # Highlight the best protocol and annotate the top-2 margin.
        order = np.argsort(np.array(means))[::-1]
        best_idx = int(order[0])
        second_idx = int(order[1])
        margin = means[best_idx] - means[second_idx]
        ax.add_patch(
            FancyBboxPatch(
                (best_idx - 0.38, 0.0),
                0.76,
                means[best_idx],
                boxstyle="round,pad=0.0,rounding_size=0.02",
                linewidth=1.1,
                edgecolor="#2F4E6A",
                facecolor="none",
                zorder=4,
            )
        )
        ax.text(
            0.98,
            0.95,
            f"top-2 margin: {margin:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color="#455A70",
            bbox={"facecolor": "#F8FBFF", "edgecolor": "#D0DCE7", "boxstyle": "round,pad=0.15"},
        )

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

    axes[1].hlines(y + 0.13, 0, gw_delta, color=PROTO_COLORS["LEACH"], linewidth=2.3, label="no_gateway - full")
    axes[1].hlines(y - 0.13, 0, cas_delta, color=PROTO_COLORS["AERIS"], linewidth=2.3, label="no_cas - full")
    axes[1].plot(gw_delta, y + 0.13, marker="o", linestyle="none", color=PROTO_COLORS["LEACH"], markersize=5.3)
    axes[1].plot(cas_delta, y - 0.13, marker="s", linestyle="none", color=PROTO_COLORS["AERIS"], markersize=5.3)
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
        curves = {}
        for proto in PROTOCOL_ORDER:
            means = np.array([stats_map[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
            ci = np.array([stats_map[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
            curves[proto] = (means, ci)
            line_width = 3.1 if proto == "AERIS" else 2.3
            marker_size = 5.9 if proto == "AERIS" else 5.0
            ax.plot(
                NODE_ORDER,
                means,
                linestyle=PROTO_LINESTYLES[proto],
                marker=PROTO_MARKERS[proto],
                color=PROTO_COLORS[proto],
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=line_width,
                markersize=marker_size,
                label=proto,
            )
            ax.fill_between(NODE_ORDER, means - ci, means + ci, color=PROTO_COLORS[proto], alpha=0.16, linewidth=0)

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
            inset = inset_axes(ax, width="43%", height="40%", loc="lower left", borderpad=1.2)
            for proto in PROTOCOL_ORDER:
                means, _ = curves[proto]
                inset.plot(
                    NODE_ORDER,
                    means,
                    linestyle=PROTO_LINESTYLES[proto],
                    marker=PROTO_MARKERS[proto],
                    markersize=3.0,
                    linewidth=1.9,
                    color=PROTO_COLORS[proto],
                )
            inset.set_xlim(95, 1005)
            inset.set_ylim(0.989, 0.996)
            inset.set_xticks([100, 500, 1000])
            inset.set_yticks([0.990, 0.993, 0.996])
            inset.tick_params(labelsize=6.7, pad=1)
            inset.grid(alpha=0.20)
            inset.set_facecolor("#FBFDFF")
            for side in ("top", "right"):
                inset.spines[side].set_visible(False)
            inset.spines["left"].set_linewidth(0.6)
            inset.spines["bottom"].set_linewidth(0.6)
            ax.text(
                0.98,
                0.06,
                "zoomed ranking",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.4,
                color="#4A5562",
                bbox={"facecolor": "white", "edgecolor": "#D6DEE7", "boxstyle": "round,pad=0.16", "alpha": 0.95},
            )
        elif env == "indoor_factory":
            ax.set_ylim(0.14, 1.02)
            aeris_slope = curves["AERIS"][0][-1] - curves["AERIS"][0][0]
            ax.text(
                0.98,
                0.06,
                f"S8 trend +{aeris_slope:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#6C3A2E",
                bbox={"facecolor": "#FFF8F3", "edgecolor": "#E7CDBA", "boxstyle": "round,pad=0.14"},
            )
        elif env == "outdoor_suburban":
            ax.set_ylim(0.52, 1.02)
            aeris_slope = curves["AERIS"][0][-1] - curves["AERIS"][0][0]
            ax.text(
                0.98,
                0.06,
                f"S8 trend +{aeris_slope:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#6C3A2E",
                bbox={"facecolor": "#FFF8F3", "edgecolor": "#E7CDBA", "boxstyle": "round,pad=0.14"},
            )
        else:
            ax.set_ylim(0.03, 0.93)
            aeris_slope = curves["AERIS"][0][-1] - curves["AERIS"][0][0]
            ax.text(
                0.98,
                0.06,
                f"S8 trend +{aeris_slope:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#6C3A2E",
                bbox={"facecolor": "#FFF8F3", "edgecolor": "#E7CDBA", "boxstyle": "round,pad=0.14"},
            )
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


def plot_fig6_s10() -> str:
    rows = load_csv(S10_SIG_FILE)

    delta = {(r["environment"], int(r["num_nodes"]), r["protocol"]): float(r["delta"]) for r in rows}
    sig = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (r["significant_005"] == "yes") for r in rows}

    node_small = [100, 500, 1000]
    vabs = np.array([abs(v) for v in delta.values()], dtype=float)
    vmax = float(max(0.02, np.quantile(vabs, 0.95)))
    vmin = -vmax

    fig = plt.figure(figsize=(13.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.28)

    cmap = LinearSegmentedColormap.from_list(
        "soft_diverging",
        ["#DCEAF6", "#EEF4FA", "#FFFFFF", "#F8F0EA", "#E6C9B4"],
    )

    # (a)-(d): per-environment heatmaps (protocol x node)
    for i, env in enumerate(ENV_ORDER):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        matrix = np.zeros((len(PROTOCOL_ORDER), len(node_small)), dtype=float)
        sig_mask = np.zeros_like(matrix, dtype=bool)
        for r, p in enumerate(PROTOCOL_ORDER):
            for c, n in enumerate(node_small):
                matrix[r, c] = delta[(env, n, p)]
                sig_mask[r, c] = sig[(env, n, p)]

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(node_small)))
        ax.set_xticklabels([str(n) for n in node_small])
        ax.set_yticks(np.arange(len(PROTOCOL_ORDER)))
        ax.set_yticklabels(PROTOCOL_ORDER)
        ax.set_xlabel("Nodes")
        ax.set_title(ENV_LABEL[env], pad=6)
        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                label = f"{val:+.3f}"
                ax.text(c, r, label, ha="center", va="center", fontsize=7.2, color="#2E2E2E")
                if not sig_mask[r, c]:
                    ax.plot(c, r, marker="o", markersize=5.4, markerfacecolor="white", markeredgecolor="#383838", markeredgewidth=0.9)

    # shared colorbar
    cax = fig.add_axes([0.44, 0.06, 0.16, 0.02])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Delta PDR (tx5 - tx15)")

    # (e): protocol-level absolute sensitivity summary across 12 cells
    ax = fig.add_subplot(gs[:, 2])
    data = [np.array([abs(delta[(env, n, p)]) for env in ENV_ORDER for n in node_small], dtype=float) for p in PROTOCOL_ORDER]
    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        tick_labels=PROTOCOL_ORDER,
        whis=(5, 95),
        medianprops={"color": "#2A2A2A", "linewidth": 1.1},
        boxprops={"linewidth": 0.8, "edgecolor": "#63707D"},
        whiskerprops={"linewidth": 0.8, "color": "#63707D"},
        capprops={"linewidth": 0.8, "color": "#63707D"},
        flierprops={"marker": "o", "markersize": 2.8, "markerfacecolor": "#8DA5BC", "markeredgecolor": "#8DA5BC", "alpha": 0.45},
    )
    for patch, proto in zip(bp["boxes"], PROTOCOL_ORDER):
        patch.set_facecolor(PROTO_COLORS[proto])
        patch.set_alpha(0.75)
    ax.set_xlabel("|Delta PDR| across 4 env × 3 node cells")
    ax.set_title("Protocol sensitivity summary")
    panel_label(ax, "(e)")
    style_axes(ax)
    ax.grid(axis="x")

    stem = f"fig6_s10_power_sensitivity_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig7_ns3_trend() -> str:
    rows = load_csv(NS3_SIG_FILE)
    node_order = [50, 100, 200, 300, 500, 800, 1000]

    grouped: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        grouped[r["environment"]][int(r["node_count"])] = r

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.8), constrained_layout=True)
    axes = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        diffs_pp = []
        sigs = []
        gs = []
        for n in node_order:
            row = grouped[env][n]
            diffs_pp.append(float(row["diff"]) * 100.0)
            sigs.append(row["sig_holm_0_05"] == "YES")
            gs.append(float(row["hedges_g"]))

        y = np.array(diffs_pp, dtype=float)
        ax.plot(node_order, y, color="#4E79A7", marker="o", linewidth=3.0, markersize=5.8)
        for x, yy, sig_ok, gval in zip(node_order, y, sigs, gs):
            if sig_ok:
                ax.plot(x, yy, marker="o", markersize=6.0, markerfacecolor="#4E79A7", markeredgecolor="#2B3948")
            else:
                ax.plot(x, yy, marker="o", markersize=6.0, markerfacecolor="white", markeredgecolor="#2B3948", markeredgewidth=1.0)
            if x in (100, 1000):
                offset = 0.18 if yy >= 0 else -0.18
                ax.text(x, yy + offset, f"g={gval:.2f}", fontsize=7.4, ha="center", color="#4C5B69")

        ax.axhline(0.0, color="#404040", linewidth=0.85)
        ax.set_xticks(node_order)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("AERIS-LEACH delta PDR (pp)")
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.grid(axis="both")
        style_axes(ax)
        panel_label(ax, f"({chr(97 + i)})")

    axes[0].text(
        0.02,
        0.08,
        "hollow marker: Holm non-significant",
        transform=axes[0].transAxes,
        fontsize=7.6,
        color="#4E5D6E",
        bbox={"facecolor": "#F8FBFF", "edgecolor": "#D2DDE8", "boxstyle": "round,pad=0.14"},
    )

    stem = f"fig7_ns3_trend_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    fig0 = plot_fig0_workflow()
    fig1 = plot_fig1()
    fig2 = plot_fig2()
    fig3 = plot_fig3()
    fig4 = plot_fig4()
    fig5 = plot_fig5()
    fig6 = plot_fig6_s10()
    fig7 = plot_fig7_ns3_trend()
    print("Generated figures:")
    for stem in [fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        print(" ", FIG_DIR / f"{stem}.pdf")


if __name__ == "__main__":
    main()
