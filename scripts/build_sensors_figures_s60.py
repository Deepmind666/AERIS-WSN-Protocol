#!/usr/bin/env python3
"""
Sensors submission figures (S60): readability and camera-ready polish
with denser low-value readability in multi-environment panels.

Outputs:
- fig0_aeris_workflow_20260222_s56.{pdf,svg,png}
- fig1_env_pdr_panel_20260222_s56.{pdf,svg,png}
- fig2_ablation_panel_20260222_s56.{pdf,svg,png}
- fig3_scalability_panel_20260222_s56.{pdf,svg,png}
- fig4_tradeoff_panel_20260222_s56.{pdf,svg,png}
- fig5_s11_patch_control_delta_20260222_s56.{pdf,svg,png}
- fig6_s10_power_sensitivity_20260222_s56.{pdf,svg,png}
- fig7_ns3_trend_panel_20260222_s56.{pdf,svg,png}
- fig8_s8_significance_heatmap_20260222_s56.{pdf,svg,png}
- fig9_s9_s11_consistency_20260222_s56.{pdf,svg,png}
- fig10_s10_absolute_profiles_20260222_s56.{pdf,svg,png}
- fig11_s11_significance_panel_20260222_s56.{pdf,svg,png}
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260225_s69"

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
    "AERIS": "#4A86B8",
    "LEACH": "#CC8A62",
    "PEGASIS": "#68AA94",
    "HEED": "#AE8CB7",
    "TEEN": "#C3A052",
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
DESC_FILE = RESULTS_DIR / "scalability_4env_v50rigor_20260222_descriptive.csv"
S11_DELTA_FILE = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_delta.csv"
S11_SIG_FILE = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_significance.csv"
S9_DELTA_FILE = RESULTS_DIR / "s9_matched_4env_patch_vs_control_20260216_delta.csv"
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
            "font.size": 11.4,
            "axes.labelsize": 12.1,
            "axes.titlesize": 12.6,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 9.6,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "axes.linewidth": 0.85,
            "lines.linewidth": 3.0,
            "lines.markersize": 6.2,
            "grid.color": "#E3E9EF",
            "grid.alpha": 0.42,
            "grid.linewidth": 0.72,
            "axes.titleweight": "semibold",
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
    ax.spines["left"].set_linewidth(0.78)
    ax.spines["bottom"].set_linewidth(0.78)
    ax.spines["left"].set_color("#687685")
    ax.spines["bottom"].set_color("#687685")


def panel_label(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.02,
        0.96,
        tag,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#E0E5EA", "pad": 0.18, "alpha": 0.95},
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
                linewidth=0.62,
                edgecolor="#6E7F90",
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
        elinewidth=1.05,
        capsize=3.2,
        zorder=3,
    )
    ax.set_xlim(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5)


def plot_fig0_workflow() -> str:
    fig, ax = plt.subplots(figsize=(11.2, 4.2), constrained_layout=True)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    def node(x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> Tuple[float, float]:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.9,
            edgecolor="#5F6F7F",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1F2A35")
        ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=8.2, color="#334455")
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

    c_a = "#ECF4FB"
    c_b = "#EEF7F2"
    c_c = "#F7F0FB"
    c_d = "#FAF3EA"
    c_e = "#F3F4F6"

    p1 = node(0.4, 2.7, 2.6, 1.45, "Sensing Nodes", "packet generation\nenergy update", c_a)
    p2 = node(3.2, 2.7, 2.8, 1.45, "CH Election", "fuzzy score + fairness\ncluster formation", c_b)
    p3 = node(6.3, 2.7, 2.8, 1.45, "CAS Switching", "direct / chain / two-hop\nmode decision", c_c)
    p4 = node(9.4, 2.7, 2.9, 1.45, "Uplink Routing", "Gateway / Skeleton\nCH-to-BS forwarding", c_d)
    p5 = node(12.6, 2.7, 2.0, 1.45, "BS", "packet sink\nmetric update", c_e)

    q1 = node(4.3, 0.6, 2.8, 1.35, "Safety Layer", "retry + power adaptation\nconstraint handling", c_a)
    q2 = node(7.4, 0.6, 2.8, 1.35, "Evidence Logger", "PDR, energy,\nhops, lifetime", c_b)
    q3 = node(10.5, 0.6, 3.0, 1.35, "Publication Bundle", "CSV/JSON + sidecar\nhash + config", c_c)

    arrow((p1[0], p1[1]), (3.2, 3.42))
    arrow((p2[0], p2[1]), (6.3, 3.42))
    arrow((p3[0], p3[1]), (9.4, 3.42))
    arrow((p4[0], p4[1]), (12.6, 3.42))
    arrow((10.85, 2.7), (10.85, 1.95))
    arrow((7.1, 1.26), (7.4, 1.26))
    arrow((10.2, 1.26), (10.5, 1.26))
    arrow((7.7, 2.7), (5.65, 1.95))

    ax.text(0.4, 4.6, "AERIS round-level workflow and evidence pipeline", fontsize=12.2, fontweight="bold", color="#1F2A35")
    ax.text(0.4, 4.28, "Routing, safety control, and publication-tier evidence recording in one compact execution loop.", fontsize=9.1, color="#4D5E70")

    panel_label(ax, "(a)")
    stem = f"fig0_aeris_workflow_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig1() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
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
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    im = axes[0].imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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
            txt_color = "#151515" if v >= (vmin + vmax) / 2.0 else "#F6F8FA"
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.3, color=txt_color)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.84, pad=0.02)
    cb.set_label("PDR")
    cb.ax.tick_params(labelsize=9.5)

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

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
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
                markeredgewidth=0.65,
                linewidth=3.1 if proto == "AERIS" else 2.45,
                markersize=6.0 if proto == "AERIS" else 5.0,
                label=proto,
            )
            # Keep confidence bands but make them subtle; n=3200 makes most bands very narrow.
            ax.fill_between(NODE_ORDER, means - ci, means + ci, color=PROTO_COLORS[proto], alpha=0.06, linewidth=0)

        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        ax.set_xlim(90, 1010)
        ax.set_xticks(NODE_ORDER)
        ax.set_ylim(0.0, 1.02)
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
        handlelength=2.5,
        columnspacing=1.05,
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

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
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

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

    ax = axes[0]
    env_curve_colors = {
        "indoor_office": "#5E86AE",
        "indoor_factory": "#A46D4E",
        "outdoor_urban": "#7F9A53",
        "outdoor_suburban": "#8A6FA8",
    }
    for env in ENV_ORDER:
        vals = np.array([delta[(env, n, "AERIS")] for n in NODE_ORDER], dtype=float)
        ax.plot(
            NODE_ORDER,
            vals,
            marker="o",
            linewidth=2.4,
            markersize=5.0,
            color=env_curve_colors[env],
            label=ENV_LABEL[env],
        )
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

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), constrained_layout=True)
    axes = axes.flatten()

    # Colorblind-safe diverging map: tx5 worse than tx15 (negative) -> warm, positive -> cool.
    cmap = LinearSegmentedColormap.from_list(
        "s10_diverging",
        ["#D98E61", "#F6EFE8", "#ECF3FA", "#5E8FB6"],
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = None

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        matrix = np.zeros((len(PROTOCOL_ORDER), len(node_small)), dtype=float)
        sig_mask = np.zeros_like(matrix, dtype=bool)
        for r, proto in enumerate(PROTOCOL_ORDER):
            for c, n in enumerate(node_small):
                matrix[r, c] = delta[(env, n, proto)]
                sig_mask[r, c] = sig[(env, n, proto)]

        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
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
                ax.text(c, r, f"{val:+.3f}", ha="center", va="center", fontsize=7.2, color="#1F1F1F")
                if not sig_mask[r, c]:
                    ax.plot(c, r, marker="x", markersize=5.8, color="#2F2F2F", markeredgewidth=1.2, zorder=3)
                else:
                    ax.plot(c, r, marker="o", markersize=2.8, markerfacecolor="white", markeredgecolor="#2A2A2A", markeredgewidth=0.8, zorder=3)

    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"Delta PDR (tx5 - tx15, percentage points)")
    fig.text(0.5, 0.014, "x marker: Holm non-significant cell", ha="center", fontsize=8.0, color="#4A5968")

    stem = f"fig6_s10_delta_maps_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig7_ns3_trend() -> str:
    rows = load_csv(NS3_SIG_FILE)
    node_order = [50, 100, 200, 300, 500, 800, 1000]

    grouped: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        grouped[r["environment"]][int(r["node_count"])] = r

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    axes = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        aeris = []
        leach = []
        diff_pp = []
        sigs = []
        for n in node_order:
            row = grouped[env][n]
            aeris.append(float(row["aeris_mean"]))
            leach.append(float(row["baseline_mean"]))
            diff_pp.append(float(row["diff"]) * 100.0)
            sigs.append(row["sig_holm_0_05"] == "YES")

        aeris = np.array(aeris, dtype=float)
        leach = np.array(leach, dtype=float)
        diff_pp = np.array(diff_pp, dtype=float)

        ax.plot(
            node_order,
            aeris,
            color=PROTO_COLORS["AERIS"],
            marker=PROTO_MARKERS["AERIS"],
            linewidth=2.9,
            markersize=5.6,
            label="AERIS",
        )
        ax.plot(
            node_order,
            leach,
            color=PROTO_COLORS["LEACH"],
            marker=PROTO_MARKERS["LEACH"],
            linewidth=2.6,
            markersize=5.2,
            label="LEACH",
        )

        # Mark node scales where AERIS-LEACH difference is not Holm-significant.
        for x, sig_ok in zip(node_order, sigs):
            if not sig_ok:
                y_mid = 0.5 * (float(aeris[node_order.index(x)]) + float(leach[node_order.index(x)]))
                ax.plot(x, y_mid, marker="x", markersize=7.0, color="#2B3948", markeredgewidth=1.25, zorder=4)

        ax.set_xticks(node_order)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="both")
        style_axes(ax)
        panel_label(ax, f"({chr(97 + i)})")

        ax.text(
            0.98,
            0.06,
            f"?@1000={diff_pp[-1]:+.2f} pp",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.6,
            color="#495C70",
            bbox={"facecolor": "#F8FBFF", "edgecolor": "#D2DDE8", "boxstyle": "round,pad=0.14"},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        framealpha=0.93,
        edgecolor="#C5D0DB",
    )

    axes[0].text(
        0.02,
        0.08,
        "x marker: Holm non-significant",
        transform=axes[0].transAxes,
        fontsize=7.6,
        color="#4E5D6E",
        bbox={"facecolor": "#F8FBFF", "edgecolor": "#D2DDE8", "boxstyle": "round,pad=0.14"},
    )

    stem = f"fig7_ns3_trend_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem

def plot_fig8_s8_significance_heatmap() -> str:
    # Use v50-rigor significance rows and pick the strongest baseline per environment-node cell.
    # "Strongest baseline" is defined by the largest baseline_mean in that cell.
    sig_rows = load_csv(RESULTS_DIR / "scalability_4env_v50rigor_20260222_significance.csv")

    diff = np.zeros((len(ENV_ORDER), len(NODE_ORDER)), dtype=float)
    pvals = np.zeros((len(ENV_ORDER), len(NODE_ORDER)), dtype=float)
    sigmask = np.zeros((len(ENV_ORDER), len(NODE_ORDER)), dtype=bool)
    gvals = np.zeros((len(ENV_ORDER), len(NODE_ORDER)), dtype=float)

    for i, env in enumerate(ENV_ORDER):
        for j, n in enumerate(NODE_ORDER):
            cell_rows = [x for x in sig_rows if x["environment"] == env and int(x["num_nodes"]) == n]
            row = max(cell_rows, key=lambda x: float(x["baseline_mean"]))
            diff[i, j] = float(row["diff"]) * 100.0
            pvals[i, j] = float(row["p_value_holm"])
            sigmask[i, j] = row["sig_holm_0_05"].lower() == "yes"
            gvals[i, j] = float(row["hedges_g"])

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), constrained_layout=True)
    cmap_a = LinearSegmentedColormap.from_list("delta_soft", ["#F1DCCF", "#FAFBFC", "#CFE1F0"])
    # Use a zero-centered diverging normalization so positive/negative deltas are equally readable.
    delta_lim = max(10.0, float(np.max(np.abs(diff))))
    im0 = axes[0].imshow(
        diff,
        aspect="auto",
        cmap=cmap_a,
        norm=TwoSlopeNorm(vmin=-delta_lim, vcenter=0.0, vmax=delta_lim),
    )
    axes[0].set_xticks(np.arange(len(NODE_ORDER)))
    axes[0].set_xticklabels([str(n) for n in NODE_ORDER])
    axes[0].set_yticks(np.arange(len(ENV_ORDER)))
    axes[0].set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    axes[0].set_xlabel("Nodes")
    axes[0].set_title("v50-rigor delta to strongest baseline (pp)")
    for i in range(len(ENV_ORDER)):
        for j in range(len(NODE_ORDER)):
            axes[0].text(j, i, f"{diff[i,j]:+.2f}", ha="center", va="center", fontsize=8.3, color="#2A2A2A")
    panel_label(axes[0], "(a)")
    style_axes(axes[0])
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.84, pad=0.02)
    cb0.set_label("Delta PDR (pp)")

    # Use Holm significance strength directly and clip at high quantile to avoid full saturation.
    p_floor = 1e-300
    p_strength = -np.log10(np.maximum(pvals, p_floor))
    p_cap = float(np.quantile(p_strength, 0.95))
    p_plot = np.minimum(p_strength, p_cap)
    cmap_b = LinearSegmentedColormap.from_list("sig_soft", ["#F3F6FB", "#D7E7F4", "#9FC3E0", "#5E8FB6"])
    im1 = axes[1].imshow(p_plot, aspect="auto", cmap=cmap_b)
    axes[1].set_xticks(np.arange(len(NODE_ORDER)))
    axes[1].set_xticklabels([str(n) for n in NODE_ORDER])
    axes[1].set_yticks(np.arange(len(ENV_ORDER)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Nodes")
    axes[1].set_title("v50-rigor significance strength")
    for i in range(len(ENV_ORDER)):
        for j in range(len(NODE_ORDER)):
            mark = "Y" if sigmask[i, j] else "N"
            g_txt = f"{gvals[i,j]:.1f}"
            axes[1].text(
                j,
                i,
                f"{mark}\ng={g_txt}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="#1F2A35",
            )
            if not sigmask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, hatch="///", edgecolor="#444444", linewidth=0.6)
                axes[1].add_patch(rect)
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.84, pad=0.02)
    cb1.set_label(r"$-\log_{10}(\mathrm{Holm}\ p)$ (clipped)")

    axes[1].text(
        0.02,
        -0.16,
        "Y: Holm-corrected p < 0.05",
        transform=axes[1].transAxes,
        fontsize=8.2,
        color="#4A5968",
    )

    stem = f"fig8_s8_significance_heatmap_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig9_s9_s11_consistency() -> str:
    s9 = load_csv(S9_DELTA_FILE)
    s11 = load_csv(S11_DELTA_FILE)
    key = lambda r: (r["environment"], int(r["num_nodes"]), r["protocol"])
    s9_map = {key(r): float(r["delta"]) for r in s9}
    s11_map = {key(r): float(r["delta"]) for r in s11}
    common = sorted(set(s9_map.keys()) & set(s11_map.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    ax = axes[0]
    for proto in PROTOCOL_ORDER:
        xs = [s9_map[k] for k in common if k[2] == proto]
        ys = [s11_map[k] for k in common if k[2] == proto]
        ax.scatter(xs, ys, s=35, alpha=0.85, color=PROTO_COLORS[proto], edgecolor="white", linewidth=0.6, label=proto)
    lim = [min(min(s9_map.values()), min(s11_map.values())) - 0.02, max(max(s9_map.values()), max(s11_map.values())) + 0.02]
    ax.plot(lim, lim, color="#596A7A", linewidth=1.0, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("S9 delta (patch-control)")
    ax.set_ylabel("S11 delta (patch-control)")
    ax.set_title("S9 vs S11 cell-wise consistency")
    ax.grid(axis="both")
    panel_label(ax, "(a)")
    style_axes(ax)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92, edgecolor="#C5D0DB")

    # Environment-level mean deltas for AERIS only, easier to read.
    ax = axes[1]
    for env in ENV_ORDER:
        s9_vals = np.array([s9_map[(env, n, "AERIS")] for n in NODE_ORDER], dtype=float)
        s11_vals = np.array([s11_map[(env, n, "AERIS")] for n in NODE_ORDER], dtype=float)
        ax.plot(NODE_ORDER, s9_vals, marker="o", linewidth=2.2, markersize=4.6, color=PROTO_COLORS["AERIS"], alpha=0.45)
        ax.plot(NODE_ORDER, s11_vals, marker="s", linewidth=2.2, markersize=4.6, color="#8C4E2B", alpha=0.85)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("AERIS delta (patch-control)")
    ax.set_title("AERIS stress deltas: S9 (o) vs S11 (s)")
    ax.set_xticks(NODE_ORDER)
    ax.axhline(0.0, color="#4A4A4A", linewidth=0.8)
    ax.grid(axis="both")
    panel_label(ax, "(b)")
    style_axes(ax)

    stem = f"fig9_s9_s11_consistency_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig10_s10_absolute_profiles() -> str:
    rows = load_csv(S10_DESC_FILE)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=True)
    axes = axes.flatten()
    node_small = [100, 500, 1000]

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        for proto in PROTOCOL_ORDER:
            y5 = [float([r for r in rows if r["environment"] == env and r["protocol"] == proto and int(r["num_nodes"]) == n and float(r["tx_power"]) == 5.0][0]["pdr_mean"]) for n in node_small]
            y15 = [float([r for r in rows if r["environment"] == env and r["protocol"] == proto and int(r["num_nodes"]) == n and float(r["tx_power"]) == 15.0][0]["pdr_mean"]) for n in node_small]
            ax.plot(node_small, y5, linestyle="--", marker=PROTO_MARKERS[proto], linewidth=1.8, markersize=4.2, color=PROTO_COLORS[proto], alpha=0.75)
            ax.plot(node_small, y15, linestyle="-", marker=PROTO_MARKERS[proto], linewidth=2.3, markersize=4.8, color=PROTO_COLORS[proto], alpha=0.95)

        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("PDR")
        ax.set_xticks(node_small)
        ax.grid(axis="both")
        # Use the same y-range for all panels to avoid clipping low-baseline series.
        ax.set_ylim(0.0, 1.02)

    # compact legend with semantics
    l1, = axes[0].plot([], [], color="#4F4F4F", linestyle="-", linewidth=2.1, label="tx15")
    l2, = axes[0].plot([], [], color="#4F4F4F", linestyle="--", linewidth=2.1, label="tx5")
    handles_proto = [plt.Line2D([0], [0], color=PROTO_COLORS[p], marker=PROTO_MARKERS[p], linestyle="-", linewidth=2.0, markersize=5, label=p) for p in PROTOCOL_ORDER]
    fig.legend(handles=[l1, l2] + handles_proto, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01), frameon=True, framealpha=0.94, edgecolor="#C5D0DB")

    stem = f"fig10_s10_absolute_profiles_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig11_s11_significance_panel() -> str:
    rows = load_csv(S11_SIG_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)

    # (a) AERIS effect size map
    ax = axes[0]
    aeris = [r for r in rows if r["protocol"] == "AERIS"]
    gmat = np.zeros((len(ENV_ORDER), len(NODE_ORDER)))
    for i, env in enumerate(ENV_ORDER):
        for j, node in enumerate(NODE_ORDER):
            row = next(r for r in aeris if r["environment"] == env and int(r["num_nodes"]) == node)
            gmat[i, j] = float(row["hedges_g"])
    vmax = max(1.0, float(np.max(np.abs(gmat))))
    cmap = LinearSegmentedColormap.from_list("s11g", ["#4C6A8A", "#F2F5F8", "#B55442"])
    im = ax.imshow(gmat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(NODE_ORDER)))
    ax.set_xticklabels(NODE_ORDER)
    ax.set_yticks(range(len(ENV_ORDER)))
    ax.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    ax.set_xlabel("Nodes")
    ax.set_title("AERIS patch-control effect size ($g$)")
    for i in range(len(ENV_ORDER)):
        for j in range(len(NODE_ORDER)):
            val = gmat[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8.3, color="#222222")
    style_axes(ax)
    panel_label(ax, "(a)")
    cb = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.04)
    cb.set_label("Hedges' $g$")

    # (b) significance counts by protocol
    ax = axes[1]
    counts = []
    for proto in PROTOCOL_ORDER:
        proto_rows = [r for r in rows if r["protocol"] == proto]
        sig = sum(1 for r in proto_rows if r["significant_005"].lower() == "yes")
        counts.append(sig)
    x = np.arange(len(PROTOCOL_ORDER))
    bars = ax.bar(x, counts, color=[PROTO_COLORS[p] for p in PROTOCOL_ORDER], edgecolor="#667788", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(PROTOCOL_ORDER)
    ax.set_ylim(0, len(ENV_ORDER) * len(NODE_ORDER) + 1)
    ax.set_ylabel("Significant cells (Holm < 0.05)")
    ax.set_title("S11 significance density by protocol")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2.0, c + 0.2, f"{c}/24", ha="center", va="bottom", fontsize=8.5)
    ax.grid(axis="y", alpha=0.28)
    style_axes(ax)
    panel_label(ax, "(b)")

    stem = f"fig11_s11_significance_panel_{SUFFIX}"
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
    fig8 = plot_fig8_s8_significance_heatmap()
    fig9 = plot_fig9_s9_s11_consistency()
    fig10 = plot_fig10_s10_absolute_profiles()
    fig11 = plot_fig11_s11_significance_panel()
    print("Generated figures:")
    for stem in [fig0, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11]:
        print(" ", FIG_DIR / f"{stem}.pdf")


if __name__ == "__main__":
    main()
