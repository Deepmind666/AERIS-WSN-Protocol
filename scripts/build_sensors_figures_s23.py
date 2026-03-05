#!/usr/bin/env python3
"""
Build publication-grade figures and summary tables for Sensors manuscript.

Design goals:
1) Colorblind-safe palette and consistent typography.
2) No text overlap or obstructed data marks.
3) Vector-first outputs (PDF + SVG) and 300 dpi PNG export.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260215_s23"

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

# Soft scientific palette (low saturation, publication friendly).
PROTO_COLORS = {
    "AERIS": "#D65A5A",   # muted red (primary method highlight)
    "LEACH": "#5D80B9",   # muted blue
    "PEGASIS": "#62A873", # muted green
    "HEED": "#8F79BA",    # muted purple
    "TEEN": "#D79A56",    # muted orange
}

PROTO_MARKERS = {
    "AERIS": "o",
    "LEACH": "s",
    "PEGASIS": "^",
    "HEED": "D",
    "TEEN": "P",
}

# Line style encoding keeps the figure readable when printed in grayscale.
PROTO_LINESTYLES = {
    "AERIS": "-",
    "LEACH": "--",
    "PEGASIS": "-.",
    "HEED": (0, (4, 1.4)),
    "TEEN": (0, (2, 1.2)),
}

SCALABILITY_FILES = {
    "indoor_office": RESULTS_DIR / "scalability_indoor_office_server_s8_20260213.json",
    "indoor_factory": RESULTS_DIR / "scalability_indoor_factory_server_s8_20260215.json",
    "outdoor_urban": RESULTS_DIR / "scalability_outdoor_urban_server_s8_20260213.json",
    "outdoor_suburban": RESULTS_DIR / "scalability_outdoor_suburban_server_s8_20260213.json",
}

ENV_FILE = RESULTS_DIR / "env_sensitivity_20260207_205317.json"
ABLATION_FILE = RESULTS_DIR / "ablation_diag_multi_20260207_205448.json"
ENERGY_FILE = RESULTS_DIR / "energy_lifetime_stats.csv"
LATENCY_FILE = RESULTS_DIR / "latency_hop_v3_20260211_stats.csv"
PRECOMP_DESC = RESULTS_DIR / "scalability_4env_s8_unified_20260215_descriptive.csv"
PRECOMP_SIG = RESULTS_DIR / "scalability_4env_s8_unified_20260215_significance.csv"


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            # Sans-serif for labels/ticks, Computer Modern for math symbols.
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.6,
            "lines.markersize": 5.6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "grid.alpha": 0.2,
            "grid.linestyle": "-",
            "axes.grid": False,
            "grid.color": "#DDE2EA",
            "axes.facecolor": "#FCFCFD",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.9},
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def draw_rounded_bars(
    ax: plt.Axes,
    x: np.ndarray,
    heights: Sequence[float],
    errors: Sequence[float],
    colors: Sequence[str],
    width: float = 0.8,
) -> None:
    """Draw rounded-corner bars to avoid harsh rectangular look."""
    for xi, h, c in zip(x, heights, colors):
        patch = FancyBboxPatch(
            (xi - width / 2, 0.0),
            width,
            max(0.0, float(h)),
            boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#7C8794",
            facecolor=c,
            mutation_aspect=1.0,
            clip_on=True,
            zorder=2,
        )
        ax.add_patch(patch)
    ax.errorbar(
        x,
        heights,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.0,
        capsize=3.0,
        zorder=3,
    )
    ax.set_xlim(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5)


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_mean_std(
    rows: Iterable[dict], value_fn, key_fields: Sequence[str]
) -> Dict[Tuple, Tuple[float, float, int]]:
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


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    pooled = ((nx - 1) * sx2 + (ny - 1) * sy2) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return 0.0
    d = (x.mean() - y.mean()) / math.sqrt(pooled)
    correction = 1.0 - 3.0 / max(4.0 * (nx + ny) - 9.0, 1.0)
    return float(d * correction)


def aggregate_scalability() -> Tuple[Path, Path, Path]:
    rows: List[dict] = []
    n_by_env: Dict[str, int] = {}

    for env, path in SCALABILITY_FILES.items():
        data = load_json(path)
        raw = [r for r in data["raw_results"] if not r.get("error")]
        for r in raw:
            rows.append(
                {
                    "environment": env,
                    "num_nodes": int(r["num_nodes"]),
                    "protocol": r["protocol"],
                    "seed": int(r["seed"]),
                    "pdr_expected": float(r["metrics"]["pdr_expected"]),
                }
            )
        cfg = data.get("config", {})
        denom = len(cfg.get("node_counts", [])) * len(cfg.get("protocols", []))
        n_by_env[env] = int(len(raw) / denom) if denom else 0

    desc_stats = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "num_nodes", "protocol"))
    desc_out = RESULTS_DIR / f"scalability_4env_mixed_{SUFFIX}_descriptive.csv"
    with desc_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["environment", "num_nodes", "protocol", "n", "pdr_mean", "pdr_std", "ci95_half_width"])
        for env in ENV_ORDER:
            for node in NODE_ORDER:
                for proto in PROTOCOL_ORDER:
                    mean, std, n = desc_stats[(env, node, proto)]
                    ci95 = 1.96 * std / math.sqrt(max(n, 1))
                    writer.writerow([env, node, proto, n, f"{mean:.6f}", f"{std:.6f}", f"{ci95:.6f}"])

    sig_rows = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            pvals = []
            temp = []
            aeris = np.asarray(
                [r["pdr_expected"] for r in rows if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == "AERIS"],
                dtype=float,
            )
            for baseline in BASELINES:
                b = np.asarray(
                    [r["pdr_expected"] for r in rows if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == baseline],
                    dtype=float,
                )
                t_stat, p_raw = stats.ttest_ind(aeris, b, equal_var=False)
                diff = float(aeris.mean() - b.mean())
                g = hedges_g(aeris, b)
                temp.append((baseline, diff, float(aeris.mean()), float(b.mean()), float(t_stat), float(p_raw), g))
                pvals.append(float(p_raw))

            p_holm = holm_bonferroni(pvals)
            for i, (baseline, diff, aeris_mean, base_mean, t_stat, p_raw, g) in enumerate(temp):
                sig_rows.append(
                    {
                        "environment": env,
                        "num_nodes": node,
                        "comparison": f"AERIS vs {baseline}",
                        "baseline": baseline,
                        "metric": "pdr_expected",
                        "aeris_mean": f"{aeris_mean:.6f}",
                        "baseline_mean": f"{base_mean:.6f}",
                        "diff": f"{diff:.6f}",
                        "hedges_g": f"{g:.6f}",
                        "t_stat": f"{t_stat:.6f}",
                        "p_value_raw": f"{p_raw:.6e}",
                        "p_value_holm": f"{p_holm[i]:.6e}",
                        "sig_holm_0_05": "yes" if p_holm[i] < 0.05 else "no",
                    }
                )

    sig_out = RESULTS_DIR / f"scalability_4env_mixed_{SUFFIX}_significance.csv"
    with sig_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "environment",
            "num_nodes",
            "comparison",
            "baseline",
            "metric",
            "aeris_mean",
            "baseline_mean",
            "diff",
            "hedges_g",
            "t_stat",
            "p_value_raw",
            "p_value_holm",
            "sig_holm_0_05",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sig_rows)

    md_out = RESULTS_DIR / f"scalability_4env_mixed_{SUFFIX}_summary.md"
    with md_out.open("w", encoding="utf-8") as f:
        f.write("# Scalability Summary (S20)\n\n")
        f.write("## Sample size by environment\n\n")
        for env in ENV_ORDER:
            f.write(f"- {env}: n={n_by_env[env]}\n")
        f.write("\n## 1000-node ranking (PDR mean)\n\n")
        for env in ENV_ORDER:
            rows_env = [r for r in desc_stats.items() if r[0][0] == env and r[0][1] == 1000]
            rank = sorted(rows_env, key=lambda kv: kv[1][0], reverse=True)
            f.write(f"- {env}: " + ", ".join(f"{k[2]}={v[0]:.4f}" for k, v in rank) + "\n")
    return desc_out, sig_out, md_out


def plot_figure1_env_pdr() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(PROTOCOL_ORDER))
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        means = [stats_map[(env, p)][0] for p in PROTOCOL_ORDER]
        stds = [stats_map[(env, p)][1] for p in PROTOCOL_ORDER]
        bars = ax.bar(
            x,
            means,
            width=0.74,
            color=[PROTO_COLORS[p] for p in PROTOCOL_ORDER],
            edgecolor="#5F6875",
            linewidth=0.85,
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#2E2E2E",
            elinewidth=1.0,
            capsize=3.0,
            zorder=3,
        )
        for b, v in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#3F4650",
            )
        panel_label(ax, f"({chr(97 + idx)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER, rotation=0)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PDR")
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Multi-environment PDR at 100 nodes (n=30 per environment)", fontsize=12)
    stem = f"fig1_env_pdr_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_figure2_ablation() -> str:
    rows = [r for r in load_json(ABLATION_FILE)["raw_results"] if not r.get("error")]
    pdr = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "ablation_config"))
    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr[(env, cfg)][0]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    soft_blues = LinearSegmentedColormap.from_list(
        "soft_blues",
        ["#FFFFFF", "#F4F9FF", "#E6F3FF", "#D3E8FA", "#BBD8F3"],
    )
    im = axes[0].imshow(matrix, cmap=soft_blues, vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=20, ha="right")
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in configs])
    panel_label(axes[0], "(a)")
    style_axes(axes[0])
    axes[0].set_title("PDR heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.45 else "black"
            axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.86, pad=0.02)
    cbar.set_label("PDR")

    full = np.array([pdr[(e, "full")][0] for e in ENV_ORDER])
    no_gw = np.array([pdr[(e, "no_gateway")][0] for e in ENV_ORDER])
    no_cas = np.array([pdr[(e, "no_cas")][0] for e in ENV_ORDER])
    y = np.arange(len(ENV_ORDER))
    gw_delta = (no_gw - full) * 100
    cas_delta = (no_cas - full) * 100
    axes[1].hlines(y + 0.14, 0, gw_delta, color="#D55E00", linewidth=2.2, label="no_gateway - full")
    axes[1].hlines(y - 0.14, 0, cas_delta, color="#0072B2", linewidth=2.2, label="no_cas - full")
    axes[1].plot(gw_delta, y + 0.14, "o", color="#D55E00", markersize=5)
    axes[1].plot(cas_delta, y - 0.14, "s", color="#0072B2", markersize=5)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    axes[1].set_xlabel("Delta PDR (percentage points)")
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    axes[1].set_title("Marginal effects")
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(axis="x", alpha=0.22)
    all_delta = np.concatenate([gw_delta, cas_delta])
    x_min = min(-2.8, float(all_delta.min()) - 0.35)
    x_max = max(2.8, float(all_delta.max()) + 0.35)
    axes[1].set_xlim(x_min, x_max)
    for yi, val in zip(y + 0.14, gw_delta):
        axoff = 0.08 if val >= 0 else -0.08
        ha = "left" if val >= 0 else "right"
        axes[1].text(val + axoff, yi, f"{val:+.2f}", va="center", ha=ha, fontsize=7.9, color="#7A3300")
    for yi, val in zip(y - 0.14, cas_delta):
        axoff = 0.08 if val >= 0 else -0.08
        ha = "left" if val >= 0 else "right"
        axes[1].text(val + axoff, yi, f"{val:+.2f}", va="center", ha=ha, fontsize=7.9, color="#003B67")

    fig.suptitle("Ablation effects by environment (n=30)", fontsize=12)
    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_figure3_scalability(desc_csv: Path) -> str:
    rows = load_csv(desc_csv)
    stats_map = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (float(r["pdr_mean"]), float(r["ci95_half_width"])) for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6), constrained_layout=True)
    axes = axes.flatten()
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for proto in PROTOCOL_ORDER:
            means = np.array([stats_map[(env, n, proto)][0] for n in NODE_ORDER], dtype=float)
            ci = np.array([stats_map[(env, n, proto)][1] for n in NODE_ORDER], dtype=float)
            ax.plot(
                NODE_ORDER,
                means,
                marker=PROTO_MARKERS[proto],
                linestyle=PROTO_LINESTYLES[proto],
                markersize=5.4,
                linewidth=2.8,
                color=PROTO_COLORS[proto],
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=proto,
            )
            ax.fill_between(NODE_ORDER, means - ci, means + ci, color=PROTO_COLORS[proto], alpha=0.18, linewidth=0)
        panel_label(ax, f"({chr(97 + idx)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=8)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        if env == "indoor_office":
            # Benign channel conditions collapse all methods near 1.0.
            # A narrow y-window preserves visible ranking differences.
            ax.set_ylim(0.989, 1.0002)
            ax.set_yticks([0.989, 0.992, 0.995, 0.998, 1.000])
            ax.text(
                0.98,
                0.05,
                "narrow y-axis",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.2", "alpha": 0.95},
            )
        elif env == "indoor_factory":
            ax.set_ylim(0.14, 1.02)
        elif env == "outdoor_suburban":
            ax.set_ylim(0.52, 1.02)
        else:
            ax.set_ylim(0.03, 0.93)
        ax.set_xlim(90, 1010)
        ax.set_xticks(NODE_ORDER)
        ax.grid(axis="both", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.01), edgecolor="0.8")
    fig.suptitle("Scalability trends with 95% CI bands (100-1000 nodes)", fontsize=12.2)
    stem = f"fig3_scalability_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_figure4_tradeoff() -> str:
    energy_rows = load_csv(ENERGY_FILE)
    latency_rows = load_csv(LATENCY_FILE)
    e_map = {(r["environment"], r["protocol"]): (float(r["pdr_mean"]), float(r["energy_mean"]), float(r["lifetime_mean"])) for r in energy_rows}
    l_map = {(r["environment"], r["protocol"]): float(r["hops_mean"]) for r in latency_rows}

    avg = {}
    for proto in PROTOCOL_ORDER:
        pdr_vals = [e_map[(env, proto)][0] for env in ENV_ORDER]
        energy_vals = [e_map[(env, proto)][1] for env in ENV_ORDER]
        life_vals = [e_map[(env, proto)][2] for env in ENV_ORDER]
        hop_vals = [l_map[(env, proto)] for env in ENV_ORDER]
        avg[proto] = {
            "pdr": float(np.mean(pdr_vals)),
            "energy": float(np.mean(energy_vals)),
            "hops": float(np.mean(hop_vals)),
            "life": float(np.mean(life_vals)),
        }

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), constrained_layout=True)
    panels = [
        ("pdr", "Average PDR", True),
        ("energy", "Average total energy (J)", False),
        ("hops", "Average hops to BS", False),
        ("life", "Average lifetime (rounds)", False),
    ]
    flat_axes = axes.flatten()
    for idx, (ax, (metric, xlabel, desc)) in enumerate(zip(flat_axes, panels)):
        vals = [avg[p][metric] for p in PROTOCOL_ORDER]
        order = np.argsort(vals)[::-1] if desc else np.argsort(vals)
        ranked_proto = [PROTOCOL_ORDER[i] for i in order]
        ranked_vals = [vals[i] for i in order]
        y = np.arange(len(ranked_proto))
        if metric == "hops":
            x0 = 1.0
            for yi, proto, vv in zip(y, ranked_proto, ranked_vals):
                ax.hlines(yi, x0, vv, color=PROTO_COLORS[proto], linewidth=3.8, alpha=0.92, zorder=2)
                ax.scatter(vv, yi, s=44, color=PROTO_COLORS[proto], edgecolors="#52606E", linewidths=0.6, zorder=3)
        else:
            bars = ax.barh(
                y,
                ranked_vals,
                color=[PROTO_COLORS[p] for p in ranked_proto],
                edgecolor="#586170",
                linewidth=0.8,
                alpha=0.95,
                zorder=2,
            )
            for b in bars:
                b.set_height(0.72)
        style_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(ranked_proto)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16)
        panel_label(ax, f"({chr(97 + idx)})")
        vmax = float(max(ranked_vals)) if ranked_vals else 1.0
        if metric == "hops":
            ax.set_xscale("log")
            ax.set_xlim(1.0, max(40.0, vmax * 1.1))
        else:
            ax.set_xlim(0.0, vmax * 1.10)
        for yi, vv in zip(y, ranked_vals):
            if metric == "life":
                label = f"{vv:.1f}"
            elif metric == "hops":
                label = f"{vv:.2f}"
            elif metric == "energy":
                label = f"{vv:.1f}"
            else:
                label = f"{vv:.3f}"
            if metric == "hops":
                x_text = vv * 1.06
            else:
                x_text = vv + vmax * 0.010
            ax.text(x_text, yi, label, va="center", ha="left", fontsize=8)
    axes[0, 0].set_title("Reliability ranking")
    axes[0, 1].set_title("Energy ranking")
    axes[1, 0].set_title("Hop-latency ranking")
    axes[1, 1].set_title("Lifetime ranking")
    fig.suptitle("Protocol trade-off profiles (environment-averaged, n=30 datasets)", fontsize=12)

    stem = f"fig4_tradeoff_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    # Use the S8 unified frozen matrix (balanced n=1000 per cell).
    if PRECOMP_DESC.exists() and PRECOMP_SIG.exists():
        desc_csv = PRECOMP_DESC
        sig_csv = PRECOMP_SIG
        md_out = RESULTS_DIR / f"scalability_4env_s8_unified_{SUFFIX}_summary.md"
        md_out.write_text(
            "# Scalability Summary (S23)\n\n"
            "- source: precomputed audited matrix `scalability_4env_s8_unified_20260215_*`\n",
            encoding="utf-8",
        )
    else:
        desc_csv, sig_csv, md_out = aggregate_scalability()
    fig1 = plot_figure1_env_pdr()
    fig2 = plot_figure2_ablation()
    fig3 = plot_figure3_scalability(desc_csv)
    fig4 = plot_figure4_tradeoff()

    print("Generated:")
    print(" ", desc_csv)
    print(" ", sig_csv)
    print(" ", md_out)
    print(" ", FIG_DIR / f"{fig1}.pdf")
    print(" ", FIG_DIR / f"{fig2}.pdf")
    print(" ", FIG_DIR / f"{fig3}.pdf")
    print(" ", FIG_DIR / f"{fig4}.pdf")


if __name__ == "__main__":
    main()
