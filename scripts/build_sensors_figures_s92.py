#!/usr/bin/env python3
"""
Sensors submission figures (S75): camera-ready readability refresh
for dense multi-protocol, multi-power panels.

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
import shutil
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
SUFFIX = "20260301_s92"

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
S10_DESC_FILE = RESULTS_DIR / "s10r_4env_merged_descriptive_20260227.csv"
S10_SIG_FILE = RESULTS_DIR / "s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260227.csv"
NS3_SIG_FILE = PROJECT_ROOT / "ns3_validation" / "results" / "ns3_scale_ext_1000_significance.csv"
NS3_5PROTO_DESC_FILE = PROJECT_ROOT / "ns3_validation" / "results" / "ns3_5proto_fullnodes_descriptive_20260226.csv"


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 12.4,
            "axes.labelsize": 12.8,
            "axes.titlesize": 13.4,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.6,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "axes.linewidth": 0.85,
            "lines.linewidth": 3.2,
            "lines.markersize": 6.5,
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
        fontsize=10.2,
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
    stem = f"fig0_aeris_workflow_{SUFFIX}"
    src_svg = FIG_DIR / "AERIS流程图_01.svg"
    src_png = FIG_DIR / "AERIS_workflow_01_embedded.png"
    archived_pdf = FIG_DIR / "fig0_aeris_workflow_20260228_s79.pdf"
    out_pdf = FIG_DIR / f"{stem}.pdf"
    out_png = FIG_DIR / f"{stem}.png"
    out_svg = FIG_DIR / f"{stem}.svg"

    if src_svg.exists():
        try:
            import cairosvg  # type: ignore

            cairosvg.svg2pdf(url=str(src_svg), write_to=str(out_pdf))
            cairosvg.svg2png(url=str(src_svg), write_to=str(out_png), dpi=360)
            shutil.copy2(src_svg, out_svg)
            return stem
        except Exception:
            # Fall through to raster/PDF fallback if conversion fails.
            pass

    if src_png.exists():
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(src_png)
            h, w = img.shape[:2]
            fig, ax = plt.subplots(figsize=(12.5, 12.5 * h / max(w, 1)))
            ax.imshow(img)
            ax.axis("off")
            fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
            fig.savefig(out_png, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return stem
        except Exception:
            pass

    if archived_pdf.exists():
        shutil.copy2(archived_pdf, out_pdf)
        return stem

    # Fallback: deterministic schematic if external SVG conversion is unavailable.
    fig, ax = plt.subplots(figsize=(11.2, 4.2), constrained_layout=True)
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.text(
        0.5,
        2.5,
        "Workflow source SVG missing or conversion failed.\nPlease provide AERIS流程图_01.svg.",
        ha="center",
        va="center",
        fontsize=12.0,
        color="#2B3A4A",
    )
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig1() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(PROTOCOL_ORDER))

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        means = [stats_map[(env, p)][0] for p in PROTOCOL_ORDER]
        stds = [stats_map[(env, p)][1] for p in PROTOCOL_ORDER]
        draw_rounded_bars(ax, x, means, stds, [PROTO_COLORS[p] for p in PROTOCOL_ORDER])
        for xi, val in zip(x, means):
            ax.text(xi, val + 0.013, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color="#3D4B5A")

        panel_label(ax, f"({chr(97 + i)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PDR")
        ax.grid(axis="y")

        if env == "outdoor_urban":
            # Keep one chart grammar across all panels; highlight low-range band without inset.
            ax.axhspan(0.0, 0.16, facecolor="#EEF5FB", alpha=0.24, zorder=0)
            ax.text(
                0.98,
                0.12,
                "low-range band",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9.0,
                color="#607487",
                bbox={"facecolor": "#F6FAFD", "edgecolor": "#D8E3ED", "pad": 0.18, "alpha": 0.85},
            )

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

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "soft_yellow_blue",
        ["#F7F2EC", "#F8FAFC", "#E3EDF6", "#C4D9EA", "#97BEDA"],
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
            txt_color = "#1A232C" if v >= (vmin + vmax) / 2.0 else "#F6F8FA"
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.0, color=txt_color, fontweight="semibold")
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

    axes[1].text(
        0.02,
        -0.18,
        "Positive delta: removing the module improves PDR; negative: removing the module degrades PDR.",
        transform=axes[1].transAxes,
        fontsize=9.0,
        color="#4A5968",
    )

    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig3() -> str:
    rows = load_csv(DESC_FILE)
    stats_map = {(r["environment"], int(r["num_nodes"]), r["protocol"]): (float(r["pdr_mean"]), float(r["ci95_half_width"])) for r in rows}

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
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

        # Keep one panel grammar per environment; details are resolved in Fig. 10.
        ax.text(
            0.98,
            0.05,
            "baseline detail: Fig. 10",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.7,
            color="#617789",
            bbox={"facecolor": "#F7FAFD", "edgecolor": "#D7E2EC", "pad": 0.16, "alpha": 0.86},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
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
        ("pdr", "Average PDR", True, "Reliability profile"),
        ("energy", "Average total energy (J)", False, "Energy profile"),
        ("hops", "Average hops to BS", False, "Hop-latency profile"),
        ("life", "Average lifetime (rounds)", False, "Lifetime profile"),
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
            ax.text(v + vmax * 0.009, yi, label, va="center", ha="left", fontsize=8.6)

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
    ax.set_ylabel("Delta PDR (absolute)")
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
    ax.set_ylabel("Delta PDR (absolute)")
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
    comparisons = ["tx5_vs_tx10", "tx10_vs_tx15", "tx5_vs_tx15"]
    cmp_label = {
        "tx5_vs_tx10": "tx5 - tx10",
        "tx10_vs_tx15": "tx10 - tx15",
        "tx5_vs_tx15": "tx5 - tx15",
    }

    delta_abs = {
        (r["comparison"], r["environment"], int(r["num_nodes"]), r["protocol"]): float(r["delta"])
        for r in rows
    }
    sig = {
        (r["comparison"], r["environment"], int(r["num_nodes"]), r["protocol"]): (r["significant_005"] == "YES")
        for r in rows
    }

    vabs = np.array([abs(v) for v in delta_abs.values()], dtype=float)
    vmax = float(max(0.12, np.quantile(vabs, 0.94)))

    fig, axes = plt.subplots(3, 4, figsize=(18.4, 12.4), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "s10r_diverging",
        ["#B55A36", "#F6E8DF", "#F2F7FC", "#4C82AF"],
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = None

    for rr, comp in enumerate(comparisons):
        for cc, env in enumerate(ENV_ORDER):
            ax = axes[rr, cc]
            matrix = np.zeros((len(PROTOCOL_ORDER), len(NODE_ORDER)), dtype=float)
            sig_mask = np.zeros_like(matrix, dtype=bool)
            for r_idx, proto in enumerate(PROTOCOL_ORDER):
                for c_idx, n in enumerate(NODE_ORDER):
                    key = (comp, env, n, proto)
                    matrix[r_idx, c_idx] = delta_abs[key]
                    sig_mask[r_idx, c_idx] = sig[key]

            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(np.arange(len(NODE_ORDER)))
            ax.set_xticklabels([str(n) for n in NODE_ORDER], fontsize=9.4)
            if rr == 2:
                ax.set_xlabel("Nodes")
            ax.set_yticks(np.arange(len(PROTOCOL_ORDER)))
            ax.set_yticklabels(PROTOCOL_ORDER, fontsize=9.2)
            if cc == 0:
                ax.set_ylabel(cmp_label[comp], fontsize=10.2)

            if rr == 0:
                ax.set_title(ENV_LABEL[env], pad=6)

            if rr == 0:
                panel_label(ax, f"({chr(97 + cc)})")
            style_axes(ax)
            ax.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(PROTOCOL_ORDER), 1), minor=True)
            ax.grid(which="minor", color="#E3EAF1", linewidth=0.55)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Show non-significant cells only.
            ns_y, ns_x = np.where(~sig_mask)
            if len(ns_x):
                ax.scatter(
                    ns_x,
                    ns_y,
                    marker="x",
                    s=26,
                    color="#2F2F2F",
                    linewidths=1.0,
                    zorder=3,
                )
            # Add compact numeric labels on the most decision-relevant row (tx5-vs-tx15).
            if comp == "tx5_vs_tx15":
                for r_idx in range(len(PROTOCOL_ORDER)):
                    for c_idx in range(len(NODE_ORDER)):
                        val = matrix[r_idx, c_idx]
                        txt_color = "#FFFFFF" if abs(val) > 0.60 * vmax else "#28313A"
                        ax.text(
                            c_idx,
                            r_idx,
                            f"{val:+.2f}",
                            ha="center",
                            va="center",
                            fontsize=8.3,
                            color=txt_color,
                            zorder=4,
                        )

    cb = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.015)
    cb.set_label(r"Delta PDR (absolute; positive means lower-tx is higher)")
    fig.text(
        0.5,
        0.005,
        "Rows: pairwise tx comparisons. Columns: environments. Cells: protocol x node-count deltas (full coverage). Bottom row includes per-cell labels; cross marker denotes Holm non-significant cell.",
        ha="center",
        va="bottom",
        fontsize=9.8,
        color="#4A5968",
    )

    stem = f"fig6_s10_delta_maps_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig7_ns3_trend() -> str:
    rows = load_csv(NS3_SIG_FILE)
    full_desc = load_csv(NS3_5PROTO_DESC_FILE)
    node_order = [50, 100, 200, 300, 500, 800, 1000]

    grouped: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        grouped[r["environment"]][int(r["node_count"])] = r

    desc_map = {
        (r["environment"], int(r["num_nodes"]), r["protocol"]): float(r["pdr_mean"])
        for r in full_desc
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.8), constrained_layout=False)
    axes = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes[i]
        sigs = []
        for n in node_order:
            row = grouped[env][n]
            sigs.append(row["sig_holm_0_05"] == "YES")

        panel_series = {}
        for proto in PROTOCOL_ORDER:
            panel_series[proto] = np.array([desc_map[(env, n, proto)] for n in node_order], dtype=float)

        for proto in PROTOCOL_ORDER:
            series = panel_series[proto]
            if proto in ("AERIS", "LEACH"):
                ax.plot(
                    node_order,
                    series,
                    color=PROTO_COLORS[proto],
                    marker=PROTO_MARKERS[proto],
                    linewidth=3.0 if proto == "AERIS" else 2.8,
                    markersize=6.8 if proto == "AERIS" else 6.3,
                    label=proto,
                    zorder=4 if proto == "AERIS" else 3,
                )
            else:
                ax.plot(
                    node_order,
                    series,
                    color=PROTO_COLORS[proto],
                    marker=PROTO_MARKERS[proto],
                    linewidth=1.35,
                    linestyle=PROTO_LINESTYLES[proto],
                    alpha=0.42,
                    markersize=4.0,
                    label=proto,
                    zorder=2,
                )

        # Mark node scales where AERIS-LEACH difference is not Holm-significant.
        aeris = panel_series["AERIS"]
        leach = panel_series["LEACH"]
        for x, sig_ok in zip(node_order, sigs):
            if not sig_ok:
                idx = node_order.index(x)
                y_mid = 0.5 * (float(aeris[idx]) + float(leach[idx]))
                ax.plot(
                    x,
                    y_mid,
                    marker="o",
                    markersize=10.5,
                    markerfacecolor="none",
                    markeredgecolor="#C23B31",
                    markeredgewidth=2.0,
                    zorder=5,
                )

        y_all = np.concatenate([panel_series[p] for p in PROTOCOL_ORDER])
        y_min = max(0.0, float(np.min(y_all)) - 0.03)
        y_max = min(1.0, float(np.max(y_all)) + 0.03)
        if y_max - y_min < 0.20:
            pad = (0.20 - (y_max - y_min)) / 2.0
            y_min = max(0.0, y_min - pad)
            y_max = min(1.0, y_max + pad)

        ax.set_xticks(node_order)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        ax.set_title(ENV_LABEL[env], pad=6)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="both")
        style_axes(ax)
        panel_label(ax, f"({chr(97 + i)})")
        delta_1000_pp = (aeris[-1] - leach[-1]) * 100.0
        ax.text(
            0.98,
            0.06,
            f"Δ@1000 = {delta_1000_pp:+.2f} pp",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.0,
            color="#566B7D",
            bbox={"facecolor": "#F7FAFD", "edgecolor": "#D8E3ED", "pad": 0.18, "alpha": 0.88},
        )

    handles = [
        plt.Line2D([0], [0], color=PROTO_COLORS[p], marker=PROTO_MARKERS[p], linestyle=PROTO_LINESTYLES[p], linewidth=2.4 if p in ("AERIS", "LEACH") else 1.9, markersize=5.6, label=p)
        for p in PROTOCOL_ORDER
    ]
    handles.append(
        plt.Line2D([0], [0], color="#C23B31", marker="o", markerfacecolor="none", linestyle="None", markersize=8.8, markeredgewidth=1.8, label="AERIS-LEACH Holm non-significant")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.025),
        frameon=True,
        framealpha=0.93,
        edgecolor="#C5D0DB",
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

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
    base_id = np.full((len(ENV_ORDER), len(NODE_ORDER)), "", dtype=object)

    baseline_tag = {
        "LEACH": "L",
        "PEGASIS": "P",
        "HEED": "H",
        "TEEN": "T",
    }

    for i, env in enumerate(ENV_ORDER):
        for j, n in enumerate(NODE_ORDER):
            cell_rows = [x for x in sig_rows if x["environment"] == env and int(x["num_nodes"]) == n]
            row = max(cell_rows, key=lambda x: float(x["baseline_mean"]))
            diff[i, j] = float(row["diff"]) * 100.0
            pvals[i, j] = float(row["p_value_holm"])
            sigmask[i, j] = row["sig_holm_0_05"].lower() == "yes"
            gvals[i, j] = float(row["hedges_g"])
            comp = row["comparison"].replace("AERIS", "").replace("aeris", "").replace("vs", "").replace("_", " ").strip()
            base_name = comp.upper()
            base_id[i, j] = baseline_tag.get(base_name, "?")

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
            axes[0].text(j, i, f"{diff[i,j]:+.2f}\n({base_id[i,j]})", ha="center", va="center", fontsize=9.0, color="#2A2A2A")
    panel_label(axes[0], "(a)")
    style_axes(axes[0])
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.84, pad=0.02)
    cb0.set_label("Delta PDR (pp)")

    # Use effect-size magnitude to avoid near-flat significance color fields at very small p-values.
    g_abs = np.abs(gvals)
    g_cap = max(5.0, float(np.quantile(g_abs, 0.92)))
    g_plot = np.minimum(g_abs, g_cap)
    cmap_b = LinearSegmentedColormap.from_list("g_soft", ["#F3F7FB", "#DCE9F5", "#A8C8E2", "#5C8EB6"])
    im1 = axes[1].imshow(g_plot, aspect="auto", cmap=cmap_b, vmin=0.0, vmax=g_cap)
    axes[1].set_xticks(np.arange(len(NODE_ORDER)))
    axes[1].set_xticklabels([str(n) for n in NODE_ORDER])
    axes[1].set_yticks(np.arange(len(ENV_ORDER)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Nodes")
    axes[1].set_title("v50-rigor effect magnitude to strongest baseline")
    for i in range(len(ENV_ORDER)):
        for j in range(len(NODE_ORDER)):
            mark = "Y" if sigmask[i, j] else "N"
            g_txt = f"{gvals[i,j]:.1f}"
            axes[1].text(
                j,
                i,
                f"{mark}\n|g|={abs(gvals[i,j]):.1f}\n{base_id[i,j]}",
                ha="center",
                va="center",
                fontsize=9.0,
                color="#1F2A35",
            )
            if not sigmask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False, hatch="///", edgecolor="#444444", linewidth=0.6)
                axes[1].add_patch(rect)
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.84, pad=0.02)
    cb1.set_label(r"$|g|$ to strongest baseline (capped)")

    axes[1].text(
        0.02,
        -0.16,
        "Y: Holm-corrected p < 0.05; baseline tags: L=LEACH, P=PEGASIS, H=HEED, T=TEEN",
        transform=axes[1].transAxes,
        fontsize=9.0,
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
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 10.6), constrained_layout=True)
    tx_styles = {
        5.0: "--",
        10.0: ":",
        15.0: "-",
    }

    baseline_set = [p for p in PROTOCOL_ORDER if p != "AERIS"]
    for i, env in enumerate(ENV_ORDER):
        top_ax = axes[0, i]
        bottom_ax = axes[1, i]
        env_max_base = 0.0

        # Row 1: AERIS-only profiles to avoid 15-line overlap.
        for tx in [5.0, 10.0, 15.0]:
            ys = [
                float(
                    [
                        r
                        for r in rows
                        if r["environment"] == env
                        and r["protocol"] == "AERIS"
                        and int(r["num_nodes"]) == n
                        and float(r["tx_power"]) == tx
                    ][0]["pdr_mean"]
                )
                for n in NODE_ORDER
            ]
            top_ax.plot(
                NODE_ORDER,
                ys,
                linestyle=tx_styles[tx],
                marker=PROTO_MARKERS["AERIS"],
                linewidth=2.85 if tx == 15.0 else 2.15,
                markersize=5.2,
                color=PROTO_COLORS["AERIS"],
                alpha=0.97 if tx == 15.0 else 0.84,
                label=f"AERIS tx{int(tx)}",
            )

        # Row 2: baseline protocols with all tx levels.
        for proto in baseline_set:
            for tx in [5.0, 10.0, 15.0]:
                ys = [
                    float(
                        [
                            r
                            for r in rows
                            if r["environment"] == env
                            and r["protocol"] == proto
                            and int(r["num_nodes"]) == n
                            and float(r["tx_power"]) == tx
                        ][0]["pdr_mean"]
                    )
                    for n in NODE_ORDER
                ]
                env_max_base = max(env_max_base, max(ys))
                bottom_ax.plot(
                    NODE_ORDER,
                    ys,
                    linestyle=tx_styles[tx],
                    marker=PROTO_MARKERS[proto],
                    linewidth=2.0 if tx == 15.0 else 1.45,
                    markersize=3.8,
                    color=PROTO_COLORS[proto],
                    alpha=0.92 if tx == 15.0 else 0.72,
                )

        for ax in (top_ax, bottom_ax):
            style_axes(ax)
            ax.set_xticks(NODE_ORDER)
            ax.set_xlim(85, 1015)
            ax.grid(axis="both")
            ax.set_xlabel("Nodes")
            ax.set_ylabel("PDR")

        top_ax.set_ylim(0.0, 1.02)
        # Baseline row uses a tighter y-range to keep low-PDR curves legible.
        bottom_cap = max(0.18, min(1.0, env_max_base + 0.08))
        bottom_ax.set_ylim(0.0, bottom_cap)
        bottom_ax.text(
            0.98,
            0.96,
            f"zoom max={bottom_cap:.2f}",
            transform=bottom_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="#5C6D7C",
        )

        top_ax.set_title(f"{ENV_LABEL[env]} (AERIS-only)", pad=5.5)
        bottom_ax.set_title(f"{ENV_LABEL[env]} (Baselines, zoomed y-axis)", pad=5.5)
        panel_label(top_ax, f"({chr(97 + i)})")

    # Build a compact two-part legend: tx linestyles + protocol colors.
    tx_handles = [
        plt.Line2D([0], [0], color="#4F4F4F", linestyle="-", linewidth=2.2, label="tx15"),
        plt.Line2D([0], [0], color="#4F4F4F", linestyle=":", linewidth=2.2, label="tx10"),
        plt.Line2D([0], [0], color="#4F4F4F", linestyle="--", linewidth=2.2, label="tx5"),
    ]
    proto_handles = [
        plt.Line2D([0], [0], color=PROTO_COLORS[p], marker=PROTO_MARKERS[p], linestyle="-", linewidth=2.0, markersize=5, label=p)
        for p in PROTOCOL_ORDER
    ]
    fig.legend(
        handles=tx_handles + proto_handles,
        loc="lower center",
        ncol=8,
        bbox_to_anchor=(0.5, 0.005),
        frameon=True,
        framealpha=0.94,
        edgecolor="#C5D0DB",
    )

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
