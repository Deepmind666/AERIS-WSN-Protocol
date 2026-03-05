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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260213_s12"

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

# Okabe-Ito palette (colorblind-safe), tuned for manuscript contrast.
PROTO_COLORS = {
    "AERIS": "#0072B2",
    "LEACH": "#D55E00",
    "PEGASIS": "#009E73",
    "HEED": "#CC79A7",
    "TEEN": "#E69F00",
}

PROTO_MARKERS = {
    "AERIS": "o",
    "LEACH": "s",
    "PEGASIS": "^",
    "HEED": "D",
    "TEEN": "P",
}

SCALABILITY_FILES = {
    "indoor_office": RESULTS_DIR / "scalability_indoor_office_server_s7_20260211.json",
    "indoor_factory": RESULTS_DIR / "scalability_indoor_factory_local_s9_20260213_010635.json",
    "outdoor_urban": RESULTS_DIR / "scalability_outdoor_urban_local_fix550_run4_20260211_210500.json",
    "outdoor_suburban": RESULTS_DIR / "scalability_outdoor_suburban_server_s7_20260211.json",
}

ENV_FILE = RESULTS_DIR / "env_sensitivity_20260207_205317.json"
ABLATION_FILE = RESULTS_DIR / "ablation_diag_multi_20260207_205448.json"
ENERGY_FILE = RESULTS_DIR / "energy_lifetime_stats.csv"
LATENCY_FILE = RESULTS_DIR / "latency_hop_v3_20260211_stats.csv"


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.1,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "grid.alpha": 0.13,
            "grid.linestyle": "-",
            "axes.grid": False,
            "axes.facecolor": "white",
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
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


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
        f.write("# Scalability Summary (S11)\n\n")
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
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3.0,
            color=[PROTO_COLORS[p] for p in PROTOCOL_ORDER],
            edgecolor="black",
            linewidth=0.7,
        )
        panel_label(ax, f"({chr(97 + idx)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER, rotation=0)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PDR")
        ax.grid(axis="y", alpha=0.16)
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
    im = axes[0].imshow(matrix, cmap="cividis", vmin=0, vmax=1, aspect="auto")
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
    axes[1].hlines(y + 0.14, 0, (no_gw - full) * 100, color="#D55E00", linewidth=2.2, label="no_gateway - full")
    axes[1].hlines(y - 0.14, 0, (no_cas - full) * 100, color="#0072B2", linewidth=2.2, label="no_cas - full")
    axes[1].plot((no_gw - full) * 100, y + 0.14, "o", color="#D55E00", markersize=5)
    axes[1].plot((no_cas - full) * 100, y - 0.14, "s", color="#0072B2", markersize=5)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER])
    axes[1].set_xlabel("Delta PDR (percentage points)")
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    axes[1].set_title("Marginal effects")
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(axis="x", alpha=0.22)

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
                markersize=4.5,
                color=PROTO_COLORS[proto],
                label=proto,
            )
            ax.fill_between(NODE_ORDER, means - ci, means + ci, color=PROTO_COLORS[proto], alpha=0.12, linewidth=0)
        panel_label(ax, f"({chr(97 + idx)})")
        style_axes(ax)
        ax.set_title(ENV_LABEL[env], pad=8)
        ax.set_xlabel("Number of nodes")
        ax.set_ylabel("PDR")
        if env == "indoor_office":
            # Dedicated zoom panel for the benign indoor setting where all
            # protocols cluster near 1.0 and small differences are meaningful.
            ax.set_ylim(0.985, 1.001)
        else:
            ax.set_ylim(0.0, 1.02)
        ax.set_xlim(90, 1010)
        ax.set_xticks(NODE_ORDER)
        ax.grid(axis="both", alpha=0.16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
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
        ax.barh(y, ranked_vals, color=[PROTO_COLORS[p] for p in ranked_proto], edgecolor="black", linewidth=0.6)
        style_axes(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(ranked_proto)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.16)
        panel_label(ax, f"({chr(97 + idx)})")
        for yi, vv in zip(y, ranked_vals):
            ax.text(vv, yi, f" {vv:.3f}" if metric != "life" else f" {vv:.1f}", va="center", fontsize=8)
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
