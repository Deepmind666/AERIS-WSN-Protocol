#!/usr/bin/env python3
"""Generate publication-quality Sensors figures from validated AERIS result files.

This script intentionally uses only publication-tier files and produces
vector + raster outputs for manuscript inclusion.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"
SUFFIX = "20260210_mdpi"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
PROTOCOL_ORDER = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]

ENV_LABEL = {
    "indoor_office": "Indoor office",
    "indoor_factory": "Indoor factory",
    "outdoor_urban": "Outdoor urban",
    "outdoor_suburban": "Outdoor suburban",
}

COLORS = {
    "AERIS": "#1f77b4",
    "LEACH": "#d62728",
    "PEGASIS": "#2ca02c",
    "HEED": "#ff7f0e",
    "TEEN": "#9467bd",
}


def apply_style() -> None:
    """Apply a consistent publication style with readable defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "grid.alpha": 0.25,
        }
    )


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    """Save each figure as PDF/SVG/PNG for manuscript + preview usage."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")


def group_mean_std(rows: List[dict], value_key: str, key_fields: Tuple[str, ...]) -> Dict[Tuple, Tuple[float, float, int]]:
    """Aggregate mean/std/sample-size for numeric fields by key tuple."""
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        val = row.get(value_key)
        if val is None:
            continue
        groups[key].append(float(val))

    out: Dict[Tuple, Tuple[float, float, int]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, int(len(arr)))
    return out


def load_env_rows() -> List[dict]:
    path = RESULTS_DIR / "env_sensitivity_20260207_205317.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["raw_results"] if not r.get("error")]


def load_ablation_rows() -> List[dict]:
    path = RESULTS_DIR / "ablation_diag_multi_20260207_205448.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["raw_results"] if not r.get("error")]


def load_scalability_rows() -> List[dict]:
    path = RESULTS_DIR / "pre_ns3_scalability_summary_20260210_231438.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_energy_rows() -> List[dict]:
    path = RESULTS_DIR / "energy_lifetime_stats.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_latency_rows() -> List[dict]:
    path = RESULTS_DIR / "latency_hop_v2_stats.csv"
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_env_pdr_panel() -> None:
    """Figure 1: 2x2 panel for 100-node PDR by environment."""
    rows = load_env_rows()
    stats = group_mean_std(rows, "pdr_expected", ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    axes = axes.flatten()

    x = np.arange(len(PROTOCOL_ORDER))
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        means = []
        stds = []
        for proto in PROTOCOL_ORDER:
            m, s, _ = stats[(env, proto)]
            means.append(m)
            stds.append(s)

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=2.0,
            color=[COLORS[p] for p in PROTOCOL_ORDER],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(ENV_LABEL[env])
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.04)
        ax.grid(axis="y", linestyle="--")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 1. PDR comparison at 100 nodes across four channel environments (n=30)", fontsize=12)
    save_all_formats(fig, f"fig1_env_pdr_panel_{SUFFIX}")
    plt.close(fig)


def plot_ablation_panel() -> None:
    """Figure 2: Heatmap + effect bars for full/no_gateway/no_cas."""
    rows = load_ablation_rows()
    pdr_stats = group_mean_std(rows, "pdr_expected", ("environment", "ablation_config"))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr_stats[(env, cfg)][0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    im = axes[0].imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels([ENV_LABEL[e].replace(" ", "\n") for e in ENV_ORDER])
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in configs])
    axes[0].set_title("(a) Ablation PDR heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt_color = "white" if matrix[i, j] < 0.45 else "black"
            axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=txt_color, fontsize=8)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.88)
    cbar.set_label("PDR")

    full_vals = [pdr_stats[(env, "full")][0] for env in ENV_ORDER]
    nogw_vals = [pdr_stats[(env, "no_gateway")][0] for env in ENV_ORDER]
    nocas_vals = [pdr_stats[(env, "no_cas")][0] for env in ENV_ORDER]
    x = np.arange(len(ENV_ORDER))
    w = 0.25
    axes[1].bar(x - w, (np.array(nogw_vals) - np.array(full_vals)) * 100.0, width=w, color="#e74c3c", edgecolor="black", linewidth=0.5, label="no_gateway - full")
    axes[1].bar(x + w, (np.array(nocas_vals) - np.array(full_vals)) * 100.0, width=w, color="#3498db", edgecolor="black", linewidth=0.5, label="no_cas - full")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ENV_LABEL[e].replace(" ", "\n") for e in ENV_ORDER])
    axes[1].set_ylabel("Delta PDR (percentage points)")
    axes[1].set_title("(b) Marginal effect relative to full")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(axis="y", linestyle="--")

    fig.suptitle("Figure 2. Multi-environment ablation results (n=30)", fontsize=12)
    save_all_formats(fig, f"fig2_ablation_panel_{SUFFIX}")
    plt.close(fig)


def plot_scalability_panel() -> None:
    """Figure 3: 2x2 scalability trend + indoor-office zoom inset."""
    rows = load_scalability_rows()

    stats: Dict[Tuple[str, int, str], Tuple[float, float]] = {}
    for r in rows:
        key = (r["environment"], int(r["num_nodes"]), r["protocol"])
        stats[key] = (float(r["pdr_mean"]), float(r["pdr_std"]))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8), constrained_layout=True)
    axes = axes.flatten()

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for proto in PROTOCOL_ORDER:
            means = [stats[(env, n, proto)][0] for n in NODE_ORDER]
            stds = [stats[(env, n, proto)][1] for n in NODE_ORDER]
            ax.errorbar(
                NODE_ORDER,
                means,
                yerr=stds,
                marker="o",
                markersize=3.8,
                capsize=2.0,
                color=COLORS[proto],
                label=proto,
            )
        ax.set_title(ENV_LABEL[env])
        ax.set_xlabel("Nodes")
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.01)
        ax.grid(True, linestyle="--")

        if env == "indoor_office":
            inset = ax.inset_axes([0.47, 0.10, 0.50, 0.36])
            for proto in ["AERIS", "PEGASIS", "TEEN", "HEED", "LEACH"]:
                means = [stats[(env, n, proto)][0] for n in NODE_ORDER]
                inset.plot(NODE_ORDER, means, marker="o", markersize=2.4, color=COLORS[proto])
            inset.set_ylim(0.988, 1.001)
            inset.set_xlim(90, 1010)
            inset.grid(True, linestyle="--", alpha=0.3)
            inset.set_title("indoor_office zoom", fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True)
    fig.suptitle("Figure 3. Scalability trends across environments (n=550)", fontsize=12)
    save_all_formats(fig, f"fig3_scalability_panel_{SUFFIX}")
    plt.close(fig)


def plot_tradeoff_panel() -> None:
    """Figure 4: PDR-energy-latency trade-off summary from n=30 data."""
    energy_rows = load_energy_rows()
    latency_rows = load_latency_rows()

    e_map = {(r["environment"], r["protocol"]): (float(r["pdr_mean"]), float(r["energy_mean"]), float(r["lifetime_mean"])) for r in energy_rows}
    l_map = {(r["environment"], r["protocol"]): float(r["hops_mean"]) for r in latency_rows}

    # Average over environments for a compact trade-off view.
    avg = {}
    for proto in PROTOCOL_ORDER:
        pdr_vals = [e_map[(env, proto)][0] for env in ENV_ORDER]
        energy_vals = [e_map[(env, proto)][1] for env in ENV_ORDER]
        life_vals = [e_map[(env, proto)][2] for env in ENV_ORDER]
        hop_vals = [l_map[(env, proto)] for env in ENV_ORDER]
        avg[proto] = {
            "pdr": float(np.mean(pdr_vals)),
            "energy": float(np.mean(energy_vals)),
            "lifetime": float(np.mean(life_vals)),
            "hops": float(np.mean(hop_vals)),
        }

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)

    # (a) PDR vs energy
    for proto in PROTOCOL_ORDER:
        axes[0].scatter(avg[proto]["energy"], avg[proto]["pdr"], s=70, color=COLORS[proto], edgecolor="black", linewidth=0.5)
        axes[0].annotate(proto, (avg[proto]["energy"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[0].set_xlabel("Average total energy consumed (J)")
    axes[0].set_ylabel("Average PDR")
    axes[0].set_title("(a) Reliability vs energy")
    axes[0].grid(True, linestyle="--")

    # (b) PDR vs hops
    for proto in PROTOCOL_ORDER:
        axes[1].scatter(avg[proto]["hops"], avg[proto]["pdr"], s=70, color=COLORS[proto], edgecolor="black", linewidth=0.5)
        axes[1].annotate(proto, (avg[proto]["hops"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("Average hop count to BS")
    axes[1].set_ylabel("Average PDR")
    axes[1].set_title("(b) Reliability vs hop-based latency")
    axes[1].grid(True, linestyle="--")

    # (c) Lifetime vs PDR
    for proto in PROTOCOL_ORDER:
        axes[2].scatter(avg[proto]["lifetime"], avg[proto]["pdr"], s=70, color=COLORS[proto], edgecolor="black", linewidth=0.5)
        axes[2].annotate(proto, (avg[proto]["lifetime"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[2].set_xlabel("Average network lifetime (rounds)")
    axes[2].set_ylabel("Average PDR")
    axes[2].set_title("(c) Reliability vs lifetime")
    axes[2].grid(True, linestyle="--")

    fig.suptitle("Figure 4. Trade-off summary across protocols (n=30, averaged over 4 environments)", fontsize=12)
    save_all_formats(fig, f"fig4_tradeoff_panel_{SUFFIX}")
    plt.close(fig)


def main() -> None:
    apply_style()
    plot_env_pdr_panel()
    plot_ablation_panel()
    plot_scalability_panel()
    plot_tradeoff_panel()
    print("Generated Sensors submission figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
