#!/usr/bin/env python3
"""Generate Sensors-ready figures and scalability tables for S10 manuscript sync.

This script does three things with explicit evidence paths:
1) Aggregate four scalability JSON files into a single descriptive CSV.
2) Compute AERIS-vs-baseline significance (Welch + Hedges g + Holm).
3) Render publication-grade figures (PDF/SVG/PNG) with consistent style.

All outputs are written under:
  - results/mega_experiments/
  - for_submission/figures/
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
SUFFIX = "20260213_s10"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
PROTOCOL_ORDER = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]

ENV_LABEL = {
    "indoor_office": "Indoor office",
    "indoor_factory": "Indoor factory",
    "outdoor_urban": "Outdoor urban",
    "outdoor_suburban": "Outdoor suburban",
}

PROTO_COLORS = {
    "AERIS": "#1B5E9A",
    "LEACH": "#D1495B",
    "PEGASIS": "#2A9D8F",
    "HEED": "#F4A261",
    "TEEN": "#7B6D8D",
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
# Prefer the latest v3 latency summary; keep v2 as fallback for compatibility.
LATENCY_FILE = RESULTS_DIR / "latency_hop_v3_20260211_stats.csv"


def apply_style() -> None:
    """Apply a consistent manuscript style (readable in print and screen)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.6,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def save_all_formats(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_latency_rows() -> List[dict]:
    if LATENCY_FILE.exists():
        return load_csv(LATENCY_FILE)
    return load_csv(RESULTS_DIR / "latency_hop_v2_stats.csv")


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
    """Return Holm-adjusted p-values in original order."""
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
    correction = 1.0 - 3.0 / max((4.0 * (nx + ny) - 9.0), 1.0)
    return float(d * correction)


def aggregate_scalability_s10() -> Tuple[Path, Path, Path]:
    """Aggregate mixed scalability files and export descriptive/significance/report."""
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
        writer.writerow(
            [
                "environment",
                "num_nodes",
                "protocol",
                "n",
                "pdr_mean",
                "pdr_std",
                "ci95_half_width",
            ]
        )
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
                    [
                        r["pdr_expected"]
                        for r in rows
                        if r["environment"] == env and r["num_nodes"] == node and r["protocol"] == baseline
                    ],
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
        f.write("# Scalability Mixed Summary (S10)\n\n")
        f.write("## Sample size by environment\n\n")
        for env in ENV_ORDER:
            f.write(f"- {env}: n={n_by_env[env]}\n")
        f.write("\n## 1000-node ranking (PDR mean)\n\n")
        for env in ENV_ORDER:
            rows_env = [r for r in desc_stats.items() if r[0][0] == env and r[0][1] == 1000]
            rank = sorted(rows_env, key=lambda kv: kv[1][0], reverse=True)
            f.write(f"- {env}: " + ", ".join(f"{k[2]}={v[0]:.4f}" for k, v in rank) + "\n")

    return desc_out, sig_out, md_out


def plot_env_pdr_panel() -> str:
    rows = [r for r in load_json(ENV_FILE)["raw_results"] if not r.get("error")]
    stats_map = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "protocol"))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(PROTOCOL_ORDER))
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        means = [stats_map[(env, p)][0] for p in PROTOCOL_ORDER]
        stds = [stats_map[(env, p)][1] for p in PROTOCOL_ORDER]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=2.5,
            color=[PROTO_COLORS[p] for p in PROTOCOL_ORDER],
            edgecolor="black",
            linewidth=0.65,
        )
        ax.set_title(ENV_LABEL[env], pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PROTOCOL_ORDER, rotation=20, ha="right")
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y")
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.013, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 1. PDR comparison at 100 nodes across four environments (n=30)", fontsize=12)
    stem = f"fig1_env_pdr_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_ablation_panel() -> str:
    rows = [r for r in load_json(ABLATION_FILE)["raw_results"] if not r.get("error")]
    pdr = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "ablation_config"))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr[(env, cfg)][0]

    fig, axes = plt.subplots(1, 2, figsize=(12.9, 5.1), constrained_layout=True)
    im = axes[0].imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels([ENV_LABEL[e].replace(" ", "\n") for e in ENV_ORDER])
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in configs])
    axes[0].set_title("(a) PDR heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = "white" if matrix[i, j] < 0.45 else "black"
            axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=c)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.88)
    cb.set_label("PDR")

    full = np.array([pdr[(e, "full")][0] for e in ENV_ORDER])
    no_gw = np.array([pdr[(e, "no_gateway")][0] for e in ENV_ORDER])
    no_cas = np.array([pdr[(e, "no_cas")][0] for e in ENV_ORDER])
    x = np.arange(len(ENV_ORDER))
    w = 0.28
    axes[1].bar(x - w / 2, (no_gw - full) * 100, width=w, color="#D1495B", edgecolor="black", linewidth=0.55, label="no_gateway - full")
    axes[1].bar(x + w / 2, (no_cas - full) * 100, width=w, color="#1B5E9A", edgecolor="black", linewidth=0.55, label="no_cas - full")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ENV_LABEL[e].replace(" ", "\n") for e in ENV_ORDER])
    axes[1].set_ylabel("Delta PDR (percentage points)")
    axes[1].set_title("(b) Marginal effects vs full")
    axes[1].grid(axis="y")
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle("Figure 2. Multi-environment ablation analysis (n=30)", fontsize=12)
    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_scalability_panel(desc_csv: Path) -> str:
    rows = load_csv(desc_csv)
    stats_map = {
        (r["environment"], int(r["num_nodes"]), r["protocol"]): (float(r["pdr_mean"]), float(r["pdr_std"]), int(r["n"]))
        for r in rows
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
    axes = axes.flatten()
    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for proto in PROTOCOL_ORDER:
            means = [stats_map[(env, n, proto)][0] for n in NODE_ORDER]
            stds = [stats_map[(env, n, proto)][1] for n in NODE_ORDER]
            ax.errorbar(
                NODE_ORDER,
                means,
                yerr=stds,
                marker="o",
                markersize=3.8,
                capsize=2.2,
                color=PROTO_COLORS[proto],
                label=proto,
            )
        env_n = stats_map[(env, NODE_ORDER[0], "AERIS")][2]
        ax.set_title(f"{ENV_LABEL[env]} (n={env_n})", pad=8)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("PDR")
        ax.set_ylim(0.0, 1.01)
        ax.grid(True)

        if env == "indoor_office":
            # Local zoom highlights the known close ranking regime.
            inset = ax.inset_axes([0.46, 0.08, 0.51, 0.38])
            for proto in ["AERIS", "PEGASIS", "TEEN", "HEED", "LEACH"]:
                y = [stats_map[(env, n, proto)][0] for n in NODE_ORDER]
                inset.plot(NODE_ORDER, y, marker="o", markersize=2.2, color=PROTO_COLORS[proto])
            inset.set_xlim(90, 1010)
            inset.set_ylim(0.988, 1.001)
            inset.set_title("office zoom", fontsize=7)
            inset.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 3. Scalability trends across environments (100-1000 nodes, n>=550)", fontsize=12)
    stem = f"fig3_scalability_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_tradeoff_panel() -> str:
    energy_rows = load_csv(ENERGY_FILE)
    latency_rows = load_latency_rows()
    e_map = {
        (r["environment"], r["protocol"]): (float(r["pdr_mean"]), float(r["energy_mean"]), float(r["lifetime_mean"]))
        for r in energy_rows
    }
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
            "lifetime": float(np.mean(life_vals)),
            "hops": float(np.mean(hop_vals)),
        }

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    for proto in PROTOCOL_ORDER:
        axes[0].scatter(avg[proto]["energy"], avg[proto]["pdr"], s=74, color=PROTO_COLORS[proto], edgecolor="black", linewidth=0.6)
        axes[0].annotate(proto, (avg[proto]["energy"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[0].set_xlabel("Average total energy (J)")
    axes[0].set_ylabel("Average PDR")
    axes[0].set_title("(a) Reliability vs energy")
    axes[0].grid(True)

    for proto in PROTOCOL_ORDER:
        axes[1].scatter(avg[proto]["hops"], avg[proto]["pdr"], s=74, color=PROTO_COLORS[proto], edgecolor="black", linewidth=0.6)
        axes[1].annotate(proto, (avg[proto]["hops"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("Average hops to BS")
    axes[1].set_ylabel("Average PDR")
    axes[1].set_title("(b) Reliability vs hop-based latency")
    axes[1].grid(True)

    for proto in PROTOCOL_ORDER:
        axes[2].scatter(avg[proto]["lifetime"], avg[proto]["pdr"], s=74, color=PROTO_COLORS[proto], edgecolor="black", linewidth=0.6)
        axes[2].annotate(proto, (avg[proto]["lifetime"], avg[proto]["pdr"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[2].set_xlabel("Average lifetime (rounds)")
    axes[2].set_ylabel("Average PDR")
    axes[2].set_title("(c) Reliability vs lifetime")
    axes[2].grid(True)

    fig.suptitle("Figure 4. Trade-off summary from publication-tier datasets (n=30)", fontsize=12)
    stem = f"fig4_tradeoff_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    desc_csv, sig_csv, md_out = aggregate_scalability_s10()
    fig1 = plot_env_pdr_panel()
    fig2 = plot_ablation_panel()
    fig3 = plot_scalability_panel(desc_csv)
    fig4 = plot_tradeoff_panel()

    print("Generated files:")
    print("  ", desc_csv)
    print("  ", sig_csv)
    print("  ", md_out)
    print("  ", FIG_DIR / f"{fig1}.pdf")
    print("  ", FIG_DIR / f"{fig2}.pdf")
    print("  ", FIG_DIR / f"{fig3}.pdf")
    print("  ", FIG_DIR / f"{fig4}.pdf")


if __name__ == "__main__":
    main()
