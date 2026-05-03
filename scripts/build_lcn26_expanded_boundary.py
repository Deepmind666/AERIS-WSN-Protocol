#!/usr/bin/env python3
"""Build the NS-3 boundary-gap figure for the LCN26 draft."""

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
RAW_FILE = (
    ROOT
    / "ns3_validation"
    / "results"
    / "lcn26_ns3_dual_combined_20260430_191527_191528"
    / "summary"
    / "ns3_focused_merged.json"
)
SUMMARY_CSV = OUT_DIR / "ns3_boundary_gap_summary.csv"
OUTPUT_PDF = OUT_DIR / "fig_lcn26_ns3_expanded_boundary.pdf"
OUTPUT_PNG = OUT_DIR / "fig_lcn26_ns3_expanded_boundary.png"

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from lcn26_style import COLUMN_WIDTH_IN, PALETTE, apply_lcn26_style  # noqa: E402

ENV_ORDER = [
    "indoor_office",
    "indoor_factory",
    "outdoor_suburban",
    "outdoor_urban",
]
ENV_TITLES = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER = [50, 100, 200, 300, 500, 800, 1000]
NODE_LABELS = ["50", "100", "200", "300", "500", "800", "1k"]
CLASSICAL_PROTOCOLS = ["AERIS", "LEACH", "HEED", "TEEN", "PEGASIS"]
ALL_PROTOCOLS = [
    "AERIS",
    "LEACH",
    "HEED",
    "TEEN",
    "PEGASIS",
    "RPL-MRHOF",
    "CTP",
]
BASELINE_PROTOCOLS = [proto for proto in ALL_PROTOCOLS if proto != "AERIS"]
PLOT_PROTOCOLS = ["AERIS", "RPL-MRHOF", "CTP", "PEGASIS", "TEEN"]


def load_grouped_values() -> dict[tuple[str, int, str], list[float]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    payload = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    for rec in payload["experiments"]:
        key = (rec["environment"], int(rec["num_nodes"]), rec["protocol"])
        grouped[key].append(float(rec["pdr"]))
    return grouped


def mean_std(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0
    if arr.size == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def rank_of_aeris(means: dict[str, float], protocols: list[str]) -> int:
    ordered = sorted(protocols, key=lambda proto: means[proto], reverse=True)
    return ordered.index("AERIS") + 1


def summarise_gap_rows(
    grouped: dict[tuple[str, int, str], list[float]]
) -> tuple[list[dict[str, float | int | str]], dict[str, float], tuple[int, int, int, int]]:
    rows: list[dict[str, float | int | str]] = []
    env_gap_means: dict[str, float] = {}
    classical_rank1 = 0
    classical_top2 = 0
    all_rank1 = 0
    all_top2 = 0

    for env in ENV_ORDER:
        env_gaps: list[float] = []
        for node in NODE_ORDER:
            means: dict[str, float] = {}
            stds: dict[str, float] = {}
            ns: dict[str, int] = {}
            for proto in ALL_PROTOCOLS:
                mean, std, n = mean_std(grouped[(env, node, proto)])
                means[proto] = mean
                stds[proto] = std
                ns[proto] = n

            best_baseline = max(BASELINE_PROTOCOLS, key=lambda proto: means[proto])
            gap_pp = (means["AERIS"] - means[best_baseline]) * 100.0
            gap_ci95 = 1.96 * math.sqrt(
                (stds["AERIS"] ** 2) / max(ns["AERIS"], 1)
                + (stds[best_baseline] ** 2) / max(ns[best_baseline], 1)
            ) * 100.0

            classical_rank = rank_of_aeris(means, CLASSICAL_PROTOCOLS)
            all_rank = rank_of_aeris(means, ALL_PROTOCOLS)
            classical_rank1 += int(classical_rank == 1)
            classical_top2 += int(classical_rank <= 2)
            all_rank1 += int(all_rank == 1)
            all_top2 += int(all_rank <= 2)
            env_gaps.append(gap_pp)

            rows.append(
                {
                    "environment": env,
                    "num_nodes": node,
                    "aeris_mean": means["AERIS"],
                    "best_baseline": best_baseline,
                    "best_baseline_mean": means[best_baseline],
                    "gap_pp": gap_pp,
                    "gap_ci95_pp": gap_ci95,
                    "classical_rank": classical_rank,
                    "all_rank": all_rank,
                }
            )

        env_gap_means[env] = float(np.mean(env_gaps))

    return rows, env_gap_means, (classical_rank1, classical_top2, all_rank1, all_top2)


def write_summary_csv(rows: list[dict[str, float | int | str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "environment",
        "num_nodes",
        "aeris_mean",
        "best_baseline",
        "best_baseline_mean",
        "gap_pp",
        "gap_ci95_pp",
        "classical_rank",
        "all_rank",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_plot(
    rows: list[dict[str, float | int | str]],
    env_gap_means: dict[str, float],
    rank_counts: tuple[int, int, int, int],
) -> None:
    plt.style.use("default")
    apply_lcn26_style()
    plt.rcParams.update(
        {
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.7,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.5,
        }
    )

    del env_gap_means
    del rank_counts
    grouped = load_grouped_values()
    row_lookup = {(str(row["environment"]), int(row["num_nodes"])): row for row in rows}
    x = np.arange(len(NODE_ORDER), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 2.70), sharex=True, sharey=True)
    axes = axes.flatten()
    compact_ticks = ["50", "100", "200", "300", "500", "800", "1k"]

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for proto in PLOT_PROTOCOLS:
            means = []
            errs = []
            for node in NODE_ORDER:
                mean, std, n = mean_std(grouped[(env, node, proto)])
                means.append(mean)
                errs.append(ci95(std, n))
            means_arr = np.asarray(means, dtype=float)
            errs_arr = np.asarray(errs, dtype=float)
            if proto == "AERIS":
                ax.fill_between(
                    x,
                    means_arr - errs_arr,
                    means_arr + errs_arr,
                    color=PALETTE["AERIS"],
                    alpha=0.08,
                    linewidth=0,
                )
            ax.plot(
                x,
                means_arr,
                color=PALETTE[proto],
                marker="o" if proto == "AERIS" else "^" if proto == "RPL-MRHOF" else "s",
                linewidth=1.55 if proto == "AERIS" else 0.95,
                markersize=2.3 if proto == "AERIS" else 2.0,
                alpha=1.0 if proto == "AERIS" else 0.82,
                zorder=4 if proto == "AERIS" else 3,
            )

        env_rows = [row_lookup[(env, node)] for node in NODE_ORDER]
        wins_all = sum(int(row["all_rank"] == 1) for row in env_rows)
        wins_classical = sum(int(row["classical_rank"] == 1) for row in env_rows)
        mean_gap = float(np.mean([float(row["gap_pp"]) for row in env_rows]))
        ax.set_title(
            f"{ENV_TITLES[env]}  C:{wins_classical}/7 A:{wins_all}/7",
            loc="left",
            pad=1.5,
            fontsize=6.7,
            fontweight="bold",
        )
        ax.axhline(0.5, color=PALETTE["grid"], linewidth=0.45, linestyle=":", zorder=0)
        ax.text(
            0.02,
            0.04,
            f"gap {mean_gap:+.1f} pp",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.4,
            color=PALETTE["muted"],
        )
        ax.set_ylim(0.0, 1.03)
        ax.set_xlim(-0.10, len(NODE_ORDER) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(compact_ticks)
        ax.grid(axis="y", linestyle="--", linewidth=0.45, color=PALETTE["grid"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.2, pad=1.5)

    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Nodes")
    axes[3].set_xlabel("Nodes")
    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE[proto],
            marker="o" if proto == "AERIS" else "^" if proto == "RPL-MRHOF" else "s",
            linewidth=1.55 if proto == "AERIS" else 0.95,
            markersize=2.8,
            label=proto,
        )
        for proto in PLOT_PROTOCOLS
    ]
    fig.legend(
        handles=handles,
        labels=PLOT_PROTOCOLS,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        frameon=False,
        columnspacing=0.60,
        handletextpad=0.25,
    )
    fig.text(
        0.50,
        0.017,
        "C/A = AERIS rank-1 cells against classical/all baselines; gap = mean AERIS minus best non-AERIS.",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color=PALETTE["muted"],
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.84, bottom=0.17, wspace=0.16, hspace=0.26)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=320)
    plt.close(fig)


def main() -> None:
    grouped = load_grouped_values()
    rows, env_gap_means, rank_counts = summarise_gap_rows(grouped)
    write_summary_csv(rows)
    build_plot(rows, env_gap_means, rank_counts)
    print(f"[LCN26] Wrote {OUTPUT_PDF}")
    print(f"[LCN26] Wrote {OUTPUT_PNG}")
    print(f"[LCN26] Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
