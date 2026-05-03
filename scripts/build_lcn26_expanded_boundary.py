#!/usr/bin/env python3
"""Build the NS-3 boundary-gap figure for the LCN26 draft."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

    classical_rank1, classical_top2, all_rank1, all_top2 = rank_counts
    env_rank = {}
    for env in ENV_ORDER:
        env_rows = [row for row in rows if row["environment"] == env]
        env_rank[env] = {
            "classical_rank1": sum(int(row["classical_rank"] == 1) for row in env_rows),
            "all_rank1": sum(int(row["all_rank"] == 1) for row in env_rows),
            "classical_top2": sum(int(row["classical_rank"] <= 2) for row in env_rows),
            "all_top2": sum(int(row["all_rank"] <= 2) for row in env_rows),
            "mean_gap": env_gap_means[env],
        }

    labels = [ENV_TITLES[env] for env in ENV_ORDER]
    y = np.arange(len(ENV_ORDER), dtype=float)

    fig, (ax_counts, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(COLUMN_WIDTH_IN, 2.58),
        gridspec_kw={"height_ratios": [0.70, 1.16]},
    )

    height = 0.34
    classical_vals = np.asarray([env_rank[env]["classical_rank1"] for env in ENV_ORDER], dtype=float)
    all_vals = np.asarray([env_rank[env]["all_rank1"] for env in ENV_ORDER], dtype=float)
    ax_counts.barh(y - height / 2, classical_vals, height=height, color=PALETTE["AERIS"], edgecolor="black", linewidth=0.35, label="Classical only")
    ax_counts.barh(y + height / 2, all_vals, height=height, color=PALETTE["RPL-MRHOF"], edgecolor="black", linewidth=0.35, label="All baselines")
    for idx, (base_val, all_val) in enumerate(zip(classical_vals, all_vals)):
        if base_val == 0 and all_val == 0:
            ax_counts.text(0.08, y[idx], "0/7", va="center", ha="left", fontsize=5.2)
            continue
        for yi, val in ((y[idx] - height / 2, base_val), (y[idx] + height / 2, all_val)):
            ax_counts.text((val + 0.08) if val > 0 else 0.08, yi, f"{int(val)}/7", va="center", ha="left", fontsize=5.2)
    ax_counts.set_xlim(0, 7.6)
    ax_counts.set_xticks([0, 2, 4, 6, 7])
    ax_counts.set_yticks(y)
    ax_counts.set_yticklabels(labels)
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("")
    ax_counts.text(0.0, 1.035, "(a) Rank-1 coverage", transform=ax_counts.transAxes, ha="left", va="bottom", fontsize=6.4, fontweight="bold")
    ax_counts.grid(axis="x", linestyle="--", linewidth=0.5, color=PALETTE["grid"])
    ax_counts.grid(axis="y", visible=False)
    ax_counts.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=True,
        facecolor="white",
        edgecolor=PALETTE["grid"],
        framealpha=0.95,
        ncol=2,
        handlelength=1.1,
        columnspacing=0.7,
    )

    gap_matrix = np.asarray(
        [
            [
                float(
                    next(
                        row["gap_pp"]
                        for row in rows
                        if row["environment"] == env and int(row["num_nodes"]) == node
                    )
                )
                for node in NODE_ORDER
            ]
            for env in ENV_ORDER
        ],
        dtype=float,
    )
    norm = TwoSlopeNorm(vmin=-9.0, vcenter=0.0, vmax=2.0)
    heat = ax_gap.imshow(gap_matrix, aspect="auto", cmap="RdBu", norm=norm)
    ax_gap.set_xticks(np.arange(len(NODE_ORDER)))
    ax_gap.set_xticklabels(NODE_LABELS)
    ax_gap.set_yticks(np.arange(len(ENV_ORDER)))
    ax_gap.set_yticklabels(labels)
    ax_gap.set_xlabel("")
    ax_gap.text(0.0, 1.035, "(b) Gap to best non-AERIS baseline (pp)", transform=ax_gap.transAxes, ha="left", va="bottom", fontsize=6.4, fontweight="bold")
    for row_idx, env in enumerate(ENV_ORDER):
        for col_idx, node in enumerate(NODE_ORDER):
            val = gap_matrix[row_idx, col_idx]
            label = r"$\approx$0" if abs(val) < 0.1 else f"{val:+.1f}"
            text_color = "white" if abs(val) > 4.5 else PALETTE["axis"]
            ax_gap.text(col_idx, row_idx, label, ha="center", va="center", fontsize=4.7, color=text_color)
    ax_gap.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
    ax_gap.set_yticks(np.arange(-0.5, len(ENV_ORDER), 1), minor=True)
    ax_gap.grid(which="minor", color="white", linewidth=0.55)
    ax_gap.tick_params(which="minor", bottom=False, left=False)
    for spine in ["top", "right", "left", "bottom"]:
        ax_gap.spines[spine].set_visible(False)
    cbar = fig.colorbar(heat, ax=ax_gap, orientation="horizontal", fraction=0.095, pad=0.18)
    cbar.set_label("AERIS - best baseline PDR (pp)", fontsize=5.5, labelpad=1)
    cbar.ax.tick_params(labelsize=5.2, length=2.0, pad=1)

    fig.subplots_adjust(left=0.20, right=0.985, top=0.90, bottom=0.18, hspace=0.42)

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
