#!/usr/bin/env python3
"""Build a compact NS-3 AERIS ablation figure for LCN26."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"
ABLATION_DIR = (
    ROOT
    / "ns3_validation"
    / "results"
    / "lcn26_ns3_ablation_combined_20260501_010355_011001"
    / "summary"
)
DELTA_FILE = ABLATION_DIR / "ns3_ablation_delta.csv"
OUTPUT_PDF = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.pdf"
OUTPUT_PNG = OUT_DIR / "fig_lcn26_ns3_ablation_expanded.png"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABEL = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
SERIES = [
    ("Full", "#C8C8C8"),
    ("-GW", "#F1D5B3"),
    ("-CAS", "#F29440"),
]
DELTA_SERIES = [
    ("-GW", "AERIS-noGW", "#F1D5B3", "o"),
    ("-CAS", "AERIS-noCAS", "#F29440", "s"),
]
COLORS = {
    "axis": "#4F5D6A",
    "grid": "#E1E6EA",
    "text": "#111111",
    "muted": "#555555",
    "err": "#333A40",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.6,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.55,
            "grid.alpha": 0.95,
            "grid.linestyle": "--",
        }
    )


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def load_ablation() -> dict[str, dict[str, tuple[float, float]]]:
    rows = load_rows(DELTA_FILE)
    data: dict[str, dict[str, tuple[float, float]]] = {}
    for env in ENV_ORDER:
        full_vals: list[float] = []
        no_gw_vals: list[float] = []
        no_cas_vals: list[float] = []
        delta_gw: list[float] = []
        delta_cas: list[float] = []
        for row in rows:
            if row["environment"] != env:
                continue
            if row["variant"] == "AERIS-noGW":
                full_vals.append(float(row["full_mean"]))
                no_gw_vals.append(float(row["variant_mean"]))
                delta_gw.append(float(row["delta_points"]))
            elif row["variant"] == "AERIS-noCAS":
                no_cas_vals.append(float(row["variant_mean"]))
                delta_cas.append(float(row["delta_points"]))
        data[env] = {
            "Full": mean_ci(full_vals),
            "-GW": mean_ci(no_gw_vals),
            "-CAS": mean_ci(no_cas_vals),
            "delta_-GW": mean_ci(delta_gw),
            "delta_-CAS": mean_ci(delta_cas),
        }
    return data


def build() -> None:
    apply_style()
    data = load_ablation()
    fig, (ax_top, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(3.50, 3.40),
        gridspec_kw={"height_ratios": [1.18, 1.0], "hspace": 0.42},
    )
    x = np.arange(len(ENV_ORDER), dtype=float)
    env_labels = [ENV_LABEL[e] for e in ENV_ORDER]
    bar_width = 0.22
    offsets = [-bar_width, 0.0, bar_width]
    handles: list[Patch] = []

    for offset, (label, color) in zip(offsets, SERIES):
        vals = np.asarray([data[env][label][0] for env in ENV_ORDER], dtype=float)
        ci = np.asarray([data[env][label][1] for env in ENV_ORDER], dtype=float)
        handles.append(Patch(facecolor=color, edgecolor="white", label=label))
        ax_top.bar(
            x + offset,
            vals,
            width=bar_width * 0.95,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax_top.errorbar(
            x + offset,
            vals,
            yerr=ci,
            fmt="none",
            ecolor=COLORS["err"],
            elinewidth=0.75,
            capsize=2.2,
            capthick=0.75,
            zorder=4,
        )

    ax_top.legend(
        handles=handles,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        columnspacing=0.80,
        handletextpad=0.35,
    )
    ax_top.set_ylabel("Mean PDR")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(env_labels)
    ax_top.set_ylim(0.0, 1.03)
    ax_top.grid(axis="y", color=COLORS["grid"], linestyle="-", linewidth=0.65)

    y_pos = np.arange(len(ENV_ORDER), dtype=float)
    for label, variant, color, marker in DELTA_SERIES:
        vals = np.asarray([data[env][f"delta_{label}"][0] for env in ENV_ORDER], dtype=float)
        ci = np.asarray([data[env][f"delta_{label}"][1] for env in ENV_ORDER], dtype=float)
        for yi, val, err in zip(y_pos, vals, ci):
            ax_delta.hlines(yi, 0.0, val, color=color, linewidth=1.75, zorder=2)
            ax_delta.errorbar(
                val,
                yi,
                xerr=err,
                fmt=marker,
                color=color,
                markeredgecolor=color,
                markersize=4.8,
                ecolor=COLORS["err"],
                elinewidth=0.70,
                capsize=1.8,
                zorder=3,
            )
    ax_delta.axvline(0.0, color="#6E7781", linewidth=0.85, linestyle="--", zorder=1)
    ax_delta.set_yticks(y_pos)
    ax_delta.set_yticklabels(env_labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("Delta vs. full (pp)")
    ax_delta.set_xlim(-8.4, 2.0)
    ax_delta.set_xticks([-8, -6, -4, -2, 0, 2])
    ax_delta.grid(axis="both", color=COLORS["grid"], linestyle="-", linewidth=0.65)

    for ax in [ax_top, ax_delta]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["axis"])
        ax.spines["bottom"].set_color(COLORS["axis"])
        ax.spines["left"].set_linewidth(0.85)
        ax.spines["bottom"].set_linewidth(0.85)
        ax.tick_params(length=2.5, width=0.75, colors=COLORS["axis"], pad=1.5)

    fig.subplots_adjust(left=0.17, right=0.985, top=0.95, bottom=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=320)
    plt.close(fig)

    print(f"[LCN26-FIG] wrote {OUTPUT_PDF}")
    print(f"[LCN26-FIG] wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    build()
