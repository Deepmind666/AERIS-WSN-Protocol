#!/usr/bin/env python3
"""
Build S11 patch-vs-control summary figure for Sensors manuscript.

Figure design goals:
1) White background and low-saturation scientific palette.
2) Clear separation of AERIS-specific trend and full protocol impact.
3) No overlapping text labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "mega_experiments"
FIG_DIR = PROJECT_ROOT / "for_submission" / "figures"

DELTA_CSV = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_delta.csv"
SIG_CSV = RESULTS_DIR / "s11_matched_4env_patch_vs_control_20260217_significance.csv"
OUT_STEM = "fig5_s11_patch_control_delta_20260217_s26"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
PROTO_ORDER = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]

ENV_LABEL = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_urban": "Outdoor Urban",
    "outdoor_suburban": "Outdoor Suburban",
}

# Soft tones aligned with manuscript figure style.
PROTO_COLORS = {
    "AERIS": "#4F7EA8",
    "LEACH": "#CF9368",
    "PEGASIS": "#69A88F",
    "HEED": "#B08BB6",
    "TEEN": "#C7A862",
}


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#E4E9EF",
            "grid.alpha": 0.22,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.4,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.97,
        label,
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
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)


def main() -> None:
    apply_style()
    rows = load_csv(DELTA_CSV)
    sig_rows = load_csv(SIG_CSV)

    # Assertion for audit safety: expected matrix size.
    if len(rows) != 120:
        raise ValueError(f"Unexpected S11 delta row count: {len(rows)} (expected 120)")

    # Build lookup tables.
    delta = {}
    for r in rows:
        key = (r["environment"], int(r["num_nodes"]), r["protocol"])
        delta[key] = float(r["delta"])

    sig = {}
    for r in sig_rows:
        key = (r["environment"], int(r["num_nodes"]), r["protocol"])
        sig[key] = (r["significant_005"] == "yes")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # (a) AERIS delta trajectory across scales.
    ax = axes[0]
    for env in ENV_ORDER:
        vals = np.array([delta[(env, n, "AERIS")] for n in NODE_ORDER], dtype=float)
        ax.plot(
            NODE_ORDER,
            vals,
            marker="o",
            markersize=5.4,
            color=PROTO_COLORS["AERIS"],
            alpha=0.95,
            label=ENV_LABEL[env],
        )
    ax.axhline(0.0, color="#5D6673", linewidth=0.9)
    ax.set_title("AERIS delta by scale (patch - control)")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Delta PDR")
    ax.set_xticks(NODE_ORDER)
    ax.set_ylim(-0.82, 0.03)
    ax.grid(axis="both")
    ax.legend(loc="lower left", frameon=True, edgecolor="#C8D0DA", framealpha=0.95)
    panel_label(ax, "(a)")
    style_axes(ax)

    # (b) Protocol delta at 1000 nodes across environments.
    ax = axes[1]
    env_idx = np.arange(len(ENV_ORDER))
    width = 0.15
    offsets = np.linspace(-2, 2, len(PROTO_ORDER)) * width
    for i, proto in enumerate(PROTO_ORDER):
        vals = [delta[(env, 1000, proto)] for env in ENV_ORDER]
        bars = ax.bar(
            env_idx + offsets[i],
            vals,
            width=width,
            color=PROTO_COLORS[proto],
            edgecolor="#5F6875",
            linewidth=0.7,
            label=proto,
        )
        # Mark non-significant cells with hollow marker above bar top.
        for j, b in enumerate(bars):
            env = ENV_ORDER[j]
            if not sig[(env, 1000, proto)]:
                ax.plot(
                    b.get_x() + b.get_width() / 2,
                    vals[j],
                    marker="o",
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgecolor="#3A3F45",
                    markeredgewidth=0.9,
                    zorder=4,
                )
    ax.axhline(0.0, color="#5D6673", linewidth=0.9)
    ax.set_title("Protocol delta at 1000 nodes")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Delta PDR")
    ax.set_xticks(env_idx)
    ax.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=15, ha="right")
    ax.set_ylim(-0.82, 0.03)
    ax.grid(axis="y")
    ax.legend(loc="lower left", ncol=3, frameon=True, edgecolor="#C8D0DA", framealpha=0.95)
    panel_label(ax, "(b)")
    style_axes(ax)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_DIR / f"{OUT_STEM}.{ext}")
    plt.close(fig)
    print(FIG_DIR / f"{OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
