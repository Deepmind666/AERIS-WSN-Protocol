#!/usr/bin/env python3
"""
S95 patch: regenerate Fig 2 (ablation) and Fig 6 (s10 delta maps) with
increased font sizes to meet MDPI Sensors 8pt minimum after scaling.

Only overwrites these two figures. All other s93 figures remain unchanged.

Sizing rationale (MDPI Sensors textwidth ≈ 17.5cm = 6.89in):
  Fig 2: figsize=(10, 6)   → scale = 0.689 → 12pt source → 8.3pt rendered
  Fig 6: figsize=(11.5, 13) → scale = 0.599 → 14pt source → 8.4pt rendered
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import everything from s93
sys.path.insert(0, str(Path(__file__).parent))
from build_sensors_figures_s93 import (
    apply_style,
    load_json, load_csv,
    style_axes, panel_label, save_all_formats, group_mean_std,
    ABLATION_FILE, S10_SIG_FILE,
    FIG_DIR, ENV_ORDER, ENV_LABEL, PROTOCOL_ORDER, NODE_ORDER,
    PROTO_COLORS,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SUFFIX = "20260302_s95"


def plot_fig2_s95() -> str:
    """Ablation panel with fonts scaled for MDPI readability."""
    rows = [r for r in load_json(ABLATION_FILE)["raw_results"] if not r.get("error")]
    pdr = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "ablation_config"))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr[(env, cfg)][0]

    # --- KEY CHANGE: smaller canvas (10, 6) → scale ≈ 0.689, all fonts ≥ 12pt ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    cmap = LinearSegmentedColormap.from_list(
        "soft_yellow_blue",
        ["#F7F2EC", "#F8FAFC", "#E3EDF6", "#C4D9EA", "#97BEDA"],
    )
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    im = axes[0].imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_xticks(np.arange(len(ENV_ORDER)))
    axes[0].set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=38, ha="right", fontsize=12)
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels([c.replace("_", " ") for c in configs], fontsize=12)
    panel_label(axes[0], "(a)")
    style_axes(axes[0])
    axes[0].set_title("Ablation PDR heatmap", fontsize=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt_color = "#1A232C" if v >= (vmin + vmax) / 2.0 else "#F6F8FA"
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=12, color=txt_color, fontweight="semibold")
    cb = fig.colorbar(im, ax=axes[0], shrink=0.84, pad=0.02)
    cb.set_label("PDR", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    full = np.array([pdr[(e, "full")][0] for e in ENV_ORDER])
    no_gw = np.array([pdr[(e, "no_gateway")][0] for e in ENV_ORDER])
    no_cas = np.array([pdr[(e, "no_cas")][0] for e in ENV_ORDER])
    y = np.arange(len(ENV_ORDER))
    gw_delta = (no_gw - full) * 100
    cas_delta = (no_cas - full) * 100

    axes[1].hlines(y + 0.13, 0, gw_delta, color=PROTO_COLORS["LEACH"], linewidth=2.3, label="no_gateway − full")
    axes[1].hlines(y - 0.13, 0, cas_delta, color=PROTO_COLORS["AERIS"], linewidth=2.3, label="no_cas − full")
    axes[1].plot(gw_delta, y + 0.13, marker="o", linestyle="none", color=PROTO_COLORS["LEACH"], markersize=6)
    axes[1].plot(cas_delta, y - 0.13, marker="s", linestyle="none", color=PROTO_COLORS["AERIS"], markersize=6)
    axes[1].axvline(0, color="#303030", linewidth=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER], fontsize=12)
    axes[1].set_xlabel("Delta PDR (percentage points)", fontsize=13)
    axes[1].set_title("Marginal effects", fontsize=14)
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#C5D0DB", fontsize=11)
    axes[1].grid(axis="x")
    axes[1].tick_params(axis="x", labelsize=12)
    panel_label(axes[1], "(b)")
    style_axes(axes[1])
    all_delta = np.concatenate([gw_delta, cas_delta])
    axes[1].set_xlim(min(-3.0, float(all_delta.min()) - 0.35), max(2.8, float(all_delta.max()) + 0.35))

    axes[1].text(
        0.02, -0.14,
        "Positive: removing the module improves PDR; negative: removing degrades PDR.",
        transform=axes[1].transAxes, fontsize=11, color="#4A5968",
    )

    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig6_s95() -> str:
    """Power-sensitivity delta maps with fonts scaled for MDPI readability."""
    rows = load_csv(S10_SIG_FILE)
    comparisons = ["tx5_vs_tx10", "tx10_vs_tx15", "tx5_vs_tx15"]
    cmp_label = {
        "tx5_vs_tx10": "tx5 − tx10",
        "tx10_vs_tx15": "tx10 − tx15",
        "tx5_vs_tx15": "tx5 − tx15",
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

    # --- KEY CHANGE: smaller canvas (11.5, 13) → scale ≈ 0.599, all fonts ≥ 14pt ---
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 13), constrained_layout=True)

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
            ax.set_xticklabels([str(n) for n in NODE_ORDER], fontsize=13, rotation=45, ha="right")
            if rr == 2:
                ax.set_xlabel("Nodes", fontsize=15)
            ax.set_yticks(np.arange(len(PROTOCOL_ORDER)))
            ax.set_yticklabels(PROTOCOL_ORDER, fontsize=14)
            if cc == 0:
                ax.set_ylabel(cmp_label[comp], fontsize=16)

            if rr == 0:
                ax.set_title(ENV_LABEL[env], fontsize=15, pad=6)

            if rr == 0:
                ax.text(
                    0.02, 0.96, f"({chr(97 + cc)})",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=14, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "#E0E5EA", "pad": 0.18, "alpha": 0.95},
                )
            style_axes(ax)
            ax.set_xticks(np.arange(-0.5, len(NODE_ORDER), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(PROTOCOL_ORDER), 1), minor=True)
            ax.grid(which="minor", color="#E3EAF1", linewidth=0.55)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Non-significant cross markers (bigger for readability)
            ns_y, ns_x = np.where(~sig_mask)
            if len(ns_x):
                ax.scatter(
                    ns_x, ns_y, marker="x",
                    s=50, color="#2F2F2F", linewidths=1.3, zorder=3,
                )
            # Numeric labels on the most decision-relevant row (tx5-vs-tx15)
            if comp == "tx5_vs_tx15":
                for r_idx in range(len(PROTOCOL_ORDER)):
                    for c_idx in range(len(NODE_ORDER)):
                        val = matrix[r_idx, c_idx]
                        txt_color = "#FFFFFF" if abs(val) > 0.60 * vmax else "#28313A"
                        ax.text(
                            c_idx, r_idx, f"{val:+.2f}",
                            ha="center", va="center",
                            fontsize=13, color=txt_color, zorder=4,
                        )

    cb = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.015)
    cb.set_label("Delta PDR (absolute; positive = lower-tx is higher)", fontsize=14)
    cb.ax.tick_params(labelsize=13)

    stem = f"fig6_s10_delta_maps_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    fig2 = plot_fig2_s95()
    fig6 = plot_fig6_s95()
    print("S95 patch generated:")
    print(f"  {FIG_DIR / fig2}.pdf")
    print(f"  {FIG_DIR / fig6}.pdf")


if __name__ == "__main__":
    main()
