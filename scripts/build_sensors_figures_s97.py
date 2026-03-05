#!/usr/bin/env python3
"""
S97: regenerate Fig 2 (ablation) and Fig 6 (s10 delta maps).

Changes from s95:
  Fig2: width_ratios=[1.45, 1.0] GridSpec layout, tighter whitespace
  Fig6: ALL cell numeric annotations REMOVED (key legibility fix)
        Caption note: "per-cell values omitted for legibility; available in CSV"

All other figures are copied from s93 with renamed suffix.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_sensors_figures_s93 import (
    apply_style,
    load_json, load_csv,
    style_axes, panel_label, save_all_formats, group_mean_std,
    ABLATION_FILE, S10_SIG_FILE,
    FIG_DIR, ENV_ORDER, ENV_LABEL, PROTOCOL_ORDER, NODE_ORDER,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SUFFIX = "20260302_s97"


def plot_fig2_s97() -> str:
    """Ablation panel with GridSpec layout for tighter whitespace."""
    rows = [r for r in load_json(ABLATION_FILE)["raw_results"] if not r.get("error")]
    pdr = group_mean_std(rows, lambda r: r["pdr_expected"], ("environment", "ablation_config"))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    matrix = np.zeros((len(configs), len(ENV_ORDER)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr[(env, cfg)][0]

    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.45, 1.0], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    cmap = LinearSegmentedColormap.from_list(
        "ablation_diverging",
        ["#C05A3C", "#E8A87C", "#F5EDE6", "#C4D9EA", "#3A7AB8"],
    )
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    im = ax0.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax0.set_xticks(np.arange(len(ENV_ORDER)))
    ax0.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=38, ha="right", fontsize=12)
    ax0.set_yticks(np.arange(len(configs)))
    ax0.set_yticklabels([c.replace("_", " ") for c in configs], fontsize=12)
    panel_label(ax0, "(a)")
    style_axes(ax0)
    ax0.set_title("Ablation PDR heatmap", fontsize=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            txt_color = "#FFFFFF" if v < (vmin + vmax) / 2.0 else "#1A232C"
            ax0.text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=12, color=txt_color, fontweight="semibold",
                     path_effects=[
                         __import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(linewidth=2.8, foreground="#00000040")
                     ] if v < (vmin + vmax) / 2.0 else [])
    cb = fig.colorbar(im, ax=ax0, shrink=0.84, pad=0.02)
    cb.set_label("PDR", fontsize=12)
    cb.ax.tick_params(labelsize=12)

    full = np.array([pdr[(e, "full")][0] for e in ENV_ORDER])
    no_gw = np.array([pdr[(e, "no_gateway")][0] for e in ENV_ORDER])
    no_cas = np.array([pdr[(e, "no_cas")][0] for e in ENV_ORDER])
    y = np.arange(len(ENV_ORDER))
    gw_delta = (no_gw - full) * 100
    cas_delta = (no_cas - full) * 100

    # 消融专用配色：Gateway=coral, CAS=steel blue（与协议色独立）
    CLR_GW = "#D35B44"
    CLR_CAS = "#3A7AB8"
    ax1.hlines(y + 0.13, 0, gw_delta, color=CLR_GW, linewidth=2.3, label="no_gateway \u2212 full")
    ax1.hlines(y - 0.13, 0, cas_delta, color=CLR_CAS, linewidth=2.3, label="no_cas \u2212 full")
    ax1.plot(gw_delta, y + 0.13, marker="o", linestyle="none", color=CLR_GW, markersize=7, zorder=5)
    ax1.plot(cas_delta, y - 0.13, marker="s", linestyle="none", color=CLR_CAS, markersize=7, zorder=5)
    ax1.axvline(0, color="#303030", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER], fontsize=12)
    ax1.set_xlabel("Delta PDR (percentage points)", fontsize=13)
    ax1.set_title("Marginal effects", fontsize=14)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#C5D0DB", fontsize=11)
    ax1.grid(axis="x")
    ax1.tick_params(axis="x", labelsize=12)
    panel_label(ax1, "(b)")
    style_axes(ax1)
    all_delta = np.concatenate([gw_delta, cas_delta])
    ax1.set_xlim(min(-3.0, float(all_delta.min()) - 0.35), max(2.8, float(all_delta.max()) + 0.35))

    ax1.text(
        0.02, -0.14,
        "Positive: removing module improves PDR; negative: removing degrades PDR.",
        transform=ax1.transAxes, fontsize=11, color="#4A5968",
    )

    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.93)

    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def plot_fig6_s97() -> str:
    """Power-sensitivity delta maps: NO cell numeric annotations for legibility."""
    rows = load_csv(S10_SIG_FILE)
    comparisons = ["tx5_vs_tx10", "tx10_vs_tx15", "tx5_vs_tx15"]
    cmp_label = {
        "tx5_vs_tx10": "tx5 \u2212 tx10",
        "tx10_vs_tx15": "tx10 \u2212 tx15",
        "tx5_vs_tx15": "tx5 \u2212 tx15",
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
            matrix_data = np.zeros((len(PROTOCOL_ORDER), len(NODE_ORDER)), dtype=float)
            sig_mask = np.zeros_like(matrix_data, dtype=bool)
            for r_idx, proto in enumerate(PROTOCOL_ORDER):
                for c_idx, n in enumerate(NODE_ORDER):
                    key = (comp, env, n, proto)
                    matrix_data[r_idx, c_idx] = delta_abs[key]
                    sig_mask[r_idx, c_idx] = sig[key]

            im = ax.imshow(matrix_data, cmap=cmap, norm=norm, aspect="auto")
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

            # Non-significant cross markers
            ns_y, ns_x = np.where(~sig_mask)
            if len(ns_x):
                ax.scatter(
                    ns_x, ns_y, marker="x",
                    s=55, color="#2F2F2F", linewidths=1.4, zorder=3,
                )
            # NO numeric annotations — key legibility improvement

    cb = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.015)
    cb.set_label("Delta PDR (absolute; positive = lower-tx is higher)", fontsize=14)
    cb.ax.tick_params(labelsize=13)

    stem = f"fig6_s10_delta_maps_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()
    fig2 = plot_fig2_s97()
    fig6 = plot_fig6_s97()
    print("S97 patch generated:")
    print(f"  {FIG_DIR / fig2}.pdf")
    print(f"  {FIG_DIR / fig6}.pdf")


if __name__ == "__main__":
    main()
