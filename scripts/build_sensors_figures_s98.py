#!/usr/bin/env python3
"""
S98: Fig 2 upgrade -- information-dense 3-panel (heatmap + marginal effects + significance overview)

Changes from s97:
  Fig2: 1x3 GridSpec layout (width_ratios=[1.55, 1.00, 0.85])
    Panel (a): PDR heatmap with delta annotations (baseline row shows 'baseline')
    Panel (b): Gateway/CAS marginal effects with 95% CI lollipop
    Panel (c): Holm significance overview (symbol matrix)
  Fig6: inherited from s97 (no changes)

All other figures copied from s97/s93.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_sensors_figures_s93 import (
    apply_style,
    load_json,
    style_axes, panel_label, save_all_formats,
    ABLATION_FILE, FIG_DIR, ENV_ORDER, ENV_LABEL,
)
from build_sensors_figures_s97 import plot_fig6_s97

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

SUFFIX = "20260304_s98"

# Ablation config order and display labels
ABLATION_CONFIGS = ["full", "no_gateway", "no_cas", "minimal"]
ABLATION_LABELS = {"full": "Full", "no_gateway": "No Gateway",
                   "no_cas": "No CAS", "minimal": "Minimal"}

# Color palette (low saturation, colorblind-safe)
CMAP_COLORS = ["#B4573D", "#E4A886", "#F6F2EA", "#B8D4E8", "#3E7CB1"]
CLR_GW = "#C56B41"
CLR_CAS = "#2F6FA3"


def _compute_welch_ci(vals_a: list, vals_b: list, confidence: float = 0.95):
    """Compute Welch t CI for mean difference. Returns (delta, ci_lo, ci_hi, p_value)."""
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    delta = float(b.mean() - a.mean())  # b - a (e.g., no_gateway - full)

    _t_stat, p_val = stats.ttest_ind(b, a, equal_var=False)

    # Welch-Satterthwaite degrees of freedom
    n_a, n_b = len(a), len(b)
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = np.sqrt(var_a / n_a + var_b / n_b)

    if se < 1e-15:
        return delta, delta, delta, 1.0

    nu_num = (var_a / n_a + var_b / n_b) ** 2
    nu_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    nu = nu_num / nu_den if nu_den > 0 else min(n_a, n_b) - 1

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, nu)
    ci_lo = delta - t_crit * se
    ci_hi = delta + t_crit * se

    return delta, float(ci_lo), float(ci_hi), float(p_val)


def _holm_bonferroni(p_values: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni correction. Returns list of bool (True = significant)."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            results[orig_idx] = True
        else:
            break
    return results


def plot_fig2_s98() -> str:
    """Three-panel ablation figure: heatmap + marginal effects + significance overview."""
    data = load_json(ABLATION_FILE)
    rows = [r for r in data["raw_results"] if not r.get("error")]

    # --- Data aggregation ---
    from collections import defaultdict
    raw_groups = defaultdict(list)
    for r in rows:
        key = (r["environment"], r["ablation_config"])
        raw_groups[key].append(r["pdr_expected"])

    pdr_stats = {}  # (env, config) -> (mean, std, n)
    for key, vals in raw_groups.items():
        arr = np.array(vals)
        pdr_stats[key] = (float(arr.mean()), float(arr.std(ddof=1)), len(vals))

    # --- Heatmap matrix ---
    matrix = np.zeros((len(ABLATION_CONFIGS), len(ENV_ORDER)))
    for i, cfg in enumerate(ABLATION_CONFIGS):
        for j, env in enumerate(ENV_ORDER):
            matrix[i, j] = pdr_stats[(env, cfg)][0]

    # Delta (pp) relative to full row
    full_row = matrix[0, :]

    # --- Marginal effect statistics ---
    marginal_data = {}  # (module, env) -> (delta_pp, ci_lo_pp, ci_hi_pp, p_val)
    p_values_all = []
    p_labels_all = []
    for module, cfg_name in [("Gateway", "no_gateway"), ("CAS", "no_cas")]:
        for env in ENV_ORDER:
            full_vals = raw_groups[(env, "full")]
            removed_vals = raw_groups[(env, cfg_name)]
            delta, ci_lo, ci_hi, p_val = _compute_welch_ci(full_vals, removed_vals)
            # Convert to percentage points
            marginal_data[(module, env)] = (delta * 100, ci_lo * 100, ci_hi * 100, p_val)
            p_values_all.append(p_val)
            p_labels_all.append((module, env))

    # Holm-Bonferroni correction (8 comparisons)
    sig_results = _holm_bonferroni(p_values_all, alpha=0.05)
    sig_map = {}
    for idx, (module, env) in enumerate(p_labels_all):
        sig_map[(module, env)] = sig_results[idx]

    # ==================== Plotting ====================
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.55, 1.00, 0.85], wspace=0.32)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # ---- Panel (a): Heatmap with delta annotations ----
    cmap = LinearSegmentedColormap.from_list("ablation_div", CMAP_COLORS)
    vmin, vmax = float(matrix.min()), float(matrix.max())
    im = ax0.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax0.set_xticks(np.arange(len(ENV_ORDER)))
    ax0.set_xticklabels([ENV_LABEL[e] for e in ENV_ORDER], rotation=25,
                        ha="right", fontsize=10)
    ax0.set_yticks(np.arange(len(ABLATION_CONFIGS)))
    ax0.set_yticklabels([ABLATION_LABELS[c] for c in ABLATION_CONFIGS], fontsize=10)

    # Per-cell annotation: PDR value + delta
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            mid = (vmin + vmax) / 2.0
            txt_color = "#FFFFFF" if v < mid - 0.03 else "#1A232C"
            # Line 1: PDR value
            pdr_txt = f"{v:.3f}"
            # Line 2: delta (full row shows 'baseline')
            if i == 0:
                delta_txt = "baseline"
            else:
                delta_pp = (v - full_row[j]) * 100
                sign = "+" if delta_pp >= 0 else ""
                delta_txt = f"{sign}{delta_pp:.1f} pp"

            stroke = []
            if v < mid - 0.03:
                stroke = [pe.withStroke(linewidth=2.5, foreground="#00000040")]

            ax0.text(j, i - 0.12, pdr_txt, ha="center", va="center",
                     fontsize=9.5, color=txt_color, fontweight="semibold",
                     path_effects=stroke)
            ax0.text(j, i + 0.18, delta_txt, ha="center", va="center",
                     fontsize=9, color=txt_color, fontstyle="italic",
                     path_effects=stroke)

    cb = fig.colorbar(im, ax=ax0, shrink=0.82, pad=0.02)
    cb.set_label("PDR (mean)", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    ax0.set_title("Ablation PDR Heatmap", fontsize=13)
    panel_label(ax0, "(a)")
    style_axes(ax0)

    # ---- Panel (b): Marginal effects lollipop ----
    y = np.arange(len(ENV_ORDER))
    offset = 0.15

    for m_idx, (module, cfg_name, color, marker) in enumerate([
        ("Gateway", "no_gateway", CLR_GW, "o"),
        ("CAS", "no_cas", CLR_CAS, "s"),
    ]):
        y_pos = y + (0.5 - m_idx) * offset
        for j, env in enumerate(ENV_ORDER):
            delta_pp, ci_lo, ci_hi, _ = marginal_data[(module, env)]
            is_sig = sig_map[(module, env)]

            # CI line segment
            ax1.plot([ci_lo, ci_hi], [y_pos[j], y_pos[j]],
                     color=color, linewidth=1.1, solid_capstyle="round")

            # Marker: filled = significant, hollow = non-significant
            if is_sig:
                ax1.plot(delta_pp, y_pos[j], marker=marker, color=color,
                         markersize=7, zorder=5)
            else:
                ax1.plot(delta_pp, y_pos[j], marker=marker, color="white",
                         markeredgecolor="#999999", markeredgewidth=1.2,
                         markersize=7, zorder=5)

            # Line from zero to point
            ax1.hlines(y_pos[j], 0, delta_pp, color=color, linewidth=2.0,
                       alpha=0.7 if is_sig else 0.35)

    ax1.axvline(0, color="#303030", linewidth=0.8, zorder=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER], fontsize=10)
    ax1.set_xlabel("Delta PDR (pp)", fontsize=11)
    ax1.set_title("Marginal Effects (95% CI)", fontsize=13)
    ax1.grid(axis="x", alpha=0.3)
    ax1.tick_params(axis="x", labelsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color=CLR_GW, linestyle="none",
               markersize=7, label="No Gateway \u2212 Full"),
        Line2D([0], [0], marker="s", color=CLR_CAS, linestyle="none",
               markersize=7, label="No CAS \u2212 Full"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#999999",
               markeredgewidth=1.2, linestyle="none", markersize=7,
               label="Non-significant (Holm)"),
    ]
    ax1.legend(handles=legend_items, loc="lower right", frameon=True,
               framealpha=0.92, edgecolor="#C5D0DB", fontsize=9)

    # x-axis range
    all_cis = [marginal_data[k][1] for k in marginal_data] + [marginal_data[k][2] for k in marginal_data]
    xmin = min(-3.0, min(all_cis) - 0.5)
    xmax = max(3.0, max(all_cis) + 0.5)
    ax1.set_xlim(xmin, xmax)

    panel_label(ax1, "(b)")
    style_axes(ax1)

    # ---- Panel (c): Significance overview matrix ----
    modules = ["Gateway", "CAS"]
    n_env = len(ENV_ORDER)
    n_mod = len(modules)

    # Build symbol matrix (y-axis order matches Panel (b))
    for m_idx, module in enumerate(modules):
        for e_idx, env in enumerate(ENV_ORDER):
            is_sig = sig_map[(module, env)]
            delta_pp = marginal_data[(module, env)][0]
            p_val = marginal_data[(module, env)][3]

            # Position (not reversed, matching Panel (b): indoor_office=0 bottom, outdoor_suburban=3 top)
            x_pos = m_idx
            y_pos_val = e_idx

            # Symbol
            if is_sig:
                marker = "o"
                color = CLR_GW if module == "Gateway" else CLR_CAS
                facecolor = color
                edgecolor = color
            else:
                marker = "o"
                facecolor = "white"
                edgecolor = "#BBBBBB"

            ax2.plot(x_pos, y_pos_val, marker=marker, markersize=12,
                     color=facecolor, markeredgecolor=edgecolor,
                     markeredgewidth=1.5, zorder=3)

            # Direction indicator (+/-)
            direction = "+" if delta_pp >= 0 else "\u2212"
            dir_color = "#2E7D32" if delta_pp >= 0 else "#C62828"
            if not is_sig:
                dir_color = "#999999"

            ax2.text(x_pos + 0.28, y_pos_val, direction,
                     ha="left", va="center", fontsize=13, fontweight="bold",
                     color=dir_color)

    # Axis settings
    ax2.set_xlim(-0.5, n_mod - 0.2)
    ax2.set_ylim(-0.6, n_env - 0.4)
    ax2.set_xticks(range(n_mod))
    ax2.set_xticklabels(modules, fontsize=10, fontweight="semibold")
    ax2.set_yticks(range(n_env))
    ax2.set_yticklabels([ENV_LABEL[e] for e in ENV_ORDER], fontsize=9.5)
    ax2.set_title("Significance (Holm)", fontsize=13)

    # Legend
    legend_items_c = [
        Line2D([0], [0], marker="o", color=CLR_GW, linestyle="none",
               markersize=9, label="Significant"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="#BBBBBB",
               markeredgewidth=1.5, linestyle="none", markersize=9,
               label="Non-significant"),
    ]
    ax2.legend(handles=legend_items_c, loc="lower center", frameon=True,
               framealpha=0.92, edgecolor="#C5D0DB", fontsize=9,
               bbox_to_anchor=(0.5, -0.18))

    # Grid
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.tick_params(axis="both", length=0)

    panel_label(ax2, "(c)")
    style_axes(ax2)

    # ---- Overall layout ----
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.13, top=0.92)

    stem = f"fig2_ablation_panel_{SUFFIX}"
    save_all_formats(fig, stem)
    plt.close(fig)
    return stem


def main() -> None:
    apply_style()

    # Fig2: new three-panel ablation
    fig2 = plot_fig2_s98()
    print(f"S98 Fig2: {FIG_DIR / fig2}.pdf")

    # Fig6: reuse s97 version
    fig6 = plot_fig6_s97()
    print(f"S98 Fig6 (from s97): {FIG_DIR / fig6}.pdf")


if __name__ == "__main__":
    main()
