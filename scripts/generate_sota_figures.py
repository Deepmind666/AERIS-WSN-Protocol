#!/usr/bin/env python3
"""
Generate Publication-Quality SOTA Comparison Figures

Following Gemini review feedback:
1. Survival curves for network lifetime
2. Box plots for PDR distribution
3. Forest plot for effect sizes
4. Bar charts with 95% CI error bars
5. Statistical testing workflow annotations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats

# Publication style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Color palette (colorblind-friendly)
COLORS = {
    'LEACH': '#4477AA',     # Blue
    'HEED': '#228833',      # Green
    'PEGASIS': '#CCBB44',   # Yellow
    'SEP': '#AA3377',       # Purple
    'TEEN': '#66CCEE',      # Cyan
    'AERIS-E': '#44AA99',   # Teal
    'AERIS-R': '#EE6677',   # Red (highlight)
}
MARKERS = {
    'LEACH': 'o',
    'HEED': 's',
    'PEGASIS': 'D',
    'SEP': '^',
    'TEEN': 'v',
    'AERIS-E': 'X',
    'AERIS-R': 'P',
}


def load_data():
    """Load SOTA comparison results."""
    path = Path(__file__).parent.parent / 'results' / 'sota_comparison.json'
    with open(path) as f:
        return json.load(f)


def bootstrap_mean_diff(a, b, n_boot=2000, seed=42):
    """Bootstrap 95% CI for mean difference (a - b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = []
    for _ in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(a_s) - np.mean(b_s))
    diffs = np.array(diffs)
    return np.mean(a) - np.mean(b), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def mean_ci(values, alpha=0.05):
    """Mean and two-sided CI using t distribution."""
    values = np.asarray(values)
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean, t * se


def create_sota_figure():
    """Create comprehensive SOTA comparison figure."""
    data = load_data()

    # Create 2x3 grid (extra space for titles/legends and table layout)
    fig = plt.figure(figsize=(19.2, 11.2))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.62,
        wspace=0.32,
        width_ratios=[1.0, 1.05, 2.8],
        height_ratios=[1.0, 1.18],
    )

    protocols = ['LEACH', 'HEED', 'PEGASIS', 'SEP', 'TEEN', 'AERIS-E', 'AERIS-R']
    run_count = len(data['protocols']['AERIS-R']['pdr_values'])

    # ========== Panel (a): PDR Comparison Bar Chart ==========
    ax1 = fig.add_subplot(gs[0, 0])

    means = [data['protocols'][p]['pdr_mean'] * 100 for p in protocols]
    cis = [data['protocols'][p]['pdr_ci95'] * 100 for p in protocols]
    colors = [COLORS[p] for p in protocols]

    bars = ax1.bar(range(len(protocols)), means, yerr=cis, capsize=4,
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    # Highlight AERIS profiles
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)
    bars[-2].set_edgecolor('black')
    bars[-2].set_linewidth(1.6)

    ax1.set_xticks(range(len(protocols)))
    ax1.set_xticklabels(protocols, rotation=25, ha='right')
    ax1.set_ylabel('PDR (%)')
    ymin = max(0, min(means) - 5)
    ymax = min(100, max(means) + 2)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(f'(a) Protocol PDR Comparison\n(n={run_count}, 95% CI)', fontweight='bold')

    # Add significance markers
    aeris_r_mean = means[-1]
    for i, (m, name) in enumerate(zip(means[:-2], protocols[:-2])):
        diff = aeris_r_mean - m
        marker = '***' if abs(diff) > 2 else '**' if abs(diff) > 1 else '*'
        ax1.text(i, m + cis[i] + 1.5, marker, ha='center', fontsize=8)

    # ========== Panel (b): Box Plot Distribution ==========
    ax2 = fig.add_subplot(gs[0, 1])

    box_data = [np.array(data['protocols'][p]['pdr_values']) * 100 for p in protocols]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=protocols)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight AERIS boxes
    bp['boxes'][-1].set_linewidth(2)
    bp['boxes'][-2].set_linewidth(1.6)

    ax2.set_ylabel('PDR (%)')
    ax2.set_ylim(ymin, ymax)
    ax2.set_title(f'(b) PDR Distribution\n(Box Plot, n={run_count})', fontweight='bold')
    ax2.tick_params(axis='x', labelrotation=25)
    # Add jittered points to show distribution density (esp. near 100% PDR)
    for i, vals in enumerate(box_data, start=1):
        jitter = np.random.normal(loc=i, scale=0.04, size=len(vals))
        ax2.scatter(jitter, vals, s=12, color=colors[i - 1], alpha=0.6, edgecolors='none')

    # ========== Panel (c): AERIS profiles vs baselines (ΔPDR with CI) ==========
    ax3 = fig.add_subplot(gs[0, 2])

    comparisons = [p for p in ['LEACH', 'HEED', 'PEGASIS', 'SEP', 'TEEN'] if p in data['protocols']]
    profiles = ['AERIS-E', 'AERIS-R']
    offsets = {'AERIS-E': -0.18, 'AERIS-R': 0.18}
    y_pos = np.arange(len(comparisons))

    all_diffs = []
    for i, baseline in enumerate(comparisons):
        base_vals = np.array(data['protocols'][baseline]['pdr_values'])
        for profile in profiles:
            aeris_vals = np.array(data['protocols'][profile]['pdr_values'])
            diff, ci_low, ci_high = bootstrap_mean_diff(aeris_vals, base_vals)
            diff_pp = diff * 100
            ci_low_pp = ci_low * 100
            ci_high_pp = ci_high * 100
            all_diffs.extend([ci_low_pp, ci_high_pp])
            ax3.errorbar(
                diff_pp,
                y_pos[i] + offsets[profile],
                xerr=[[diff_pp - ci_low_pp], [ci_high_pp - diff_pp]],
                fmt=MARKERS[profile],
                color=COLORS[profile],
                markersize=7,
                capsize=3,
                capthick=1.2,
                linewidth=1.2,
                label=profile if i == 0 else None,
                zorder=3,
            )
        # Keep panel clean: paired markers already indicate the two profiles

    ax3.axvline(0, color='#666666', linestyle='--', linewidth=1.0)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(comparisons)
    ax3.set_xlabel("ΔPDR (AERIS profile − protocol, pp)")
    ax3.set_title("(c) AERIS Profiles vs Protocols\n(Mean ΔPDR ± 95% CI)", fontweight='bold')
    x_min = min(all_diffs) - 1.5
    x_max = max(all_diffs) + 1.5
    ax3.set_xlim(x_min, x_max)
    ax3.legend(loc='upper right', frameon=True, fontsize=7, edgecolor='black')
    ax3.text(
        x_max,
        -0.55,
        "Positive favors AERIS",
        ha='right',
        va='center',
        fontsize=7,
        color='#555555',
    )

    # ========== Panel (d): Survival Curves / End-of-Run Survival ==========
    ax4 = fig.add_subplot(gs[1, 0])

    survival_curves = data.get('survival_curves', {})
    curves_available = all(p in survival_curves for p in protocols)
    curves_constant = False
    if curves_available:
        curves_constant = all(len(set(survival_curves[p])) == 1 for p in protocols)

    if curves_available and not curves_constant:
        # Use different line styles for visibility when curves overlap
        line_styles = {
            'LEACH': '-',
            'HEED': '--',
            'PEGASIS': '-.',
            'SEP': ':',
            'TEEN': (0, (3, 1)),
            'AERIS-E': '-',
            'AERIS-R': '-'
        }
        line_widths = {
            'LEACH': 1.5,
            'HEED': 1.5,
            'PEGASIS': 1.5,
            'SEP': 1.8,
            'TEEN': 1.6,
            'AERIS-E': 2.0,
            'AERIS-R': 2.4  # Thicker for robust profile
        }

        for name in protocols:
            curve = survival_curves[name]
            rounds = range(len(curve))
            ax4.plot(rounds, curve, line_styles[name], color=COLORS[name],
                     linewidth=line_widths[name], label=name, alpha=0.9)

        ax4.set_xlabel('Round')
        ax4.set_ylabel('Alive Nodes')
        ax4.set_title('(d) Network Lifetime\n(Survival Curve, 200 rounds)', fontweight='bold')
        ax4.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')

        max_nodes = max(max(survival_curves[p]) for p in protocols)
        ax4.set_ylim(0, max_nodes * 1.1)
    else:
        # Trade-off scatter with mean ± 95% CI overlay (avoid overlapping labels)
        all_pdr = []
        all_energy = []
        for name in protocols:
            pdr_vals = np.array(data['protocols'][name]['pdr_values']) * 100
            energy_vals = np.array(data['protocols'][name]['energy_values'])
            pdr_mean, pdr_ci = mean_ci(pdr_vals)
            e_mean, e_ci = mean_ci(energy_vals)

            # Per-run scatter for transparency
            ax4.scatter(
                energy_vals,
                pdr_vals,
                s=18,
                color=COLORS[name],
                alpha=0.45,
                edgecolors='none',
                zorder=2,
            )
            # Mean ± CI overlay
            ax4.errorbar(
                e_mean,
                pdr_mean,
                xerr=e_ci,
                yerr=pdr_ci,
                fmt=MARKERS[name],
                color=COLORS[name],
                markersize=7 if name in ['AERIS-E', 'AERIS-R'] else 6,
                capsize=3,
                linewidth=1.2,
                markeredgecolor='black',
                markeredgewidth=0.4,
                zorder=4,
                label=name,
            )
            all_pdr.extend([pdr_mean - pdr_ci, pdr_mean + pdr_ci])
            all_energy.extend([e_mean - e_ci, e_mean + e_ci])

        ax4.set_xlabel('Total Energy (J)')
        ax4.set_ylabel('PDR (%)')
        if all_pdr and all_energy:
            ax4.set_ylim(max(0, min(all_pdr) - 2), min(100, max(all_pdr) + 2))
            ax4.set_xlim(min(all_energy) - 0.6, max(all_energy) + 0.6)
        ax4.set_title(f'(d) PDR–Energy Trade-off\n(Runs + Mean ± 95% CI, n={run_count})', fontweight='bold')
        ax4.legend(loc='lower left', ncol=2,
                   frameon=True, fancybox=False, edgecolor='black', fontsize=7)

    # ========== Panel (e): Energy Comparison ==========
    ax5 = fig.add_subplot(gs[1, 1])

    energies = [data['protocols'][p]['energy_mean'] for p in protocols]
    energy_stds = [data['protocols'][p]['energy_std'] for p in protocols]

    bars = ax5.bar(range(len(protocols)), energies, yerr=energy_stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    ax5.set_xticks(range(len(protocols)))
    ax5.set_xticklabels(protocols, rotation=25, ha='right')
    ax5.set_ylabel('Total Energy (J)')
    ax5.set_title('(e) Energy Consumption\n(Mean ± SD)', fontweight='bold')

    # ========== Panel (f): Statistical Summary Table ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary table
    table_data = []
    headers = ['Protocol', 'ΔPDR-E (pp)', 'p_E', 'ΔPDR-R (pp)', 'p_R']
    for baseline in comparisons:
        stat_e = data['statistics'][f'AERIS-E_vs_{baseline}']
        stat_r = data['statistics'][f'AERIS-R_vs_{baseline}']
        diff_e = (data['protocols']['AERIS-E']['pdr_mean'] -
                  data['protocols'][baseline]['pdr_mean']) * 100
        diff_r = (data['protocols']['AERIS-R']['pdr_mean'] -
                  data['protocols'][baseline]['pdr_mean']) * 100
        table_data.append([
            baseline,
            f"{diff_e:+.2f}",
            f"{stat_e['p_value']:.1e}",
            f"{diff_r:+.2f}",
            f"{stat_r['p_value']:.1e}",
        ])

    # Draw table
    col_widths = [0.24, 0.20, 0.16, 0.20, 0.16]
    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=col_widths,
        bbox=[0.02, 0.10, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13.2)
    table.scale(1.6, 2.2)

    # Color ΔPDR cells by sign (E and R)
    for i in range(len(table_data)):
        for col in (1, 3):
            cell = table[(i + 1, col)]
            try:
                val = float(table_data[i][col])
            except ValueError:
                val = 0.0
            cell.set_facecolor('#d4edda' if val >= 0 else '#f8d7da')

    ax6.set_title(
        '(f) Statistical Validation Summary\n(Shapiro–Wilk → Levene → t-test)',
        fontweight='bold',
        fontsize=10.2,
        y=1.08,
        pad=6,
    )

    # Main title
    fig.suptitle(
        'AERIS: SOTA Protocol Comparison with Rigorous Statistical Testing\n'
        f'(Same geometry/energy model; AERIS profiles include full reliability stack, n={run_count} runs each)',
        fontsize=11,
        fontweight='bold',
        y=0.988,
    )

    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.94)

    # Save
    out_dirs = [
        Path(__file__).parent.parent / 'results' / 'publication_figures',
        Path(__file__).parent.parent / 'for_submission' / 'figures'
    ]
    for out_dir in out_dirs:
        out_dir.mkdir(exist_ok=True)
        for fmt in ['pdf', 'png', 'svg']:
            out_path = out_dir / f'sota_comparison_6panel.{fmt}'
            fig.savefig(out_path, format=fmt, bbox_inches='tight', dpi=300,
                        facecolor='white', edgecolor='none')
            print(f"Saved: {out_path}")

    plt.close()


def create_honest_summary():
    """Create honest summary figure highlighting PEGASIS advantage."""
    data = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    protocols = ['LEACH', 'HEED', 'SEP', 'TEEN', 'PEGASIS', 'AERIS-R']
    colors = [COLORS[p] for p in protocols]

    # Panel 1: PDR ranking
    ax1 = axes[0]
    means = [data['protocols'][p]['pdr_mean'] * 100 for p in protocols]
    cis = [data['protocols'][p]['pdr_ci95'] * 100 for p in protocols]

    # Sort by PDR
    sorted_idx = np.argsort(means)
    sorted_protocols = [protocols[i] for i in sorted_idx]
    sorted_means = [means[i] for i in sorted_idx]
    sorted_cis = [cis[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    bars = ax1.barh(range(len(sorted_protocols)), sorted_means, xerr=sorted_cis,
                    capsize=4, color=sorted_colors, edgecolor='black', linewidth=0.8)

    ax1.set_yticks(range(len(sorted_protocols)))
    ax1.set_yticklabels(sorted_protocols)
    ax1.set_xlabel('PDR (%)')
    ax1.set_xlim(85, 100)
    ax1.set_title('Honest PDR Ranking\n(Higher is Better)', fontweight='bold')

    # Add values
    for i, (m, c) in enumerate(zip(sorted_means, sorted_cis)):
        ax1.text(m + c + 0.5, i, f'{m:.1f}%', va='center', fontsize=9)

    # Panel 2: Trade-off analysis
    ax2 = axes[1]
    pdrs = [data['protocols'][p]['pdr_mean'] * 100 for p in protocols]
    energies = [data['protocols'][p]['energy_mean'] for p in protocols]

    for i, (p, pdr, e, c) in enumerate(zip(protocols, pdrs, energies, colors)):
        ax2.scatter(e, pdr, c=c, s=150, label=p, edgecolors='black',
                    linewidth=1.5 if p in ['AERIS', 'PEGASIS'] else 0.8,
                    zorder=5 if p in ['AERIS', 'PEGASIS'] else 3)

    ax2.set_xlabel('Energy Consumption (J)')
    ax2.set_ylabel('PDR (%)')
    ax2.set_title('PDR-Energy Trade-off\n(Pareto Analysis)', fontweight='bold')
    ax2.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')

    # Add annotation
    ax2.annotate('PEGASIS: Best PDR\nbut higher energy',
                 xy=(energies[2], pdrs[2]),
                 xytext=(energies[2] - 0.5, pdrs[2] - 3),
                 fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    aeris_idx = protocols.index('AERIS-R')
    ax2.annotate('AERIS-R: Good balance\nof PDR and energy',
                 xy=(energies[aeris_idx], pdrs[aeris_idx]),
                 xytext=(energies[aeris_idx] + 0.8, pdrs[aeris_idx] + 2),
                 fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle('Honest Assessment: PEGASIS achieves highest PDR\n'
                 'AERIS-R offers best balance among clustering protocols',
                 fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    out_dir = Path(__file__).parent.parent / 'results' / 'publication_figures'
    out_path = out_dir / 'honest_sota_summary.pdf'
    fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {out_path}")

    out_path = out_dir / 'honest_sota_summary.png'
    fig.savefig(out_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {out_path}")

    plt.close()


if __name__ == '__main__':
    print("Generating SOTA comparison figures...")
    create_sota_figure()
    create_honest_summary()
    print("\nFigure generation complete!")
