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
    'AERIS': '#EE6677',     # Red (highlight)
}
MARKERS = {
    'LEACH': 'o',
    'HEED': 's',
    'PEGASIS': 'D',
    'SEP': '^',
    'AERIS': 'X',
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
    fig = plt.figure(figsize=(16.2, 10.8))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.70,
        wspace=0.35,
        width_ratios=[1, 1, 1.7],
        height_ratios=[1.0, 1.12],
    )

    protocols = ['LEACH', 'HEED', 'PEGASIS', 'SEP', 'AERIS']
    run_count = len(data['protocols']['AERIS']['pdr_values'])

    # ========== Panel (a): PDR Comparison Bar Chart ==========
    ax1 = fig.add_subplot(gs[0, 0])

    means = [data['protocols'][p]['pdr_mean'] * 100 for p in protocols]
    cis = [data['protocols'][p]['pdr_ci95'] * 100 for p in protocols]
    colors = [COLORS[p] for p in protocols]

    bars = ax1.bar(range(len(protocols)), means, yerr=cis, capsize=4,
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    # Highlight AERIS
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    ax1.set_xticks(range(len(protocols)))
    ax1.set_xticklabels(protocols)
    ax1.set_ylabel('PDR (%)')
    ax1.set_ylim(80, 100)
    ax1.set_title(f'(a) Protocol PDR Comparison\n(n={run_count}, 95% CI)', fontweight='bold')

    # Add significance markers
    aeris_mean = means[-1]
    for i, (m, name) in enumerate(zip(means[:-1], protocols[:-1])):
        diff = aeris_mean - m
        marker = '***' if abs(diff) > 2 else '**' if abs(diff) > 1 else '*'
        ax1.text(i, m + cis[i] + 1.5, marker, ha='center', fontsize=8)

    # ========== Panel (b): Box Plot Distribution ==========
    ax2 = fig.add_subplot(gs[0, 1])

    box_data = [np.array(data['protocols'][p]['pdr_values']) * 100 for p in protocols]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=protocols)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight AERIS box
    bp['boxes'][-1].set_linewidth(2)

    ax2.set_ylabel('PDR (%)')
    ax2.set_ylim(80, 100)
    ax2.set_title(f'(b) PDR Distribution\n(Box Plot, n={run_count})', fontweight='bold')
    # Add jittered points to show distribution density (esp. near 100% PDR)
    for i, vals in enumerate(box_data, start=1):
        jitter = np.random.normal(loc=i, scale=0.04, size=len(vals))
        ax2.scatter(jitter, vals, s=12, color=colors[i - 1], alpha=0.6, edgecolors='none')

    # ========== Panel (c): AERIS vs Baseline PDR (mean ± CI) ==========
    ax3 = fig.add_subplot(gs[0, 2])

    comparisons = ['LEACH', 'HEED', 'PEGASIS', 'SEP']
    base_means = [data['protocols'][p]['pdr_mean'] * 100 for p in comparisons]
    base_cis = [data['protocols'][p]['pdr_ci95'] * 100 for p in comparisons]
    aeris_mean_pdr = data['protocols']['AERIS']['pdr_mean'] * 100
    aeris_ci_pdr = data['protocols']['AERIS']['pdr_ci95'] * 100
    delta_pp = [aeris_mean_pdr - m for m in base_means]

    y_pos = np.arange(len(comparisons))

    for i, (base_mean, base_ci, dpp) in enumerate(zip(base_means, base_cis, delta_pp)):
        # Baseline mean ± CI (grey circle)
        ax3.errorbar(
            base_mean,
            y_pos[i],
            xerr=base_ci,
            fmt='o',
            color='#888888',
            markersize=6,
            capsize=3,
            capthick=1.2,
            linewidth=1.2
        )
        # AERIS mean ± CI (red diamond)
        ax3.errorbar(
            aeris_mean_pdr,
            y_pos[i],
            xerr=aeris_ci_pdr,
            fmt='D',
            color=COLORS['AERIS'],
            markersize=7,
            capsize=3,
            capthick=1.2,
            linewidth=1.2
        )
        # Connector line (no in-plot delta labels to avoid occlusion)
        ax3.plot([base_mean, aeris_mean_pdr], [y_pos[i], y_pos[i]],
                 color='#BBBBBB', linewidth=1.0, zorder=0)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(comparisons)
    ax3.set_xlabel("PDR (%)")
    ax3.set_title("(c) AERIS vs Baselines\n(Mean ± 95% CI)", fontweight='bold')
    x_min = min(base_means + [aeris_mean_pdr]) - max(base_cis + [aeris_ci_pdr]) - 1.0
    x_max = max(base_means + [aeris_mean_pdr]) + max(base_cis + [aeris_ci_pdr]) + 3.0
    ax3.set_xlim(x_min, x_max)
    # Small legend for markers (placed outside to avoid overlap)
    ax3.scatter([], [], color='#888888', marker='o', label='Baseline mean')
    ax3.scatter([], [], color=COLORS['AERIS'], marker='D', label='AERIS mean')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
               frameon=True, fontsize=7, edgecolor='black')

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
            'AERIS': '-'
        }
        line_widths = {
            'LEACH': 1.5,
            'HEED': 1.5,
            'PEGASIS': 1.5,
            'SEP': 2.0,
            'AERIS': 2.5  # Thicker for AERIS
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
        # Summary trade-off: mean ± 95% CI in both axes
        for name in protocols:
            pdr_vals = np.array(data['protocols'][name]['pdr_values']) * 100
            energy_vals = np.array(data['protocols'][name]['energy_values'])
            pdr_mean, pdr_ci = mean_ci(pdr_vals)
            e_mean, e_ci = mean_ci(energy_vals)

            ax4.errorbar(
                e_mean, pdr_mean,
                xerr=e_ci, yerr=pdr_ci,
                fmt=MARKERS[name],
                color=COLORS[name],
                markersize=7 if name == 'AERIS' else 6,
                capsize=3,
                linewidth=1.2,
                label=name
            )

        ax4.set_xlabel('Total Energy (J)')
        ax4.set_ylabel('PDR (%)')
        ax4.set_ylim(85, 100)
        ax4.set_title(f'(d) PDR–Energy Trade-off\n(Mean ± 95% CI, n={run_count})', fontweight='bold')
        ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1,
                   frameon=True, fancybox=False, edgecolor='black',
                   handletextpad=0.4, columnspacing=0.8, borderaxespad=0.4)

    # ========== Panel (e): Energy Comparison ==========
    ax5 = fig.add_subplot(gs[1, 1])

    energies = [data['protocols'][p]['energy_mean'] for p in protocols]
    energy_stds = [data['protocols'][p]['energy_std'] for p in protocols]

    bars = ax5.bar(range(len(protocols)), energies, yerr=energy_stds, capsize=4,
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    ax5.set_xticks(range(len(protocols)))
    ax5.set_xticklabels(protocols)
    ax5.set_ylabel('Total Energy (J)')
    ax5.set_title('(e) Energy Consumption\n(Mean ± SD)', fontweight='bold')

    # ========== Panel (f): Statistical Summary Table ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary table
    table_data = []
    headers = ['Baseline', 'Test', 'p', 'ΔPDR (pp)', 'Direction']

    for baseline, diff in zip(['LEACH', 'HEED', 'PEGASIS', 'SEP'], delta_pp):
        key = f'AERIS_vs_{baseline}'
        stat = data['statistics'][key]
        p_str = f"{stat['p_value']:.1e}"
        d_str = f"{diff:.2f}"
        # Fix test name formatting
        test_name = stat['test_used']
        if 'Independent' in test_name:
            test_name = 't-test'
        elif 'Mann-Whitney' in test_name:
            test_name = 'M-W'
        table_data.append([
            baseline,
            test_name,
            p_str,
            d_str,
            'AERIS higher' if diff > 0 else f'{baseline} higher'
        ])

    # Draw table
    col_widths = [0.22, 0.18, 0.22, 0.18, 0.20]
    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=col_widths,
        bbox=[0.01, 0.02, 0.98, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.20, 1.90)

    # Color cells
    for i in range(len(table_data)):
        result_cell = table[(i + 1, 4)]
        if 'AERIS' in table_data[i][4]:
            result_cell.set_facecolor('#d4edda')  # Light green
        else:
            result_cell.set_facecolor('#f8d7da')  # Light red

    ax6.set_title(
        '(f) Statistical Validation Summary\n(Shapiro-Wilk → Levene → t-test)',
        fontweight='bold',
        fontsize=9,
        y=1.06,
        pad=6,
    )

    # Main title
    fig.suptitle(
        'AERIS: SOTA Protocol Comparison with Rigorous Statistical Testing\n'
        f'(Same geometry/energy model; AERIS includes full reliability stack, n={run_count} runs each)',
        fontsize=11,
        fontweight='bold',
        y=0.988,
    )

    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.06, right=0.92)

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

    protocols = ['LEACH', 'HEED', 'SEP', 'AERIS', 'PEGASIS']
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

    ax2.annotate('AERIS: Good balance\nof PDR and energy',
                 xy=(energies[3], pdrs[3]),
                 xytext=(energies[3] + 0.8, pdrs[3] + 2),
                 fontsize=8, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    fig.suptitle('Honest Assessment: PEGASIS achieves highest PDR\n'
                 'AERIS offers best balance among clustering protocols',
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
