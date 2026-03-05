#!/usr/bin/env python3
"""
MEGA Panel Figures v2 for AERIS Sensors Paper.
Usage: conda run -n aether-wsn python scripts/plot_mega_panels.py
"""
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08, 'axes.linewidth': 0.5,
    'axes.grid': True, 'grid.alpha': 0.15, 'grid.linewidth': 0.3,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
})

PROTOCOLS = ['AERIS', 'LEACH', 'PEGASIS', 'HEED', 'TEEN']
ENVS = ['indoor_office', 'indoor_factory', 'outdoor_urban', 'outdoor_suburban']
NODES = [100, 200, 300, 500, 800, 1000]
PCOL = {'AERIS': '#1a5276', 'LEACH': '#cb4335', 'PEGASIS': '#b7950b',
        'HEED': '#6c3483', 'TEEN': '#1e8449'}
PMARK = {'AERIS': 'o', 'LEACH': 's', 'PEGASIS': '^', 'HEED': 'D', 'TEEN': 'v'}
ENV_SHORT = {'indoor_office': 'Indoor Office', 'indoor_factory': 'Indoor Factory',
             'outdoor_urban': 'Outdoor Urban', 'outdoor_suburban': 'Outdoor Suburban'}

EVIDENCE = Path('for_submission/submission_bundle_v54_20260222/evidence')
OUTDIR = Path('for_submission/figures_mega')

def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / name
    fig.savefig(out, format='pdf')
    fig.savefig(out.with_suffix('.png'), format='png')
    plt.close(fig)
    print(f"  -> {out}")


def plot_mega_a():
    """v50-rigor scalability: 2x2 line chart, 5 protocols per panel."""
    print("[A] v50-rigor scalability line-panel ...")
    rows = read_csv(EVIDENCE / 'scalability_4env_v50rigor_20260222_descriptive.csv')
    d = {}
    for r in rows:
        d[(r['environment'], int(r['num_nodes']), r['protocol'])] = float(r['pdr_mean'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for idx, env in enumerate(ENVS):
        ax = axes.flat[idx]
        for p in PROTOCOLS:
            ys = [d.get((env, n, p), 0) for n in NODES]
            ax.plot(NODES, ys, marker=PMARK[p], color=PCOL[p],
                    label=p, linewidth=1.4, markersize=5, zorder=3)
        ax.set_title(ENV_SHORT[env], fontweight='bold', fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(50, 1050)
        ax.set_ylabel('PDR')
        ax.set_xlabel('Node Count')
        a1k = d.get((env, 1000, 'AERIS'), 0)
        ax.annotate(f'{a1k:.3f}', xy=(1000, a1k),
                    xytext=(820, min(a1k + 0.08, 0.95)), fontsize=6,
                    arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),
                    color=PCOL['AERIS'], fontweight='bold')

    h = [Line2D([0], [0], color=PCOL[p], marker=PMARK[p],
         label=p, linewidth=1.4, markersize=5) for p in PROTOCOLS]
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=8,
               frameon=True, edgecolor='#ccc', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Primary Scalability Matrix \u2014 PDR across Node Counts\n'
                 '(n=3200 per cell, MAC-collision + multi-hop relay)',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, 'mega_fig_a_v50rigor_panorama.pdf')


def plot_mega_b():
    """S11 patch-vs-control delta: 2x2 line chart."""
    print("[B] S11 patch-vs-control delta panel ...")
    rows = read_csv(EVIDENCE / 's11_matched_4env_patch_vs_control_20260217_delta.csv')
    sr = read_csv(EVIDENCE / 's11_matched_4env_patch_vs_control_20260217_significance.csv')
    dd = {}
    for r in rows:
        dd[(r['environment'], int(r['num_nodes']), r['protocol'])] = float(r['delta'])
    sig = {}
    for r in sr:
        sig[(r['environment'], int(r['num_nodes']), r['protocol'])] = (
            r['significant_005'].strip().lower() == 'yes')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for idx, env in enumerate(ENVS):
        ax = axes.flat[idx]
        ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
        for p in PROTOCOLS:
            ys = [dd.get((env, n, p), 0) for n in NODES]
            ss = [sig.get((env, n, p), False) for n in NODES]
            ax.plot(NODES, ys, marker=PMARK[p], color=PCOL[p],
                    label=p, linewidth=1.4, markersize=5, zorder=3)
            for nn, y, s in zip(NODES, ys, ss):
                if not s:
                    ax.plot(nn, y, marker=PMARK[p], color='white',
                            markeredgecolor=PCOL[p], markersize=5, zorder=4)
        ax.set_title(ENV_SHORT[env], fontweight='bold', fontsize=9)
        ax.set_ylabel('\u0394PDR (patch \u2212 control)')
        ax.set_xlabel('Node Count')
        ax.set_xlim(50, 1050)

    h = [Line2D([0], [0], color=PCOL[p], marker=PMARK[p],
         label=p, linewidth=1.4, markersize=5) for p in PROTOCOLS]
    h.append(Line2D([0], [0], marker='o', color='white',
             markeredgecolor='gray', label='not sig.', markersize=5, lw=0))
    fig.legend(handles=h, loc='lower center', ncol=6, fontsize=7,
               frameon=True, edgecolor='#ccc', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Matched Patch-vs-Control \u0394PDR across Node Counts\n'
                 '(n=1000 per arm; hollow = not significant after Holm)',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    _save(fig, 'mega_fig_b_s11_patch_control.pdf')


def plot_mega_c():
    """S10 power sensitivity: 4x3 grouped bar chart."""
    print("[C] S10 power sensitivity matrix ...")
    rows = read_csv(EVIDENCE / 's10_4env_merged_descriptive_20260216.csv')
    d = {}
    for r in rows:
        d[(r['environment'], float(r['tx_power']),
           int(r['num_nodes']), r['protocol'])] = float(r['pdr_mean'])
    s10n = [100, 500, 1000]
    fig, axes = plt.subplots(4, 3, figsize=(9, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.28)
    x = np.arange(len(PROTOCOLS))
    w = 0.35
    for ri, env in enumerate(ENVS):
        for ci, nn in enumerate(s10n):
            ax = axes[ri, ci]
            v5 = [d.get((env, 5.0, nn, p), 0) for p in PROTOCOLS]
            v15 = [d.get((env, 15.0, nn, p), 0) for p in PROTOCOLS]
            ax.bar(x - w/2, v5, w, color='#5dade2', edgecolor='black', lw=0.3)
            ax.bar(x + w/2, v15, w, color='#e74c3c', edgecolor='black',
                   lw=0.3, hatch='///')
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x)
            ax.set_xticklabels([p[:3] for p in PROTOCOLS], fontsize=5.5,
                               rotation=45)
            if ci == 0:
                ax.set_ylabel(ENV_SHORT[env], fontsize=7.5, fontweight='bold')
            if ri == 0:
                ax.set_title(f'N={nn}', fontsize=8, fontweight='bold')
            if ci > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis='y', labelsize=6)
    h = [Patch(facecolor='#5dade2', edgecolor='black', label='5 dBm'),
         Patch(facecolor='#e74c3c', edgecolor='black', hatch='///',
               label='15 dBm')]
    fig.legend(handles=h, loc='lower center', ncol=2, fontsize=8,
               frameon=True, edgecolor='#ccc', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('TX-Power Sensitivity \u2014 PDR at 5 vs 15 dBm\n'
                 '(n=600 per cell; hatched = 15 dBm)',
                 fontsize=10, fontweight='bold', y=1.01)
    _save(fig, 'mega_fig_c_s10_power_sensitivity.pdf')


def plot_mega_d():
    """Dual heatmap: v50 PDR + S11 delta, 20 rows x 6 cols each."""
    print("[D] Dual heatmap matrix ...")
    v50 = read_csv(EVIDENCE / 'scalability_4env_v50rigor_20260222_descriptive.csv')
    s11 = read_csv(EVIDENCE / 's11_matched_4env_patch_vs_control_20260217_delta.csv')

    pairs = [(p, e) for p in PROTOCOLS for e in ENVS]
    labels = [f'{p} / {ENV_SHORT[e]}' for p, e in pairs]
    nR, nC = len(pairs), len(NODES)

    v50d = {(r['protocol'], r['environment'], int(r['num_nodes'])): float(r['pdr_mean']) for r in v50}
    s11d = {(r['protocol'], r['environment'], int(r['num_nodes'])): float(r['delta']) for r in s11}

    mat_a = np.array([[v50d.get((p, e, n), np.nan) for n in NODES] for p, e in pairs])
    mat_b = np.array([[s11d.get((p, e, n), np.nan) for n in NODES] for p, e in pairs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(wspace=0.35)

    # Left: absolute PDR
    im1 = ax1.imshow(mat_a, aspect='auto', cmap='YlOrRd_r', vmin=0, vmax=1)
    ax1.set_xticks(range(nC)); ax1.set_xticklabels([str(n) for n in NODES], fontsize=7)
    ax1.set_yticks(range(nR)); ax1.set_yticklabels(labels, fontsize=5.5)
    ax1.set_xlabel('Node Count', fontsize=9)
    ax1.set_title('(a) Absolute PDR', fontsize=9, fontweight='bold')
    for ri in range(nR):
        for ci in range(nC):
            v = mat_a[ri, ci]
            ax1.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                     fontsize=4.5, color='white' if v < 0.3 else 'black')
    for i in range(1, len(PROTOCOLS)):
        ax1.axhline(y=i * len(ENVS) - 0.5, color='white', lw=1.5)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cb1.set_label('PDR', fontsize=8); cb1.ax.tick_params(labelsize=6)

    # Right: S11 delta
    vm = max(abs(np.nanmin(mat_b)), abs(np.nanmax(mat_b)))
    im2 = ax2.imshow(mat_b, aspect='auto', cmap='RdBu', vmin=-vm, vmax=vm)
    ax2.set_xticks(range(nC)); ax2.set_xticklabels([str(n) for n in NODES], fontsize=7)
    ax2.set_yticks(range(nR)); ax2.set_yticklabels(labels, fontsize=5.5)
    ax2.set_xlabel('Node Count', fontsize=9)
    ax2.set_title('(b) Patch \u2212 Control \u0394PDR', fontsize=9, fontweight='bold')
    for ri in range(nR):
        for ci in range(nC):
            v = mat_b[ri, ci]
            ax2.text(ci, ri, f'{v:+.2f}', ha='center', va='center',
                     fontsize=4.5, color='white' if abs(v) > 0.3 else 'black')
    for i in range(1, len(PROTOCOLS)):
        ax2.axhline(y=i * len(ENVS) - 0.5, color='white', lw=1.5)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
    cb2.set_label('\u0394PDR', fontsize=8); cb2.ax.tick_params(labelsize=6)

    fig.suptitle('Full Evidence Heatmap \u2014 120 Cells per Panel\n'
                 '(Left: n=3200; Right: matched \u0394 n=1000 per arm)',
                 fontsize=10, fontweight='bold', y=1.02)
    _save(fig, 'mega_fig_d_dual_heatmap.pdf')


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent.parent)
    plot_mega_a()
    plot_mega_b()
    plot_mega_c()
    plot_mega_d()
    print("\nAll 4 mega figures generated.")
