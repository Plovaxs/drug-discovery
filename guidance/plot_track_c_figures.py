"""Publication-quality figures for the Track C statistical findings
(guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md), for direct use in the
thesis/paper. Uses the validated colorblind-safe palette from this
project's dataviz reference (light-mode categorical + status slots).

Outputs both vector PDF (for LaTeX inclusion, infinitely scalable, small
file size for line/bar charts) and 300 DPI PNG (for quick preview/slides)
per panel, plus one 3-panel composite figure.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

RESULTS_PATH = './guidance/track_c_analysis/track_c_analysis_results.json'
OUT_DIR = './guidance/track_c_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# -- validated palette (guidance dataviz skill, references/palette.md) --
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
SURFACE = '#fcfcfb'

STATUS_GOOD = '#0ca30c'
STATUS_WARNING = '#fab219'
STATUS_SERIOUS = '#ec835a'
STATUS_CRITICAL = '#d03b3b'
CAT_BLUE = '#2a78d6'
CAT_ORANGE = '#eb6834'

VERDICT_COLOR = {
    'clear_signal': STATUS_GOOD,
    'signal_but_confounded_by_size': STATUS_WARNING,
    'signal_but_quality_degraded': STATUS_SERIOUS,
    'signal_wrong_direction': STATUS_CRITICAL,
    'no_signal': INK_MUTED,
}
VERDICT_LABEL = {
    'clear_signal': 'Clear signal',
    'signal_but_confounded_by_size': 'Confounded by size',
    'signal_but_quality_degraded': 'Quality degraded',
    'signal_wrong_direction': 'Wrong direction',
    'no_signal': 'No signal',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10.5,
    'text.color': INK_PRIMARY,
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK_SECONDARY,
    'axes.titlecolor': INK_PRIMARY,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'grid.color': GRID,
    'grid.linewidth': 0.7,
    'axes.linewidth': 0.8,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,  # embed as real text, not paths
})

SHORT_NAME = {
    'BSD_ASPTE_1_130_0': 'BSD_ASPTE', 'HDAC8_HUMAN_1_377_0': 'HDAC8',
    'CD38_HUMAN_44_300_0': 'CD38', 'MENE_BACSU_2_486_0': 'MENE_BACSU',
    'NQO1_HUMAN_2_274_0': 'NQO1', 'PAK4_HUMAN_291_591_ATP_0': 'PAK4',
    'PNTM_STRAE_2_398_0': 'PNTM_STRAE', 'RIBB_VIBCH_2_218_0': 'RIBB_VIBCH',
    'DPP2_HUMAN_27_492_0': 'DPP2', 'XANLY_BACGL_26_777_0': 'XANLY_BACGL',
    'IMA1_HUMAN_68_497_0': 'IMA1', 'NOS2_HUMAN_78_505_0': 'NOS2',
    'P2Y12_HUMAN_1_342_0': 'P2Y12', 'PYRD_TRYCC_2_314_catalytic_0': 'PYRD',
    'SQHC_ALIAD_1_631_0': 'SQHC_ALIAD',
}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')


def panel_a_forest(ax, data):
    """Forest plot: raw vina_dock effect size + 95% CI, per pocket,
    colored by verdict. Sorted by effect size (most improved at top)."""
    rows = []
    for target, r in data.items():
        rows.append((SHORT_NAME.get(target, target), r['effect_size'],
                     r['effect_size_ci95'][0], r['effect_size_ci95'][1], r['verdict']))
    rows.sort(key=lambda x: x[1])

    y = np.arange(len(rows))
    for i, (name, eff, lo, hi, verdict) in enumerate(rows):
        color = VERDICT_COLOR[verdict]
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle='round', zorder=2)
        ax.plot(eff, i, 'o', color=color, ms=6, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.8)

    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel('Effect size: mean top-10% vina_dock − full-pool mean\n(more negative = apparently better)')
    ax.set_title('A. Raw Vina Dock “improvement” looks real — by itself', loc='left',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)

    handles = [plt.Line2D([0], [0], marker='o', color=c, linestyle='', markersize=7, label=VERDICT_LABEL[v])
               for v, c in VERDICT_COLOR.items() if v in {r[4] for r in rows}]
    ax.legend(handles=handles, loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def panel_b_ligand_efficiency(ax, data):
    """Slope chart: ligand efficiency, full pool -> top-10%, per pocket.
    Nearly all pockets slope the WRONG way (worse per-atom efficiency)."""
    rows = []
    for target, r in data.items():
        vh = r['vina_hacking_check']
        rows.append((SHORT_NAME.get(target, target), vh['mean_ligand_efficiency_full'],
                     vh['mean_ligand_efficiency_topk'], vh.get('ligand_efficiency_effect', 0) or 0))
    rows.sort(key=lambda x: x[3])

    for i, (name, full, topk, eff) in enumerate(rows):
        worse = eff > 0
        color = STATUS_CRITICAL if worse else STATUS_GOOD
        ax.plot([0, 1], [full, topk], color=color, lw=1.6, alpha=0.85, zorder=2)
        ax.plot(0, full, 'o', color=INK_MUTED, ms=5, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.plot(1, topk, 'o', color=color, ms=5, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Full pool', 'Top-10% by\npredicted affinity'])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel('Ligand efficiency (vina_dock / heavy atoms)\nmore negative = better per-atom binding')
    ax.set_title('B. Size-normalized: top-10% is worse in 14/15 pockets', loc='left',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    handles = [plt.Line2D([0], [0], color=STATUS_CRITICAL, lw=2, label='Worse after filtering (14/15)'),
               plt.Line2D([0], [0], color=STATUS_GOOD, lw=2, label='Better after filtering (1/15)')]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def panel_c_posebusters(ax, data):
    """Grouped bars: PoseBusters valid rate, full pool vs top-10%, per
    pocket, sorted by drop magnitude."""
    rows = []
    for target, r in data.items():
        vh = r['vina_hacking_check']
        drop = vh['pb_valid_rate_full'] - vh['pb_valid_rate_topk']
        rows.append((SHORT_NAME.get(target, target), vh['pb_valid_rate_full'], vh['pb_valid_rate_topk'], drop))
    rows.sort(key=lambda x: -x[3])

    y = np.arange(len(rows))
    h = 0.36
    ax.barh(y + h / 2, [r[1] for r in rows], height=h, color=CAT_BLUE, label='Full pool', zorder=2)
    ax.barh(y - h / 2, [r[2] for r in rows], height=h, color=STATUS_CRITICAL, label='Top-10%', zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel('PoseBusters valid rate')
    ax.set_xlim(0, 1)
    ax.set_title('C. Top-ranked molecules are structurally less valid', loc='left',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc='lower right', frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def main():
    with open(RESULTS_PATH) as f:
        summary = json.load(f)
    data = summary['per_pocket_primary']

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    panel_a_forest(ax, data)
    fig.tight_layout()
    save(fig, 'panel_a_effect_size_forest')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 6.2))
    panel_b_ligand_efficiency(ax, data)
    fig.tight_layout()
    save(fig, 'panel_b_ligand_efficiency')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    panel_c_posebusters(ax, data)
    fig.tight_layout()
    save(fig, 'panel_c_posebusters')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    panel_a_forest(axes[0], data)
    panel_b_ligand_efficiency(axes[1], data)
    panel_c_posebusters(axes[2], data)
    fig.suptitle('Track C: rejection-sampling "improvement" is a size artifact, not a real affinity signal',
                 fontsize=13, fontweight='bold', color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    save(fig, 'track_c_combined_figure')
    plt.close(fig)

    print(f'Wrote figures to {OUT_DIR}/')


if __name__ == '__main__':
    main()
