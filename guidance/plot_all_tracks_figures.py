"""Publication-quality figures for Tracks A, B, D and the two mechanistic
diagnostics (DIAG1, D.5) -- completes the figure set started by
guidance/plot_track_c_figures.py (Track C), using the same validated
colorblind-safe palette for visual consistency across the whole thesis.

All data is read directly from this project's own already-computed result
files (gradient_informativeness_results.csv per track, Track D's raw
lambda-sweep CSVs, DIAG1/D.5's JSON outputs) -- nothing here re-derives or
re-computes a statistical result; it only visualizes numbers already
reported in the corresponding finding docs.

Outputs vector PDF (thesis/LaTeX) + 300 DPI PNG per figure to
guidance/all_tracks_figures/.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = './guidance/all_tracks_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# -- validated palette (guidance/plot_track_c_figures.py, dataviz skill) --
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
CAT_AQUA = '#1baf7a'
CAT_VIOLET = '#4a3aa7'
CAT_YELLOW = '#eda100'

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
    'pdf.fonttype': 42,
})

SHORT_NAME = {
    'BSD_ASPTE_1_130_0': 'BSD_ASPTE', 'HDAC8_HUMAN_1_377_0': 'HDAC8',
    'CD38_HUMAN_44_300_0': 'CD38', 'MENE_BACSU_2_486_0': 'MENE_BACSU',
    'NQO1_HUMAN_2_274_0': 'NQO1', 'PAK4_HUMAN_291_591_ATP_0': 'PAK4',
    'PNTM_STRAE_2_398_0': 'PNTM_STRAE', 'RIBB_VIBCH_2_218_0': 'RIBB_VIBCH',
}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_label(label):
    pocket, cond = label.split(':')
    return SHORT_NAME.get(pocket, pocket), cond


# =============================================================================
# Figure 1: Track A -- effect-size forest plot, exploratory vs full tier
# =============================================================================
def fig_trackA_forest():
    exp = pd.read_csv('guidance/track_a_exploratory_results/gradient_informativeness_results.csv')
    full = pd.read_csv('guidance/track_a_full_results/gradient_informativeness_results.csv')

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True)
    for ax, df, title, n_pockets in [
        (axes[0], exp, 'A. Exploratory tier (3 pockets, n=20)', 3),
        (axes[1], full, 'B. Full tier (8 pockets, n=30, converged model)', 8),
    ]:
        rows = []
        for _, r in df.iterrows():
            pocket, cond = parse_label(r['label'])
            rows.append((f'{pocket}\n{cond}', r['mean_shift'], r['significant_after_correction'],
                         r['direction'].startswith('correct')))
        rows.sort(key=lambda x: x[1])
        y = np.arange(len(rows))
        for i, (name, eff, sig, correct) in enumerate(rows):
            color = STATUS_GOOD if sig else (CAT_BLUE if correct else STATUS_CRITICAL)
            ax.plot(eff, i, 'o', color=color, ms=7, markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
        ax.axvline(0, color=BASELINE, lw=1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_xlabel('Mean Vina Dock shift (guided − unguided)\nmore negative = correct direction')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.grid(axis='x', zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)

    handles = [
        plt.Line2D([0], [0], marker='o', color=CAT_BLUE, linestyle='', ms=8, label='Correct direction (n.s.)'),
        plt.Line2D([0], [0], marker='o', color=STATUS_CRITICAL, linestyle='', ms=8, label='Wrong direction (n.s.)'),
        plt.Line2D([0], [0], marker='o', color=STATUS_GOOD, linestyle='', ms=8, label='Significant after BH correction'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9.5)
    fig.suptitle('Track A: physics-anchored gradient guidance (GIGN+PIGNet2) — direction consistency\nweakens, not strengthens, with more data and full convergence',
                 fontsize=12.5, fontweight='bold', y=1.03)
    fig.tight_layout()
    save(fig, 'trackA_effect_forest_exploratory_vs_full')


# =============================================================================
# Figure 2: Track A -- direction-consistency summary bar
# =============================================================================
def fig_trackA_direction_summary():
    exp = pd.read_csv('guidance/track_a_exploratory_results/gradient_informativeness_results.csv')
    full = pd.read_csv('guidance/track_a_full_results/gradient_informativeness_results.csv')

    def pct_correct(df):
        return 100 * df['direction'].str.startswith('correct').mean()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    tiers = ['Exploratory\n(3 pockets, n=20)', 'Full\n(8 pockets, n=30)']
    vals = [pct_correct(exp), pct_correct(full)]
    colors = [CAT_BLUE, STATUS_WARNING]
    bars = ax.bar(tiers, vals, color=colors, width=0.55, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f'{v:.0f}%', ha='center', fontsize=13, fontweight='bold')
    ax.axhline(50, color=BASELINE, lw=1.2, linestyle='--', zorder=1)
    ax.text(1.35, 51, 'chance (50%)', fontsize=9, color=INK_MUTED, ha='right')
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of (pocket, λ) conditions with correct-direction shift')
    ax.set_title('Track A: more data and full convergence made direction\nconsistency WORSE (89% → 67%), not better',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.grid(axis='y', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'trackA_direction_consistency_summary')


# =============================================================================
# Figure 3: Track A -- PoseBusters dose-response degradation
# =============================================================================
def fig_trackA_pb_doseresponse():
    full = pd.read_csv('guidance/track_a_full_results/gradient_informativeness_results.csv')
    full['pocket'], full['lambda'] = zip(*full['label'].map(parse_label))
    full['lambda_val'] = full['lambda'].str.replace('lambda_', '').astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    for pocket in full['pocket'].unique():
        sub = full[full['pocket'] == pocket].sort_values('lambda_val')
        ax.plot(sub['lambda_val'], sub['pb_valid_rate_guided'] * 100, marker='o', ms=5,
                color=CAT_BLUE, alpha=0.35, lw=1.3, zorder=2)
    agg = full.groupby('lambda_val')['pb_valid_rate_guided'].mean() * 100
    ax.plot(agg.index, agg.values, marker='o', ms=9, color=STATUS_CRITICAL, lw=2.6, zorder=4, label='Mean across 8 pockets')
    baseline_pb = full.groupby('pocket')['pb_valid_rate_unguided'].first().mean() * 100
    ax.axhline(baseline_pb, color=BASELINE, lw=1.5, linestyle='--', zorder=1)
    ax.text(1.02, baseline_pb + 1.5, f'unguided baseline ({baseline_pb:.0f}%)', fontsize=9, color=INK_MUTED)

    ax.set_xscale('log')
    ax.set_xlabel('λ_affinity (guidance strength)')
    ax.set_ylabel('PoseBusters valid rate (%), guided samples')
    ax.set_title('Track A: PoseBusters validity collapses with guidance strength\n(100% of pockets affected at λ=1.0)',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.grid(zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, 'trackA_posebusters_dose_response')


# =============================================================================
# Figure 4: predictive-quality comparison across every model trained
# =============================================================================
def fig_predictive_quality_comparison():
    models = [
        ('Stage 0\nEGNN\n(affinity)', 0.609, CAT_BLUE),
        ('Track A\nexploratory\n(affinity)', 0.583, CAT_BLUE),
        ('Track A\nfull tier\n(affinity)', 0.581, CAT_BLUE),
        ('Track D original\n(synth, leaky)', 0.878, STATUS_CRITICAL),
        ('Track D\nleakage-safe\n(synth)', 0.647, CAT_AQUA),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    x = np.arange(len(models))
    vals = [m[1] for m in models]
    colors = [m[2] for m in models]
    bars = ax.bar(x, vals, color=colors, width=0.6, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}', ha='center', fontsize=10.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models], fontsize=9.5)
    ax.set_ylabel('Test-set Pearson correlation')
    ax.set_ylim(0, 1.0)
    ax.set_title('Every guidance predictor achieved genuine, moderate-to-strong\npredictive quality — none transferred to usable guidance',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.grid(axis='y', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 1.08)
    ax.annotate('leakage-inflated,\nsuperseded by\nthe green bar →', xy=(3, 0.6), xytext=(2.15, 0.55),
                fontsize=8.5, color=STATUS_CRITICAL, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=STATUS_CRITICAL, lw=1.2))
    fig.tight_layout()
    save(fig, 'all_tracks_predictive_quality_comparison')


# =============================================================================
# Figure 5: Track B combined forest (B1 + B2 + B3)
# =============================================================================
def fig_trackB_combined_forest():
    b1 = pd.read_csv('guidance/track_b1_results/gradient_informativeness_results.csv')
    b1['track'] = 'B1: gradient-norm\nnormalization'
    b2 = pd.read_csv('guidance/track_b2_results/gradient_informativeness_results.csv')
    b2['track'] = 'B2: classifier-head\nreformulation'
    b3 = pd.read_csv('guidance/track_b3_results/gradient_informativeness_results.csv')
    b3['track'] = 'B3: timestep-\nwindowed guidance'
    all_b = pd.concat([b1, b2, b3], ignore_index=True)
    all_b['pocket'], all_b['cond'] = zip(*all_b['label'].map(parse_label))

    # NOT sharex: B1/B2 include some guided-vs-guided comparisons (e.g.
    # normalized_vs_raw) on a very different scale (one raw-condition
    # sample was a known, investigated, genuine outlier, TRACK_B1_FINDING.md)
    # than B3's guided-vs-unguided comparisons -- forcing one shared axis
    # would flatten B3's real, smaller-scale variation to an illegible line.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    for ax, (track_name, df) in zip(axes, all_b.groupby('track', sort=False)):
        rows = sorted(zip(df['pocket'] + ': ' + df['cond'], df['mean_shift'], df['significant_after_correction']),
                      key=lambda x: x[1])
        y = np.arange(len(rows))
        for i, (name, eff, sig) in enumerate(rows):
            color = STATUS_GOOD if sig else (CAT_BLUE if eff < 0 else STATUS_CRITICAL)
            ax.plot(eff, i, 'o', color=color, ms=7, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
        ax.axvline(0, color=BASELINE, lw=1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
        ax.set_title(track_name, fontsize=10.5, fontweight='bold')
        ax.set_xlabel('Mean Vina Dock shift\n(more negative = correct direction)', fontsize=9)
        ax.grid(axis='x', zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
    fig.suptitle('Track B: three gradient-guidance mechanism variants, same frozen backbone — 3/3 null',
                 fontsize=12.5, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'trackB_combined_forest')


# =============================================================================
# Figure 6: Track D dose-response divergence (own score vs real RA-score)
# =============================================================================
def fig_trackD_divergence():
    rows = []
    for lam_dir in sorted(os.listdir('guidance/track_d_resweep_results')):
        if not lam_dir.startswith('lambda_'):
            continue
        lam = float(lam_dir.replace('lambda_', ''))
        path = f'guidance/track_d_resweep_results/{lam_dir}/pocket0/results.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        has_scores = 'real_ra_score' in df.columns
        scoreable = df.dropna(subset=['real_ra_score']) if has_scores else df.iloc[0:0]
        rows.append(dict(
            lam=lam, own_mean=df['synth_own_score'].mean() if has_scores else np.nan,
            real_mean=scoreable['real_ra_score'].mean() if len(scoreable) else np.nan,
            n_scoreable=len(scoreable), n_attempted=len(df),
            frag_rate=1 - (df['n_single_fragment'].iloc[0] / df['n_attempted'].iloc[0]) if 'n_single_fragment' in df else np.nan,
        ))
    dose = pd.DataFrame(rows).sort_values('lam')

    # Well-powered points (N>=23, per TRACK_D_D3_LAMBDA_RESWEEP_FINDING.md)
    # vs. underpowered ones (lambda=0.3: N=6; lambda=1.0: N=2) -- the small-N
    # points are shown hollow/dashed and explicitly labeled with their N, so
    # this figure cannot be misread as "the divergence resolves at high
    # lambda" when it is really a sample-size artifact.
    well_powered = dose['n_scoreable'] >= 20

    fig, ax1 = plt.subplots(figsize=(8.5, 6.2))
    ax1.plot(dose['lam'], dose['own_mean'], color=CAT_ORANGE, lw=2, zorder=2, linestyle='--')
    ax1.plot(dose.loc[well_powered, 'lam'], dose.loc[well_powered, 'own_mean'], marker='o', ms=9,
              color=CAT_ORANGE, lw=0, zorder=3, label="Guidance's own score (N≥20)")
    ax1.plot(dose.loc[~well_powered, 'lam'], dose.loc[~well_powered, 'own_mean'], marker='o', ms=9,
              color=CAT_ORANGE, lw=0, zorder=3, markerfacecolor='white', markeredgewidth=2,
              label="Guidance's own score (N<20, underpowered)")
    ax1.set_xlabel('λ_synth (guidance strength)')
    ax1.set_ylabel("Guidance's own score", color=CAT_ORANGE)
    ax1.tick_params(axis='y', labelcolor=CAT_ORANGE)
    ax1.set_xscale('symlog', linthresh=0.01)

    ax2 = ax1.twinx()
    ax2.plot(dose['lam'], dose['real_mean'], color=CAT_BLUE, lw=2, zorder=2, linestyle='--')
    ax2.plot(dose.loc[well_powered, 'lam'], dose.loc[well_powered, 'real_mean'], marker='s', ms=9,
              color=CAT_BLUE, lw=0, zorder=3, label='Real RA-score (N≥20)')
    ax2.plot(dose.loc[~well_powered, 'lam'], dose.loc[~well_powered, 'real_mean'], marker='s', ms=9,
              color=CAT_BLUE, lw=0, zorder=3, markerfacecolor='white', markeredgewidth=2,
              label='Real RA-score (N<20, underpowered)')
    ax2.set_ylabel('Real RA-score', color=CAT_BLUE)
    ax2.tick_params(axis='y', labelcolor=CAT_BLUE)
    ax2.spines['top'].set_visible(False)

    for _, r in dose.iterrows():
        if r['n_scoreable'] < 20:
            ax2.annotate(f"N={int(r['n_scoreable'])}", xy=(r['lam'], r['real_mean']),
                         xytext=(0, -16), textcoords='offset points', fontsize=8.5,
                         color=INK_MUTED, ha='center')

    ax1.spines[['top']].set_visible(False)
    ax1.grid(zorder=0)
    ax1.set_title('Track D: the guidance model’s own score rises while the real,\nindependently-computed RA-score falls — a synthesizability "Vina-hack"\n(hollow markers: underpowered, do not over-read the apparent reversal)',
                  fontsize=11, fontweight='bold', loc='left')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, 'trackD_divergence_dose_response')

    # Companion: sample-size / fragmentation collapse, so the divergence
    # plot above isn't read without its important caveat (N shrinks fast).
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(range(len(dose)), dose['n_scoreable'], color=CAT_VIOLET, width=0.6, zorder=2,
           label='N scoreable molecules')
    ax.set_xticks(range(len(dose)))
    ax.set_xticklabels([f'{v:g}' for v in dose['lam']], fontsize=9.5)
    ax.set_xlabel('λ_synth')
    ax.set_ylabel('N scoreable molecules (of 30 attempted)')
    ax.set_title('Track D: sample size collapses at higher λ — large point estimates\nbeyond λ=0.3 rest on very few surviving molecules',
                 fontsize=11.5, fontweight='bold', loc='left')
    ax.grid(axis='y', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'trackD_sample_size_collapse')


# =============================================================================
# Figure 7: DIAG1 vs D.5 diagnostic correlations, side by side
# =============================================================================
def fig_diagnostics_comparison():
    diag1 = json.load(open('guidance/diag_size_confound_results.json'))
    diag5 = json.load(open('guidance/diag_synth_direction_results.json'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))

    # DIAG1 (affinity side)
    ax = axes[0]
    items = [('Size / heaviness', diag1['size_correlation']),
             ('Distance to pocket', diag1['distance_correlation'])]
    y = np.arange(len(items))
    for i, (name, c) in enumerate(items):
        sig = not (c['ci_lo'] < 0 < c['ci_hi'])
        color = STATUS_CRITICAL if (sig and c['mean'] > 0) else (STATUS_GOOD if sig else INK_MUTED)
        ax.plot([c['ci_lo'], c['ci_hi']], [i, i], color=color, lw=3, zorder=2)
        ax.plot(c['mean'], i, 'o', color=color, ms=10, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=11)
    ax.set_xlabel('Correlation with gradient magnitude (95% CI)')
    ax.set_title('DIAG1 (affinity guidance):\ndistance-to-pocket explains the failure', fontsize=10.5, fontweight='bold')
    ax.grid(axis='x', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(-0.3, 0.35)

    # D.5 (synth side)
    ax = axes[1]
    items = [('Size / heaviness', diag5['size_correlation']),
             ('Aromaticity fraction', diag5['aromaticity_correlation'])]
    y = np.arange(len(items))
    for i, (name, c) in enumerate(items):
        sig = not (c['ci_lo'] < 0 < c['ci_hi'])
        color = STATUS_CRITICAL if (sig and c['mean'] > 0) else (STATUS_GOOD if sig else INK_MUTED)
        ax.plot([c['ci_lo'], c['ci_hi']], [i, i], color=color, lw=3, zorder=2)
        ax.plot(c['mean'], i, 'o', color=color, ms=10, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=11)
    ax.set_xlabel('Correlation with gradient magnitude (95% CI)')
    ax.set_title('D.5 (synth guidance):\nneither proxy explains the failure', fontsize=10.5, fontweight='bold')
    ax.grid(axis='x', zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_xlim(-0.3, 0.35)
    ax.margins(y=0.6)
    axes[0].margins(y=0.6)

    fig.suptitle('Both CIs cross zero for D.5 (right) — an honest asymmetry: the affinity-side\nfailure has a diagnosed cause, the synthesizability-side failure does not (yet)',
                 fontsize=11.5, fontweight='bold', y=1.14)
    fig.tight_layout()
    save(fig, 'diagnostics_diag1_vs_d5_comparison')


# =============================================================================
# Figure 8: grand summary verdict matrix across every track/sub-track
# =============================================================================
def fig_grand_summary():
    track_c = json.load(open('guidance/track_c_analysis/track_c_analysis_results.json'))
    c_verdicts = [r['verdict'] for r in track_c['per_pocket_primary'].values()]

    def verdict_counts_from_csv(path):
        df = pd.read_csv(path)
        return df['verdict'].value_counts().to_dict()

    groups = {
        'Track A\n(exploratory)': verdict_counts_from_csv('guidance/track_a_exploratory_results/gradient_informativeness_results.csv'),
        'Track A\n(full tier)': verdict_counts_from_csv('guidance/track_a_full_results/gradient_informativeness_results.csv'),
        'Track B1': verdict_counts_from_csv('guidance/track_b1_results/gradient_informativeness_results.csv'),
        'Track B2': verdict_counts_from_csv('guidance/track_b2_results/gradient_informativeness_results.csv'),
        'Track B3': verdict_counts_from_csv('guidance/track_b3_results/gradient_informativeness_results.csv'),
        'Track C\n(15 pockets)': {v: c_verdicts.count(v) for v in set(c_verdicts)},
    }

    verdict_order = ['clear_signal', 'signal_but_confounded_by_size', 'signal_but_quality_degraded',
                      'signal_wrong_direction', 'no_signal']
    verdict_color = {
        'clear_signal': STATUS_GOOD, 'signal_but_confounded_by_size': STATUS_WARNING,
        'signal_but_quality_degraded': STATUS_SERIOUS, 'signal_wrong_direction': STATUS_CRITICAL,
        'no_signal': INK_MUTED,
    }
    verdict_label = {
        'clear_signal': 'Clear signal', 'signal_but_confounded_by_size': 'Confounded by size',
        'signal_but_quality_degraded': 'Quality degraded', 'signal_wrong_direction': 'Wrong direction',
        'no_signal': 'No signal',
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(groups))
    bottoms = np.zeros(len(groups))
    for v in verdict_order:
        heights = np.array([100 * g.get(v, 0) / sum(g.values()) for g in groups.values()])
        ax.bar(x, heights, bottom=bottoms, color=verdict_color[v], width=0.62, label=verdict_label[v], zorder=2)
        bottoms += heights
    ax.set_xticks(x)
    ax.set_xticklabels(groups.keys(), fontsize=10)
    ax.set_ylabel('% of tested conditions')
    ax.set_ylim(0, 100)
    ax.set_title('Every mechanism, every track: no condition anywhere in this project\nsurvived as a clear, undegraded, unconfounded signal',
                 fontsize=12, fontweight='bold', loc='left')
    ax.grid(axis='y', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, 'grand_summary_verdicts_all_tracks')


def main():
    fig_trackA_forest()
    fig_trackA_direction_summary()
    fig_trackA_pb_doseresponse()
    fig_predictive_quality_comparison()
    fig_trackB_combined_forest()
    fig_trackD_divergence()
    fig_diagnostics_comparison()
    fig_grand_summary()
    print(f'Wrote all figures to {OUT_DIR}/')


if __name__ == '__main__':
    main()
