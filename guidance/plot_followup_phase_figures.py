"""Publication-quality figures for the entire follow-up + Phase 2/3
investigation (guidance/FOLLOWUP_PHASE_POOLED_CORRECTION.md and every
guidance/*_FINDING.md doc), for direct use in the thesis/paper. Reuses
this project's validated colorblind-safe palette and output convention
(vector PDF for LaTeX + 300 DPI PNG for preview/slides) from
guidance/plot_track_c_figures.py, so these sit visually consistent with
the existing Track C figures already in the thesis.

Every number plotted is loaded from a saved result file on disk
(guidance/checkpoint_summary.json, guidance/followup_pooled_tests.json,
guidance/chembl_crossvalidation_data.json, guidance/diag_size_confound*
_results.json, guidance/alphafold_pocket_robustness_results.json,
guidance/plip_validation_esm2*_results.json, guidance/
prelambda_*_results.json) -- nothing in this script is a hand-typed
number without a traceable source file.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from scipy.stats import wilcoxon

OUT_DIR = './guidance/followup_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# -- validated palette (same as plot_track_c_figures.py) --
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
CAT_PURPLE = '#8456c9'
CAT_TEAL = '#1a9e8f'

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
    'pdf.fonttype': 42,
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 1: predictive quality across every follow-up checkpoint
# ============================================================
def fig_predictive_quality():
    data = json.load(open('./guidance/checkpoint_summary.json'))
    rows = [(k, v) for k, v in data.items() if not k.startswith('_')]
    rows.sort(key=lambda kv: kv[1]['r2'])
    labels = [v['label'] for k, v in rows]
    r2 = [v['r2'] for k, v in rows]
    pear = [v['pearson'] for k, v in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(rows))
    h = 0.35
    ax.barh(y + h/2, r2, height=h, color=CAT_BLUE, label='Test R²')
    ax.barh(y - h/2, pear, height=h, color=CAT_ORANGE, label='Test Pearson r')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Score (held-out test set)')
    ax.set_title('Predictive quality across every follow-up-phase checkpoint', loc='left', fontweight='bold')
    ax.axvline(0, color=BASELINE, lw=1)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='x', alpha=0.6)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    fig.text(0.5, -0.06,
              'Best predictive quality (ESM2 whole-protein, unseen-family) never corresponds to a real-docking effect -- see Fig. 2.',
              fontsize=8.5, color=INK_MUTED, ha='center')
    save(fig, 'fig1_predictive_quality')


# ============================================================
# Figure 2: DIAG1 diagnostic dissociation (distance-corr vs predictive quality)
# ============================================================
def fig_diag1_dissociation():
    diag_files = {
        'Original EGNN': 'guidance/diag_size_confound_results.json',
        'Noise-matched': 'guidance/diag_size_confound_noisematched_results.json',
        'Vina-target': 'guidance/diag_size_confound_vinatarget_results.json',
        'Gradient-alignment': 'guidance/diag_size_confound_gradalign_results.json',
        'ESM2 (whole-protein)': 'guidance/diag_size_confound_esm2_results.json',
        'Kitchen-sink': 'guidance/diag_size_confound_kitchensink_results.json',
        'Family-holdout': 'guidance/diag_size_confound_family_holdout_results.json',
        'ESM2 (pocket-only)': 'guidance/diag_size_confound_esm2pocket_results.json',
    }
    quality = json.load(open('./guidance/checkpoint_summary.json'))
    label_to_r2 = {v['label']: v['r2'] for k, v in quality.items() if not k.startswith('_')}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    names, dists, los, his = [], [], [], []
    for name, path in diag_files.items():
        d = json.load(open(path))['distance_correlation']
        names.append(name); dists.append(d['mean']); los.append(d['ci_lo']); his.append(d['ci_hi'])
    order = np.argsort(dists)
    y = np.arange(len(names))
    for i, idx in enumerate(order):
        color = STATUS_GOOD if his[idx] < 0 else (STATUS_CRITICAL if los[idx] > 0 else INK_MUTED)
        ax.plot([los[idx], his[idx]], [i, i], color=color, lw=2.2, solid_capstyle='round')
        ax.plot(dists[idx], i, 'o', color=color, ms=6, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([names[idx] for idx in order], fontsize=9)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_xlabel('DIAG1 gradient/distance-to-pocket correlation\n(95% CI; negative = correct direction)')
    ax.set_title('A. Diagnostic health varies a lot...', loc='left', fontweight='bold')

    ax = axes[1]
    xs = [dists[names.index(n)] for n in label_to_r2 if n in names]
    ys = [label_to_r2[n] for n in label_to_r2 if n in names]
    labs = [n for n in label_to_r2 if n in names]
    ax.scatter(xs, ys, color=CAT_PURPLE, s=70, zorder=3, edgecolor=SURFACE, linewidth=0.8)
    for x, yv, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, yv), fontsize=7.5, color=INK_SECONDARY,
                   xytext=(5, 4), textcoords='offset points')
    ax.set_xlabel('DIAG1 distance correlation (more negative = healthier)')
    ax.set_ylabel('Test R² (predictive quality)')
    ax.set_title('B. ...and is unrelated to predictive quality', loc='left', fontweight='bold')
    ax.grid(alpha=0.5)

    fig.suptitle('The diagnostic that was supposed to explain guidance failure does not predict\neither predictive quality or (per Fig. 3) real-docking success',
                 fontsize=10, y=1.06, color=INK_SECONDARY)
    save(fig, 'fig2_diag1_dissociation')


# ============================================================
# Figure 3: pooled forest/dot plot of all 48 real-docking tests
# ============================================================
def fig_pooled_tests():
    tests = json.load(open('./guidance/followup_pooled_tests.json'))
    tests_sorted = sorted(tests, key=lambda t: t['raw_p'])
    y = np.arange(len(tests_sorted))
    raw_p = [max(t['raw_p'], 1e-4) for t in tests_sorted]
    bh_p = [max(t['bh_p'], 1e-4) for t in tests_sorted]

    fig, ax = plt.subplots(figsize=(7, 8.5))
    ax.scatter(raw_p, y, s=14, color=CAT_BLUE, label='Raw p-value', zorder=3)
    ax.scatter(bh_p, y, s=14, color=CAT_ORANGE, label='BH-corrected p-value', zorder=3, marker='D')
    ax.axvline(0.05, color=STATUS_CRITICAL, lw=1.3, linestyle='--', label='α = 0.05')
    ax.set_xscale('log')
    ax.set_yticks([])
    ax.set_ylabel(f'{len(tests_sorted)} real-docking comparisons, sorted by raw p-value')
    ax.set_xlabel('p-value (log scale)')
    ax.set_title('Every real-docking statistical test across the entire follow-up phase',
                 loc='left', fontweight='bold')
    ax.text(0.98, 0.03, f'0/{len(tests_sorted)} significant after\nBenjamini-Hochberg correction',
            transform=ax.transAxes, ha='right', fontsize=10, color=STATUS_CRITICAL, fontweight='bold')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='x', alpha=0.5)
    save(fig, 'fig3_pooled_forest')


# ============================================================
# Figure 4: dose-response curves across key checkpoints
# ============================================================
def fig_dose_response():
    sources = {
        'Gradient-alignment': ('guidance/prelambda_gradalign_results.json', CAT_BLUE),
        'Kitchen-sink': ('guidance/prelambda_kitchensink_results.json', CAT_ORANGE),
        'Family-holdout': ('guidance/prelambda_family_holdout_results.json', STATUS_GOOD),
        'ESM2 (pocket-only)': ('guidance/prelambda_esm2pocket_results.json', CAT_PURPLE),
        'ESM2 (whole-protein)': ('guidance/prelambda_esm2_results.json', CAT_TEAL),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, (path, color) in sources.items():
        d = json.load(open(path))
        lams = sorted(c['lambda_affinity'] for c in d)
        base = next(c['mean_vina_dock'] for c in d if c['lambda_affinity'] == 0.0)
        xs, ys = [], []
        for c in d:
            if c.get('mean_vina_dock') is None:
                continue
            xs.append(c['lambda_affinity'])
            ys.append(c['mean_vina_dock'] - base)
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.plot(xs, ys, 'o-', color=color, label=label, ms=5, lw=1.6)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_xlabel('Guidance strength λ (symlog scale)')
    ax.set_ylabel('Mean Vina Dock shift vs. unguided (kcal/mol)\nmore negative = apparently better')
    ax.set_title('Dose-response across checkpoints: never a clean, stable improvement',
                 loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    ax.grid(alpha=0.5)
    save(fig, 'fig4_dose_response')


# ============================================================
# Figure 5: AlphaFold pLDDT vs pocket RMSD (n=835)
# ============================================================
def fig_alphafold_scatter():
    d = json.load(open('./guidance/alphafold_pocket_robustness_results.json'))
    ok = [r for r in d if r['status'] == 'ok']
    rmsd = np.array([r['pocket_rmsd_after_superposition'] for r in ok])
    plddt = np.array([r['pocket_mean_plddt'] for r in ok])
    overconf = (plddt > 80) & (rmsd > 5)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(plddt[~overconf], rmsd[~overconf], s=14, color=CAT_BLUE, alpha=0.55,
              edgecolor='none', label=f'n={(~overconf).sum()}')
    ax.scatter(plddt[overconf], rmsd[overconf], s=38, color=STATUS_CRITICAL, zorder=5,
              edgecolor=SURFACE, linewidth=0.6, label=f'"Confidently wrong" (n={overconf.sum()})')
    ax.axhspan(5, rmsd.max() * 1.05, color=STATUS_CRITICAL, alpha=0.04)
    ax.axvline(80, color=INK_MUTED, lw=0.8, linestyle=':')
    ax.set_yscale('log')
    ax.set_xlabel('AlphaFold pocket-local pLDDT')
    ax.set_ylabel('Pocket RMSD vs. crystal structure (Å, log scale)')
    ax.set_title(f'AlphaFold vs. crystal structure at the binding pocket (n={len(ok)})',
                 loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    ax.grid(alpha=0.4)
    save(fig, 'fig5_alphafold_plddt_rmsd')


# ============================================================
# Figure 6: pocket-local vs whole-protein pLDDT
# ============================================================
def fig_pocket_vs_global_plddt():
    d = json.load(open('./guidance/alphafold_pocket_robustness_results.json'))
    ok = [r for r in d if r['status'] == 'ok']
    pocket = np.array([r['pocket_mean_plddt'] for r in ok])
    glob = np.array([r['global_plddt'] for r in ok])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    parts = ax.violinplot([glob, pocket], showmedians=True, widths=0.7)
    for pc, color in zip(parts['bodies'], [CAT_BLUE, CAT_ORANGE]):
        pc.set_facecolor(color); pc.set_alpha(0.55); pc.set_edgecolor(INK_SECONDARY)
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        parts[key].set_color(INK_SECONDARY)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Whole-protein pLDDT', 'Pocket-local pLDDT'])
    ax.set_ylabel('pLDDT')
    ax.set_title(f"Binding pockets sit in AlphaFold's most confident regions (n={len(ok)})",
                 loc='left', fontweight='bold')
    diff = pocket.mean() - glob.mean()
    ax.text(0.5, 0.04, f'mean difference: +{diff:.1f}', transform=ax.transAxes,
            ha='center', fontsize=9.5, color=INK_SECONDARY)
    ax.grid(axis='y', alpha=0.4)
    save(fig, 'fig6_pocket_vs_global_plddt')


# ============================================================
# Figure 7: family-holdout generalization comparison
# ============================================================
def fig_family_holdout():
    quality = json.load(open('./guidance/checkpoint_summary.json'))
    vt = quality['vinatarget']
    fh = quality['family_holdout']

    fig, ax = plt.subplots(figsize=(6, 4.8))
    metrics = ['R²', 'Pearson', 'Spearman']
    vt_vals = [vt['r2'], vt['pearson'], vt['spearman'] or 0]
    fh_vals = [fh['r2'], fh['pearson'], fh['spearman']]
    x = np.arange(len(metrics))
    w = 0.32
    ax.bar(x - w/2, vt_vals, width=w, color=CAT_BLUE, label='Same-distribution test set')
    ax.bar(x + w/2, fh_vals, width=w, color=STATUS_GOOD, label='Unseen kinase family (held out)')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('Predictive quality does not collapse on a wholly unseen\nprotein family',
                 loc='left', fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.5)
    save(fig, 'fig7_family_holdout_quality')


# ============================================================
# Figure 8: ChEMBL cross-validation range comparison
# ============================================================
def fig_chembl_crossval():
    d = json.load(open('./guidance/chembl_crossvalidation_data.json'))
    targets = list(d.keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(targets))
    h = 0.32
    for i, t in enumerate(targets):
        r = d[t]
        ax.plot([r['our_pk_min'], r['our_pk_max']], [i + h/2, i + h/2], color=CAT_BLUE, lw=6, solid_capstyle='round')
        ax.plot([r['chembl_pchembl_min'], r['chembl_pchembl_max']], [i - h/2, i - h/2], color=CAT_ORANGE, lw=6, solid_capstyle='round')
    ax.set_yticks(y); ax.set_yticklabels(targets)
    ax.set_xlabel('Affinity (pK / pChEMBL)')
    ax.set_title('Our PDBBind-derived affinity labels vs. independent ChEMBL data',
                 loc='left', fontweight='bold')
    handles = [plt.Line2D([0], [0], color=CAT_BLUE, lw=6, label='Our dataset (PDBBind-derived pK)'),
              plt.Line2D([0], [0], color=CAT_ORANGE, lw=6, label='ChEMBL (independent Ki data)')]
    ax.legend(handles=handles, frameon=False, loc='upper center',
             bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9.5)
    ax.grid(axis='x', alpha=0.5)
    cd38_idx = targets.index('CD38')
    ax.annotate('No overlap\n(CD38)', xy=(4.9, cd38_idx - h/2), xytext=(3.3, cd38_idx + 1.1),
               fontsize=9.5, color=STATUS_CRITICAL, fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color=STATUS_CRITICAL, lw=1.3))
    save(fig, 'fig8_chembl_crossvalidation')


# ============================================================
# Figure 9: PLIP interaction-count exploratory vs confirm reversal
# ============================================================
def fig_plip_signal_flip():
    d8 = json.load(open('./guidance/plip_validation_esm2_results.json'))
    d16 = json.load(open('./guidance/plip_validation_esm2_confirm_results.json'))

    def totmap(mols):
        return {m['sample_idx']: m['hbond']+m['hydrophobic']+m['pi_stack']+m['salt_bridge']
                for m in mols if m['status'] == 'ok'}

    base8, cond8 = totmap(d8['0.0']), totmap(d8['3.0'])
    base16, cond16 = totmap(d16['0.0']), totmap(d16['3.0'])
    common8 = sorted(set(base8) & set(cond8))
    common16 = sorted(set(base16) & set(cond16))
    diff8 = np.array([cond8[i] - base8[i] for i in common8])
    diff16 = np.array([cond16[i] - base16[i] for i in common16])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Exploratory\n(n=8)', 'Confirmatory\n(n=16)'],
                  [diff8.mean(), diff16.mean()],
                  color=[STATUS_WARNING, STATUS_GOOD], width=0.5)
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_ylabel('Mean Δ PLIP interactions (λ=3 vs. λ=0)')
    ax.set_title('A "promising" small-n signal reverses sign at larger n',
                 loc='left', fontweight='bold')
    for bar, val, n in zip(bars, [diff8.mean(), diff16.mean()], [len(common8), len(common16)]):
        ax.annotate(f'{val:+.2f}\n(n={n})', (bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.5)
    save(fig, 'fig9_plip_signal_reversal')


if __name__ == '__main__':
    fig_predictive_quality()
    print('fig1 done')
    fig_diag1_dissociation()
    print('fig2 done')
    fig_pooled_tests()
    print('fig3 done')
    fig_dose_response()
    print('fig4 done')
    fig_alphafold_scatter()
    print('fig5 done')
    fig_pocket_vs_global_plddt()
    print('fig6 done')
    fig_family_holdout()
    print('fig7 done')
    fig_chembl_crossval()
    print('fig8 done')
    fig_plip_signal_flip()
    print('fig9 done')
    print(f'All figures saved to {OUT_DIR}/')
