"""Thesis figures that are not already in guidance/thesis_figures/: schematics (Chapter III, VIII) and the Track E
result figures (Chapter VIII). Reads only committed JSON results; writes PNGs to guidance/thesis/figures/.
Run: PYTHONPATH=. python guidance/thesis/make_figures.py"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = 'guidance/thesis/figures'
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GRAY, DARK, GREEN, RED = '#2a78d6', '#eb6834', '#8a8f98', '#222222', '#2e8b57', '#c0392b'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
                     'savefig.dpi': 300})


def box(ax, x, y, w, h, text, fc='#f2f4f7', ec=DARK, fs=9, bold=False, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=DARK)


def arrow(ax, x1, y1, x2, y2, text=None, style='-|>', color=DARK, ls='-', fs=8, tx=0, ty=0.06):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, lw=1.1, color=color, linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha='center', va='bottom', fontsize=fs, color=color)


# ------------------------------------------------------------------------------------------ Fig 3.1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.6); ax.axis('off')
    box(ax, 0.2, 5.9, 2.5, 1.3, 'Pocket + current\nstate $x_t$', fs=8.5)
    box(ax, 3.5, 5.9, 3.0, 1.3, 'Frozen TargetDiff\ndenoiser (unmodified)', fc='#dbe8fa', fs=8.5, bold=True)
    box(ax, 7.6, 5.9, 3.0, 1.3, 'Clean-data estimate $\\hat{x}_0$\n(positions, atom types)', fs=8.5)
    box(ax, 7.6, 3.6, 3.0, 1.4, 'Guidance predictor $f$\n(affinity or\nsynthesizability)', fc='#fde3d9', fs=8.5, bold=True)
    box(ax, 3.5, 3.6, 3.0, 1.4, 'Guided estimate\n$\\hat{x}_0+\\lambda\\,\\nabla_{pos}f(\\hat{x}_0)$', fs=8.5)
    box(ax, 0.2, 3.6, 2.5, 1.4, 'Posterior step\n$x_t\\rightarrow x_{t-1}$', fs=8.5)
    arrow(ax, 2.7, 6.55, 3.5, 6.55)
    arrow(ax, 6.5, 6.55, 7.6, 6.55)
    arrow(ax, 9.1, 5.9, 9.1, 5.0)
    ax.text(9.25, 5.35, '$\\hat{x}_0$', fontsize=8.5)
    arrow(ax, 7.6, 4.3, 6.5, 4.3)
    ax.text(7.05, 4.42, '$\\nabla f$', fontsize=8.5, ha='center')
    arrow(ax, 3.5, 4.3, 2.7, 4.3)
    arrow(ax, 1.45, 5.0, 1.45, 5.9)
    ax.text(1.6, 5.35, 'repeat, $t=T\\ldots1$', fontsize=7.5, color=GRAY)
    box(ax, 0.2, 0.5, 2.6, 1.3, 'Generated ligands\n($t=0$, reconstructed)', fs=8.2)
    box(ax, 3.6, 0.5, 4.0, 1.3, 'Independent evaluation:\nVina Dock, PoseBusters,\nligand efficiency, RA-score', fc='#e3f3e8', fs=8.2, bold=True)
    box(ax, 8.4, 0.5, 3.4, 1.3, 'Statistics: KS + BH,\nbootstrap CI, negative\ncontrols', fc='#e3f3e8', fs=8.2, bold=True)
    arrow(ax, 1.45, 3.6, 1.45, 1.8)
    ax.text(1.6, 2.5, 'final state', fontsize=7.5, color=GRAY)
    arrow(ax, 2.8, 1.15, 3.6, 1.15); arrow(ax, 7.6, 1.15, 8.4, 1.15)
    fig.savefig(f'{OUT}/fig_pipeline.png', bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------------------------------ Fig 3.2
def fig_verdict_flow():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis('off')
    box(ax, 0.2, 5.9, 3.0, 1.1, 'Null-consistency check\n(scrambled vs random)', fs=8.5)
    box(ax, 4.2, 5.9, 3.4, 1.1, 'KS test of guided vs.\nunguided, BH-corrected', fs=8.5)
    box(ax, 8.6, 5.9, 3.2, 1.1, 'Bootstrap 95% CI on\nthe effect size', fs=8.5)
    arrow(ax, 3.2, 6.45, 4.2, 6.45); arrow(ax, 7.6, 6.45, 8.6, 6.45)
    box(ax, 0.2, 3.5, 2.6, 1.3, '$p_{null}<0.01$:\nsuspect\nimplementation bug', fc='#f9dcd9', fs=8.5)
    box(ax, 3.3, 3.5, 2.6, 1.3, 'Fails BH:\nno signal', fc='#eceff3', fs=8.5)
    box(ax, 6.4, 3.5, 2.6, 1.3, 'Passes BH,\nwrong direction:\nwrong-direction signal', fc='#fde3d9', fs=8.5)
    box(ax, 9.5, 3.5, 2.4, 1.3, 'Passes BH,\ncorrect direction', fs=8.5)
    arrow(ax, 1.5, 5.9, 1.5, 4.8); arrow(ax, 5.9, 5.9, 4.6, 4.8); arrow(ax, 6.2, 5.9, 7.7, 4.8); arrow(ax, 7.0, 5.9, 10.6, 4.8)
    box(ax, 6.6, 1.3, 2.4, 1.4, 'LE worse or\nheavy atoms up:\nconfounded by size', fc='#fde3d9', fs=8.5)
    box(ax, 9.4, 1.3, 2.5, 1.4, 'PoseBusters drop\n> 10 points: quality\ndegraded', fc='#fde3d9', fs=8.5)
    box(ax, 8.6, 0.05, 3.3, 0.9, 'Otherwise: clear signal', fc='#e3f3e8', fs=8.5, bold=True)
    arrow(ax, 10.7, 3.5, 7.8, 2.7); arrow(ax, 10.7, 3.5, 10.7, 2.7)
    arrow(ax, 10.7, 1.3, 10.7, 0.95)
    ax.text(0.3, 2.2, 'Quality checks (ligand efficiency, heavy atoms,\nPoseBusters) are applied to every apparent positive.', fontsize=8, color=GRAY, ha='left', style='italic')
    fig.savefig(f'{OUT}/fig_verdict_flow.png', bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------------------------------ Fig 8.1
def fig_track_e_design():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
    box(ax, 0.2, 2.9, 2.4, 1.6, 'Pocket + ligand\ncomplex (docked\npose)', fs=8.5)
    box(ax, 3.4, 2.7, 2.6, 2.0, 'Shared encoder\n(GIGN heterogeneous\ninteraction layers ×3)', fc='#dbe8fa', fs=8.5, bold=True)
    arrow(ax, 2.6, 3.7, 3.4, 3.7)
    box(ax, 7.2, 5.1, 2.9, 1.2, 'E1 head (physics sum):\n$\\hat{\\Delta}=b+s\\cdot(-\\Sigma E_{phys})$', fc='#fde3d9', fs=8.2)
    box(ax, 7.2, 3.3, 2.9, 1.2, 'E1 readout ablation:\nscalar MLP on pooled\nembeddings', fc='#fde3d9', fs=8.2)
    box(ax, 7.2, 1.2, 2.9, 1.4, 'E2 head (not run):\nper-ligand-atom PLIP\nlabels (H-bond, hydrophobic,\nsalt bridge)', fc='#eceff3', fs=8.0)
    arrow(ax, 6.0, 4.3, 7.2, 5.6); arrow(ax, 6.0, 3.7, 7.2, 3.9); arrow(ax, 6.0, 3.1, 7.2, 2.1, ls='--', color=GRAY)
    box(ax, 10.6, 5.1, 1.35, 1.2, '$\\hat{y}=pK_{Vina}$\n$+\\hat{\\Delta}$', fc='#e3f3e8', fs=8.2, bold=True)
    arrow(ax, 10.1, 5.7, 10.6, 5.7)
    ax.text(6.0, 0.35, 'Training target: $\\Delta = pK_{exp} - \\max(-E_{Vina},0)/1.364$ (standardised); the absolute-target control uses $pK_{exp}$.',
            ha='center', fontsize=8, color=GRAY, style='italic')
    fig.savefig(f'{OUT}/fig_trackE_design.png', bbox_inches='tight'); plt.close(fig)


# ------------------------------------------------------------------------------------------ Fig 8.2 / 8.3
def fig_track_e_r2():
    d = json.load(open('guidance/track_e/e1_gate_results.json'))
    R, A = d['references'], d['arms']
    rows = [('E1 (Δ target, physics head)', A['E1']['r2'], A['E1']['r2_ci95'], A['E1']['r2_per_seed'], ORANGE),
            ('E1-MLP (Δ target, MLP head)', A['E1_MLP']['r2'], A['E1_MLP']['r2_ci95'], A['E1_MLP']['r2_per_seed'], ORANGE),
            ("A0′ (absolute target)", A['A0prime']['r2'], A['A0prime']['r2_ci95'], A['A0prime']['r2_per_seed'], BLUE),
            ('EGNN, 3 seeds', R['egnn_3seed_mean']['r2'], R['egnn_3seed_mean']['r2_ci95'], R['egnn_3seed_mean']['r2_per_seed'], GRAY),
            ('EGNN Stage 0 (reference)', R['stage0_egnn']['r2'], R['stage0_egnn']['r2_ci95'], None, GRAY),
            ('Linear: Vina + heavy atoms', R['vina_plus_ha']['r2'], R['vina_plus_ha']['r2_ci95'], None, GRAY),
            ('Linear: heavy atoms only', R['heavy_atoms']['r2'], R['heavy_atoms']['r2_ci95'], None, GRAY),
            ('Vina only (calibrated)', R['vina_calibrated']['r2'], R['vina_calibrated']['r2_ci95'], None, GRAY),
            ('GIGN+PIGNet2 (Track A, no bias)', R['trackA_gign_pignet']['r2'], R['trackA_gign_pignet']['r2_ci95'], None, GRAY)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, (name, m, ci, seeds, c) in enumerate(rows[::-1]):
        ax.plot(ci, [i, i], color=c, lw=2.2, solid_capstyle='butt')
        ax.plot([m], [i], marker='o' if c != GRAY else 's', color=c, ms=7, mec='white', mew=0.8, zorder=3)
        if seeds:
            ax.scatter(seeds, [i] * len(seeds), marker='|', s=90, color=DARK, zorder=4, lw=1.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.5)
    ax.axvline(R['stage0_egnn']['r2'], color=GRAY, ls=':', lw=1); ax.axvline(R['vina_plus_ha']['r2'], color=GRAY, ls='--', lw=1)
    ax.axvline(0, color=DARK, lw=0.6)
    ax.set_xlabel('Test R² on 127 held-out targets (bar = target-clustered 95% CI; tick marks = individual seeds)')
    ax.set_xlim(-0.9, 0.6)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackE_r2.png', bbox_inches='tight'); plt.close(fig)


def fig_track_e_learnability():
    import glob
    from guidance.track_e.core import build_anchor_table, split_arrays, delta_target
    t = build_anchor_table()
    tr, va = split_arrays(t, 'train'), split_arrays(t, 'val')
    mtr, mva = tr['vina'] <= 0, va['vina'] <= 0
    out = {}
    for tgt in ('abs', 'delta'):
        y_tr = tr['pk'][mtr] if tgt == 'abs' else delta_target(tr['pk'], tr['vina'])[mtr]
        y_va = va['pk'][mva] if tgt == 'abs' else delta_target(va['pk'], va['vina'])[mva]
        out[tgt] = float(((y_va - y_tr.mean()) ** 2).mean() / y_tr.var())
    vals = {}
    for arm in ('abs_phys', 'e1', 'e1mlp'):
        vals[arm] = [json.load(open(f'runs_track_e/{arm}_s{s}/train_complete.json'))['best_val'] for s in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = {'abs_phys': 0, 'e1': 1, 'e1mlp': 2}
    labels = ['A0′\n(absolute target)', 'E1\n(Δ target,\nphysics head)', 'E1-MLP\n(Δ target,\nMLP head)']
    cols = [BLUE, ORANGE, ORANGE]
    for (arm, v), c in zip(vals.items(), cols):
        ax.scatter([xs[arm]] * 3 + np.linspace(-0.12, 0.12, 3), v, color=c, s=45, zorder=3, edgecolor='white')
    ax.hlines(out['abs'], -0.35, 0.35, color=BLUE, ls='--', lw=1.3)
    ax.hlines(out['delta'], 0.65, 2.35, color=ORANGE, ls='--', lw=1.3)
    ax.text(0.37, out['abs'] + 0.015, 'constant\n(mean) predictor', fontsize=7.5, color=BLUE, va='bottom')
    ax.text(2.37, out['delta'] + 0.015, 'constant (mean Δ)\npredictor', fontsize=7.5, color=ORANGE, va='bottom', ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Best validation MSE\n(standardised units; 1.0 ≈ variance)')
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.5, 1.15)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackE_learnability.png', bbox_inches='tight'); plt.close(fig)
    json.dump({'constant_predictor_val_mse_z': out, 'best_val_by_arm': vals}, open('guidance/thesis/figures/learnability_numbers.json', 'w'), indent=1)


def fig_track_e_within_target():
    d = json.load(open('guidance/track_e/e1_gate_results.json'))
    R, A = d['references'], d['arms']
    rows = [('E1', A['E1']['within_target_spearman'], A['E1']['sp_ci95'], ORANGE),
            ('E1-MLP', A['E1_MLP']['within_target_spearman'], A['E1_MLP']['sp_ci95'], ORANGE),
            ("A0′", A['A0prime']['within_target_spearman'], A['A0prime']['sp_ci95'], BLUE),
            ('EGNN Stage 0', R['stage0_egnn']['sp'], R['stage0_egnn']['sp_ci95'], GRAY),
            ('Linear: Vina + HA', R['vina_plus_ha']['sp'], R['vina_plus_ha']['sp_ci95'], GRAY),
            ('Linear: HA only', R['heavy_atoms']['sp'], R['heavy_atoms']['sp_ci95'], GRAY),
            ('Vina only', R['vina_raw']['sp'], R['vina_raw']['sp_ci95'], GRAY)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (n, m, ci, c) in enumerate(rows[::-1]):
        ax.plot(ci, [i, i], color=c, lw=2.2); ax.plot([m], [i], 'o' if c != GRAY else 's', color=c, ms=7, mec='white', zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8.5)
    ax.axvline(0, color=DARK, lw=0.6)
    ax.set_xlabel('Mean within-target Spearman ρ (73 targets with ≥ 10 complexes; 95% bootstrap CI)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackE_within_target.png', bbox_inches='tight'); plt.close(fig)


def fig_plip_jitter():
    d = json.load(open('guidance/track_e/plip_scale_check_results.json'))['summary']['jitter']
    cls = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']
    names = ['H-bond', 'Hydrophobic', 'π-stacking\n(+ π-cation)', 'Salt bridge']
    j = [d['atom_level_jaccard_mean_over_complexes_with_label'][c] for c in cls]
    p = [d['complex_level_presence_agreement'][c] for c in cls]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, v, ttl in zip(axs, (j, p), ('(A) Atom-level Jaccard', '(B) Complex-level presence agreement')):
        bars = ax.bar(range(4), v, color=[BLUE, BLUE, RED, BLUE], edgecolor='white')
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.02, f'{x:.2f}', ha='center', fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=7.5); ax.set_title(ttl, fontsize=9); ax.set_ylim(0, 1.1)
    axs[0].set_ylabel('Agreement with unperturbed\nlabels (0.2 Å jitter, n=100)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_plip_jitter.png', bbox_inches='tight'); plt.close(fig)


def fig_vina_semantics():
    d = json.load(open('guidance/track_e/d6_vina_semantics_results.json'))
    st = np.array([r['stored'] for r in d['rows']]); so = np.array([r['score_only'] for r in d['rows']]); mn = np.array([r['minimize'] for r in d['rows']])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for ax, x, ttl, c in zip(axs, (so, mn), ('(A) Recomputed score-only', '(B) Recomputed minimised'), (BLUE, ORANGE)):
        ax.scatter(x, st, s=14, color=c, alpha=0.8, edgecolor='none')
        lim = [min(st.min(), x.min()) - 0.5, max(st.max(), x.max()) + 0.5]
        ax.plot(lim, lim, color=DARK, lw=0.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel('Recomputed Vina 1.2.6 (kcal/mol)')
        ax.text(0.04, 0.93, f'MAE {np.abs(st - x).mean():.2f}\nmean(stored − recomp.) {np.mean(st - x):+.2f}', transform=ax.transAxes, fontsize=7.5, va='top')
    axs[0].set_ylabel('Stored `vina` value (kcal/mol)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_vina_semantics.png', bbox_inches='tight'); plt.close(fig)


def fig_split_artifact():
    labels = ['Old split\nEGNN\n(deployed)', 'Old split\nEGNN\n(lr 3e-5)', 'Old split\nligand-only RF',
              'LP split\nEGNN', 'LP split\nligand-only RF']
    vals = [-0.537, 0.046, 0.047, 0.342, 0.195]
    lo = [-1.769, -0.601, None, 0.322, None]
    hi = [0.183, 0.445, None, 0.361, None]
    cols = [ORANGE, ORANGE, '#c9ced6', BLUE, '#c9ced6']
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.62, edgecolor=DARK, lw=0.6)
    for i, v in enumerate(vals):
        if lo[i] is not None:
            ax.errorbar(i, v, yerr=[[v - lo[i]], [hi[i] - v]], color=DARK, capsize=3, lw=1)
        ax.text(i + 0.36, v, f'{v:.3f}', fontsize=8, va='center', ha='left')
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Test R$^2$'); ax.set_xlim(-0.6, len(vals) - 0.1)
    ax.text(0.99, 0.03, 'Old split: n = 27 pockets; LP split: n = 11,855 entries.\nError bars: bootstrap 95% CI (not available for RF).',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_split_artifact.png', bbox_inches='tight'); plt.close(fig)


def fig_predictive_quality():
    labels = ['Stage 0 EGNN\n(affinity)', 'GIGN+PIGNet2\npartial (affinity)', 'GIGN+PIGNet2\nfull (affinity)',
              'SynthPredNet\noriginal split', 'SynthPredNet\nleakage-safe']
    vals = [0.609, 0.583, 0.581, 0.878, 0.647]
    cols = [BLUE, BLUE, BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cols, width=0.6, edgecolor=DARK, lw=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Test Pearson correlation'); ax.set_ylim(0, 1.0)
    ax.text(3, 0.5, 'target leakage\n(superseded)', ha='center', fontsize=7.5, color='white')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_predictive_quality.png', bbox_inches='tight'); plt.close(fig)


def _read_csv(path):
    import csv
    return list(csv.DictReader(open(path)))


def fig_track_a_full():
    rows = _read_csv('guidance/track_a_full_results/gradient_informativeness_results.csv')
    pockets = []
    for r in rows:
        p = r['label'].split(':')[0].replace('_HUMAN', '').split('_')[0] if False else r['label'].split(':')[0]
        if p not in pockets:
            pockets.append(p)
    short = {p: p.split('_1_')[0].split('_2_')[0].split('_44_')[0].split('_291_')[0] for p in pockets}
    lams = ['0.1', '0.3', '1.0']
    cols = {'0.1': '#7fb3e6', '0.3': BLUE, '1.0': '#0b3d80'}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axs[0]
    for i, p in enumerate(pockets):
        for j, l in enumerate(lams):
            r = [q for q in rows if q['label'] == f'{p}:lambda_{l}'][0]
            ax.scatter(float(r['mean_shift']), i + (j - 1) * 0.2, s=22, color=cols[l], zorder=3,
                       label=f'$\\lambda$ = {l}' if i == 0 else None)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_yticks(range(len(pockets))); ax.set_yticklabels([short[p] for p in pockets], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('Mean Vina Dock shift, guided − unguided (kcal/mol);\nnegative = better', fontsize=8)
    ax.set_title('(A) Effect on docking score', fontsize=9); ax.legend(fontsize=7, loc='upper right', frameon=False)
    ax = axs[1]
    xs = [0, 1, 2, 3]
    allv = []
    for p in pockets:
        base = float([q for q in rows if q['label'].startswith(p)][0]['pb_valid_rate_unguided'])
        v = [base] + [float([q for q in rows if q['label'] == f'{p}:lambda_{l}'][0]['pb_valid_rate_guided']) for l in lams]
        allv.append(v); ax.plot(xs, v, color='#b9c0cc', lw=0.9, zorder=1)
    m = np.mean(allv, axis=0)
    ax.plot(xs, m, color=RED, lw=2, marker='o', zorder=3)
    for x, y in zip(xs, m):
        ax.text(x, y + 0.04, f'{y:.2f}', ha='center', fontsize=7.5, color=RED)
    ax.set_xticks(xs); ax.set_xticklabels(['0\n(unguided)', '0.1', '0.3', '1.0'], fontsize=8)
    ax.set_xlabel('Guidance strength $\\lambda$', fontsize=8); ax.set_ylabel('PoseBusters pass rate', fontsize=8)
    ax.set_ylim(0, 1); ax.set_title('(B) Physical validity', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackA_full.png', bbox_inches='tight'); plt.close(fig)


def fig_track_b():
    groups = [('B1 gradient normalisation', 'b1', BLUE), ('B2 classifier head', 'b2', ORANGE), ('B3 timestep window', 'b3', GREEN)]
    items = []
    for name, t, c in groups:
        for r in _read_csv(f'guidance/track_{t}_results/gradient_informativeness_results.csv'):
            pk, cmp_ = r['label'].split(':')
            items.append((name, c, pk.split('_')[0] + ': ' + cmp_.replace('_', ' '), float(r['median_shift']), r['pb_degraded'] == 'True'))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for i, (name, c, lab, v, pb) in enumerate(items):
        ax.scatter(v, i, s=28, facecolor=('white' if pb else c), edgecolor=c, lw=1.4, zorder=3)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[2] for it in items], fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color=DARK, lw=0.8)
    edges = [0]
    for k in range(1, len(items)):
        if items[k][0] != items[k - 1][0]:
            ax.axhline(k - 0.5, color='#c9ced6', lw=0.8)
    for name, _, c in groups:
        idx = [i for i, it in enumerate(items) if it[0] == name]
        ax.text(1.01, np.mean(idx), name, transform=ax.get_yaxis_transform(), fontsize=8, va='center', color=c, rotation=270)
    ax.set_xlabel('Median Vina Dock shift (kcal/mol); positive = worse', fontsize=8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='w', mec=DARK, label='PoseBusters degraded'),
                       Line2D([], [], marker='o', ls='', color=DARK, label='PoseBusters not degraded')],
              fontsize=7, loc='lower right', frameon=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackB.png', bbox_inches='tight'); plt.close(fig)


def fig_diag1():
    labs = ['Gradient magnitude vs.\nheaviness of atom type', 'Gradient magnitude vs.\ndistance to pocket']
    m = [-0.127, 0.240]; lo = [-0.178, 0.194]; hi = [-0.075, 0.285]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    for i in range(2):
        ax.errorbar(m[i], i, xerr=[[m[i] - lo[i]], [hi[i] - m[i]]], fmt='o', color=BLUE if i else GRAY, capsize=4, lw=1.4)
        ax.text(hi[i] + 0.012, i, f'{m[i]:+.3f}', va='center', fontsize=8)
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labs, fontsize=8); ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-0.3, 0.4); ax.set_xlabel('Mean per-sample Pearson correlation (95% bootstrap CI)', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_diag1.png', bbox_inches='tight'); plt.close(fig)


def fig_track_c():
    rows = json.load(open('guidance/thesis/trackc_rows.json'))
    rows = sorted(rows, key=lambda r: r['raw'])
    y = np.arange(len(rows))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 4.2), sharey=True)
    ax = axs[0]
    for i, r in enumerate(rows):
        ax.errorbar(r['raw'], i, xerr=[[r['raw'] - r['ci'][0]], [r['ci'][1] - r['raw']]], fmt='o', color=BLUE, capsize=2, lw=1, ms=4)
    ax.axvline(0, color=DARK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([r['pocket'] for r in rows], fontsize=8)
    ax.set_xlabel('Vina Dock shift (kcal/mol)', fontsize=8); ax.set_title('(A) Raw docking score', fontsize=9)
    ax = axs[1]
    ax.barh(y, [r['le'] for r in rows], color=[RED if r['le'] > 0 else GREEN for r in rows], height=0.6)
    ax.axvline(0, color=DARK, lw=0.8); ax.set_xlabel('Ligand-efficiency shift\n(positive = worse)', fontsize=8)
    ax.set_title('(B) Per-atom efficiency', fontsize=9)
    ax = axs[2]
    ax.barh(y, [100 * (r['pb_full'] - r['pb_top']) for r in rows], color=ORANGE, height=0.6)
    ax.axvline(0, color=DARK, lw=0.8); ax.set_xlabel('PoseBusters pass-rate drop\n(percentage points)', fontsize=8)
    ax.set_title('(C) Physical validity', fontsize=9)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackC.png', bbox_inches='tight'); plt.close(fig)


def fig_track_d():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axs[0]
    lam = [0, 0.01, 0.03, 0.1, 0.3]
    own = [0.458, 0.487, 0.591, 0.645, 0.735]
    real = [0.361, 0.365, 0.168, 0.154, 0.126]
    xs = np.arange(len(lam))
    ax.plot(xs, own, 'o-', color=BLUE, label='Predictor\'s own score')
    ax.plot(xs, real, 's-', color=RED, label='Real RA-score')
    ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in lam], fontsize=8)
    ax.set_xlabel('$\\lambda_{synth}$', fontsize=9); ax.set_ylabel('Mean score', fontsize=8); ax.set_ylim(0, 0.85)
    ax.set_title('(A) Original design, n = 8, 1 pocket', fontsize=9); ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax = axs[1]
    lam2 = ['0.01', '0.03', '0.1', '0.3', '1.0']
    m = [-0.009, -0.066, -0.071, -0.192, -0.018]
    lo = [-0.161, -0.207, -0.212, -0.315, -0.194]; hi = [0.140, 0.079, 0.072, -0.078, 0.148]
    n = [27, 26, 23, 6, 2]
    for i in range(5):
        ax.errorbar(i, m[i], yerr=[[m[i] - lo[i]], [hi[i] - m[i]]], fmt='o', color=BLUE if n[i] > 20 else GRAY, capsize=3, lw=1.2)
        ax.text(i, 0.19, f'n = {n[i]}', ha='center', fontsize=7)
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(range(5)); ax.set_xticklabels(lam2, fontsize=8); ax.set_ylim(-0.4, 0.25); ax.set_xlim(-0.6, 4.6)
    ax.set_xlabel('$\\lambda_{synth}$', fontsize=9); ax.set_ylabel('Shift of real RA-score vs. unguided', fontsize=8)
    ax.set_title('(B) Leakage-safe predictor, n = 30, 1 pocket', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trackD.png', bbox_inches='tight'); plt.close(fig)


def fig_overview_significance():
    """New figure. Fraction of comparisons significant after Benjamini-Hochberg
    correction, one bar per checkpoint that used the shared KS+BH gradient-
    informativeness protocol of Section III.6.1. Counts are read directly from
    each checkpoint's own results file (or, where noted, copied from the
    already-published table in the corresponding chapter) -- no new statistics
    are computed here beyond a fraction and a mean effect size.
    Sources: guidance/track_a_exploratory_results/, guidance/track_a_full_results/,
    guidance/track_b{1,2,3}_results/, guidance/track_e/d6.. (Chapter VII Table 7.3),
    guidance/track_c_analysis/track_c_analysis_results.json.
    """
    import csv, json
    rows_src = {
        'Track A\nexploratory (9)': ('guidance/track_a_exploratory_results/gradient_informativeness_results.csv', None),
        'Track A\nfull tier (24)': ('guidance/track_a_full_results/gradient_informativeness_results.csv', None),
        'Track B1 (9)': ('guidance/track_b1_results/gradient_informativeness_results.csv', None),
        'Track B2 (6)': ('guidance/track_b2_results/gradient_informativeness_results.csv', None),
        'Track B3 (8)': ('guidance/track_b3_results/gradient_informativeness_results.csv', None),
    }
    labels, sig_frac, n_tests, note_color = [], [], [], []
    for lab, (path, _) in rows_src.items():
        rs = list(csv.DictReader(open(path)))
        sig = sum(r['significant_after_correction'] == 'True' for r in rs)
        labels.append(lab); sig_frac.append(sig / len(rs)); n_tests.append(len(rs)); note_color.append(BLUE)
    # Track D, D.3 (leakage-safe predictor, pocket BSD_ASPTE, 5 testable lambdas; all BH p = 0.997 -- Chapter VII, Table 7.3)
    labels.append('Track D\nD.3 (5)'); sig_frac.append(0 / 5); n_tests.append(5); note_color.append(BLUE)
    # Track C: KS test of top-10% selection vs. full pool, BH-corrected across 15 pockets (Chapter VI)
    d = json.load(open('guidance/track_c_analysis/track_c_analysis_results.json'))
    pp = d['per_pocket_primary']
    sigC = sum(v['ks_reject_bh'] for v in pp.values())
    labels.append('Track C\nranker (15)'); sig_frac.append(sigC / len(pp)); n_tests.append(len(pp)); note_color.append(ORANGE)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(labels))
    bars = ax.bar(x, sig_frac, color=note_color, width=0.6, edgecolor=DARK, lw=0.7)
    for i, (f, n) in enumerate(zip(sig_frac, n_tests)):
        ax.text(i, f + 0.02, f'{int(round(f * n))}/{n}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fraction significant after BH correction', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.text(len(labels) - 1, 1.0 + 0.03, 'confounded by size\n(Chapter VI)', ha='center', fontsize=7.5, color=ORANGE)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_overview_significance.png', bbox_inches='tight'); plt.close(fig)


def fig_power_curve():
    """New figure. Minimum Vina Dock shift detectable at 80% power for a
    two-sample comparison, as a function of the number of molecules per
    condition, computed from the standard normal-approximation formula
    MDD = (z_(1-a/2) + z_(1-b)) * SD * sqrt(2/n), using the range of
    within-pocket Vina Dock standard deviations actually measured in this
    project's sweeps (1.1-1.6 kcal/mol; e.g. guidance/LPSPLIT_LAMBDA_RESWEEP_FINDING.md,
    guidance/AFFINITY_LAMBDA_RESWEEP_FINDING.md). This is a textbook power
    calculation, not a new empirical result; it is provided to make the
    statistical-power argument of Section IX.2.3 quantitative and visual.
    """
    from scipy.stats import norm
    z_beta = norm.ppf(0.8)
    n = np.arange(8, 41)
    def mdd(alpha, sd):
        z_a = norm.ppf(1 - alpha / 2)
        return (z_a + z_beta) * sd * np.sqrt(2 / n)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for sd, ls in [(1.1, '-'), (1.6, '--')]:
        ax.plot(n, mdd(0.05, sd), color=BLUE, ls=ls, lw=1.6,
                label=f'$\\alpha=0.05$, SD={sd}')
        ax.plot(n, mdd(0.05 / 24, sd), color=RED, ls=ls, lw=1.6,
                label=f'$\\alpha=0.05/24$, SD={sd}')
    for nn in (20, 30):
        ax.axvline(nn, color=GRAY, lw=0.7, ls=':')
    ax.text(20, ax.get_ylim()[1] * 0.96, 'n = 20\n(Track B)', fontsize=7, ha='center', color=GRAY)
    ax.text(30, ax.get_ylim()[1] * 0.96, 'n = 30\n(Track A full,\nTask F)', fontsize=7, ha='center', color=GRAY)
    ax.set_xlabel('Molecules per condition (n)', fontsize=9)
    ax.set_ylabel('Minimum detectable Vina Dock shift\nat 80% power (kcal/mol)', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_power_curve.png', bbox_inches='tight'); plt.close(fig)


def fig_scale_by_track():
    """New figure. Number of pockets vs. molecules (or complexes) per
    condition for every checkpoint reported in Chapters IV-VIII, on a log
    y-axis, to make visible how much the evidentiary base differs between
    an exploratory tier and a full tier, and why Track C's ranker result
    and Track E's scoring result are much better powered than the single-
    pocket checkpoints. Values are the pocket/molecule counts already
    stated in the corresponding chapter; no new counts are introduced.
    """
    pts = [
        ('Task F baseline\n(Ch. IV)', 20, 30, BLUE, (6, 6)),
        ('Repaired-predictor\nconfirmation (Ch. IV)', 8, 8, BLUE, (6, -18)),
        ('Track A exploratory\n(Ch. V)', 3, 20, BLUE, (6, 10)),
        ('Track A full\n(Ch. V)', 8, 30, BLUE, (6, 6)),
        ('Track B1/B2\n(Ch. V)', 3, 20, BLUE, (-70, -20)),
        ('Track B3\n(Ch. V)', 2, 20, BLUE, (10, -22)),
        ('Track C ranker\n(Ch. VI)', 15, 594, ORANGE, (6, 6)),  # mean pool size 593.9, see track_c_analysis_results.json
        ('Track D exploratory\n(Ch. VII)', 1, 8, GREEN, (6, -18)),
        ('Track D powered\n(Ch. VII)', 1, 30, GREEN, (6, 6)),
        ('Track E test set\n(Ch. VIII)', 127, 14, '#7a4fa3', (-95, 6)),  # median complexes/target (Chapter III); mean is 93 but skewed by one target with 1,392
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lab, npk, nmol, c, off in pts:
        ax.scatter(npk, nmol, s=60, color=c, zorder=3, edgecolor=DARK, lw=0.6)
        ax.annotate(lab, (npk, nmol), fontsize=7, xytext=off, textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Pockets or targets', fontsize=9)
    ax.set_ylabel('Molecules or complexes per pocket/condition', fontsize=9)
    ax.set_xlim(0.7, 250); ax.set_ylim(5, 1500)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_scale_by_track.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    for f in (fig_pipeline, fig_verdict_flow, fig_track_e_design, fig_track_e_r2, fig_track_e_learnability,
              fig_track_e_within_target, fig_plip_jitter, fig_vina_semantics, fig_split_artifact, fig_predictive_quality, fig_track_a_full, fig_track_b, fig_diag1, fig_track_c, fig_track_d, fig_overview_significance, fig_power_curve, fig_scale_by_track):
        f(); print('ok', f.__name__)
