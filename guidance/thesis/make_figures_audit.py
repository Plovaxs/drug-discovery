"""Figures for the material added during the 2026-10 audit and follow-up measurements.

Every panel here is drawn from a committed result file, never from a number typed into this script, so
the figures cannot drift from the tables they accompany. The source path for each is named in the
function's docstring, and `guidance/thesis/check_numbers_against_sources.py` pins the same files.

These fill the chapters that carried the new material but almost no graphics: Chapter IV gained four
sections and had two figures against nine tables, and the positive control of Chapter VI had none.

Run:  PYTHONPATH=. python guidance/thesis/make_figures_audit.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = 'guidance/thesis/figures'
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GRAY, DARK, GREEN, RED = '#2a78d6', '#eb6834', '#8a8f98', '#222222', '#2e8b57', '#c0392b'
PURPLE = '#7a4fa3'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
                     'savefig.dpi': 300})


def load(rel):
    with open(rel) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------- Chapter IV.6
def fig_representation_ladder():
    """Held-out R2 of each information source, with target-clustered bootstrap intervals.
    Source: representation_ladder_results.json (block R2 and its ci95, plus the paired
    comparisons against the ligand-only block); the class-prior and label-noise reference
    levels come from noise_ceiling_results.json.

    The intervals are the point of the figure. Read without them, the bar ordering looks like a
    ranking and suggests that a pocket representation beats the ligand-only baseline; every
    paired test that exists against that baseline in fact returns a tie after Benjamini-Hochberg
    correction, which is the claim the chapter makes.
    """
    res = load('guidance/cheminformatics/representation_ladder_results.json')
    lad, paired = res['blocks'], res['paired']
    nc = load('guidance/cheminformatics/noise_ceiling_results.json')
    prior = nc['decomposition']['class_level_prior']
    ceiling = nc['decomposition']['ceiling']

    order = [('plip_raw', 'PLIP typed counts, raw', ORANGE),
             ('plip_norm', 'PLIP typed counts, normalised', ORANGE),
             ('ecif_norm', 'ECIF contacts, size-normalised', ORANGE),
             ('pocket', 'Pocket only (no ligand)', BLUE),
             ('plip_norm+size', 'PLIP counts + size', GREEN),
             ('ligand', 'Ligand only (14 descriptors)', GREEN),
             ('ecif_raw', 'ECIF contacts, raw', GREEN)]
    labels = [l for _, l, _ in order]
    vals = [lad[k]['test_r2'] for k, _, _ in order]
    cis = [lad[k]['ci95'] for k, _, _ in order]
    cols = [c for _, _, c in order]
    err = np.array([[v - lo for v, (lo, _) in zip(vals, cis)],
                    [hi - v for v, (_, hi) in zip(vals, cis)]])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(vals))
    ax.barh(y, vals, xerr=err, color=cols, edgecolor=DARK, lw=0.6, height=0.6, zorder=3,
            error_kw={'lw': 1.1, 'capsize': 3, 'ecolor': DARK, 'zorder': 5})
    for yi, (k, _, _), v, (lo, hi) in zip(y, order, vals, cis):
        cmp_key = f'{k}__minus__ligand'
        if k == 'ligand':
            tag = '(reference)'
        elif cmp_key in paired:
            vd = paired[cmp_key]['verdict']
            tag = 'ligand better' if vd.startswith('b >') else ('better' if vd.startswith('a >') else 'tie vs ligand-only')
            tag = f'{tag}, p(BH) = {paired[cmp_key]["boot_p_bh"]:.2g}'
        else:
            tag = 'not tested vs ligand-only'
        ax.text(hi + 0.012, yi, f'{v:.3f}  [{lo:.3f}, {hi:.3f}]   {tag}',
                va='center', fontsize=7.2, color=DARK)
    ax.axvline(prior, color=RED, ls='--', lw=1.2, zorder=4)
    ax.text(prior - 0.008, -0.62, f'class-level prior {prior:.3f}', color=RED, fontsize=7.5,
            ha='right', va='bottom')
    ax.axvline(ceiling, color=GRAY, ls=':', lw=1.4, zorder=4)
    ax.text(ceiling - 0.008, -0.62, f'label-noise ceiling {ceiling:.3f}', color=GRAY,
            fontsize=7.5, ha='right', va='bottom')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Held-out $R^2$ (target-disjoint test set); bars show the 95% '
                  'target-clustered bootstrap interval', fontsize=8.5)
    ax.set_xlim(-0.05, 1.02); ax.set_ylim(-0.95, len(vals) - 0.3)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_representation_ladder.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter IV.8
def fig_class_prior_decomposition():
    """Where each model sits between chance, the class-level prior and the label-noise ceiling.
    Source: noise_ceiling_results.json."""
    nc = load('guidance/cheminformatics/noise_ceiling_results.json')
    dec, models = nc['decomposition'], nc['models']
    chance, prior, ceiling = dec['chance'], dec['class_level_prior'], dec['ceiling']
    pick = [('egnn_a1c_ensemble', 'EGNN ensemble\n(structure)', BLUE),
            ('egnn_stage0', 'EGNN Stage 0\n(structure)', BLUE),
            ('desc_ridge (lig)', 'Ridge, 14 descriptors\n(no protein)', GREEN),
            ('heavy_atoms (lig)', 'Heavy-atom count\n(one feature)', GREEN),
            ('vina (lig)', 'AutoDock Vina\n(docking score)', GRAY)]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(chance, prior, color='#f6e2e2', zorder=0)
    ax.axvspan(prior, ceiling, color='#e8f1e8', zorder=0)
    ax.text((chance + prior) / 2, len(pick) - 0.18, 'reachable without any\nwithin-target SAR',
            ha='center', va='top', fontsize=7.5, color=RED)
    ax.text((prior + ceiling) / 2, len(pick) - 0.18, 'the window real structure-activity\nlearning could occupy',
            ha='center', va='top', fontsize=7.5, color=GREEN)
    for i, (k, lab, c) in enumerate(pick):
        v = models[k]['test_r2']
        ax.plot([chance, v], [i, i], color=c, lw=2.4, solid_capstyle='round', zorder=3)
        ax.scatter([v], [i], s=46, color=c, edgecolor=DARK, lw=0.6, zorder=4)
        ax.text(v + 0.008, i, f"{v:.3f}  ({models[k]['pct_of_headroom']:.0f}% of the window)",
                va='center', fontsize=8, color=DARK)
    # Reference-line labels are drawn INSIDE the axes, rotated along their lines. Placing them below
    # the axis collides with the x tick labels, which is what a first version of this figure did.
    for x, lab, c, ls in ((chance, 'chance', GRAY, ':'), (prior, 'class-level prior', RED, '--'),
                          (ceiling, 'label-noise ceiling', GRAY, '-.')):
        ax.axvline(x, color=c, ls=ls, lw=1.2, zorder=2)
        ax.text(x + 0.006, -0.78, f'{lab} = {x:.3f}', rotation=90, ha='left', va='bottom',
                fontsize=7.2, color=c, zorder=5)
    ax.set_yticks(range(len(pick))); ax.set_yticklabels([l for _, l, _ in pick], fontsize=8)
    ax.set_xlim(-0.01, 0.72); ax.set_ylim(-0.95, len(pick) - 0.05)
    ax.set_xlabel('Held-out $R^2$', fontsize=9)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_class_prior.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter VI.5
def fig_rq2_positive_control():
    """The positive control's multi-objective outcome. Source: rq2_table.json (regenerated from the
    per-molecule result files).

    Colour encodes IMPROVEMENT, not the sign of the change: Vina Dock and ligand efficiency are
    quantities where a more negative value is better, so colouring by sign alone would paint the
    headline improvement red and tell the reader the opposite of what the data show. Bars whose
    paired test does not reach p < 0.05 are drawn hollow, so that a visible bar is never mistaken
    for a supported effect.
    """
    d = load('guidance/rq2_table.json')['metrics']
    # (key, label, lower_is_better)
    rows = [('vina_dock', 'Vina Dock\n(lower is better)', True),
            ('ligand_efficiency', 'Ligand efficiency\n(lower is better)', True),
            ('pb_valid', 'PoseBusters validity\n(higher is better)', False),
            ('qed', 'QED\n(higher is better)', False),
            ('sa', 'SA score\n(higher is better)', False),
            ('ra_score', 'RA-score\n(higher is better)', False),
            ('heavy_atoms', 'Heavy atoms\n(size; not a quality metric)', None)]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    for i, (k, lab, lower_better) in enumerate(rows):
        m = d[k]
        b, r = m['baseline_mean'], m['reranked_mean']
        rel = (r - b) / (abs(b) if abs(b) > 1e-9 else 1.0) * 100
        p = m['wilcoxon_paired_p']
        sig = p is not None and p < 0.05
        if lower_better is None:
            col = ORANGE                      # size: neither good nor bad on its own
        else:
            improved = (rel < 0) if lower_better else (rel > 0)
            col = GREEN if improved else RED
        ax.barh(i, rel, color=col if sig else 'none', edgecolor=col if sig else col,
                lw=1.2, height=0.6, zorder=3, hatch=None if sig else '///')
        tag = (f"{rel:+.1f}%   p = {p:.2g}" + ('' if sig else '  (n.s.)')).replace('-', '\u2212')
        ax.text(rel + (1.5 if rel >= 0 else -1.5), i, tag, va='center',
                ha='left' if rel >= 0 else 'right', fontsize=7.3, color=DARK)
    ax.axvline(0, color=DARK, lw=0.9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l, _ in rows], fontsize=7.5)
    ax.set_xlabel('Change from unguided baseline (%)', fontsize=8.5)
    ax.set_xlim(-46, 46)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)
    ax.text(-44, len(rows) - 0.35, 'filled = significant (p < 0.05)\nhollow = not significant',
            fontsize=7, color=DARK, va='top')

    ax2 = axes[1]
    vd = d['vina_dock']
    means = [vd['baseline_mean'], vd['reranked_mean']]
    sds = [vd['baseline_sd'], vd['reranked_sd']]
    xs = [0, 1]
    ax2.bar(xs, means, yerr=sds, capsize=5, color=[GRAY, GREEN], edgecolor=DARK, lw=0.7,
            zorder=3, width=0.55, error_kw={'lw': 1.2})
    for x, m, sd in zip(xs, means, sds):
        ax2.text(x + 0.33, m, f'{m:.2f}\n± {sd:.2f}'.replace('-', '\u2212'), ha='left',
                 va='center', fontsize=8, color=DARK)
    ax2.set_xticks(xs); ax2.set_xticklabels(['Unguided\nbaseline', 'Vina-reranked'], fontsize=8.5)
    ax2.set_ylabel('Vina Dock (kcal/mol), lower is better', fontsize=8.5)
    ax2.set_ylim(-21, 6); ax2.set_xlim(-0.6, 1.9)
    ax2.axhline(0, color=DARK, lw=0.8)
    ax2.set_title(f"improvement: {abs(means[1] - means[0]):.2f} kcal/mol more negative\n"
                  f"Wilcoxon (paired by pocket) p = {vd['wilcoxon_paired_p']:.1e}", fontsize=8.5)
    ax2.grid(axis='y', color='#e6e8ec', zorder=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_rq2_control.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter IV.7
def fig_arm_budget():
    """Arm A against Arm C on quality and on the compute each received.
    Sources: armAC_table.json and compute_budget_table.json.

    Two deliberate choices. The quality panel states that the two intervals overlap and that no
    paired test between the arms exists in the source file, so that the vertical ordering is not
    read as a result. The budget panel is drawn in hours alone, split into active compute and
    thermally suspended time, with the epoch counts written on the bars: plotting hours and
    epochs as neighbouring bars on one axis would put two incommensurable units on one scale.
    """
    arms = load('guidance/surrogate_data/armAC_table.json')['groups']
    budget = load('guidance/surrogate_data/compute_budget_table.json')['totals']
    rows = [('surrogate_arms/armA_cd20000_bn0_bs1', 'Arm A\n20k CrossDocked', 'armA', BLUE),
            ('surrogate_arms/armC_cd6000_bn14000_bs1', 'Arm C\n6k CD + 14k BindingNet', 'armC', ORANGE)]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    ax = axes[0]
    los, his = [], []
    for i, (k, lab, tag, col) in enumerate(rows):
        g = arms[k]
        lo, hi = g['ci95']; los.append(lo); his.append(hi)
        ax.errorbar(i, g['mean'], yerr=[[g['mean'] - lo], [hi - g['mean']]], fmt='o', color=col,
                    ecolor=col, capsize=6, markersize=9, markeredgecolor=DARK, lw=1.6, zorder=3)
        ax.text(i + 0.13, g['mean'], f"{g['mean']:.3f}\n[{lo:.3f}, {hi:.3f}]", fontsize=7.8,
                va='center', color=DARK)
    ov_lo, ov_hi = max(los), min(his)
    if ov_lo < ov_hi:
        ax.axhspan(ov_lo, ov_hi, color='#e6e8ec', zorder=0)
        ax.text(1.55, (ov_lo + ov_hi) / 2, 'intervals\noverlap', fontsize=7.5, color=GRAY,
                ha='right', va='center')
    ax.set_xticks([0, 1]); ax.set_xticklabels([r[1] for r in rows], fontsize=8)
    ax.set_ylabel('Validation $R^2$', fontsize=8.5)
    ax.set_xlim(-0.5, 1.6)
    ax.set_title('3 seeds per arm; 95% CI over seeds.\nNo paired test between arms was computed,\n'
                 'so no difference is claimed here.', fontsize=7.8)
    ax.grid(axis='y', color='#e6e8ec', zorder=0)

    ax2 = axes[1]
    tags, cols = ['armA', 'armC'], [BLUE, ORANGE]
    x = np.arange(2)
    total = [budget[t]['wall_clock_h'] for t in tags]
    susp = [budget[t]['thermally_suspended_h'] for t in tags]
    active = [tt - ss for tt, ss in zip(total, susp)]
    ax2.bar(x, active, width=0.5, color=cols, edgecolor=DARK, lw=0.6, zorder=3, label='active compute')
    ax2.bar(x, susp, bottom=active, width=0.5, color=cols, edgecolor=DARK, lw=0.6, alpha=0.4,
            zorder=3, hatch='//', label='thermally suspended')
    for xi, t in enumerate(tags):
        bt = budget[t]
        ax2.text(xi, total[xi] + 0.5, f"{total[xi]:.2f} h total", ha='center', fontsize=8, color=DARK)
        ax2.text(xi, total[xi] + 2.4, f"{bt['epochs']} epochs @ {bt['hours_per_epoch']:.2f} h/epoch",
                 ha='center', fontsize=7.5, color=DARK)
    ax2.set_xticks(x); ax2.set_xticklabels(['Arm A', 'Arm C'], fontsize=8.5)
    ax2.set_ylabel('Wall-clock time (hours)', fontsize=8.5)
    ax2.set_ylim(0, max(total) * 1.33)
    ax2.set_title('Compute actually consumed\n(single RTX 3050, thermally gated)', fontsize=8)
    ax2.legend(fontsize=7.2, frameon=False, loc='upper left')
    ax2.grid(axis='y', color='#e6e8ec', zorder=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_arm_budget.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter V.4
def fig_gradient_direction():
    """The signed direction result with the control that had to pass first.
    Source: diag_gradient_direction_results.json.

    The right-hand panel is deliberately drawn in a single neutral colour. The near/far split has
    no interval and no test attached to it in the source file, so colouring the two halves as
    'good' and 'bad' would present a descriptive number as a directional finding. The only
    inferential claim in this figure is the left-hand panel.
    """
    res = load('guidance/diag_gradient_direction_results.json')
    d = res['checkpoints']
    r = next(iter(d.values()))
    lo, hi = r['ci95']
    rlo, rhi = r['random_ci95']

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={'width_ratios': [1.3, 1]})
    ax = axes[0]
    items = [('Affinity gradient', r['mean_cos_toward_contact'], lo, hi, BLUE),
             ('Random-direction\ncontrol', r['random_direction_control_mean_cos'], rlo, rhi, GRAY)]
    for i, (lab, m, l, h, c) in enumerate(items):
        ax.errorbar(m, i, xerr=[[m - l], [h - m]], fmt='o', color=c, ecolor=c, capsize=6,
                    markersize=9, markeredgecolor=DARK, lw=1.7, zorder=3)
        ax.text(-0.085, i + 0.3, f'{m:+.3f}  95% CI [{l:+.3f}, {h:+.3f}]'.replace('-', '\u2212'),
                ha='left', fontsize=7.5, color=DARK)
    ax.axvline(0, color=RED, ls='--', lw=1.2, zorder=2)
    ax.text(0.003, 1.45, 'no preferred direction', ha='left', fontsize=7.5, color=RED, va='top')
    ax.set_yticks([0, 1]); ax.set_yticklabels([lab for lab, *_ in items], fontsize=8.5)
    ax.set_xlabel('Mean cosine toward nearest pocket contact\n'
                  f"(n = {r['n_atoms']:,} atom-states over {res['n_pockets']} pockets; "
                  f"{r['frac_pointing_toward'] * 100:.1f}% point toward, chance = 50%)", fontsize=8)
    ax.set_xlim(-0.09, 0.09); ax.set_ylim(-0.6, 1.75)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)

    ax2 = axes[1]
    near, far = r['mean_cos_near_half'], r['mean_cos_far_half']
    ax2.bar(['Nearer half', 'Farther half'], [near, far], color=BLUE,
            edgecolor=DARK, lw=0.7, width=0.5, zorder=3)
    for i, v in enumerate([near, far]):
        ax2.text(i, v + (0.004 if v >= 0 else -0.004), f'{v:+.3f}'.replace('-', '\u2212'),
                 ha='center', va='bottom' if v >= 0 else 'top', fontsize=8.5, color=DARK)
    ax2.axhline(0, color=DARK, lw=0.9)
    ax2.set_ylabel('Mean cosine toward contact', fontsize=8.5)
    ax2.set_title('split by distance to the pocket: descriptive only\n'
                  'no interval or test computed for this split\n'
                  f"(median contact distance {r['median_nn_distance_A']:.2f} \u00c5)", fontsize=7.8)
    ax2.set_ylim(-0.11, 0.07)
    ax2.grid(axis='y', color='#e6e8ec', zorder=0)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_gradient_direction.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter VIII
def fig_m2_subgroup():
    """What the M2 subgroup leaves out. Source: m2_subgroup_characterisation.json.

    The affinity panel carries the within-group standard deviation, because the 0.79 pK unit
    difference between the two groups is small beside a spread of about 1.65 to 1.75 pK units
    inside each of them; drawn as two bare bars it would read as a clean separation.
    """
    s = load('guidance/track_e/m2_subgroup_characterisation.json')
    thr = s['min_complexes_threshold']
    cats = [f'included\n(\u2265 {thr} complexes)', f'excluded\n(< {thr} complexes)']
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2))

    a = axes[0]
    inc, exc = s['n_targets_included'], s['n_targets_excluded']
    a.bar(cats, [inc, exc], color=[BLUE, GRAY], edgecolor=DARK, lw=0.6, width=0.55, zorder=3)
    for i, v in enumerate([inc, exc]):
        a.text(i, v + 1.5, f"{v}  ({v / s['n_targets_total'] * 100:.0f}%)", ha='center', fontsize=8)
    a.set_ylabel('Targets', fontsize=8.5)
    a.set_title(f"Targets ({s['n_targets_total']} in total)", fontsize=9)
    a.set_ylim(0, max(inc, exc) * 1.32)
    a.grid(axis='y', color='#e6e8ec', zorder=0)

    b = axes[1]
    ci, ce = s['n_complexes_included'], s['n_complexes_total'] - s['n_complexes_included']
    b.bar(cats, [ci, ce], color=[BLUE, GRAY], edgecolor=DARK, lw=0.6, width=0.55, zorder=3)
    b.text(0, ci * 1.03, f"{ci:,}\n({s['pct_complexes_covered']:.1f}%)", ha='center', fontsize=8)
    b.text(1, ce * 1.05 + 200, f"{ce:,}\n({100 - s['pct_complexes_covered']:.1f}%)",
           ha='center', fontsize=8)
    b.set_ylabel('Complexes', fontsize=8.5)
    b.set_title(f"Complexes ({s['n_complexes_total']:,} in total)\n"
                f"median per target: {s['median_complexes_included']:.0f} vs "
                f"{s['median_complexes_excluded']:.0f}", fontsize=8.5)
    b.set_ylim(0, ci * 1.38)
    b.grid(axis='y', color='#e6e8ec', zorder=0)

    c = axes[2]
    pi, pe = s['pk_mean_included'], s['pk_mean_excluded']
    si, se = s['pk_sd_included'], s['pk_sd_excluded']
    c.bar(cats, [pi, pe], yerr=[si, se], capsize=5, color=[BLUE, GRAY], edgecolor=DARK, lw=0.6,
          width=0.55, zorder=3, error_kw={'lw': 1.1, 'ecolor': DARK, 'zorder': 5})
    for i, (v, sd) in enumerate(((pi, si), (pe, se))):
        c.text(i, v + sd + 0.25, f'{v:.2f} \u00b1 {sd:.2f}', ha='center', va='bottom', fontsize=8)
    c.set_ylabel('Mean affinity (pK)', fontsize=8.5)
    c.set_title(f"Mean affinity (pK)\ndifference {s['pk_mean_difference']:.2f}, "
                f"within-group SD \u2248 {(si + se) / 2:.2f}", fontsize=8.5)
    c.set_ylim(0, (max(pi, pe) + max(si, se)) * 1.3)
    c.grid(axis='y', color='#e6e8ec', zorder=0)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=7.8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_m2_subgroup.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Chapter III
def fig_clustering_levels():
    """Audit item F037: the same comparison at three clustering levels.
    Source: family_cluster_bootstrap_results.json.

    The point of the figure is the first step, not the second. Only the resampling unit changes
    between the three rows, so the point estimate is identical in all of them and the entire
    difference is in the width of the interval.
    """
    d = load('guidance/cheminformatics/family_cluster_bootstrap_results.json')
    order = [('complex (no clustering)', 'Complex\n(no clustering)', RED),
             ('target (as reported in this thesis)', 'Target\n(used in this thesis)', BLUE),
             ('Pfam family', 'Pfam family', GREEN)]

    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    for i, (key, lab, col) in enumerate(order):
        v = d['levels'][key]
        lo, hi = v['ci95']
        y = len(order) - 1 - i
        sig = v['boot_p'] < 0.05
        ax.plot([lo, hi], [y, y], color=col, lw=3.2, solid_capstyle='butt', zorder=3)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=col, lw=1.6, zorder=3)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=col, lw=1.6, zorder=3)
        ax.plot([v['d_r2']], [y], 'o', color=col, markeredgecolor=DARK, markersize=8, zorder=4)
        ax.text(hi + 0.006, y,
                f"width {v['ci_width']:.4f},  p = {v['boot_p']:.3f}"
                + ('  \u2014 would be called significant' if sig else '  \u2014 tie'),
                va='center', fontsize=7.6, color=DARK)
        ax.text(lo - 0.014, y, f"{v['n_clusters']:,} clusters", va='center', ha='right',
                fontsize=7.2, color=GRAY)
    ax.axvline(0, color=DARK, ls='--', lw=1.1, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for _k, lab, _c in reversed(order)], fontsize=8.5)
    ax.set_xlabel('\u0394R\u00b2 (structure-aware ensemble \u2212 ligand-only ridge), '
                  'with the 95% bootstrap interval', fontsize=8.5)
    ax.set_xlim(-0.13, 0.33); ax.set_ylim(-0.6, len(order) - 0.3)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)
    ax.set_title('The point estimate is identical in all three rows: only the resampling unit changes',
                 fontsize=8.4)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_clustering_levels.png', bbox_inches='tight'); plt.close(fig)


# --------------------------------------------------------------------------------- Chapter VII
def fig_synth_gradient():
    """Track D's gradient diagnostic, as the counterpart to the affinity one of Chapter V.
    Sources: diag_synth_direction_results.json and diag_synth_direction_followup_results.json.

    All three intervals include zero, and the figure is drawn so that this reads as "no correlate
    was found" rather than as three separate near-misses. The shaded band marks the magnitude no
    interval reaches beyond, which is the positive content of the measurement: correlations larger
    than about 0.09 with these three proxies are excluded, while nothing smaller is resolved.
    """
    a = load('guidance/diag_synth_direction_results.json')
    b = load('guidance/diag_synth_direction_followup_results.json')
    items = [('Heaviness of\nthe atom type', a['size_correlation']),
             ('Aromatic\nfraction', a['aromaticity_correlation']),
             ('Heteroatom fraction\n(N, O, S, P, halogens)', b['heteroatom_correlation'])]
    n = items[0][1]['n']

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    widest = max(max(abs(v['ci_lo']), abs(v['ci_hi'])) for _l, v in items)
    ax.axvspan(-widest, widest, color='#eef2f7', zorder=0)
    for i, (lab, v) in enumerate(items):
        y = len(items) - 1 - i
        ax.errorbar(v['mean'], y, xerr=[[v['mean'] - v['ci_lo']], [v['ci_hi'] - v['mean']]],
                    fmt='o', color=BLUE, ecolor=BLUE, capsize=6, markersize=8,
                    markeredgecolor=DARK, lw=1.7, zorder=3)
        ax.text(v['ci_hi'] + 0.006, y,
                f"{v['mean']:+.3f}  95% CI [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]".replace('-', '\u2212'),
                va='center', fontsize=7.6, color=DARK)
    ax.axvline(0, color=RED, ls='--', lw=1.2, zorder=2)
    ax.text(0, -0.72, 'no correlation', ha='center', va='top', fontsize=7.5, color=RED)
    ax.text(-widest, len(items) - 0.42,
            f'  shaded: |r| < {widest:.2f} \u2014 no interval reaches beyond this',
            fontsize=7.2, color=GRAY, va='bottom')
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([l for l, _v in reversed(items)], fontsize=8.2)
    ax.set_xlabel(f'Mean per-atom correlation with the guidance gradient magnitude '
                  f'(n = {n} captured states)', fontsize=8.5)
    ax.set_xlim(-0.14, 0.28); ax.set_ylim(-0.95, len(items) - 0.3)
    ax.grid(axis='x', color='#e6e8ec', zorder=1)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_synth_gradient.png', bbox_inches='tight'); plt.close(fig)


# --------------------------------------------------------------------------------- Appendix E.5
def fig_trace_distribution():
    """Audit item F060: the per-molecule trajectory gap over many molecules, not one.
    Sources: trace_guidance_trajectory_multi_results.json and trace_guidance_trajectory_results.json.

    The single-molecule trace of Appendix E.5 is drawn as one marked point inside the
    distribution, because the question the audit raised is precisely whether that molecule was
    representative. Plotting the distribution without it would answer a different question.
    """
    d = load('guidance/trace_guidance_trajectory_multi_results.json')
    mols = d['molecules']
    one = load('guidance/trace_guidance_trajectory_results.json')
    gaps = np.array([m['final_gap_mean'] for m in mols])
    ratios = np.array([m['gap_over_nudge'] * 100 for m in mols])
    single_gap = float(one['all_atom_final_gap_mean'])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    rng = np.random.default_rng(0)

    ax = axes[0]
    ax.scatter(gaps, rng.normal(0, 0.045, len(gaps)), s=34, color=BLUE, edgecolor=DARK, lw=0.5,
               zorder=3, label=f'{len(gaps)} molecules')
    ax.axvline(float(np.median(gaps)), color=DARK, lw=1.2, zorder=4)
    ax.text(float(np.median(gaps)) + 0.012, -0.21, f'median {np.median(gaps):.4f} \u00c5',
            ha='left', fontsize=7.4, color=DARK)
    ax.scatter([single_gap], [0], s=130, marker='*', color=ORANGE, edgecolor=DARK, lw=0.6,
               zorder=5, label='the single molecule of E.5')
    ax.set_yticks([]); ax.set_ylim(-0.25, 0.3)
    ax.set_xlabel('Mean all-atom position gap after 1,000 steps (\u00c5)', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc='upper right', ncol=1)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)
    ax.set_title(f'range {gaps.min():.4f} to {gaps.max():.4f} \u00c5 '
                 f'({gaps.max() / max(gaps.min(), 1e-9):.0f}\u00d7)', fontsize=8.4)

    ax2 = axes[1]
    order = np.argsort(ratios)
    ax2.barh(np.arange(len(ratios)), ratios[order], color=BLUE, edgecolor=DARK, lw=0.4,
             height=0.72, zorder=3)
    ax2.axvline(1.0, color=RED, ls='--', lw=1.2, zorder=4)
    # Placed low, where the bars are short: at the top it would sit on the two longest bars.
    ax2.annotate('the \u22481% quoted\nfrom one molecule', xy=(1.0, 3.0),
                 xytext=(6.5, 5.0), fontsize=7.2, color=RED, va='center',
                 arrowprops=dict(arrowstyle='->', color=RED, lw=0.9))
    ax2.set_yticks([]); ax2.set_ylabel('molecules, sorted', fontsize=8.5)
    ax2.set_xlabel('Final gap as a percentage of the cumulative nudge per atom', fontsize=8.5)
    ax2.grid(axis='x', color='#e6e8ec', zorder=0)
    ax2.set_title(f'median {np.median(ratios):.2f}%, max {ratios.max():.2f}%', fontsize=8.4)

    fig.tight_layout(); fig.savefig(f'{OUT}/fig_trace_distribution.png', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------------------------- Appendix E
def fig_chembl_expansion():
    """Audit item F051: this project's pK range against ChEMBL's independent Ki range,
    for every resolved target of the 20-target random sample.
    Source: chembl_crossvalidation_expanded_results.json.

    Targets are sorted by this project's own pK range so the eye reads down a gradient; the
    ChEMBL range is drawn alongside each one, and a target is flagged red only where the two
    ranges fail to touch at all, which is a stricter bar than most would apply to two label
    sources measured under different assay conditions.
    """
    d = load('guidance/chembl_crossvalidation_expanded_results.json')
    rows = {k: v for k, v in d['rows'].items() if v['status'] == 'resolved_with_data'}
    order = sorted(rows, key=lambda t: rows[t]['our_pk_min'])

    def draw(y, lo, hi, color, label):
        # A range with lo == hi (a single measurement, or several identical ones) draws as a
        # zero-length line and disappears entirely -- this affected four rows on the first render,
        # including the one row (RNAS2) the whole subsection exists to show. A marker at the point
        # keeps every row visible regardless of how wide its range is.
        ax.plot([lo, hi], [y, y], color=color, lw=4, solid_capstyle='butt', zorder=3, label=label)
        if hi - lo < 0.03:
            ax.plot([lo], [y], 'o', color=color, markersize=5, zorder=4)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, t in enumerate(order):
        r = rows[t]
        draw(i - 0.17, r['our_pk_min'], r['our_pk_max'], BLUE, 'this project' if i == 0 else None)
        col = RED if r['agreement'] == 'no_overlap' else GRAY
        draw(i + 0.17, r['chembl_pchembl_min'], r['chembl_pchembl_max'], col,
             'ChEMBL (Ki)' if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([rows[t]['mnemonic'] for t in order], fontsize=8)
    ax.set_xlabel('pK / pChEMBL (−log₁₀ of the molar affinity)', fontsize=8.5)
    ax.grid(axis='x', color='#e6e8ec', zorder=0)
    n_overlap = sum(1 for t in order if rows[t]['agreement'] == 'overlap')
    ax.set_title(f'{n_overlap} of {len(order)} targets overlap; red marks the exception (RNAS2), '
                 'the same pattern as CD38 in the original four-target check', fontsize=8.2)
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_chembl_expansion.png', bbox_inches='tight'); plt.close(fig)


FIGURES = [fig_representation_ladder, fig_class_prior_decomposition, fig_rq2_positive_control,
           fig_arm_budget, fig_gradient_direction, fig_m2_subgroup,
           fig_clustering_levels, fig_synth_gradient, fig_trace_distribution,
           fig_chembl_expansion]

if __name__ == '__main__':
    for f in FIGURES:
        f()
        print('ok', f.__name__)
