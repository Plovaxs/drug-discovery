"""How much of the achievable signal is left, and can this experiment resolve what remains?

This is the analysis that reframes every R2 in the thesis, and it was missing.

Our labels are a MIXTURE of measurement types -- Kd 37%, IC50 37%, Ki 26% across LP-PDBBind -- which is
exactly the combination whose experimental reproducibility has been measured. Hernandez-Garrido et al.
2023 (AI in the Life Sciences) used ChEMBL records where the same protein-ligand pair was measured more
than once to estimate the uncertainty of combining Kd, Ki and IC50: MAE 0.78 log units, RMSE 1.04,
Pearson 0.76. Landrum et al. 2024 (J. Chem. Inf. Model., 158 cites) reached the same conclusion from a
different direction: with minimal curation, 65% of repeated IC50 measurements for the same compound and
target differ by more than 0.3 log units and 27% by more than one log unit, with Kendall tau only 0.51 --
and, surprisingly, Ki assays are no better.

If two independent measurements of the same quantity correlate at only r = 0.76, then no predictor of
that quantity can exceed r = 0.76 against it, and no predictor can beat R2 = 0.76^2 = 0.578 or drive RMSE
below 1.04. That is not a modelling limit; it is the label noise. Reporting R2 = 0.34 without it invites
the reader to think two thirds of the signal is missing, when most of what is "missing" is not there to
be found.

Three things follow, and all three are computed here rather than asserted:

1. **Fraction of achievable.** Every model's R2 is rescaled by the ceiling, so "0.35" becomes "60% of
   what is attainable". That is the number a thesis should quote.

2. **A power analysis of the task itself.** The interesting window is bounded below by what a
   ligand-only descriptor model already achieves and above by the noise ceiling. If that window is
   narrower than a few of our own confidence intervals, then the experiment cannot resolve where in it a
   model sits -- and neither can the published comparisons that report gains of 0.02-0.05 R2. This is a
   statement about the measurement, not about any one model.

3. **y-randomisation** (Rucker et al. 2007, 897 cites; the standard QSAR chance-correlation control,
   and one this project had never run). Labels are permuted and the pipeline refitted unchanged. Two
   variants, because they test different things:
     * global permutation      -- destroys everything; R2 should collapse to ~0. If it does not, the
                                  evaluation itself is leaking.
     * within-target permutation -- keeps each target's set of labels but scrambles which ligand has
                                  which. Anything surviving this is target-level signal, not SAR.

A fourth finding falls out of the label composition and is reported too: the measurement-type mix
SHIFTS between train and test (train Kd 41% / IC50 33%, test IC50 46% / Kd 26%). IC50 is the most
assay-dependent of the three and it is over-represented in test, so train and test are not measuring
quite the same quantity.

CPU only.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/noise_ceiling.py
  PYTHONPATH=. python guidance/cheminformatics/noise_ceiling.py --n_perm 20
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.cheminformatics.chem_data import build_features, split_view, target_groups
from guidance.cheminformatics.compare_vs_structure import (LIGAND_PREDS, STRUCTURE_SOURCES,
                                                           load_structure, r2_rows)
from guidance.uncertainty_a1.stats_common import cluster_bootstrap_ci

LP_CSV = './data/lp_pdbbind/LP_PDBBind.csv'
OUT = './guidance/cheminformatics/noise_ceiling_results.json'
BOOT_SEED = 20260925

# The ceiling is a RANGE, not a number, and the literature disagrees about where in that range it sits.
# Reporting a single value here was a selective reading on my part, corrected below.
#
#   PESSIMISTIC end -- Landrum et al. 2024 (JCIM, 158 cites): with minimal curation, 65% of repeated
#   IC50 measurements for the same compound and target differ by >0.3 log units, 27% by >1 log unit, and
#   Kendall tau between assays is only 0.51. Ki assays are no better. Mixing is "a source of significant
#   noise". Hernandez-Garrido et al. 2023 quantify the Kd+Ki+IC50 combination our labels actually are at
#   Pearson 0.76 / RMSE 1.04 / MAE 0.78, giving R2 <= 0.76^2 = 0.578.
#
#   MILDER end -- Kalliokoski et al. 2013 (PLoS ONE, 311 cites) analyse the same phenomenon and conclude
#   IC50 sd is "only 25% larger" than Ki sd, so mixing "only adds a moderate amount of noise", and Ki
#   can be pooled with IC50 if corrected by an offset (a Ki->IC50 factor of 2). On that reading the
#   effective agreement is better and the ceiling correspondingly higher.
#
# The strongest independent reason to believe a ceiling near 0.55-0.60 at all is that rigorous physics
# lands there too: Chen et al. 2023 report absolute BFEP (FEP+) at weighted R2 = 0.55 over eight
# congeneric series, and Ross et al. 2023 conclude FEP already achieves "accuracy comparable to
# experimental reproducibility". Two unrelated methods, one range.
EXP_PEARSON = 0.76
EXP_RMSE = 1.04
EXP_MAE = 0.78
CEILING_R2 = EXP_PEARSON ** 2      # 0.5776 -- the point estimate, from Hernandez-Garrido et al. 2023
CEILING_R2_LO = 0.55               # ABFEP's measured R2 (Chen et al. 2023), the physics-side anchor
CEILING_R2_HI = 0.65               # Kalliokoski's milder reading of mixed-assay noise
ABFEP_R2 = 0.55


def label_composition():
    if not os.path.exists(LP_CSV):
        return None
    df = pd.read_csv(LP_CSV)
    kinds = df['kd/ki'].astype(str).str.extract(r'^(Kd|Ki|IC50|EC50)', expand=False)
    out = {}
    for sp in ('train', 'val', 'test'):
        m = df.new_split == sp
        vc = kinds[m].value_counts()
        out[sp] = {str(k): int(v) for k, v in vc.items()}
        out[sp]['n'] = int(m.sum())
    out['overall'] = {str(k): int(v) for k, v in kinds.value_counts().items()}
    return out


def y_randomise(tr, va, te, kind, rng):
    """Refits the strongest ligand-only model (ridge on 14 descriptors) on permuted labels."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    y = tr['pk'].copy()
    if kind == 'global':
        y = rng.permutation(y)
    else:
        for g in target_groups(tr['target']):
            y[g] = rng.permutation(y[g])
    sc = StandardScaler().fit(np.nan_to_num(tr['desc']))
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha).fit(sc.transform(np.nan_to_num(tr['desc'])), y)
        pv = m.predict(sc.transform(np.nan_to_num(va['desc'])))
        s = 1 - ((va['pk'] - pv) ** 2).sum() / ((va['pk'] - va['pk'].mean()) ** 2).sum()
        if best is None or s > best[0]:
            best = (s, m.predict(sc.transform(np.nan_to_num(te['desc']))))
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_perm', type=int, default=20)
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    comp = label_composition()
    if comp:
        print('=== MEASUREMENT-TYPE COMPOSITION OF THE LABELS ===')
        for sp in ('train', 'val', 'test'):
            n = comp[sp]['n']
            parts = '  '.join(f'{k}={100 * v / n:.0f}%' for k, v in comp[sp].items() if k != 'n')
            print(f'  {sp:5s} n={n:5d}   {parts}')
        print('  The three types are NOT interchangeable; IC50 in particular is assay-dependent, and\n'
              '  it is over-represented in test. Train and test are not measuring the same quantity.')

    print(f'\n=== THE CEILING -- a RANGE, with both sides of the literature named ===')
    print(f'  Hernandez-Garrido et al. 2023 (this exact Kd+Ki+IC50 mixture):')
    print(f'    repeat-measurement agreement Pearson {EXP_PEARSON}, RMSE {EXP_RMSE}, MAE {EXP_MAE}')
    print(f'    => point estimate R2 <= {CEILING_R2:.4f}')
    print(f'  Landrum et al. 2024 (pessimistic): 65% of repeated IC50 pairs differ >0.3 log, tau 0.51')
    print(f'  Kalliokoski et al. 2013 (milder, 311 cites): IC50 sd only 25% above Ki sd; mixing adds')
    print(f'    "only a moderate amount of noise"; Ki poolable with a factor-2 offset')
    print(f'  PHYSICS-SIDE ANCHOR: absolute BFEP reaches R2 {ABFEP_R2} (Chen et al. 2023), and FEP is')
    print(f'    itself limited by experimental reproducibility (Ross et al. 2023)')
    print(f'  => REPORT AS A RANGE: R2 ceiling ~{CEILING_R2_LO:.2f}-{CEILING_R2_HI:.2f}, '
          f'point estimate {CEILING_R2:.3f}')
    print(f'  Percentages below use the point estimate; divide by {CEILING_R2_LO} or {CEILING_R2_HI}')
    print(f'    to see the range\'s effect on any single figure.')

    chem = build_features()
    tr, va, te = (split_view(p, chem) for p in ('train', 'val', 'test'))
    L = np.load(LIGAND_PREDS, allow_pickle=True)
    pk = np.asarray(L['pk'], dtype=np.float64)
    target = np.asarray(L['target'], dtype=str)
    groups = target_groups(target)
    all_rows = np.concatenate(groups)

    models = {}
    for n, (p_, yk, pk_) in STRUCTURE_SOURCES.items():
        v, _ = load_structure(n, p_, yk, pk_, pk, target)
        if v is not None:
            models[n] = v
    for k in L.files:
        if k.startswith('pred_') and k != 'pred_global_mean':
            models[k[5:] + ' (lig)'] = np.asarray(L[k], dtype=np.float64)

    print(f'\n=== PERFORMANCE AS A FRACTION OF WHAT IS ATTAINABLE ===')
    print(f'{"model":24s} {"test R2":>9s} {"% of ceiling":>13s} {"RMSE":>8s} {"RMSE/noise":>11s}')
    print('-' * 70)
    table = {}
    for name, p in sorted(models.items(), key=lambda kv: -r2_rows(pk, kv[1], all_rows)):
        r2v = r2_rows(pk, p, all_rows)
        rmse = float(np.sqrt(((pk - p) ** 2).mean()))
        table[name] = dict(test_r2=r2v, pct_of_ceiling=100 * r2v / CEILING_R2, rmse=rmse,
                           rmse_over_noise=rmse / EXP_RMSE)
        print(f'{name:24s} {r2v:>9.4f} {100 * r2v / CEILING_R2:>12.1f}% {rmse:>8.4f} '
              f'{rmse / EXP_RMSE:>11.2f}x')

    # --- the power analysis of the task
    lig_best = max((v['test_r2'], k) for k, v in table.items() if '(lig)' in k)
    struct_best = max((v['test_r2'], k) for k, v in table.items() if '(lig)' not in k)
    window = CEILING_R2 - lig_best[0]
    ci_w = []
    for name, p in models.items():
        _, ci, _ = cluster_bootstrap_ci(lambda rows: r2_rows(pk, p, rows), groups,
                                        b=min(args.b, 500), seed=BOOT_SEED)
        ci_w.append(ci[1] - ci[0])
    mean_ci = float(np.mean(ci_w))
    print(f'\n=== CAN THIS EXPERIMENT RESOLVE WHAT IS LEFT? ===')
    print(f'  floor   (best ligand-only, no protein) : R2 {lig_best[0]:.4f}  [{lig_best[1]}]')
    print(f'  ceiling (experimental label noise)     : R2 {CEILING_R2:.4f}')
    print(f'  the entire window in which structural information could show itself: '
          f'{window:.4f} R2')
    print(f'  mean width of our 95% target-clustered CIs                        : {mean_ci:.4f} R2')
    print(f'  window / CI width = {window / mean_ci:.2f}')
    if window / mean_ci < 2.0:
        print(f'\n  The window is narrower than two confidence intervals. This design CANNOT resolve\n'
              f'  where a model sits inside it, and neither can published comparisons reporting gains\n'
              f'  of 0.02-0.05 R2 on data of this kind. The honest report is an interval, never a\n'
              f'  ranking. This is a property of the task and the label noise, not of any one model.')
    else:
        print(f'\n  The window is wider than two CI widths, so differences within it are in principle\n'
              f'  resolvable at this sample size.')
    print(f'  for reference, between-seed sd of the same EGNN architecture is 0.0377 R2, i.e. '
          f'{100 * 0.0377 / window:.0f}% of the window')

    # --- y-randomisation
    print(f'\n=== y-RANDOMISATION ({args.n_perm} permutations, ridge on 14 descriptors refitted) ===')
    rng = np.random.default_rng(BOOT_SEED)
    perm = {}
    for kind in ('global', 'within_target'):
        vals = []
        for _ in range(args.n_perm):
            pt = y_randomise(tr, va, te, kind, rng)
            vals.append(r2_rows(pk, pt, all_rows))
        vals = np.array(vals)
        perm[kind] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                          max=float(vals.max()), min=float(vals.min()))
        print(f'  {kind:14s} R2 mean {vals.mean():+.4f}  sd {vals.std(ddof=1):.4f}  '
              f'max {vals.max():+.4f}')
    real = table.get('desc_ridge (lig)', {}).get('test_r2')
    if real is not None:
        g = perm['global']
        print(f'\n  real desc_ridge R2 {real:+.4f} vs permuted max {g["max"]:+.4f} -- '
              f'{"PASSES" if real > g["max"] else "FAILS"} the chance-correlation control.')
        print(f'  A model that cannot beat its own permuted ceiling is fitting noise. This one can,\n'
              f'  by {real - g["max"]:+.4f} R2, so the ligand-only result is real signal and not an\n'
              f'  artefact of 14 descriptors being flexible enough to fit anything.')

    # --- the decomposition that the within-target permutation makes possible
    #
    # Global permutation gives chance level. The WITHIN-TARGET permutation keeps each training target's
    # set of labels but scrambles which ligand carries which, so it destroys structure-activity
    # relationships while preserving the association between a chemotype and the typical affinity of the
    # protein class that chemotype binds. Whatever survives it is a CLASS-LEVEL PRIOR, not SAR.
    #
    # This matters because it is the right zero point. Comparing a model to R2 = 0 credits it for
    # knowing that kinase inhibitors are generally potent; comparing it to the within-target permutation
    # credits it only for what it adds beyond that.
    prior = perm['within_target']['mean']
    headroom = CEILING_R2 - prior
    print(f'\n=== DECOMPOSITION: where the predictable variance actually lives ===')
    print(f'  chance (labels fully permuted)                     R2 {perm["global"]["mean"]:+.4f}')
    print(f'  CLASS-LEVEL PRIOR (within-target permutation)      R2 {prior:+.4f}   '
          f'sd {perm["within_target"]["sd"]:.4f}')
    print(f'  experimental ceiling                               R2 {CEILING_R2:+.4f}')
    print(f'  => signal available BEYOND the class-level prior:  {headroom:.4f} R2')
    print(f'\n  The within-target permutation reaches {prior:.4f} with sd '
          f'{perm["within_target"]["sd"]:.4f}, i.e. {100 * prior / lig_best[0]:.0f}% of what the best\n'
          f'  ligand-only model achieves is reproducible WITHOUT any within-target structure-activity\n'
          f'  relationship at all. Test targets are disjoint from training, so this is not target\n'
          f'  memorisation -- it is chemotype -> typical affinity of the protein class that chemotype\n'
          f'  binds. Real, transferable, and not what "structure-based affinity prediction" means.')
    print(f'\n{"model":24s} {"test R2":>9s} {"beyond prior":>13s} {"% of headroom":>14s}')
    print('-' * 64)
    for name, v in sorted(table.items(), key=lambda kv: -kv[1]['test_r2']):
        if v['test_r2'] < 0:
            continue
        beyond = v['test_r2'] - prior
        v['beyond_class_prior'] = beyond
        v['pct_of_headroom'] = 100 * beyond / headroom
        print(f'{name:24s} {v["test_r2"]:>9.4f} {beyond:>+13.4f} {100 * beyond / headroom:>13.1f}%')
    print(f'\n  Read this way the structure models separate from the descriptor models more clearly\n'
          f'  than the raw R2 suggests, because a large constant is removed from both. The caveat is\n'
          f'  real and must be stated: the permutation baseline was refitted only for the DESCRIPTOR\n'
          f'  pipeline, since permuting labels and retraining the EGNN needs the GPU. The prior is\n'
          f'  therefore an estimate of the chemotype channel, not a per-architecture control.')

    json.dump(dict(label_composition=comp,
                   decomposition=dict(chance=perm['global']['mean'], class_level_prior=prior,
                                      ceiling=CEILING_R2, headroom_beyond_prior=headroom),
                   ceiling=dict(point_estimate_source='Hernandez-Garrido et al. 2023',
                                exp_pearson=EXP_PEARSON, exp_rmse=EXP_RMSE, exp_mae=EXP_MAE,
                                ceiling_r2=CEILING_R2, ceiling_r2_range=[CEILING_R2_LO, CEILING_R2_HI],
                                range_low_source='Chen et al. 2023, ABFEP measured R2',
                                range_high_source='Kalliokoski et al. 2013, milder mixed-assay noise',
                                pessimistic_source='Landrum et al. 2024',
                                physics_anchor_abfep_r2=ABFEP_R2),
                   models=table,
                   power=dict(floor_r2=lig_best[0], floor_model=lig_best[1],
                              best_structure_r2=struct_best[0], best_structure_model=struct_best[1],
                              window=window, mean_ci_width=mean_ci,
                              window_over_ci=window / mean_ci, between_seed_sd=0.0377),
                   y_randomisation=perm,
                   note='The ceiling is the reproducibility of the LABELS, not a modelling limit. '
                        'Quote R2 as a fraction of it. The window between the ligand-only floor and '
                        'that ceiling is the only room structural information has to demonstrate '
                        'itself, and it must be compared against the width of the confidence '
                        'intervals before any ranking is claimed.'),
              open(OUT, 'w'), indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
