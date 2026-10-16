"""Audit item F065: is the residual variance constant, and does R-squared hide a subrange where
the models are useless?

R-squared is a single number summarising a whole error distribution, and it is the number every
predictor comparison in this thesis turns on. It is only a fair summary if the error behaves
roughly the same way across the range of the label. If the residual spread grows with affinity,
or if the models track the middle of the range and fail at the extremes, then two things follow:
R-squared flatters whichever model is better in the densest part of the range, and a comparison
that comes out a tie overall could still hide a subrange where one model is clearly better.

The audit measures three things on the leakage-safe test set, for the structure-aware ensemble and
for the ligand-only ridge alike:

1. Residual standard deviation by decile of the true label, and the ratio of the widest to the
   narrowest decile. A ratio near 1 means homoscedastic.
2. Spearman correlation computed *within* each decile. R-squared was tried here first and is
   useless at this resolution: inside a narrow slice of the label the denominator is the variance
   within the slice, which is tiny, and the values come out between -4 and -119 for every model,
   which says more about the slicing than about the models. A rank correlation has no such
   denominator and answers the question actually being asked, which is whether a model still
   orders molecules correctly inside a narrow affinity band.
3. A Breusch-Pagan style check: the squared residual regressed on the prediction, reported as the
   correlation between them, with a target-clustered bootstrap interval so that it is tested at
   the same resampling unit as every other comparison in the thesis.

Run:  PYTHONPATH=. python guidance/cheminformatics/residual_variance_audit.py
"""
import argparse
import json

import numpy as np

from guidance.cheminformatics.compare_vs_structure import LIGAND_PREDS, STRUCTURE_SOURCES, load_structure
from scipy.stats import spearmanr

from guidance.uncertainty_a1.stats_common import cluster_bootstrap_ci

BOOT_SEED = 20260925
OUT = 'guidance/cheminformatics/residual_variance_audit_results.json'
N_BINS = 10


def r2(y, p):
    denom = float(((y - y.mean()) ** 2).sum())
    if denom <= 0:
        return float('nan')
    return 1.0 - float(((y - p) ** 2).sum()) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    pk, target = L['pk'].astype(np.float64), L['target'].astype(str)

    structure = {}
    for name, (path, ykey, pkey) in STRUCTURE_SOURCES.items():
        p, _note = load_structure(name, path, ykey, pkey, pk, target)
        if p is not None:
            structure[name] = p
    s_main = 'egnn_a1c_ensemble' if 'egnn_a1c_ensemble' in structure else next(iter(structure))
    models = {
        'structure (EGNN ensemble)': structure[s_main],
        'ligand-only (14-descriptor ridge)': np.asarray(L['pred_desc_ridge'], dtype=np.float64),
        'heavy-atom count alone': np.asarray(L['pred_heavy_atoms'], dtype=np.float64),
    }

    groups = [np.where(target == t)[0] for t in np.unique(target)]
    edges = np.quantile(pk, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    binid = np.clip(np.digitize(pk, edges) - 1, 0, N_BINS - 1)

    out = {'structure_model': s_main, 'n_rows': int(len(pk)), 'n_targets': len(groups),
           'n_bins': N_BINS, 'b': args.b, 'seed': BOOT_SEED,
           'bin_edges': [float(e) for e in edges], 'models': {}}

    print(f'rows: {len(pk)}   targets: {len(groups)}   structure model: {s_main}')
    for name, pred in models.items():
        resid = pk - pred
        print(f'\n=== {name} ===')
        print(f'  overall R2 = {r2(pk, pred):.4f}   residual SD = {resid.std():.4f}')
        hdr = f'  {"decile":>7s} {"n":>6s} {"pK range":>16s} {"resid SD":>9s} {"Spearman":>10s}'
        print(hdr); print('  ' + '-' * (len(hdr) - 2))
        bins = []
        for b in range(N_BINS):
            m = binid == b
            sd = float(resid[m].std())
            rho = float(spearmanr(pk[m], pred[m]).statistic)
            bins.append(dict(bin=b, n=int(m.sum()), lo=float(edges[b]), hi=float(edges[b + 1]),
                             resid_sd=sd, spearman_in_bin=rho, mean_resid=float(resid[m].mean())))
            print(f'  {b + 1:>7d} {int(m.sum()):>6d} {edges[b]:>7.2f}–{edges[b + 1]:<8.2f} '
                  f'{sd:>9.4f} {rho:>10.4f}')
        sds = np.array([x['resid_sd'] for x in bins])
        ratio = float(sds.max() / sds.min())

        # Breusch-Pagan style: does the squared residual track the prediction? Tested at the same
        # resampling unit as every other comparison in the thesis, not per complex.
        def stat(rows):
            rr, pp = resid[rows] ** 2, pred[rows]
            if rr.std() == 0 or pp.std() == 0:
                return float('nan')
            return float(np.corrcoef(rr, pp)[0, 1])

        corr, ci, _boot = cluster_bootstrap_ci(stat, groups, args.b, BOOT_SEED)
        excludes_zero = ci is not None and (ci[0] > 0 or ci[1] < 0)
        print(f'  widest/narrowest decile residual SD: {ratio:.2f}x')
        print(f'  corr(residual^2, prediction) = {corr:+.4f}  '
              f'95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  '
              f'{"excludes zero" if excludes_zero else "includes zero"}')
        out['models'][name] = dict(overall_r2=float(r2(pk, pred)),
                                   residual_sd=float(resid.std()), bins=bins,
                                   sd_ratio_max_over_min=ratio,
                                   corr_sq_resid_vs_pred=float(corr),
                                   corr_ci95=[float(ci[0]), float(ci[1])],
                                   ci_excludes_zero=bool(excludes_zero))

    # Does the ranking of the two main models flip inside any decile?
    s_bins = out['models']['structure (EGNN ensemble)']['bins']
    l_bins = out['models']['ligand-only (14-descriptor ridge)']['bins']
    s_better_overall = (out['models']['structure (EGNN ensemble)']['overall_r2']
                        > out['models']['ligand-only (14-descriptor ridge)']['overall_r2'])
    flips = [b + 1 for b in range(N_BINS)
             if (s_bins[b]['spearman_in_bin'] > l_bins[b]['spearman_in_bin']) != s_better_overall]
    out['deciles_where_ranking_flips'] = flips
    out['n_deciles_where_ranking_flips'] = len(flips)
    print(f'\ndeciles where the structure/ligand ranking flips against the overall ordering: '
          f'{len(flips)} of {N_BINS} {flips if flips else ""}')

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
