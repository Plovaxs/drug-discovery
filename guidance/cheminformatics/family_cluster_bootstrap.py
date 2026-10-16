"""Audit item F037: does clustering the bootstrap by protein family, rather than by target,
widen the interval on the headline structure-versus-ligand-only comparison?

Every predictor comparison in this thesis resamples *targets*, on the argument that complexes
sharing a target are not independent. The audit asked the obvious next question: targets sharing a
protein family are not independent either. Two kinases are more alike than a kinase and a
protease, so if a family's worth of targets is resampled as a block, the interval should widen.
If it widens a great deal, every target-clustered interval in the thesis is optimistic and the
comparisons that were called ties are *more* securely ties, while any comparison called a win
becomes less secure.

This script answers that by running the identical paired comparison at three clustering levels:
the complex (no clustering, which is the naive analysis the thesis argues against), the target
(what the thesis actually reports), and the Pfam family. Only the grouping changes; the data,
the models, the statistic, the number of resamples and the seed are held fixed, so the three
intervals are directly comparable and any difference between them is attributable to the
clustering alone.

A caveat belongs with the result and is printed with it: Pfam annotation does not cover every
target, and unannotated targets have to be handled explicitly rather than silently dropped. They
are kept as singleton clusters, which is the conservative choice for this question, because
dropping them would change the sample as well as the clustering and the comparison would no
longer be like for like.

Run:  PYTHONPATH=. python guidance/cheminformatics/family_cluster_bootstrap.py
"""
import argparse
import json

import numpy as np

from guidance.cheminformatics.compare_vs_structure import LIGAND_PREDS, STRUCTURE_SOURCES, load_structure
from guidance.uncertainty_a1.conditional_conformal_coverage import family_by_target
from guidance.uncertainty_a1.stats_common import paired_r2_comparison

BOOT_SEED = 20260925
OUT = 'guidance/cheminformatics/family_cluster_bootstrap_results.json'


def groups_of(labels):
    """One row-index array per distinct label.

    `paired_r2_comparison` takes a list of row-index arrays, one per cluster, not a per-row code
    vector; passing the latter fails with a confusing concatenate error rather than a type error.
    """
    labels = np.asarray(labels)
    return [np.where(labels == u)[0] for u in np.unique(labels)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    pk, target = L['pk'].astype(np.float64), L['target'].astype(str)
    fam = np.asarray(family_by_target(target), dtype=object)

    structure = {}
    for name, (path, ykey, pkey) in STRUCTURE_SOURCES.items():
        p, _note = load_structure(name, path, ykey, pkey, pk, target)
        if p is not None:
            structure[name] = p
    champion = 'desc_ridge' if 'pred_desc_ridge' in L.files else \
        [k[5:] for k in L.files if k.startswith('pred_') and k != 'pred_global_mean'][0]
    s_main = 'egnn_a1c_ensemble' if 'egnn_a1c_ensemble' in structure else next(iter(structure))
    sp, lp = structure[s_main], np.asarray(L['pred_' + champion], dtype=np.float64)

    # Unannotated targets become their own cluster rather than one shared "unknown" block: lumping
    # them together would assert a relatedness the annotation does not support, and would make the
    # family interval look wider for a reason that is an artefact of the missing annotation.
    n_unannotated_targets = len({t for f, t in zip(fam, target) if f == 'no_annotation'})
    fam_eff = np.array([t if f == 'no_annotation' else f for f, t in zip(fam, target)], dtype=object)

    levels = {
        'complex (no clustering)': [np.array([i]) for i in range(len(pk))],
        'target (as reported in this thesis)': groups_of(target),
        'Pfam family': groups_of(fam_eff),
    }

    print(f'structure model: {s_main}   ligand-only champion: {champion}')
    print(f'rows: {len(pk)}   targets: {len(set(target))}   '
          f'family clusters: {len(set(fam_eff))} '
          f'({n_unannotated_targets} unannotated targets kept as singleton clusters)')
    print(f'\n=== F037: same comparison, three clustering levels (B={args.b}, seed={BOOT_SEED}) ===')
    hdr = f'{"clustering":38s} {"clusters":>9s} {"dR2":>8s} {"95% CI":>22s} {"width":>8s} {"p":>7s}'
    print(hdr); print('-' * len(hdr))

    out = {'structure_model': s_main, 'ligand_champion': champion, 'b': args.b,
           'seed': BOOT_SEED, 'n_rows': int(len(pk)), 'n_targets': len(set(target)),
           'n_unannotated_targets': n_unannotated_targets, 'levels': {}}
    widths = {}
    for label, g in levels.items():
        r = paired_r2_comparison(pk, sp, lp, g, args.b, BOOT_SEED, min_clusters=2)
        lo, hi = r['ci95'] if r['ci95'] else (float('nan'), float('nan'))
        width = hi - lo
        widths[label] = width
        print(f'{label:38s} {r["n_target_clusters"]:>9d} {r["d_r2"]:>8.4f} '
              f'[{lo:>8.4f}, {hi:>8.4f}] {width:>8.4f} {r["boot_p"]:>7.3f}')
        out['levels'][label] = dict(n_clusters=int(r['n_target_clusters']),
                                    structure_r2=float(r['r2_a']), ligand_r2=float(r['r2_b']),
                                    d_r2=float(r['d_r2']), ci95=[float(lo), float(hi)],
                                    ci_width=float(width), boot_p=float(r['boot_p']),
                                    underpowered=bool(r['underpowered']),
                                    degenerate=bool(r['degenerate']))

    tgt, famw = widths['target (as reported in this thesis)'], widths['Pfam family']
    cpx = widths['complex (no clustering)']
    out['width_ratio_family_over_target'] = float(famw / tgt)
    out['width_ratio_target_over_complex'] = float(tgt / cpx)
    print(f'\nfamily / target interval width: {famw / tgt:.2f}x')
    print(f'target / complex interval width: {tgt / cpx:.2f}x')

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
