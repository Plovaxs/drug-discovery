"""Characterise the subgroup that Track E's M2 metric is computed on.

M2 (within-target mean Spearman) needs enough complexes per target for a rank correlation to mean
anything, so it is restricted to test targets with at least 10 complexes: 73 of 127. A metric reported
on 57% of the targets invites the question of what the excluded 54 would have contributed, and the
thesis answers it with the numbers this script produces rather than asserting representativeness.

The finding is that the subgroup is NOT a random sample of the test set: excluded targets are small by
construction, but they are also systematically lower-affinity, which means M2 is measured on the
better-populated and higher-affinity part of the data.

Run:  PYTHONPATH=. python guidance/track_e/m2_subgroup_characterisation.py
Cheap: reads the cached anchor table, no training, no GPU.
"""
import argparse
import json

import numpy as np

from guidance.track_e.core import build_anchor_table, split_arrays

MIN_COMPLEXES = 10


def characterise(min_complexes=MIN_COMPLEXES):
    anchor = build_anchor_table()
    test = split_arrays(anchor, 'test')
    target = np.asarray(test['target'])
    pk = np.asarray(test['pk'], dtype=float)

    uniq, counts = np.unique(target, return_counts=True)
    keep_mask = counts >= min_complexes
    included = set(uniq[keep_mask])
    row_in = np.array([t in included for t in target])

    return {
        'min_complexes_threshold': int(min_complexes),
        'n_targets_total': int(len(uniq)),
        'n_targets_included': int(keep_mask.sum()),
        'n_targets_excluded': int((~keep_mask).sum()),
        'n_complexes_total': int(counts.sum()),
        'n_complexes_included': int(counts[keep_mask].sum()),
        'pct_complexes_covered': round(100.0 * counts[keep_mask].sum() / counts.sum(), 2),
        'median_complexes_included': float(np.median(counts[keep_mask])),
        'median_complexes_excluded': float(np.median(counts[~keep_mask])),
        'pk_mean_included': round(float(pk[row_in].mean()), 3),
        'pk_mean_excluded': round(float(pk[~row_in].mean()), 3),
        'pk_sd_included': round(float(pk[row_in].std()), 3),
        'pk_sd_excluded': round(float(pk[~row_in].std()), 3),
        'pk_mean_difference': round(float(pk[row_in].mean() - pk[~row_in].mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_complexes', type=int, default=MIN_COMPLEXES)
    ap.add_argument('--out', type=str, default='./guidance/track_e/m2_subgroup_characterisation.json')
    args = ap.parse_args()

    res = characterise(args.min_complexes)
    print(json.dumps(res, indent=2))
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print('wrote', args.out)

    # The interpretation the thesis draws, stated here so the script and the text cannot drift apart.
    print(f"\nM2 covers {res['pct_complexes_covered']}% of test complexes across "
          f"{res['n_targets_included']}/{res['n_targets_total']} targets. Excluded targets are smaller "
          f"(median {res['median_complexes_excluded']:.0f} vs {res['median_complexes_included']:.0f} "
          f"complexes) AND lower-affinity (mean pK {res['pk_mean_excluded']} vs "
          f"{res['pk_mean_included']}, difference {res['pk_mean_difference']}).")


if __name__ == '__main__':
    main()
