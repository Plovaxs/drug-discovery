"""Stage 0 CI correction: the original rigor pass (stage0_checkpoint_analysis.py) bootstrapped Pearson/R^2
by resampling individual test COMPLEXES i.i.d. (rng.randint over all 11,855 rows). That treats complexes
from the same target as independent, which they are not (~93 complexes/target on average, sharing one
pocket/protein) -- so it understates the true variance and the reported CI is too narrow.

This script recomputes the same two statistics with the resampling unit changed to the TEST TARGET (127
clusters), matching the convention already pre-registered and used for A1/A1b/A1c (cluster bootstrap, B=2,000,
seed 20260925, imported directly from guidance.uncertainty_a1.mc_dropout_calibration so the seed/B cannot
drift from those three tests). No retraining or GPU needed: reuses the cached test predictions already saved
at guidance/track_e/_stage0_test_preds.npz for this exact checkpoint
(logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt).

Usage:
  python guidance/lp_split/stage0_cluster_bootstrap_ci.py
"""
import json

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from guidance.uncertainty_a1.mc_dropout_calibration import B, BOOT_SEED

CACHE = './guidance/track_e/_stage0_test_preds.npz'
OUT = './guidance/lp_split/stage0_cluster_bootstrap_ci_results.json'


def iid_bootstrap(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    """The original (incorrect-unit) bootstrap, kept for a direct before/after comparison in the thesis."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_bootstrap(y_true, y_pred, target, metric_fn):
    uniq, inv = np.unique(target, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[k] for k in pick])
        vals[b] = metric_fn(y_true[idx], y_pred[idx])
    point = metric_fn(y_true, y_pred)
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def main():
    z = np.load(CACHE, allow_pickle=True)
    y_true, y_pred, target = z['y_true'], z['y_pred'], z['tgt']
    n_targets = len(np.unique(target))
    print(f'n_test={len(y_true)}  n_targets={n_targets}  B={B}  boot_seed={BOOT_SEED}')

    pearson_fn = lambda a, b: pearsonr(a, b)[0]

    p_point, p_lo_c, p_hi_c, _ = cluster_bootstrap(y_true, y_pred, target, pearson_fn)
    r2_point, r2_lo_c, r2_hi_c, _ = cluster_bootstrap(y_true, y_pred, target, r2_score)
    p_lo_i, p_hi_i = iid_bootstrap(y_true, y_pred, pearson_fn)
    r2_lo_i, r2_hi_i = iid_bootstrap(y_true, y_pred, r2_score)

    print(f'Pearson = {p_point:.3f}  i.i.d.-per-complex CI [{p_lo_i:.3f}, {p_hi_i:.3f}] (width {p_hi_i - p_lo_i:.3f})  '
          f'cluster-per-target CI [{p_lo_c:.3f}, {p_hi_c:.3f}] (width {p_hi_c - p_lo_c:.3f})')
    print(f'R2      = {r2_point:.3f}  i.i.d.-per-complex CI [{r2_lo_i:.3f}, {r2_hi_i:.3f}] (width {r2_hi_i - r2_lo_i:.3f})  '
          f'cluster-per-target CI [{r2_lo_c:.3f}, {r2_hi_c:.3f}] (width {r2_hi_c - r2_lo_c:.3f})')

    results = dict(
        n_test=int(len(y_true)), n_targets=int(n_targets), B=B, boot_seed=int(BOOT_SEED),
        pearson=dict(point=float(p_point), ci95_iid_per_complex=[p_lo_i, p_hi_i], ci95_cluster_per_target=[p_lo_c, p_hi_c]),
        r2=dict(point=float(r2_point), ci95_iid_per_complex=[r2_lo_i, r2_hi_i], ci95_cluster_per_target=[r2_lo_c, r2_hi_c]),
    )
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
