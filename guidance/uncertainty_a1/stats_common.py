"""Shared statistical helpers for the A1-A1g uncertainty-quantification suite (TASKS.md's long-standing
"berguna" item: these were duplicated/imported ad hoc across mc_dropout_calibration.py,
conformal_prediction.py, analyze_evidential_corrected.py, and guidance/track_e/analyze_track_e.py).

mc_dropout_calibration.py now imports bh/boot_p/partial_spearman from here instead of defining its own
copies, with the exact same signatures and behavior (verified by tests/test_stats_common.py against the
original inline implementations) -- so every script that does
`from guidance.uncertainty_a1 import mc_dropout_calibration as a1` and uses a1.bh/a1.boot_p keeps working
unchanged. conformal_prediction.py and analyze_evidential_corrected.py's own inline cluster-bootstrap loops
are intentionally left as-is (not retrofitted to call cluster_bootstrap_indices below) -- their numeric
outputs are already reported as final results in TASKS.md, and swapping their internals for a "cleaner"
shared call with no rerun to verify byte-for-byte equivalence would risk silently changing already-reported
numbers for a refactor-only benefit. New analyses should use this module directly.
"""
import numpy as np
from scipy.stats import rankdata

# Re-exported, not redefined: guidance/track_e/analyze_track_e.py's bh() is the one pre-existing canonical
# copy (already used by mc_dropout_calibration.py and by every Track E analysis) -- duplicating it here
# would recreate exactly the "same helper defined twice" problem this module exists to fix.
from guidance.track_e.analyze_track_e import bh  # noqa: F401


def boot_p(samples, null, b=None):
    """Two-sided bootstrap percentile p-value for a statistic's null value, floored at 1/B so p is never
    reported as exactly 0 from a finite bootstrap. b defaults to len(samples) if not given."""
    b = len(samples) if b is None else b
    lo, hi = np.mean(samples <= null), np.mean(samples >= null)
    return max(min(1.0, 2 * min(lo, hi)), 1 / b)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after linearly regressing out the rank of z from both ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def cluster_bootstrap_indices(n_groups, b, seed):
    """Yields B arrays of group indices (sampled with replacement, 0..n_groups-1) -- the resampling step
    shared by every cluster-bootstrap test in A1-A1g (resample TARGET clusters, not individual complexes,
    per this project's standing rigor rule that complexes within one target are not independent)."""
    rng = np.random.default_rng(seed)
    for _ in range(b):
        yield rng.integers(0, n_groups, n_groups)


def cluster_bootstrap_ci(stat_fn, groups, b, seed, levels=(2.5, 97.5)):
    """stat_fn(idx) -> float, evaluated on np.concatenate of the picked groups' member-index arrays for
    each of B resamples. Returns (point_estimate, [percentile_lo, percentile_hi], boot_samples)."""
    all_idx = np.concatenate(groups)
    point = stat_fn(all_idx)
    boot = np.empty(b)
    for i, pick in enumerate(cluster_bootstrap_indices(len(groups), b, seed)):
        boot[i] = stat_fn(np.concatenate([groups[k] for k in pick]))
    ci = [float(np.percentile(boot, levels[0])), float(np.percentile(boot, levels[1]))]
    return float(point), ci, boot
