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


def r2_rows(y, p, rows):
    """R2 on a row subset, against the SUBSET's own mean (not the global mean) -- the formulation used
    everywhere in this project a per-stratum R2 is reported (novelty_tiers.py's tiers, noise_ceiling.py's
    headroom, representation_ladder.py's blocks). Returns NaN, not an error, for a subset with literally
    zero variance (denom <= 0) -- see paired_r2_comparison below for the case this alone does not catch.

    Canonical home as of 2026-10-13: previously defined independently in
    guidance/cheminformatics/compare_vs_structure.py (byte-identical formula). That module's docstring
    claimed, as of 2026-10-13, that it had already been switched to import this definition; it had not
    -- it still carried its own copy, so nothing kept the two in sync, and a future fix made to one
    would not have reached the other. This was found and fixed on 2026-10-16: compare_vs_structure.py
    now imports r2_rows from here, and the five scripts that use it transitively through that module
    (label_harmonisation.py, scaffold_audit.py, representation_ladder.py, noise_ceiling.py and
    novelty_tiers.py) follow automatically. The two bodies were verified byte-identical before the
    change, so no previously reported number in any of those scripts' output moves. New callers should
    import from here, not redefine it again."""
    yy, pp = y[rows], p[rows]
    denom = ((yy - yy.mean()) ** 2).sum()
    if denom <= 0:
        return np.nan
    return 1.0 - ((yy - pp) ** 2).sum() / denom


def paired_r2_comparison(y, pred_a, pred_b, groups, b, seed, min_clusters=10,
                         degenerate_std_fraction=0.01):
    """dR2 = R2(pred_a) - R2(pred_b) on a stratum (groups' union), with a target-clustered bootstrap CI
    computed PAIRED: the SAME resampled rows feed both R2 terms in every draw, so the CI is about the
    DIFFERENCE, not an eyeballed comparison of two separate marginal CIs.

    Written after that exact mistake was made and caught twice in one session (both initially compared
    two independently-bootstrapped point estimates / marginal CIs instead of this): once in
    guidance/cheminformatics/matched_molecular_pairs.py (sign-agreement of two models, compared via two
    separate CIs before being fixed to a paired bootstrap on the identical matched pairs), and again,
    more plainly, in guidance/cheminformatics/activity_cliffs.py (`if champ_cliff > struct_cliff` with no
    significance test at all). Both call sites now reuse this function instead of hand-rolling the
    pattern a third and fourth time.

    Also guards a SECOND, independent bug found the same session
    (guidance/cheminformatics/protein_family_stratification.py): r2_rows' own `denom <= 0` guard only
    catches EXACT zero variance, and a stratum with nonzero-but-near-duplicate y values (measured case:
    variance 1.13e-16) survives it and blows up to values like +5.8e16, which is division by a number
    indistinguishable from zero, not a finding. Any stratum, or any individual bootstrap resample, whose
    local y-std falls below `degenerate_std_fraction` of y's GLOBAL std (computed over the full array
    passed as `y`, every time -- never just the stratum) is treated as degenerate: reported explicitly
    via the `degenerate` flag, never silently returned as a finite-looking number.

    `groups`: row-index arrays already restricted to the stratum under test, one array per target
    cluster (e.g. target_groups(target[rows]) with rows re-mapped to absolute indices, exactly as every
    existing caller already builds for cluster_bootstrap_ci). Kept as `groups` rather than
    `target`+`rows` so this module stays dependency-free of how any particular caller defines "target".

    Returns a dict: n, n_target_clusters, r2_a, r2_b, d_r2, ci95, boot_p, underpowered, degenerate,
    local_std. `underpowered` is True whenever no CI was computed, whether the cause was too few target
    clusters OR degenerate variance OR every bootstrap resample landing on a degenerate/non-finite
    subset -- `degenerate` and the `note` field (when present) distinguish which."""
    all_rows = np.concatenate(groups) if len(groups) else np.array([], dtype=int)
    n, n_clusters = len(all_rows), len(groups)
    degenerate_std = degenerate_std_fraction * float(y.std())
    local_std = float(y[all_rows].std()) if n else 0.0

    base = dict(n=int(n), n_target_clusters=int(n_clusters), r2_a=None, r2_b=None, d_r2=None,
               ci95=None, boot_p=None, underpowered=True, degenerate=False, local_std=local_std)
    if n == 0 or local_std < degenerate_std:
        base['degenerate'] = True
        return base

    r2_a, r2_b = r2_rows(y, pred_a, all_rows), r2_rows(y, pred_b, all_rows)
    base.update(r2_a=float(r2_a), r2_b=float(r2_b), d_r2=float(r2_a - r2_b))
    if n_clusters < min_clusters:
        return base

    boot = []
    for pick in cluster_bootstrap_indices(n_clusters, b, seed):
        dr = np.concatenate([groups[k] for k in pick])
        if float(y[dr].std()) < degenerate_std:
            continue
        v = r2_rows(y, pred_a, dr) - r2_rows(y, pred_b, dr)
        if np.isfinite(v):
            boot.append(v)
    if not boot:
        base['note'] = 'every bootstrap resample was degenerate or non-finite'
        return base

    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    pval = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    pval = min(1.0, max(pval, 1.0 / (len(boot) + 1)))
    base.update(ci95=[float(lo), float(hi)], boot_p=float(pval), underpowered=False)
    return base
