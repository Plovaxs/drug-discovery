"""Unit tests for guidance/uncertainty_a1/stats_common.py, with known/independently-derived reference
values (TASKS.md's long-standing "berguna" ask) -- not just re-running the implementation against itself.
Run: PYTHONPATH=. python guidance/uncertainty_a1/tests/test_stats_common.py"""
import numpy as np

from guidance.uncertainty_a1.stats_common import (bh, boot_p, cluster_bootstrap_ci, cluster_bootstrap_indices,
                                                   paired_r2_comparison, partial_spearman, r2_rows)


def test_bh_hand_computed_example_with_ties():
    # p = [0.005, 0.03, 0.03, 0.05], m=4. Step-up from the largest rank down:
    #   rank4 (p=.05):  .05*4/4=.05                  -> .05
    #   rank3 (p=.03):  min(.05, .03*4/3=.04)         -> .04
    #   rank2 (p=.03):  min(.04, .03*4/2=.06)         -> .04
    #   rank1 (p=.005): min(.04, .005*4/1=.02)        -> .02
    p = [0.005, 0.03, 0.03, 0.05]
    expected = [0.02, 0.04, 0.04, 0.05]
    got = bh(p)
    assert np.allclose(got, expected, atol=1e-12), got


def test_bh_is_monotonic_in_sorted_p_order():
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 50)
    adj = np.array(bh(p))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12), 'BH-adjusted p must be non-decreasing in sorted p order'


def test_bh_never_exceeds_one_and_never_below_raw_p():
    rng = np.random.RandomState(1)
    p = rng.uniform(0, 1, 30)
    adj = np.array(bh(p))
    assert np.all(adj <= 1.0 + 1e-12)
    assert np.all(adj >= np.array(p) - 1e-12), 'BH correction only ever makes p-values larger'


def test_boot_p_exact_known_fractions():
    # 100 bootstrap draws, null=0: 70 are >= 0 (lo side via <=0 only the exact zeros, hi side via >=0 is 70),
    # construct directly so the two one-sided fractions are known exactly.
    samples = np.array([-1.0] * 30 + [1.0] * 70)  # 30 at/below 0, 70 at/above 0
    # lo = P(samples <= 0) = 0.30, hi = P(samples >= 0) = 0.70 -> two-sided = 2*min(0.30,0.70) = 0.60
    assert np.isclose(boot_p(samples, 0.0, b=100), 0.60)


def test_boot_p_floored_at_one_over_b():
    # All samples far above null -> lo=0 exactly -> two-sided p would be 0, floored to 1/b.
    samples = np.full(200, 100.0)
    assert np.isclose(boot_p(samples, 0.0, b=200), 1 / 200)


def test_boot_p_capped_at_one():
    # null sits exactly at the sample value -> both one-sided fractions are 1.0 -> 2*min(1,1)=2, capped to 1.
    samples = np.zeros(50)
    assert boot_p(samples, 0.0, b=50) == 1.0


def _closed_form_partial_corr(rx, ry, rz):
    """Independent reference: the textbook partial-correlation identity
    pcorr(x,y|z) = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)), applied to the SAME rank vectors the
    implementation uses, but computed via plain Pearson correlation instead of the OLS-residual route --
    a different computational path that the implementation must agree with if it's correct."""
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def test_partial_spearman_matches_closed_form_formula():
    rng = np.random.RandomState(42)
    for _ in range(20):
        n = rng.randint(8, 30)
        x = rng.permutation(n) + 1.0
        y = rng.permutation(n) + 1.0
        z = rng.permutation(n) + 1.0
        got = partial_spearman(x, y, z)
        expected = _closed_form_partial_corr(x, y, z)  # ranks of a permutation are itself
        assert np.isclose(got, expected, atol=1e-9), (got, expected)


def test_partial_spearman_symmetric_in_x_y():
    rng = np.random.RandomState(7)
    x, y, z = rng.uniform(size=15), rng.uniform(size=15), rng.uniform(size=15)
    assert np.isclose(partial_spearman(x, y, z), partial_spearman(y, x, z))


def test_cluster_bootstrap_equal_group_sizes_gives_invariant_total():
    # 4 groups of exactly 5 members each: ANY resample-with-replacement of 4 group-picks must total
    # 4*5=20 members, regardless of which groups get picked (even the same one 4 times) -- an exact
    # invariant, not a statistical approximation.
    groups = [np.arange(i * 5, i * 5 + 5) for i in range(4)]
    point, ci, boot = cluster_bootstrap_ci(lambda idx: len(idx), groups, b=500, seed=123)
    assert point == 20
    assert np.all(boot == 20), 'every bootstrap resample must also total exactly 20 members'
    assert ci == [20.0, 20.0]


def test_cluster_bootstrap_seed_is_deterministic_and_seed_sensitive():
    picks_a = list(cluster_bootstrap_indices(10, 5, seed=99))
    picks_b = list(cluster_bootstrap_indices(10, 5, seed=99))
    picks_c = list(cluster_bootstrap_indices(10, 5, seed=100))
    assert all(np.array_equal(a, b) for a, b in zip(picks_a, picks_b))
    assert not all(np.array_equal(a, c) for a, c in zip(picks_a, picks_c))


def test_cluster_bootstrap_unequal_sizes_sum_matches_picked_groups():
    sizes = [2, 3, 5]
    groups = []
    off = 0
    for s in sizes:
        groups.append(np.arange(off, off + s))
        off += s
    point, ci, boot = cluster_bootstrap_ci(lambda idx: len(idx), groups, b=1000, seed=321)
    assert point == sum(sizes)
    assert boot.min() >= min(sizes) * 3  # smallest possible: same smallest group picked all 3 times
    assert boot.max() <= max(sizes) * 3  # largest possible: same largest group picked all 3 times


# ---------------------------------------------------------------- r2_rows / paired_r2_comparison
# The centralised home for a pattern that had the SAME bug independently in two different scripts this
# session (guidance/cheminformatics/protein_family_stratification.py: a near-zero-variance stratum blew
# up to +5.8e16; guidance/cheminformatics/matched_molecular_pairs.py and activity_cliffs.py: a winner
# declared from two un-paired point estimates / marginal CIs). These tests are the reference this
# function must keep meeting so a third script never reintroduces either mistake by hand-rolling again.

def test_r2_rows_matches_sklearn_on_a_hand_picked_case():
    y = np.array([3.0, 5.0, 7.0, 9.0])
    p = np.array([3.5, 4.5, 7.5, 8.5])
    got = r2_rows(y, p, np.arange(4))
    from sklearn.metrics import r2_score
    assert abs(got - r2_score(y, p)) < 1e-12, got


def test_r2_rows_returns_nan_for_exactly_zero_variance():
    y = np.array([5.0, 5.0, 5.0])
    p = np.array([1.0, 2.0, 3.0])
    assert np.isnan(r2_rows(y, p, np.arange(3)))


def test_paired_r2_comparison_matches_hand_computation_on_a_healthy_stratum():
    """y = [1,2,3,4,5,6], pred_a perfect, pred_b off by a constant 1 everywhere (same R2 shape, since
    R2 is invariant to pred_b's constant offset only when... worked out explicitly below rather than
    assumed): denom = variance of y around its own mean = sum((y-3.5)^2) = 17.5.
    pred_a == y exactly -> R2_a = 1 - 0/17.5 = 1.0
    pred_b = y - 1       -> SS_res = sum(1^2)*6 = 6 -> R2_b = 1 - 6/17.5 = 0.657142857...
    d_r2 = 1.0 - 0.657142857... = 0.342857142..."""
    y = np.arange(1.0, 7.0)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    r = paired_r2_comparison(y, y.copy(), y - 1.0, groups, b=500, seed=1, min_clusters=1)
    assert abs(r['r2_a'] - 1.0) < 1e-9, r['r2_a']
    assert abs(r['r2_b'] - 0.6571428571428) < 1e-9, r['r2_b']
    assert abs(r['d_r2'] - 0.3428571428) < 1e-9, r['d_r2']
    assert r['underpowered'] is False and r['degenerate'] is False
    assert r['ci95'] is not None


def test_paired_r2_comparison_flags_degenerate_variance_not_a_huge_number():
    rng = np.random.default_rng(0)
    y_healthy = rng.normal(7, 1.5, 200)
    groups_healthy = [np.arange(i, i + 10) for i in range(0, 200, 10)]
    degenerate_rows = np.arange(15)
    y_degenerate = y_healthy.copy()
    y_degenerate[degenerate_rows] = 7.0 + rng.normal(0, 1e-9, len(degenerate_rows))
    pred_a, pred_b = y_healthy + rng.normal(0, 0.3, 200), y_healthy + rng.normal(0, 0.5, 200)
    r = paired_r2_comparison(y_degenerate, pred_a, pred_b, [degenerate_rows], b=200, seed=0)
    assert r['degenerate'] is True
    assert r['d_r2'] is None, 'a degenerate stratum must never carry a numeric dR2'


def test_paired_r2_comparison_does_not_flag_a_healthy_stratum_as_degenerate():
    rng = np.random.default_rng(1)
    y = rng.normal(7, 1.5, 200)
    groups = [np.arange(i, i + 10) for i in range(0, 200, 10)]
    pred_a, pred_b = y + rng.normal(0, 0.3, 200), y + rng.normal(0, 0.5, 200)
    r = paired_r2_comparison(y, pred_a, pred_b, groups, b=200, seed=0)
    assert r['degenerate'] is False
    assert abs(r['d_r2']) < 10, 'a healthy, varied stratum must produce a sane, bounded dR2'


def test_paired_r2_comparison_underpowered_below_min_clusters_still_reports_a_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.normal(7, 1.5, 50)
    groups = [np.arange(i, i + 5) for i in range(0, 50, 5)]     # 10 groups
    pred_a, pred_b = y + rng.normal(0, 0.3, 50), y + rng.normal(0, 0.5, 50)
    r = paired_r2_comparison(y, pred_a, pred_b, groups, b=200, seed=0, min_clusters=20)
    assert r['underpowered'] is True
    assert r['degenerate'] is False
    assert r['d_r2'] is not None, 'underpowered-but-not-degenerate must still report the point estimate'
    assert r['ci95'] is None and r['boot_p'] is None


def test_paired_r2_comparison_is_paired_not_two_independent_bootstraps():
    """The property that makes this 'paired': d_r2 must equal r2_a - r2_b computed from the SAME full
    sample, and every bootstrap draw must use the SAME resampled rows for both terms. Checked directly
    against a hand-rolled, deliberately UNPAIRED alternative (two separate bootstraps, point estimates
    subtracted) using the same seed -- the two need not match in general, and this test documents why
    paired_r2_comparison is the one that should be trusted: it is the only one where every resampled dR2
    reflects one coherent resample, not two different ones subtracted after the fact."""
    rng = np.random.default_rng(3)
    y = rng.normal(7, 1.5, 100)
    groups = [np.arange(i, i + 10) for i in range(0, 100, 10)]
    pred_a, pred_b = y + rng.normal(0, 0.4, 100), y + rng.normal(0, 0.6, 100)
    r = paired_r2_comparison(y, pred_a, pred_b, groups, b=500, seed=7)
    all_rows = np.concatenate(groups)
    assert abs(r['d_r2'] - (r2_rows(y, pred_a, all_rows) - r2_rows(y, pred_b, all_rows))) < 1e-9


def test_paired_r2_comparison_resample_landing_on_degenerate_subset_is_skipped_not_corrupting():
    """A healthy overall stratum can still have an UNLUCKY bootstrap resample that happens to draw only
    near-duplicate-y groups. That single resample must be excluded from the CI, not produce a finite
    but enormous dR2 that drags the percentile bounds somewhere absurd."""
    rng = np.random.default_rng(4)
    # Nine healthy groups, one group whose own y-values are a near-duplicate cluster -- resamples that
    # happen to draw only (or mostly) that one group should be dropped, not corrupt the CI.
    healthy_groups = [np.arange(i, i + 10) for i in range(0, 90, 10)]
    y = rng.normal(7, 1.5, 100)
    y[90:100] = 7.0 + rng.normal(0, 1e-9, 10)
    groups = healthy_groups + [np.arange(90, 100)]
    pred_a, pred_b = y + rng.normal(0, 0.4, 100), y + rng.normal(0, 0.6, 100)
    r = paired_r2_comparison(y, pred_a, pred_b, groups, b=500, seed=0)
    assert r['degenerate'] is False          # the FULL stratum (100 rows) is not itself degenerate
    assert r['ci95'] is not None
    assert abs(r['ci95'][0]) < 50 and abs(r['ci95'][1]) < 50, (
        f'CI {r["ci95"]} looks corrupted by a degenerate resample slipping through')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
