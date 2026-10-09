"""Unit tests for guidance/uncertainty_a1/stats_common.py, with known/independently-derived reference
values (TASKS.md's long-standing "berguna" ask) -- not just re-running the implementation against itself.
Run: PYTHONPATH=. python guidance/uncertainty_a1/tests/test_stats_common.py"""
import numpy as np

from guidance.uncertainty_a1.stats_common import bh, boot_p, cluster_bootstrap_ci, cluster_bootstrap_indices, partial_spearman


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


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
