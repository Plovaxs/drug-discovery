"""Pins this project's statistical machinery against independently written references.

Every conclusion in this thesis is an interval or a corrected p-value, so a quiet error in one of these
functions would not crash anything -- it would change what the thesis claims. The references here are
written from each method's definition rather than imported from a library, so an algebra slip in the
implementation cannot be mirrored by an identical slip in the check. (statsmodels is not in this
environment, which forced the hand-written BH reference; that turned out to be the better test.)

The specific things being guarded, each because getting it wrong is both easy and invisible:
  * BH step-up over TIES -- the common implementation error, and this project runs BH over 40 paired
    comparisons where three p-values sat at 0.039-0.049.
  * sd with ddof=1 and an `undefined` at n=1 -- reporting 0.0000 there reads as perfect reproducibility,
    and this project twice drew a wrong conclusion from too few seeds.
  * the t critical value rather than 1.96 -- at n=3, t(df=2) = 4.303, so using z would understate every
    interval by more than half.
  * the conformal order statistic ceil((n+1)(1-alpha)) -- off by one and coverage is no longer valid,
    which matters because split-conformal is the ONLY uncertainty method that passed A1.
  * cluster-bootstrap invariants -- resampling targets rather than complexes is the project's standing
    rigor rule; the uniform-group case must preserve n exactly.
  * the AVE sign convention -- the first version of ave_bias.py had it inverted and reported -1.04 for a
    random split, i.e. the right magnitude with the wrong sign.

Usage:
  PYTHONPATH=. python guidance/tests/test_statistics_against_references.py
"""
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.aggregate_results import mde, summarize
from guidance.cheminformatics.ave_bias import ave, cdf_auc
from guidance.uncertainty_a1.conformal_prediction import conformal_quantile
from guidance.uncertainty_a1.stats_common import (bh, cluster_bootstrap_ci,
                                                  cluster_bootstrap_indices, partial_spearman)


def bh_reference(p):
    """Benjamini-Hochberg step-up written from the definition: sort ascending, then
    q_i = min over j >= i of (n/j) * p_j, clipped to 1 and restored to the original order."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ps = p[order]
    q = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, n * ps[i] / (i + 1))
        q[i] = running
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def test_bh_matches_an_independent_reference_over_random_cases():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = rng.uniform(0, 1, int(rng.integers(2, 40)))
        worst = max(worst, float(np.abs(np.array(bh(list(p))) - bh_reference(p)).max()))
    assert worst < 1e-12, f'bh() diverges from the reference by {worst:.2e}'


def test_bh_is_correct_on_ties():
    """Ties are where step-up implementations usually break, and this project's 40-pair correction had
    three p-values clustered at 0.039-0.049."""
    for case in ([0.01, 0.02, 0.02, 0.05], [0.04, 0.04, 0.04], [0.5] * 6, [0.001, 0.001, 0.9, 0.9]):
        assert np.abs(np.array(bh(case)) - bh_reference(case)).max() < 1e-12, case


def test_bh_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = np.sort(rng.uniform(0, 1, 20))
        q = np.array(bh(list(p)))
        assert (q >= 0).all() and (q <= 1).all()
        assert (np.diff(q) >= -1e-12).all(), 'adjusted p-values must be non-decreasing in sorted p'
        assert (q >= p - 1e-12).all(), 'adjustment can only increase a p-value'


def test_summarize_uses_sample_sd_not_population():
    v = [0.30, 0.34, 0.39]
    s = summarize(v)
    assert abs(s['sd'] - float(np.std(v, ddof=1))) < 1e-12
    assert abs(s['sd'] - float(np.std(v, ddof=0))) > 1e-6, 'ddof=0 would understate between-seed spread'


def test_summarize_reports_sd_undefined_at_n1():
    """0.0000 there would read as perfect reproducibility. This project reported 'QAT matches FP32' off
    one seed and was wrong."""
    s = summarize([0.3])
    assert s['sd'] is None and s['n'] == 1
    assert 'ci95' not in s


def test_summarize_ci_uses_t_not_z():
    v = [0.30, 0.34, 0.39]
    s = summarize(v)
    t = float(st.t.ppf(0.975, len(v) - 1))
    assert abs(s['ci95'][1] - (s['mean'] + t * s['sem'])) < 1e-12
    assert abs(t - 1.96) > 2.0, 'at n=3, t(df=2)=4.303; using z would halve every interval'


def test_mde_matches_its_formula_and_refuses_impossible_input():
    m = mde(0.05, 3)
    ta, tb = float(st.t.ppf(0.975, 4)), float(st.t.ppf(0.80, 4))
    assert abs(m - (ta + tb) * 0.05 * np.sqrt(2 / 3)) < 1e-12
    assert mde(None, 3) is None and mde(0.05, 1) is None


def test_cluster_bootstrap_preserves_n_for_uniform_groups():
    g = [np.arange(i * 5, (i + 1) * 5) for i in range(8)]
    point, _ci, boot = cluster_bootstrap_ci(lambda idx: len(idx), g, b=500, seed=1)
    assert point == 40
    assert boot.min() == 40 and boot.max() == 40, 'equal-sized groups must resample to the same total'


def test_cluster_bootstrap_varies_for_unequal_groups():
    g = [np.arange(0, 2), np.arange(2, 20)]       # sizes 2 and 18
    _p, _ci, boot = cluster_bootstrap_ci(lambda idx: len(idx), g, b=400, seed=1)
    assert boot.min() == 4 and boot.max() == 36, (boot.min(), boot.max())


def test_cluster_bootstrap_indices_are_deterministic_per_seed():
    a = [x.tolist() for x in cluster_bootstrap_indices(5, 3, 42)]
    assert a == [x.tolist() for x in cluster_bootstrap_indices(5, 3, 42)]
    assert a != [x.tolist() for x in cluster_bootstrap_indices(5, 3, 43)]


def test_cluster_bootstrap_indices_stay_in_range():
    for gi in cluster_bootstrap_indices(7, 50, 3):
        assert gi.min() >= 0 and gi.max() < 7 and len(gi) == 7


def test_conformal_quantile_is_the_right_order_statistic():
    """Off by one and the coverage guarantee is gone. Split-conformal is the only uncertainty method
    that passed A1, so this one must be exact."""
    for n, alpha in [(100, 0.1), (19, 0.05), (9, 0.1), (20, 0.05), (1000, 0.05)]:
        scores = np.arange(1.0, n + 1.0)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        assert conformal_quantile(scores, alpha) == float(k), (n, alpha)


def test_conformal_quantile_is_monotone_in_alpha():
    scores = np.arange(1.0, 201.0)
    qs = [conformal_quantile(scores, a) for a in (0.2, 0.1, 0.05, 0.01)]
    assert all(b >= a for a, b in zip(qs, qs[1:])), 'a smaller alpha must not give a smaller quantile'


def test_cdf_auc_equals_mean_similarity():
    """The AVE proximity term is the area under the survival curve, which is the mean. Getting the
    orientation wrong inverted the whole statistic once already."""
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = rng.uniform(0, 1, 20000)
        assert abs(cdf_auc(x) - float(x.mean())) < 3e-3


def test_ave_sign_convention():
    assert abs(ave(0.5, 0.5, 0.5, 0.5)) < 1e-12, 'symmetric case must be exactly zero'
    assert ave(0.9, 0.3, 0.9, 0.3) > 0, 'closer to own class => positive (biased split)'
    assert ave(0.3, 0.9, 0.3, 0.9) < 0, 'closer to the other class => negative (adversarial)'


def test_partial_spearman_is_symmetric_in_its_two_variables():
    rng = np.random.default_rng(7)
    x, y, z = (rng.normal(size=300) for _ in range(3))
    assert abs(partial_spearman(x, y, z) - partial_spearman(y, x, z)) < 1e-10


def test_partial_spearman_removes_a_pure_confounder():
    """If x and y are related only through z, the partial correlation should collapse toward zero while
    the raw correlation stays high."""
    rng = np.random.default_rng(11)
    z = rng.normal(size=4000)
    x = z + 0.05 * rng.normal(size=4000)
    y = z + 0.05 * rng.normal(size=4000)
    raw = float(st.spearmanr(x, y).statistic)
    partial = partial_spearman(x, y, z)
    assert raw > 0.9, raw
    assert abs(partial) < 0.3, f'confounder not removed: partial={partial:.3f} vs raw={raw:.3f}'


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
