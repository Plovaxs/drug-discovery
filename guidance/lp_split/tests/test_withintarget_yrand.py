"""Tests for guidance/lp_split/train_egnn_stage0_withintarget_yrand.py::permute_pk_within_target --
the ONE function this training-script variant adds relative to train_egnn_stage0.py. The training loop
itself is untouched copy-paste (same as every other train_egnn_stage0_*.py variant in this project) and
is not re-tested here; what is new, and what must be exactly right, is the permutation.

Three properties matter, in order of how badly getting them wrong would corrupt the result:
  1. Val/test pK must be returned UNCHANGED. If this failed, the reported test R2 would not be measuring
     "trained on permuted labels, evaluated on real ones" -- it would be measuring something else
     entirely, silently.
  2. The permutation must be WITHIN target: the multiset of pK values a target owns must be identical
     before and after, only the assignment to ligands changes. If this failed (e.g. a global shuffle
     leaked in), the experiment would no longer isolate "class-level prior" from genuine SAR signal.
  3. Determinism for a fixed seed, and sensitivity to the seed -- the same correctness bar every
     bootstrap/permutation in this project is already held to.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def _fake_anchor_table(monkeypatch_target):
    """A tiny, hand-built (idx, target, pk, vina, n_lig) table so this test does not depend on the real
    anchor_table.npz cache or on build_lp_splits' expensive dataset load."""
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m
    rows = {
        'train': np.array([
            (0, 'T1', 5.0, 0.0, 1), (1, 'T1', 6.0, 0.0, 1), (2, 'T1', 7.0, 0.0, 1),
            (3, 'T2', 8.0, 0.0, 1), (4, 'T2', 9.0, 0.0, 1),
            (5, 'T3', 3.0, 0.0, 1),            # singleton target -- nothing to permute
        ], dtype=object),
    }
    monkeypatch_target(m, rows)


def test_val_and_test_pk_are_returned_unchanged():
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m

    def fake_build_anchor_table(force=False):
        return {'train': np.array([
            (0, 'T1', 5.0, 0.0, 1), (1, 'T1', 6.0, 0.0, 1), (2, 'T1', 7.0, 0.0, 1),
        ], dtype=object)}
    m.build_anchor_table = fake_build_anchor_table

    pk_by_idx = {0: 5.0, 1: 6.0, 2: 7.0, 100: 42.0, 200: 99.0}   # 100, 200 stand in for val/test rows
    train_idx = [0, 1, 2]
    out = m.permute_pk_within_target(pk_by_idx, train_idx, seed=1)
    assert out[100] == 42.0 and out[200] == 99.0, 'a non-train index was altered'


def test_permutation_preserves_the_label_multiset_per_target():
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m

    def fake_build_anchor_table(force=False):
        return {'train': np.array([
            (0, 'T1', 5.0, 0.0, 1), (1, 'T1', 6.0, 0.0, 1), (2, 'T1', 7.0, 0.0, 1),
            (3, 'T2', 8.0, 0.0, 1), (4, 'T2', 9.0, 0.0, 1),
        ], dtype=object)}
    m.build_anchor_table = fake_build_anchor_table

    pk_by_idx = {0: 5.0, 1: 6.0, 2: 7.0, 3: 8.0, 4: 9.0}
    out = m.permute_pk_within_target(pk_by_idx, [0, 1, 2, 3, 4], seed=7)
    assert sorted(out[i] for i in (0, 1, 2)) == [5.0, 6.0, 7.0], 'T1 label set changed'
    assert sorted(out[i] for i in (3, 4)) == [8.0, 9.0], 'T2 label set changed'
    # and cross-target leakage must not happen: T1's values must never land in T2's indices or vice versa
    assert out[3] in (8.0, 9.0) and out[4] in (8.0, 9.0)
    assert out[0] in (5.0, 6.0, 7.0) and out[1] in (5.0, 6.0, 7.0) and out[2] in (5.0, 6.0, 7.0)


def test_singleton_target_is_left_unpermuted_not_errored():
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m

    def fake_build_anchor_table(force=False):
        return {'train': np.array([(5, 'T3', 3.0, 0.0, 1)], dtype=object)}
    m.build_anchor_table = fake_build_anchor_table

    out = m.permute_pk_within_target({5: 3.0}, [5], seed=1)
    assert out[5] == 3.0


def test_permutation_is_deterministic_for_a_fixed_seed_and_seed_sensitive():
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m

    def fake_build_anchor_table(force=False):
        return {'train': np.array([(i, 'T1', float(i), 0.0, 1) for i in range(12)], dtype=object)}
    m.build_anchor_table = fake_build_anchor_table

    pk_by_idx = {i: float(i) for i in range(12)}
    idx = list(range(12))
    a = m.permute_pk_within_target(pk_by_idx, idx, seed=5)
    b = m.permute_pk_within_target(pk_by_idx, idx, seed=5)
    c = m.permute_pk_within_target(pk_by_idx, idx, seed=6)
    assert a == b, 'same seed produced different permutations'
    assert a != c, 'different seeds produced the identical permutation -- vanishingly unlikely unless the seed is ignored'


def test_abort_if_a_train_index_is_missing_from_the_anchor_table():
    import guidance.lp_split.train_egnn_stage0_withintarget_yrand as m

    def fake_build_anchor_table(force=False):
        return {'train': np.array([(0, 'T1', 5.0, 0.0, 1)], dtype=object)}
    m.build_anchor_table = fake_build_anchor_table

    try:
        m.permute_pk_within_target({0: 5.0, 999: 1.0}, [0, 999], seed=1)
    except SystemExit as e:
        assert '999' in str(e)
        return
    raise AssertionError('expected an explicit abort for a train index missing from the anchor table')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
