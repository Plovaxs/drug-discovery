"""Tests for guidance/lp_split/lp_split_loader.py -- the function that defines train/val/test everywhere.

Why this file exists. An import audit of our own code found 55 modules imported by real scripts but
touched by no test. Most are one-shot experiment drivers whose test is the experiment. This one is not:
`build_lp_splits` has 37 importers and decides which complexes are training data and which are held out,
so a fault in it does not produce a wrong experiment -- it produces a thesis in which every number is
quietly invalid. It is 40 lines, it is load-bearing, and it was untested.

The audit also turned up a latent hazard in it, now guarded: `pk_by_idx[idx] = pk` was written for all
three partitions with no collision check, so an index present in both train and test would have been
collapsed to a single label, hiding the leakage inside the structure built to prevent it. The shipped
split has zero overlaps, so the guard changes nothing today; these tests are what keep that true if the
split is ever rebuilt.

PocketLigandPairDataset is stubbed throughout. The 3 GB CrossDocked dataset is irrelevant to the index
bookkeeping under test, and requiring it would turn a 1-second suite into one that only runs on this
laptop.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

SPLIT_JSON = os.path.join(ROOT, 'guidance/lp_split/leakage_safe_split.json')
SKIP_EXIT = 2

if not os.path.exists(SPLIT_JSON):
    print(f'SKIP: {SPLIT_JSON} absent')
    sys.exit(SKIP_EXIT)

from guidance.lp_split import lp_split_loader as L

# The loader builds the real dataset object; nothing under test reads it.
L.PocketLigandPairDataset = lambda root: ('stub-dataset', root)

# Counts quoted in the thesis and in guidance/cheminformatics results (AVE's "46,964 training
# fingerprints" among them). Pinned so a silent change to the split file is a test failure, not a
# discrepancy someone notices in a table months later.
EXPECT = {'train': 46964, 'val': 6069, 'test': 11855}
EXPECT_TARGETS = {'train': 779, 'val': 65, 'test': 127}


def _real():
    with open(SPLIT_JSON) as f:
        return json.load(f)


def _write_split(tmpdir, train, val, test):
    """A minimal split file. Values are (index, pK) pairs, exactly the shipped file's shape."""
    path = os.path.join(tmpdir, 'split.json')
    with open(path, 'w') as f:
        json.dump({'train': train, 'val': val, 'test': test}, f)
    return path


# --------------------------------------------------------------- the real split file's invariants

def test_real_split_indices_are_pairwise_disjoint():
    """The one property that makes every downstream number meaningful."""
    d = _real()
    s = {p: {i for i, _ in d[p]} for p in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = s[a] & s[b]
        assert not overlap, f'{len(overlap)} indices shared between {a} and {b}: {sorted(overlap)[:10]}'


def test_real_split_has_no_duplicate_indices_within_a_partition():
    d = _real()
    for p in ('train', 'val', 'test'):
        idxs = [i for i, _ in d[p]]
        assert len(idxs) == len(set(idxs)), f'{p} has {len(idxs) - len(set(idxs))} duplicated indices'


def test_real_split_targets_are_pairwise_disjoint():
    """LP-PDBBind splits by TARGET, not by complex. Disjoint indices with shared targets would still
    leak, so this is the stronger statement and the one the split's name claims."""
    d = _real()
    t = {p: set(d[f'{p}_targets']) for p in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = t[a] & t[b]
        assert not overlap, f'targets shared between {a} and {b}: {sorted(overlap)[:5]}'


def test_real_split_sizes_match_the_numbers_we_report():
    d = _real()
    got = {p: len(d[p]) for p in ('train', 'val', 'test')}
    assert got == EXPECT, f'split sizes changed: {got} vs reported {EXPECT}'
    got_t = {p: len(d[f'{p}_targets']) for p in ('train', 'val', 'test')}
    assert got_t == EXPECT_TARGETS, f'target counts changed: {got_t} vs reported {EXPECT_TARGETS}'


def test_real_split_loads_and_pk_covers_every_index():
    base, splits, pk = L.build_lp_splits(split_json=SPLIT_JSON, train_subsample=None)
    assert base == ('stub-dataset', './data/crossdocked_v1.1_rmsd1.0_pocket10')
    assert {p: len(v) for p, v in splits.items()} == EXPECT
    for p in ('train', 'val', 'test'):
        missing = [i for i in splits[p] if i not in pk]
        assert not missing, f'{len(missing)} indices in {p} have no pK'
    assert len(pk) == sum(EXPECT.values()), (len(pk), sum(EXPECT.values()))


def test_real_split_pk_values_match_the_file():
    d = _real()
    _, _, pk = L.build_lp_splits(split_json=SPLIT_JSON, train_subsample=None)
    for p in ('train', 'val', 'test'):
        for idx, v in d[p]:
            assert pk[idx] == v, f'pK for index {idx} is {pk[idx]}, file says {v}'


# --------------------------------------------------------------- the collision guard

def test_guard_rejects_an_index_present_in_train_and_test():
    """The leakage case. Before the guard this returned normally, with pk_by_idx silently keeping the test
    label and the index counted in both partitions."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_split(td, train=[[1, 5.0], [2, 6.0]], val=[[3, 7.0]], test=[[2, 9.0], [4, 8.0]])
        try:
            L.build_lp_splits(split_json=path, train_subsample=None)
        except ValueError as e:
            msg = str(e)
            assert 'index 2' in msg and 'leakage' in msg, msg
            assert "'train'" in msg and "'test'" in msg, msg
            return
        raise AssertionError('an index in both train and test was accepted')


def test_guard_rejects_a_duplicate_inside_one_partition():
    """Distinct failure, distinct message: a repeated index inflates that partition and makes its pK
    ambiguous, but it is not leakage, and conflating the two would mislead whoever hits it."""
    with tempfile.TemporaryDirectory() as td:
        path = _write_split(td, train=[[1, 5.0], [1, 6.0]], val=[[3, 7.0]], test=[[4, 8.0]])
        try:
            L.build_lp_splits(split_json=path, train_subsample=None)
        except ValueError as e:
            msg = str(e)
            assert 'index 1' in msg and "twice inside 'train'" in msg, msg
            assert 'leakage' not in msg, f'duplicate misreported as leakage: {msg}'
            return
        raise AssertionError('a duplicated index inside train was accepted')


def test_guard_accepts_a_clean_split():
    with tempfile.TemporaryDirectory() as td:
        path = _write_split(td, train=[[1, 5.0], [2, 6.0]], val=[[3, 7.0]], test=[[4, 8.0]])
        _, splits, pk = L.build_lp_splits(split_json=path, train_subsample=None)
        assert splits == {'train': [1, 2], 'val': [3], 'test': [4]}, splits
        assert pk == {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0}, pk


# --------------------------------------------------------------- train subsampling

def _big_split(td, n=100):
    return _write_split(td, train=[[i, float(i)] for i in range(n)],
                        val=[[1000 + i, 1.0] for i in range(7)],
                        test=[[2000 + i, 2.0] for i in range(9)])


def test_subsample_is_deterministic_for_a_given_seed():
    """Every experiment that reports a train_subsample depends on this. If it drifted, two runs labelled
    the same seed would not be the same run."""
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td)
        a = L.build_lp_splits(split_json=path, train_subsample=30, seed=2021)[1]['train']
        b = L.build_lp_splits(split_json=path, train_subsample=30, seed=2021)[1]['train']
        assert a == b, 'same seed produced different subsamples'
        assert len(a) == 30, len(a)


def test_subsample_actually_depends_on_the_seed():
    """The companion check. A determinism test alone would pass just as happily if the seed were ignored
    and the function always returned the first N."""
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td)
        a = L.build_lp_splits(split_json=path, train_subsample=30, seed=2021)[1]['train']
        c = L.build_lp_splits(split_json=path, train_subsample=30, seed=2023)[1]['train']
        assert a != c, 'the seed has no effect on which examples are selected'
        assert a != list(range(30)), 'subsample is just the first N, not a random draw'


def test_subsample_is_a_subset_of_the_full_train_split():
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td)
        _, splits, _ = L.build_lp_splits(split_json=path, train_subsample=30, seed=2021)
        assert set(splits['train']) <= set(range(100)), 'subsample invented indices'
        assert len(set(splits['train'])) == 30, 'subsample repeated an index'


def test_subsampling_never_touches_val_or_test():
    """Stage 0's stated point is a complete evaluation on the held-out set; shrinking it would quietly
    change what every reported metric means."""
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td)
        for n in (None, 5, 30, 1000):
            _, splits, _ = L.build_lp_splits(split_json=path, train_subsample=n, seed=2021)
            assert splits['val'] == [1000 + i for i in range(7)], (n, splits['val'])
            assert splits['test'] == [2000 + i for i in range(9)], (n, splits['test'])


def test_no_subsample_keeps_the_file_order():
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td)
        _, splits, _ = L.build_lp_splits(split_json=path, train_subsample=None)
        assert splits['train'] == list(range(100)), 'train was reordered with subsampling disabled'


def test_subsample_at_and_above_the_train_size_is_a_no_op_in_file_order():
    """Boundary on the `train_subsample < len(train)` comparison: at exactly len(train) the branch is
    skipped, so the order stays the file's rather than becoming a permutation. Pinned because a `<=`
    here would silently shuffle the training order for runs that asked for no subsampling at all."""
    with tempfile.TemporaryDirectory() as td:
        path = _big_split(td, n=100)
        for n in (100, 101, 10_000):
            _, splits, _ = L.build_lp_splits(split_json=path, train_subsample=n, seed=2021)
            assert splits['train'] == list(range(100)), f'train_subsample={n} altered the split'


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
