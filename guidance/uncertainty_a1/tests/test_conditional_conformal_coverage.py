"""Tests for the NEW logic in conditional_conformal_coverage.py -- not for cluster_bootstrap_coverage,
which is reused unchanged from conformal_prediction.py and already has its own tests. What is new here
is: mapping targets to Pfam family, collapsing rare families without letting unannotated targets crowd
out real ones, and the underpowered/gate_pass decision logic.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from guidance.uncertainty_a1.conditional_conformal_coverage import (
    collapse_rare_families, family_by_target, stratum_coverage)


def _write_cache(tmpdir, records):
    path = os.path.join(tmpdir, 'uniprot_cache.json')
    with open(path, 'w') as f:
        json.dump(records, f)
    return path


# --------------------------------------------------------------- family_by_target

def test_family_by_target_uses_first_two_underscore_fields_as_the_lookup_key():
    """Must match build_family_holdout_split.py's convention exactly -- 'GENE_SPECIES_n_n_n' -> first
    two fields -- since this is reusing an established mapping, not defining a new one."""
    import guidance.uncertainty_a1.conditional_conformal_coverage as m
    with tempfile.TemporaryDirectory() as td:
        m.UNIPROT_CACHE_PATH = _write_cache(td, {
            'BSD_ASPTE': {'found': True, 'pfam_ids': ['PF00001']},
            'CAH7_HUMAN': {'found': True, 'pfam_ids': ['PF00002', 'PF00099']},
        })
        target = np.array(['BSD_ASPTE_1_130_0', 'CAH7_HUMAN_2_263_0', 'CAH7_HUMAN_9_1_0'], dtype=object)
        fam = family_by_target(target)
        assert list(fam) == ['PF00001', 'PF00002', 'PF00002']


def test_family_by_target_takes_the_first_listed_pfam_id_when_several_exist():
    import guidance.uncertainty_a1.conditional_conformal_coverage as m
    with tempfile.TemporaryDirectory() as td:
        m.UNIPROT_CACHE_PATH = _write_cache(td, {'X_HUMAN': {'found': True, 'pfam_ids': ['PF999', 'PF111']}})
        fam = family_by_target(np.array(['X_HUMAN_1_1_0']))
        assert fam[0] == 'PF999'


def test_family_by_target_marks_missing_or_unfound_entries_explicitly():
    """Three ways a target can lack a family: absent from the cache entirely, found=False, or
    found=True with an empty pfam_ids list. All three must produce the literal string
    'no_annotation', never None and never a KeyError, so a missing target shows up as its own
    reported stratum instead of crashing or silently vanishing."""
    import guidance.uncertainty_a1.conditional_conformal_coverage as m
    with tempfile.TemporaryDirectory() as td:
        m.UNIPROT_CACHE_PATH = _write_cache(td, {
            'B_HUMAN': {'found': False, 'pfam_ids': ['PF00001']},
            'C_HUMAN': {'found': True, 'pfam_ids': []},
        })
        target = np.array(['A_HUMAN_1_1_0', 'B_HUMAN_1_1_0', 'C_HUMAN_1_1_0'], dtype=object)
        fam = family_by_target(target)
        assert list(fam) == ['no_annotation', 'no_annotation', 'no_annotation']


# --------------------------------------------------------------- collapse_rare_families

def test_collapse_ranks_by_distinct_targets_not_by_row_count():
    """The point of ranking by targets rather than rows: a family with ONE target and a hundred
    ligand rows must not outrank a family with ten targets and ten rows -- the bootstrap's unit is
    the target, so target count is what determines whether a stratum is even usable."""
    fam = np.array(['small_many_rows'] * 100 + ['big_few_rows'] * 10, dtype=object)
    target = np.array([f't_small_1'] * 100 + [f't_big_{i}' for i in range(10)], dtype=object)
    collapsed, top = collapse_rare_families(fam, target, n_top=1)
    assert top[0][0] == 'big_few_rows', top
    # 'small_many_rows' has only ONE distinct target, so despite 100 rows it is not in top-1 and
    # every one of its rows must be relabelled 'other_family'.
    assert set(collapsed[:100]) == {'other_family'}
    assert set(collapsed[100:]) == {'big_few_rows'}


def test_no_annotation_never_counts_toward_n_top_and_is_preserved_as_its_own_label():
    """'no_annotation' must never compete for a top-N slot (it is not a real family), and it must
    stay labelled 'no_annotation' after collapsing -- not get relabelled 'other_family', which would
    conflate 'we have no idea' with 'a real but rare family'."""
    fam = np.array(['no_annotation'] * 50 + ['real_rare'] * 3, dtype=object)
    target = np.array([f't_na_{i}' for i in range(50)] + ['t_rare_1', 't_rare_2', 't_rare_3'], dtype=object)
    collapsed, top = collapse_rare_families(fam, target, n_top=5)
    assert 'no_annotation' not in {f for f, _ in top}
    assert set(collapsed[:50]) == {'no_annotation'}
    # real_rare has only 3 targets but there is room in top-5, so it is NOT collapsed to other_family.
    assert set(collapsed[50:]) == {'real_rare'}


def test_collapse_keeps_exactly_n_top_families_and_pools_the_rest():
    fam = np.array(sum(([f'fam{i}'] * 5 for i in range(6)), []), dtype=object)
    target = np.array(sum(([f't{i}_{j}' for j in range(5)] for i in range(6)), []), dtype=object)
    collapsed, top = collapse_rare_families(fam, target, n_top=2)
    assert len(top) == 2
    kept = {f for f, _ in top}
    assert len(kept) == 2
    for f, c in zip(fam, collapsed):
        assert (c == f) if f in kept else (c == 'other_family')


# --------------------------------------------------------------- stratum_coverage

def test_stratum_below_min_clusters_is_flagged_underpowered_not_given_a_verdict():
    import guidance.uncertainty_a1.conditional_conformal_coverage as m
    rng = np.random.default_rng(0)
    n = 40
    target = np.array([f't{i}' for i in range(5)] * (n // 5))   # only 5 distinct targets
    assert m.MIN_CLUSTERS > 5, 'test assumes the real MIN_CLUSTERS threshold exceeds 5'
    covered = (rng.random(n) < 0.9).astype(float)
    r = stratum_coverage(covered, target, np.arange(n), q=1.0, B=200)
    assert r['underpowered'] is True
    assert r['ci95'] is None and r['gate_pass'] is None
    assert r['point'] is not None, 'a point estimate should still be reported even when underpowered'


def test_stratum_at_or_above_min_clusters_gets_a_real_ci_and_gate_verdict():
    import guidance.uncertainty_a1.conditional_conformal_coverage as m
    rng = np.random.default_rng(1)
    n_targets = m.MIN_CLUSTERS + 5
    rows_per_target = 8
    target = np.array([f't{i}' for i in range(n_targets) for _ in range(rows_per_target)])
    covered = (rng.random(len(target)) < 0.9).astype(float)
    r = stratum_coverage(covered, target, np.arange(len(target)), q=1.0, B=500)
    assert r['underpowered'] is False
    assert r['ci95'] is not None and len(r['ci95']) == 2
    assert r['ci95'][0] <= r['ci95'][1]
    assert r['gate_pass'] == bool(r['ci95'][0] <= 0.90 <= r['ci95'][1])


def test_stratum_coverage_handles_an_empty_selection_without_crashing():
    """A tier or family with zero rows (e.g. a novelty tier with no test molecules in this split) must
    report n=0 and underpowered, not divide by zero or index out of range."""
    target = np.array(['t0', 't1'])
    covered = np.array([1.0, 0.0])
    r = stratum_coverage(covered, target, np.array([], dtype=int), q=1.0, B=100)
    assert r['n'] == 0 and r['underpowered'] is True and r['point'] is None


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
