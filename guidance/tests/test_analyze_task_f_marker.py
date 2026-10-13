"""Tests for guidance/analyze_task_f.py::load_all_results's DONE-marker parsing -- written after this
failed live in the real RQ2 queue run (rq2_analyze, exit 1, IndexError on 'n_attempted='.split()[1]).

Root cause: run_triple_vina_rerank wrote its DONE marker as 'n_pool=... n_select_failed=... n_kept=...
n_scored=...' -- a DIFFERENT, incompatible key set from run_triple's 'n_attempted=... n_reconstructed=...
n_scored=...', which load_all_results parses by splitting on the literal string 'n_attempted='. The
synthetic/dummy verifications run earlier in the same session (tiny pocket counts, mocked predictions)
never exercised this function at all, so the incompatibility was invisible until the real 60-triple run
reached the analysis step. Fixed by making run_triple_vina_rerank's marker ALSO carry n_attempted=
(set equal to samples_per_pocket, the same quantity the other variants' n_attempted means), and the
60 already-written DONE files on disk were patched in place rather than re-running the underlying
Vina/sampling work.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from guidance.analyze_task_f import load_all_results


def _write_triple(base, variant, pocket_id, seed, done_line, n_rows=3):
    d = os.path.join(base, variant, f'pocket{pocket_id}_seed{seed}')
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({'vina_dock': [-1.0] * n_rows, 'qed': [0.5] * n_rows, 'file': [f'{i:03d}.sdf' for i in range(n_rows)],
                 'pocket_data_id': [pocket_id] * n_rows, 'seed': [seed] * n_rows}).to_csv(
        os.path.join(d, 'honest_eval.csv'), index=False)
    with open(os.path.join(d, 'DONE'), 'w') as f:
        f.write(done_line)


def test_load_all_results_parses_run_triple_style_marker():
    """baseline/affinity_only's own marker format -- must keep working exactly as before."""
    with tempfile.TemporaryDirectory() as td:
        _write_triple(td, 'baseline', 0, 1, 'n_attempted=10 n_reconstructed=10 n_scored=10\n')
        df, attempted = load_all_results(td)
        assert attempted[('baseline', 0, 1)] == 10
        assert len(df) == 3


def test_load_all_results_parses_vina_rerank_style_marker_after_the_fix():
    """The exact marker format run_triple_vina_rerank now writes -- n_attempted first, diagnostic
    fields after. This is the regression test for the bug: before the fix, this marker had no
    'n_attempted=' at all and load_all_results raised IndexError."""
    with tempfile.TemporaryDirectory() as td:
        _write_triple(td, 'vina_rerank', 0, 1,
                     'n_attempted=10 n_pool=20 n_select_failed=0 n_kept=10 n_scored=10\n')
        df, attempted = load_all_results(td)
        assert attempted[('vina_rerank', 0, 1)] == 10
        assert len(df) == 3


def test_load_all_results_raises_a_clear_error_not_a_bare_indexerror_for_a_malformed_marker():
    """Documents the CURRENT behaviour (a bare IndexError) as a known sharp edge: any future DONE
    marker format that omits 'n_attempted=' will fail the same way. This test exists so that if
    load_all_results is ever hardened to raise a clearer error instead, the change is deliberate and
    this test is updated alongside it -- not so the bare IndexError is endorsed as desirable."""
    with tempfile.TemporaryDirectory() as td:
        _write_triple(td, 'some_future_variant', 0, 1, 'n_pool=20 n_kept=10 n_scored=10\n')
        try:
            load_all_results(td)
        except IndexError:
            return
        raise AssertionError('expected an IndexError for a marker with no n_attempted= key -- if this '
                             'now passes, load_all_results was hardened and this test should be updated '
                             'to check for the new, clearer error instead')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
