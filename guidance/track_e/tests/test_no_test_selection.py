"""Positive control against test-set checkpoint selection / leakage (Track E methodology).
Static check on the trainer's source: the test split may be *read* only where datasets are constructed and inside
Runner.final_test (called once, after training has completed, on best.pt chosen by validation loss). best.pt may be
written only inside the validation-improvement branch, and neither the scheduler nor early stopping may see test metrics.
Run: PYTHONPATH=. python guidance/track_e/tests/test_no_test_selection.py"""
import ast
import inspect

from guidance.track_e import train_e


def _functions_using(name_literal):
    tree = ast.parse(inspect.getsource(train_e))
    hits = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,))]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == name_literal:
                hits.setdefault(fn.name, 0)
                hits[fn.name] += 1
    return hits


def test_test_split_only_touched_at_construction_and_final_test():
    hits = _functions_using('test')
    allowed = {'build_everything', 'final_test'}
    assert set(hits) <= allowed, f"test split referenced outside {allowed}: {hits}"


def test_run_loop_never_reads_test_split():
    src = inspect.getsource(train_e.Runner.run)
    assert "['test']" not in src and '"test"' not in src and 'test_' not in src.replace('test_preds', ''), 'Runner.run must not touch test data/metrics'


def test_best_checkpoint_written_only_on_validation_improvement():
    src = inspect.getsource(train_e.Runner.run)
    assert src.count("'best.pt'") == 1
    before = src[:src.index("'best.pt'")]
    assert "improved = val['loss'] < st['best_val']" in before and "if improved:" in before[before.rindex("improved = "):]


def test_final_test_uses_best_checkpoint_and_runs_once_after_training():
    src = inspect.getsource(train_e.Runner.final_test)
    assert "best.pt" in src and src.index("best.pt") < src.index("evaluate(self.ctx['test']")
    run_src = inspect.getsource(train_e.Runner.run)
    assert run_src.count('final_test()') == 1 and run_src.index('train_complete') < run_src.index('final_test()')


def test_scheduler_and_early_stopping_use_validation_loss_only():
    src = inspect.getsource(train_e.Runner.run)
    assert "self.sched.step(val['loss'])" in src and "patience'] >= self.patience_max" in src


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f(); print('PASS', k)
    print(f'{len(fns)} tests passed')
