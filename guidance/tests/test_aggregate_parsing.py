"""Regression tests for aggregate_results.py's log parsing.

Why these exist: the aggregator was written and validated against the Stage-0 / surrogate-arm validate
line, then queued as a step that pointed it at the heteroscedastic trainers, whose validate line has an
entirely different shape. parse_log found nothing, collect returned an empty list, and the queue step
exited 1 after eight training runs had already completed. The queue's fail-closed design caught it, but
a test would have caught it before it was queued.

Both formats are pinned here, with the exact strings taken from real logs, so adding a third trainer with
a third format fails loudly in the test suite rather than silently in a queue step hours later.

Usage:
  PYTHONPATH=. python guidance/tests/test_aggregate_parsing.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.aggregate_results import A1E_RE, EPOCH_RE, METRIC_RE, group_tag, parse_log

# Verbatim from logs_lp_split_stage0 / logs_surrogate_arms.
STAGE0 = """[2026-09-08 16:30:00,000::train::INFO] [Validate] Epoch 001 | Loss 2.500000
[2026-09-08 16:30:01,000::train::INFO] [Validate] num: 6069, RMSE: 1.500, MAE: 1.200, R^2 score: 0.200, Pearson: 0.500, Spearman: 0.510, mean/std: 6.5/1.4
[2026-09-08 17:00:00,000::train::INFO] [Validate] Epoch 002 | Loss 1.780000
[2026-09-08 17:00:01,000::train::INFO] [Validate] num: 6069, RMSE: 1.417, MAE: 1.090, R^2 score: 0.317, Pearson: 0.615, Spearman: 0.620, mean/std: 7.2/1.4
"""

# Verbatim from logs_a1e_heteroscedastic / logs_a1e_beta_heteroscedastic.
A1E = """[2026-10-11 08:31:30,048::train_egnn_heteroscedastic::INFO] [Validate] Epoch 002 | NLL 2.232975 | RMSE(mu) 1.4308 | R2(mu) 0.3034 | mean_sigma 0.6154 | std_sigma 0.0749
[2026-10-11 08:43:03,529::train_egnn_heteroscedastic::INFO] [Validate] Epoch 003 | NLL 1.404833 | RMSE(mu) 1.3668 | R2(mu) 0.3643 | mean_sigma 0.7399 | std_sigma 0.0667
"""


def _parse(text):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'log.txt')
        open(p, 'w').write(text)
        return parse_log(p)


def test_stage0_format_parses():
    r = _parse(STAGE0)
    assert r is not None, 'Stage-0 format no longer parses'
    assert r['n_epochs'] == 2
    # best = lowest validation LOSS, not best metric -- selecting on the reported metric would be a
    # selection bias, so epoch 2 (loss 1.78) must win over epoch 1 (loss 2.50).
    assert r['best']['epoch'] == 2
    assert abs(r['best']['R2'] - 0.317) < 1e-9
    assert abs(r['best']['MAE'] - 1.090) < 1e-9
    assert r['best']['n'] == 6069


def test_a1e_format_parses():
    """The case that broke the queue."""
    r = _parse(A1E)
    assert r is not None, 'A1e heteroscedastic format does not parse -- this is the bug that exited 1'
    assert r['n_epochs'] == 2
    assert r['best']['epoch'] == 3                      # NLL 1.4048 < 2.2330
    assert abs(r['best']['R2'] - 0.3643) < 1e-9
    assert abs(r['best']['RMSE'] - 1.3668) < 1e-9
    assert abs(r['best']['loss'] - 1.404833) < 1e-9


def test_a1e_sigma_statistics_are_captured():
    """std_sigma is the quantity of interest for heteroscedastic runs: near zero is the
    variance-collapse signature behind A1's finding that point-wise sigma cannot rank error."""
    r = _parse(A1E)
    assert abs(r['best']['mean_sigma'] - 0.7399) < 1e-9
    assert abs(r['best']['std_sigma'] - 0.0667) < 1e-9


def test_a1e_absent_metrics_are_none_not_missing():
    """The A1e line has no MAE/Pearson/Spearman/num. They must be present as None so downstream
    formatting and summarising do not raise."""
    r = _parse(A1E)
    for k in ('MAE', 'Pearson', 'Spearman', 'n'):
        assert k in r['best'], f'{k} missing entirely; downstream code does .get but relies on the key'
        assert r['best'][k] is None


def test_the_two_formats_do_not_cross_match():
    """A1E_RE must not fire on a Stage-0 line and vice versa, or epochs would be double counted."""
    s0_epoch = '[Validate] Epoch 002 | Loss 1.780000'
    a1e_line = '[Validate] Epoch 002 | NLL 2.232975 | RMSE(mu) 1.4308 | R2(mu) 0.3034'
    assert EPOCH_RE.search(s0_epoch) and not A1E_RE.search(s0_epoch)
    assert A1E_RE.search(a1e_line) and not EPOCH_RE.search(a1e_line)
    assert not METRIC_RE.search(a1e_line)


def test_a1e_without_sigma_fields_still_parses():
    """The sigma fields are optional in the regex; a trainer that omits them must not break parsing."""
    r = _parse('[Validate] Epoch 004 | NLL 1.100000 | RMSE(mu) 1.2000 | R2(mu) 0.4000\n')
    assert r is not None and r['best']['epoch'] == 4
    assert r['best'].get('mean_sigma') is None


def test_mixed_log_counts_each_epoch_once():
    r = _parse(STAGE0 + A1E)
    assert r['n_epochs'] == 4, f"expected 4 epochs across both formats, got {r['n_epochs']}"


def test_empty_or_unparseable_log_returns_none():
    assert _parse('') is None
    assert _parse('nothing of interest here\nanother line\n') is None


def test_group_tag_separates_plain_from_beta():
    """--compare a1e_beta a1e_hetero relies on these tags being distinct substrings."""
    plain = group_tag('logs_a1e_heteroscedastic/crossdocked_affinity_egnn_2026_10_08__17_21_42_a1e_s2021')
    beta = group_tag('logs_a1e_beta_heteroscedastic/crossdocked_affinity_egnn_2026_10_11__01_40_17_a1eb_s2021')
    assert plain != beta
    assert 'a1e_beta' in beta and 'a1e_beta' not in plain
    assert 'a1e_hetero' in plain and 'a1e_hetero' not in beta


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
