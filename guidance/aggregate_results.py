"""Aggregates multi-seed run results into one table, with the small-n guards this project learned the
hard way.

WHY THIS EXISTS -- the specific failure it prevents:

Multi-seed tables in this project were computed by hand, and that produced two wrong conclusions about
the SAME experiment, three days apart:
  * at n=1, A1g ternary QAT was reported as "matches FP32";
  * at n=2, it was reported as "a systematic ~0.02 R^2 cost";
  * at n=3 (seed 2023 = 0.3872) both statements were refuted. Mean 0.3437, sd 0.0377 -- the between-seed
    spread was LARGER than the effect being claimed, and had been all along.

The error was not arithmetic. It was reporting a mean difference without first asking whether n was large
enough for that difference to mean anything. So this tool does not merely compute mean +- sd: it REFUSES
to present an sd-based claim at n<3, and for any comparison it prints the minimum difference that the
observed spread could actually have detected. If the claimed effect is smaller than that number, the
honest report is "underpowered", not a point estimate with a confident sign.

WHAT IT READS (no GPU, no model loading, safe to run while training occupies the card):
  * `log.txt` in each run directory -- the per-epoch `[Validate]` blocks. The reported row is the epoch
    with the LOWEST validation loss, i.e. the checkpoint that `best.pt` actually holds, not the last
    epoch and not the best value of the metric being tabulated (picking the epoch that maximises the
    metric you then report is a selection bias that inflates every number).
  * `checkpoints/best.pt` metadata, if present, for the authoritative seed and epoch. The seed is read
    from the checkpoint rather than from the directory name wherever possible, because directory names
    have been renamed by hand in this project (`..._RANDOM_aborted`) and a renamed directory must not be
    able to misattribute a result to the wrong seed.
  * `metrics.json` in a run directory, if present -- held-out TEST metrics, which the training logs do
    not contain (training never touches test).

Usage:
  PYTHONPATH=. python guidance/aggregate_results.py logs_a1g_qat_ternary logs_lp_split_stage0
  PYTHONPATH=. python guidance/aggregate_results.py 'logs_surrogate_arms/*' --metric R2
  PYTHONPATH=. python guidance/aggregate_results.py logs_a1g_qat_ternary --compare a1g stage0
  PYTHONPATH=. python guidance/aggregate_results.py logs_surrogate_arms --json results.json
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scipy import stats as _st
except Exception:                                     # scipy is present in this env, but the tool is
    _st = None                                        # useful enough that it should not hard-depend on it

# `[Validate] num: 6069, RMSE: 1.436, MAE: 1.133, R^2 score: 0.298, Pearson: 0.622, Spearman: 0.623, ...`
METRIC_RE = re.compile(
    r'\[(?P<phase>\w+)\]\s+num:\s*(?P<n>\d+),\s*RMSE:\s*(?P<RMSE>[-\d.]+),\s*MAE:\s*(?P<MAE>[-\d.]+),'
    r'\s*R\^2 score:\s*(?P<R2>[-\d.]+),\s*Pearson:\s*(?P<Pearson>[-\d.]+),\s*Spearman:\s*(?P<Spearman>[-\d.]+)')
# `[Validate] Epoch 005 | Loss 2.311473`
EPOCH_RE = re.compile(r'\[(?P<phase>\w+)\]\s+Epoch\s+(?P<epoch>\d+)\s+\|\s+Loss\s+(?P<loss>[-\d.]+)')
TIMESTAMP_RE = re.compile(r'_?\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_?')
SEED_RE = re.compile(r'_s(\d{3,5})(?:_|$)')

METRICS = ('R2', 'RMSE', 'MAE', 'Pearson', 'Spearman')


def group_tag(run_dir):
    """Collapse a run directory name to the experiment arm it belongs to, so sibling seeds land in one
    group and nothing else does.

    The parent log directory is always part of the tag. Without it, runs whose directory name is only a
    config name plus a timestamp (the Stage 0 convention) collapse to a bare timestamp, which groups
    nothing and leaves `--compare` unable to address them by experiment name. With it every group is
    addressable by the experiment it belongs to -- `--compare a1g stage0` resolves because the parent
    directories are `logs_a1g_qat_ternary` and `logs_lp_split_stage0`."""
    run_dir = run_dir.rstrip('/')
    name = os.path.basename(run_dir)
    parent = re.sub(r'^logs_?', '', os.path.basename(os.path.dirname(run_dir))) or 'runs'
    name = TIMESTAMP_RE.sub('_', name)
    name = SEED_RE.sub('_', name)
    name = re.sub(r'^crossdocked_affinity_\w+?_', '', name)
    name = re.sub(r'_(RANDOM_)?aborted$', '_ABORTED', name)
    name = name.strip('_')
    return f'{parent}/{name}' if name else parent


def read_checkpoint_meta(run_dir):
    """Seed / epoch / val_loss straight from best.pt. torch is imported lazily and the load is CPU-only
    with the weights left on disk where possible -- this tool must stay runnable while the GPU is busy."""
    path = os.path.join(run_dir, 'checkpoints', 'best.pt')
    if not os.path.exists(path):
        return {}
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return {k: ckpt[k] for k in ('seed', 'epoch', 'val_loss', 'best_val_epoch') if k in ckpt}
    except Exception as e:
        return {'ckpt_error': f'{type(e).__name__}: {e}'}


def parse_log(log_path):
    """Returns the metrics of the epoch with the lowest validation loss, plus a count of epochs seen.

    The pairing rule matters: an `Epoch NNN | Loss` line is immediately followed by the `num: ...` metric
    line for that same epoch, so the loss line arms the epoch and the next metric line fills it. Parsing
    the two independently and zipping them would silently misalign whenever a run was resumed and the
    log picked up mid-epoch."""
    epochs, pending, test_metrics = [], None, None
    with open(log_path, errors='replace') as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m and m.group('phase').lower().startswith('valid'):
                pending = dict(epoch=int(m.group('epoch')), loss=float(m.group('loss')))
                continue
            m = METRIC_RE.search(line)
            if not m:
                continue
            vals = {k: float(m.group(k)) for k in METRICS}
            vals['n'] = int(m.group('n'))
            phase = m.group('phase').lower()
            if phase.startswith('test'):
                test_metrics = vals
            elif pending is not None:
                epochs.append({**pending, **vals})
                pending = None
    if not epochs:
        return None
    best = min(epochs, key=lambda e: e['loss'])
    return dict(best=best, n_epochs=len(epochs), test=test_metrics)


def collect(roots):
    """Finds every run directory under the given roots or globs. A run directory is one containing
    log.txt; roots may themselves be run directories."""
    runs, seen = [], set()
    candidates = []
    for root in roots:
        expanded = glob.glob(root) or [root]
        for e in expanded:
            if os.path.isfile(os.path.join(e, 'log.txt')):
                candidates.append(e)
            else:
                candidates.extend(os.path.dirname(p) for p in
                                  glob.glob(os.path.join(e, '**', 'log.txt'), recursive=True))
    for run_dir in sorted(set(candidates)):
        real = os.path.realpath(run_dir)
        if real in seen:
            continue
        seen.add(real)
        parsed = parse_log(os.path.join(run_dir, 'log.txt'))
        if parsed is None:
            continue
        meta = read_checkpoint_meta(run_dir)
        dir_seed = SEED_RE.search(os.path.basename(run_dir.rstrip('/')))
        rec = dict(run_dir=os.path.relpath(run_dir),
                   group=group_tag(run_dir),
                   seed=meta.get('seed', int(dir_seed.group(1)) if dir_seed else None),
                   seed_source='checkpoint' if 'seed' in meta else ('dirname' if dir_seed else 'unknown'),
                   best_epoch=parsed['best']['epoch'], n_epochs=parsed['n_epochs'],
                   val_loss=parsed['best']['loss'], val_n=parsed['best']['n'],
                   **{f'val_{k}': parsed['best'][k] for k in METRICS})
        if parsed['test']:
            rec.update({f'test_{k}': parsed['test'][k] for k in METRICS}, test_n=parsed['test']['n'])
        mpath = os.path.join(run_dir, 'metrics.json')
        if os.path.exists(mpath):
            try:
                for k, v in json.load(open(mpath)).items():
                    if isinstance(v, (int, float)):
                        rec.setdefault(f'test_{k}', v)
            except Exception:
                pass
        runs.append(rec)
    return runs


def summarize(values):
    """Mean with an explicit statement of what n supports. sd uses ddof=1 because these are seeds drawn
    from a population of seeds, not the whole population; at n=1 it is undefined and reported as such
    rather than as 0.0, which is the number that invites a false claim of precision."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    out = dict(n=int(v.size), mean=float(v.mean()) if v.size else None,
               min=float(v.min()) if v.size else None, max=float(v.max()) if v.size else None)
    out['sd'] = float(v.std(ddof=1)) if v.size >= 2 else None
    if v.size >= 2:
        sem = out['sd'] / math.sqrt(v.size)
        out['sem'] = sem
        tcrit = _st.t.ppf(0.975, v.size - 1) if _st is not None else 1.96
        out['ci95'] = [out['mean'] - tcrit * sem, out['mean'] + tcrit * sem]
    return out


def fmt_summary(s):
    """One line per summary. At n<2 the sd slot says 'undefined', never '0.0000': a zero there reads as
    perfect reproducibility, which is the opposite of what one run tells you."""
    if s['n'] == 0:
        return 'no values'
    if s['sd'] is None:
        return f'mean {s["mean"]:.4f}  sd undefined at n={s["n"]}'
    return (f'mean {s["mean"]:.4f}  sd {s["sd"]:.4f}  '
            f'95% CI [{s["ci95"][0]:.4f}, {s["ci95"][1]:.4f}]  range [{s["min"]:.4f}, {s["max"]:.4f}]')


def small_n_warning(n):
    if n >= 3:
        return None
    if n == 1:
        return ('n=1: no spread is observable, so NO comparative claim is supportable from this group. '
                'This project has already published "matches FP32" off a single seed and been wrong.')
    return ('n=2: the sd is computed from two points and is itself nearly uninformative; a difference of '
            'means here flipped sign when a third seed was added (A1g, 2026-10-09). Treat as a pilot.')


def mde(sd, n, alpha=0.05, power=0.80):
    """Minimum detectable difference between two groups of n seeds each, at the observed sd.

    Printed next to every comparison on purpose. A difference smaller than this is not a small effect
    that we measured -- it is an effect this design could not have measured at all, and the honest label
    for it is "underpowered", not a point estimate carrying a confident sign."""
    if sd is None or n < 2:
        return None
    if _st is not None:
        t_a = _st.t.ppf(1 - alpha / 2, 2 * n - 2)
        t_b = _st.t.ppf(power, 2 * n - 2)
    else:
        t_a, t_b = 1.96, 0.84
    return float((t_a + t_b) * sd * math.sqrt(2.0 / n))


def compare(groups, a, b, metric):
    ga = [r for r in groups if a in r['group']]
    gb = [r for r in groups if b in r['group']]
    if not ga or not gb:
        print(f'\ncompare: no runs matched {"a=" + a if not ga else ""} {"b=" + b if not gb else ""}')
        return
    key = f'val_{metric}' if f'val_{metric}' in ga[0] else metric
    va = np.array([r[key] for r in ga if r.get(key) is not None])
    vb = np.array([r[key] for r in gb if r.get(key) is not None])
    sa, sb = summarize(va), summarize(vb)
    diff = sa['mean'] - sb['mean']
    print(f'\n=== COMPARISON on {key} ===')
    print(f'  {a:<28} n={sa["n"]}  mean={sa["mean"]:.4f}' +
          (f'  sd={sa["sd"]:.4f}' if sa['sd'] is not None else '  sd=undefined'))
    print(f'  {b:<28} n={sb["n"]}  mean={sb["mean"]:.4f}' +
          (f'  sd={sb["sd"]:.4f}' if sb['sd'] is not None else '  sd=undefined'))
    print(f'  difference (a - b)        : {diff:+.4f}')
    pooled = None
    if sa['sd'] is not None and sb['sd'] is not None:
        pooled = math.sqrt((sa['sd'] ** 2 + sb['sd'] ** 2) / 2)
    n_min = min(sa['n'], sb['n'])
    m = mde(pooled, n_min)
    if m is not None:
        print(f'  pooled sd                 : {pooled:.4f}')
        print(f'  min detectable difference : {m:.4f}  (alpha=0.05, power=0.80, n={n_min}/group)')
        if abs(diff) < m:
            print(f'  VERDICT: UNDERPOWERED. |{diff:+.4f}| < {m:.4f}. The sign of this difference is not '
                  f'supported.\n           Report as "indistinguishable at n={n_min}", not as an effect.')
        else:
            print(f'  VERDICT: difference exceeds the detection floor at n={n_min}. Still report the CI.')
    if _st is not None and sa['n'] >= 2 and sb['n'] >= 2:
        t, p = _st.ttest_ind(va, vb, equal_var=False)
        print(f'  Welch t-test              : t={t:+.3f}  p={p:.4f}  '
              f'(seed-level only; says nothing about target-level generalisation)')
    for lbl, s in ((a, sa), (b, sb)):
        w = small_n_warning(s['n'])
        if w:
            print(f'  WARNING [{lbl}]: {w}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('roots', nargs='+', help='log directories, run directories, or globs')
    ap.add_argument('--metric', default='R2', choices=METRICS)
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'),
                    help='substring-match two group tags and compare them with a power check')
    ap.add_argument('--json', help='write the full per-run and per-group record here')
    ap.add_argument('--min_epochs', type=int, default=0,
                    help='ignore runs with fewer validated epochs than this (filters crashed starts)')
    args = ap.parse_args()

    runs = collect(args.roots)
    if args.min_epochs:
        runs = [r for r in runs if r['n_epochs'] >= args.min_epochs]
    if not runs:
        print('no runs with a parseable log.txt found under: ' + ', '.join(args.roots))
        return 1

    key = f'val_{args.metric}'
    by_group = {}
    for r in runs:
        by_group.setdefault(r['group'], []).append(r)

    print(f'{len(runs)} run(s) in {len(by_group)} group(s); reporting {key} at the lowest-val-loss epoch\n')
    for g in sorted(by_group):
        rs = sorted(by_group[g], key=lambda r: (r['seed'] is None, r['seed']))
        s = summarize([r.get(key) for r in rs])
        head = f'{g}  (n={s["n"]})'
        print(head)
        print('-' * len(head))
        for r in rs:
            test = f"  test_{args.metric}={r[f'test_{args.metric}']:.4f}" if f'test_{args.metric}' in r else ''
            src = '' if r['seed_source'] == 'checkpoint' else f" [seed from {r['seed_source']}]"
            print(f"  seed {str(r['seed']):>5}  ep {r['best_epoch']:>3}/{r['n_epochs']:<3} "
                  f"val_loss={r['val_loss']:.4f}  {key}={r.get(key, float('nan')):.4f}{test}{src}")
        print('  VAL   ' + fmt_summary(s))
        # Test is summarised separately and labelled, because the test mean is the number a thesis
        # quotes while the val mean is only what model selection optimised. Conflating the two, or
        # quoting whichever is higher, is the easiest way to overstate a result.
        test_key = f'test_{args.metric}'
        if any(test_key in r for r in rs):
            st = summarize([r.get(test_key) for r in rs])
            print(f'  TEST  ' + fmt_summary(st) +
                  ('' if st['n'] == s['n'] else f'   (only {st["n"]}/{s["n"]} run(s) have test metrics)'))
        w = small_n_warning(s['n'])
        if w:
            print(f'  WARNING: {w}')
        # Duplicate seeds mean the "n" above is not n independent draws.
        seeds = [r['seed'] for r in rs if r['seed'] is not None]
        if len(seeds) != len(set(seeds)):
            dup = sorted({x for x in seeds if seeds.count(x) > 1})
            print(f'  WARNING: duplicate seed(s) {dup} in this group -- n is inflated, these are not '
                  f'independent draws')
        print()

    if args.compare:
        compare(runs, args.compare[0], args.compare[1], args.metric)

    if args.json:
        payload = dict(metric=key, runs=runs,
                       groups={g: summarize([r.get(key) for r in rs]) for g, rs in by_group.items()})
        with open(args.json, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
