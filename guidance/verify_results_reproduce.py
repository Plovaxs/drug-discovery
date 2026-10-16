"""Audit item F077: do the committed result files actually regenerate from the committed code?

The audit asked for a clean-room reproducibility run. A true clean room, a fresh machine and a
fresh environment built from the lockfiles, is outside what this project can run on one laptop,
and claiming one that was not performed would be worse than not doing it. What is both affordable
and worth more than it may sound is the question underneath it: if someone re-ran the analyses
today, from the code and data in this repository, would they get the numbers the thesis quotes?

That is not guaranteed by anything else in the project. The number checker verifies that the
thesis agrees with the committed JSON; it cannot tell whether the JSON still agrees with the code
that claims to produce it. A script edited after its output was written, a dependency whose
default changed, or an unseeded random path would all slip past every other guard here and leave
a thesis whose numbers no longer reproduce.

This harness re-runs deterministic, CPU-only analyses into a temporary directory and compares the
result against the committed file, field by field, with a numerical tolerance. GPU training runs
and sampling runs are deliberately excluded: they are not deterministic across runs at this
precision and re-running them is hours of compute, so claiming they reproduce would be a claim
this harness cannot support.

Exit status is 0 when every selected analysis reproduces, 1 otherwise. Every mismatch is printed
with its path, the committed value and the regenerated one, so a failure says what moved.

Run:  PYTHONPATH=. python guidance/verify_results_reproduce.py
      PYTHONPATH=. python guidance/verify_results_reproduce.py --only family_cluster
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: name -> (script, committed result path, argv for a faster but still deterministic run)
#: Only analyses that are seeded, CPU-only and finish in minutes belong here.
C = 'guidance/cheminformatics'
ANALYSES = {
    'family_cluster':     (f'{C}/family_cluster_bootstrap.py',
                           f'{C}/family_cluster_bootstrap_results.json', []),
    'residual_variance':  (f'{C}/residual_variance_audit.py',
                           f'{C}/residual_variance_audit_results.json', []),
    'representation_ladder': (f'{C}/representation_ladder.py',
                              f'{C}/representation_ladder_results.json', []),
    'noise_ceiling':      (f'{C}/noise_ceiling.py', f'{C}/noise_ceiling_results.json', []),
    'activity_cliffs':    (f'{C}/activity_cliffs.py', f'{C}/activity_cliffs_results.json', []),
    'matched_pairs':      (f'{C}/matched_molecular_pairs.py',
                           f'{C}/matched_molecular_pairs_results.json', []),
    'family_stratification': (f'{C}/protein_family_stratification.py',
                              f'{C}/protein_family_stratification_results.json', []),
}

#: Keys whose value is expected to differ between runs and is not a result.
VOLATILE = {'out', 'timestamp', 'generated_at', 'elapsed_s', 'wall_clock_h'}


def flatten(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in VOLATILE:
                continue
            yield from flatten(v, f'{prefix}.{k}' if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f'{prefix}[{i}]')
    else:
        yield prefix, obj


def compare(committed, fresh, tol):
    a, b = dict(flatten(committed)), dict(flatten(fresh))
    problems = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            problems.append(f'{k}: absent from the committed file, present now = {b[k]!r}')
            continue
        if k not in b:
            problems.append(f'{k}: present in the committed file = {a[k]!r}, absent now')
            continue
        x, y = a[k], b[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) \
                or not isinstance(y, (int, float)):
            if x != y:
                problems.append(f'{k}: committed {x!r}, regenerated {y!r}')
            continue
        if math.isnan(x) and math.isnan(y):
            continue
        if abs(x - y) > tol * max(1.0, abs(x)):
            problems.append(f'{k}: committed {x!r}, regenerated {y!r} '
                            f'(difference {abs(x - y):.3g})')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', action='append', choices=sorted(ANALYSES),
                    help='run only these analyses (repeatable)')
    ap.add_argument('--tol', type=float, default=1e-9,
                    help='relative tolerance; the default is tight because these are seeded')
    args = ap.parse_args()
    names = args.only or sorted(ANALYSES)

    failed, summary = 0, []
    for name in names:
        script, committed_path, extra = ANALYSES[name]
        if not os.path.exists(committed_path):
            print(f'[SKIP] {name}: no committed result at {committed_path}')
            summary.append((name, 'SKIP', 'no committed file'))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            fresh_path = os.path.join(tmp, os.path.basename(committed_path))
            backup = os.path.join(tmp, 'committed_backup.json')
            shutil.copy2(committed_path, backup)
            env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
            t0 = time.time()
            # These scripts write to a fixed path, so the committed file is moved aside, the script
            # is run, the fresh output is captured and the committed file is put back. The restore
            # happens in `finally` so an interrupted run cannot leave the repository with the
            # regenerated file silently standing in for the committed one.
            try:
                proc = subprocess.run([sys.executable, script] + extra, cwd=ROOT, env=env,
                                      capture_output=True, text=True)
                if proc.returncode != 0:
                    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
                    print(f'[FAIL] {name}: script exited {proc.returncode}')
                    for line in tail:
                        print(f'        {line}')
                    failed += 1
                    summary.append((name, 'FAIL', f'exit {proc.returncode}'))
                    continue
                shutil.copy2(committed_path, fresh_path)
            finally:
                shutil.copy2(backup, committed_path)
            dt = time.time() - t0
            problems = compare(json.load(open(backup)), json.load(open(fresh_path)), args.tol)
        if problems:
            print(f'[FAIL] {name}: {len(problems)} field(s) differ ({dt:.0f}s)')
            for p in problems[:20]:
                print(f'        {p}')
            if len(problems) > 20:
                print(f'        ... and {len(problems) - 20} more')
            failed += 1
            summary.append((name, 'FAIL', f'{len(problems)} fields differ'))
        else:
            n = len(dict(flatten(json.load(open(committed_path)))))
            print(f'[OK]   {name}: {n} values reproduce exactly ({dt:.0f}s)')
            summary.append((name, 'OK', f'{n} values'))

    print('\n=== F077 reproducibility summary ===')
    for name, status, detail in summary:
        print(f'  [{status:4s}] {name}: {detail}')
    print(f'\n{sum(1 for s in summary if s[1] == "OK")} reproduced, {failed} failed, '
          f'{sum(1 for s in summary if s[1] == "SKIP")} skipped')
    print('Scope: seeded CPU analyses only. GPU training and sampling runs are not covered here '
          'and are not claimed to reproduce bit-for-bit.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
