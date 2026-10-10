"""Runs every test suite in this project and reports one summary.

This repo has no pytest in its conda environment and follows a plain-function convention instead: each
suite is a module with `test_*` functions and an `if __name__ == '__main__'` runner. That is fine for a
single suite but gives no way to ask "is everything still green", which is the question that matters
before a commit. So suites are discovered and each is run as a SUBPROCESS, deliberately:

  * a suite that imports heavy modules (torch, pandas over a 48 MB CSV) cannot slow or pollute the others;
  * a suite that crashes at import time is reported as a failure of that suite rather than aborting the
    whole run, which is what happens if you import them all into one process;
  * exit codes stay meaningful -- in particular exit 2 means "could not run", not "passed" (see below).

Exit-code contract for suites:
  0  all tests passed
  1  a test failed
  2  the suite could not run because required data artifacts are absent (gitignored multi-MB CSVs).
     Reported as a VISIBLE SKIP. It never counts as a pass, and never blocks a commit on a machine that
     simply has not built the data pool.

Usage:
  PYTHONPATH=. python guidance/run_tests.py
  PYTHONPATH=. python guidance/run_tests.py --quiet       # only failures and the summary
  PYTHONPATH=. python guidance/run_tests.py -k leakage    # run suites whose path matches
"""
import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Explicit, not a bare recursive glob: guidance/reference_code/ is vendored third-party code (GIGN and
# friends) and contains its own test_logger.py files that are not ours and are not expected to pass here.
SUITE_GLOBS = (
    'guidance/*/tests/test_*.py',
    'guidance/tests/test_*.py',
)
EXCLUDE_PARTS = ('reference_code',)

SKIP_EXIT = 2


def discover(pattern=None):
    suites = []
    for g in SUITE_GLOBS:
        for path in glob.glob(os.path.join(ROOT, g)):
            rel = os.path.relpath(path, ROOT)
            if any(part in rel.split(os.sep) for part in EXCLUDE_PARTS):
                continue
            if pattern and pattern not in rel:
                continue
            suites.append(rel)
    return sorted(set(suites))


def run_suite(rel, quiet):
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    t0 = time.time()
    proc = subprocess.run([sys.executable, rel], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode == 0:
        n = sum(1 for l in proc.stdout.splitlines() if l.startswith('PASS'))
        status, detail = 'PASS', f'{n} test(s)'
    elif proc.returncode == SKIP_EXIT:
        status, detail = 'SKIP', 'required data artifacts absent'
    else:
        status, detail = 'FAIL', f'exit {proc.returncode}'
    print(f'[{status}] {rel}  ({detail}, {dt:.1f}s)')
    if status != 'PASS' or not quiet:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        shown = tail if status == 'FAIL' else tail[-3:]
        for line in shown[-40:]:
            print(f'       {line}')
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-k', '--pattern', help='only run suites whose relative path contains this')
    ap.add_argument('--quiet', action='store_true', help='show output only for failing suites')
    args = ap.parse_args()

    suites = discover(args.pattern)
    if not suites:
        print('no test suites discovered' + (f' matching {args.pattern!r}' if args.pattern else ''))
        return 1

    print(f'running {len(suites)} suite(s)\n')
    results = [run_suite(s, args.quiet) for s in suites]
    passed, failed, skipped = (results.count(x) for x in ('PASS', 'FAIL', 'SKIP'))

    print(f'\n{passed} passed, {failed} failed, {skipped} skipped')
    if skipped:
        print('NOTE: a skipped suite is not a passing suite. The leakage guard in particular only runs '
              'where the BindingNet data pool has been built.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
