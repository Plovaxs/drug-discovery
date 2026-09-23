"""Checkpointed driver for Diagnostic 1 (training-size learning curve):
runs guidance/train_affinity_diag1.py at target fractions [0.25, 0.5, 0.75]
using the sweep's best config (lr=3e-5, 6 layers), full 20-epoch budget.
The 100% point is the already-completed main-sweep run
(logs_affinity_sweep/lr_3e-5_layers6/...) -- reused, not retrained.

Resumable at fraction granularity, same pattern as run_affinity_sweep.py.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

BEST_CONFIG = 'configs/prop/affinity_sweep/lr_3e-5.yml'
REFERENCE_100PCT_LOG = './logs_affinity_sweep/lr_3e-5_layers6/lr_3e-5_2026_09_06__23_57_10/log.txt'
FRACTIONS = [0.25, 0.5, 0.75]


def parse_log_for_best(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    best_epoch = None
    best_val_loss = None
    for line in lines:
        m = re.search(r'Best val achieved at epoch (\d+), val loss: ([\d.]+)', line)
        if m:
            best_epoch = int(m.group(1))
            best_val_loss = float(m.group(2))
    if best_epoch is None:
        return None

    def find_eval(prefix, epoch):
        pat_summary = re.compile(
            rf'\[\s*{prefix}\s*\] num:\s*(\d+), RMSE: ([\d.]+), MAE: ([\d.]+), '
            rf'R\^2 score: (-?[\d.]+), Pearson: (-?[\d.]+), Spearman: (-?[\d.]+)')
        found = False
        for line in lines:
            if f'[{prefix}] Epoch {epoch:03d}' in line:
                found = True
            if found:
                sm = pat_summary.search(line)
                if sm:
                    return {'n': int(sm.group(1)), 'rmse': float(sm.group(2)), 'mae': float(sm.group(3)),
                           'r2': float(sm.group(4)), 'pearson': float(sm.group(5)), 'spearman': float(sm.group(6))}
        return None

    n_targets_m = re.search(r'n_targets=(\d+)', ''.join(lines))
    return {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'n_targets': int(n_targets_m.group(1)) if n_targets_m else None,
        'val': find_eval('Validate', best_epoch), 'test': find_eval('Test', best_epoch),
    }


def find_completed_log(logdir_root):
    if not os.path.isdir(logdir_root):
        return None
    for sub in sorted(glob.glob(os.path.join(logdir_root, '*'))):
        best_ckpt = os.path.join(sub, 'checkpoints', 'best.pt')
        log_path = os.path.join(sub, 'log.txt')
        if os.path.exists(best_ckpt) and os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if content.strip().splitlines() and 'Best val loss:' in content.strip().splitlines()[-1]:
                return log_path
    return None


def run_fraction(fraction, logdir_root, device):
    os.makedirs(logdir_root, exist_ok=True)
    existing = find_completed_log(logdir_root)
    if existing:
        print(f'[fraction={fraction}] already completed, found {existing}')
        return existing

    print(f'[fraction={fraction}] launching training')
    t0 = time.time()
    cmd = [sys.executable, '-u', 'guidance/train_affinity_diag1.py', BEST_CONFIG,
           '--target_fraction', str(fraction), '--device', device, '--logdir', logdir_root]
    env = dict(os.environ, PYTHONPATH='.')
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f'[fraction={fraction}] training subprocess failed with code {result.returncode}')
    print(f'[fraction={fraction}] finished in {elapsed/60:.1f} min')

    log_path = find_completed_log(logdir_root)
    if log_path is None:
        raise RuntimeError(f'[fraction={fraction}] training subprocess exited 0 but no completed log.txt found')
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir_root', type=str, default='./logs_affinity_diag1')
    parser.add_argument('--out', type=str, default='./guidance/diag1_learning_curve_results.json')
    args = parser.parse_args()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done_fractions = {r['fraction'] for r in results}

    if 1.0 not in done_fractions:
        metrics = parse_log_for_best(REFERENCE_100PCT_LOG)
        metrics['n_targets'] = 507  # from the main sweep's train_subsample=6000 run
        results.append({'fraction': 1.0, 'log_path': REFERENCE_100PCT_LOG, **metrics})
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[fraction=1.0] reused main-sweep result: test_pearson={metrics["test"]["pearson"]} '
             f'test_r2={metrics["test"]["r2"]}')

    for fraction in FRACTIONS:
        if fraction in done_fractions:
            print(f'[fraction={fraction}] already in {args.out}, skipping')
            continue
        logdir_root = os.path.join(args.logdir_root, f'frac{fraction}')
        log_path = run_fraction(fraction, logdir_root, args.device)
        metrics = parse_log_for_best(log_path)
        if metrics is None:
            raise RuntimeError(f'[fraction={fraction}] could not parse metrics from {log_path}')
        entry = {'fraction': fraction, 'log_path': log_path, **metrics}
        results.append(entry)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[fraction={fraction}] recorded: n_targets={metrics["n_targets"]} '
             f'test_pearson={metrics["test"]["pearson"]} test_r2={metrics["test"]["r2"]}')

    print(f'\nLearning curve complete. Results in {args.out}')
    for r in sorted(results, key=lambda x: x['fraction']):
        t = r['test']
        print(f'  fraction={r["fraction"]} n_targets={r.get("n_targets")}: '
             f'test_pearson={t["pearson"]:.3f} test_r2={t["r2"]:.3f}')


if __name__ == '__main__':
    main()
