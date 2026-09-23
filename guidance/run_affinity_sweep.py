"""Checkpointed driver for Part 3.3's affinity-model hyperparameter sweep
(full 20-epoch budget per config, per the addendum's "full sweep" scope
decision).

Each config is a full, independent invocation of
guidance/train_affinity_model.py (unmodified) via subprocess, writing to
its own --logdir root. Resumability is at config granularity: a config is
skipped if its logdir root already contains a completed run (a
checkpoints/best.pt file and a log.txt whose last line confirms the run
finished all epochs or early-stopped) -- if interrupted mid-run, that one
config's partial log dir is left in place and this script will re-run it
from scratch on the next invocation (train_affinity_model.py itself has no
mid-run resume), but no completed config is ever redone.

After each config finishes, its final best-epoch Val/Test metrics are
parsed out of its log.txt and appended to --out (a JSON list), so the
sweep can be aggregated at any point without re-parsing everything.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

CONFIGS = [
    # (name, config_path, is_baseline_already_trained)
    ('baseline_lr1e-4_layers6', 'configs/prop/crossdocked_affinity_egnn.yml', True),
    ('lr_3e-5_layers6', 'configs/prop/affinity_sweep/lr_3e-5.yml', False),
    ('lr_3e-4_layers6', 'configs/prop/affinity_sweep/lr_3e-4.yml', False),
    ('arch_lr1e-4_layers4', 'configs/prop/affinity_sweep/arch_4layers.yml', False),
]

BASELINE_LOG_DIR = 'logs_affinity/crossdocked_affinity_egnn_2026_09_04__01_23_11/log.txt'


def parse_log_for_best(log_path):
    """Returns dict with the metrics logged at the best (final-selected) val
    epoch: val + test Pearson/Spearman/R2/RMSE, plus which epoch."""
    with open(log_path) as f:
        lines = f.readlines()

    best_epoch = None
    for line in lines:
        m = re.search(r'Best val achieved at epoch (\d+), val loss: ([\d.]+)', line)
        if m:
            best_epoch = int(m.group(1))
            best_val_loss = float(m.group(2))

    if best_epoch is None:
        return None

    def find_eval(prefix, epoch):
        pat_epoch = re.compile(rf'\[{prefix}\] Epoch {epoch:03d}')
        pat_summary = re.compile(
            rf'\[\s*{prefix}\s*\] num:\s*(\d+), RMSE: ([\d.]+), MAE: ([\d.]+), '
            rf'R\^2 score: (-?[\d.]+), Pearson: (-?[\d.]+), Spearman: (-?[\d.]+)')
        found_epoch_marker = False
        for i, line in enumerate(lines):
            if f'[{prefix}] Epoch {epoch:03d}' in line:
                found_epoch_marker = True
            if found_epoch_marker:
                sm = pat_summary.search(line)
                if sm:
                    return {
                        'n': int(sm.group(1)), 'rmse': float(sm.group(2)), 'mae': float(sm.group(3)),
                        'r2': float(sm.group(4)), 'pearson': float(sm.group(5)), 'spearman': float(sm.group(6)),
                    }
        return None

    val_metrics = find_eval('Validate', best_epoch)
    test_metrics = find_eval('Test', best_epoch)
    return {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'val': val_metrics, 'test': test_metrics,
    }


def find_completed_log(logdir_root):
    """Returns the log.txt path of a completed run under logdir_root, or
    None if no completed run is found there."""
    if not os.path.isdir(logdir_root):
        return None
    for sub in sorted(glob.glob(os.path.join(logdir_root, '*'))):
        best_ckpt = os.path.join(sub, 'checkpoints', 'best.pt')
        log_path = os.path.join(sub, 'log.txt')
        if os.path.exists(best_ckpt) and os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if 'Best val loss:' in content.splitlines()[-1] if content.splitlines() else False:
                return log_path
    return None


def run_config(name, config_path, logdir_root, device):
    os.makedirs(logdir_root, exist_ok=True)
    existing = find_completed_log(logdir_root)
    if existing:
        print(f'[{name}] already completed, found {existing}')
        return existing

    print(f'[{name}] launching training: config={config_path} logdir={logdir_root}')
    t0 = time.time()
    cmd = [sys.executable, '-u', 'guidance/train_affinity_model.py', config_path,
           '--device', device, '--logdir', logdir_root]
    env = dict(os.environ, PYTHONPATH='.')
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f'[{name}] training subprocess failed with code {result.returncode}')
    print(f'[{name}] finished in {elapsed/60:.1f} min')

    log_path = find_completed_log(logdir_root)
    if log_path is None:
        raise RuntimeError(f'[{name}] training subprocess exited 0 but no completed log.txt found')
    return log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sweep_logdir_root', type=str, default='./logs_affinity_sweep')
    parser.add_argument('--out', type=str, default='./guidance/affinity_sweep_results.json')
    args = parser.parse_args()

    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    done_names = {r['name'] for r in results}

    for name, config_path, is_baseline in CONFIGS:
        if name in done_names:
            print(f'[{name}] already in {args.out}, skipping')
            continue

        if is_baseline:
            log_path = BASELINE_LOG_DIR
            print(f'[{name}] reusing already-trained baseline log: {log_path}')
        else:
            logdir_root = os.path.join(args.sweep_logdir_root, name)
            log_path = run_config(name, config_path, logdir_root, args.device)

        metrics = parse_log_for_best(log_path)
        if metrics is None:
            raise RuntimeError(f'[{name}] could not parse best-epoch metrics from {log_path}')
        entry = {'name': name, 'config': config_path, 'log_path': log_path, **metrics}
        results.append(entry)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[{name}] recorded: val_pearson={metrics["val"]["pearson"] if metrics["val"] else None} '
              f'test_pearson={metrics["test"]["pearson"] if metrics["test"] else None} '
              f'test_r2={metrics["test"]["r2"] if metrics["test"] else None}')

    print(f'\nSweep complete. Results in {args.out}')
    for r in results:
        t = r['test']
        v = r['val']
        print(f'  {r["name"]}: val_pearson={v["pearson"]:.3f} test_pearson={t["pearson"]:.3f} '
              f'test_r2={t["r2"]:.3f} test_rmse={t["rmse"]:.3f}')


if __name__ == '__main__':
    main()
