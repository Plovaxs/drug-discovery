"""Resume-aware driver for the A1c deep-ensemble training chain (5 EGNN Stage 0 seeds: 2021 reused +
2022-2025 trained here). Written so the whole chain can be killed at any point -- e.g. to let a laptop GPU
cool down from thermal throttling -- and continued later by just rerunning this exact script: it inspects
each seed's run directory and decides to skip (already finished), resume (--resume last.pt), or start fresh.

Per-seed decision:
  - No run directory for this seed yet -> start fresh.
  - A run directory exists but has no checkpoints/last.pt -> it was killed before completing even one
    validation epoch; nothing to resume from, so start a fresh run (old partial directory is left in place,
    harmless, just not reused).
  - checkpoints/last.pt exists -> load it (CPU, just to read the small metadata fields) and check whether
    its own patience_count had already reached early_stop_patience (i.e. it had already finished/early-
    stopped before being killed, or validate() was mid-way through what turned out to be the final val
    epoch) -- if so, treat as done; otherwise resume training with --resume pointing at that file.

Requires train_egnn_stage0.py's --resume support (added alongside this script, 2026-10-08) -- a run started
under the OLD version of that script (no --resume) has a best.pt but no last.pt and is correctly detected
as "start fresh" above only if truly nothing was ever saved; if it has a best.pt from a COMPLETED old-style
run (no last.pt, because the run never needed to resume), see the explicit completed-seed check below.

Usage:
  python guidance/surrogate_data/run_a1c_ensemble.py
"""
import glob
import os
import subprocess
import sys

import torch

CONFIG = 'configs/prop/crossdocked_affinity_egnn.yml'
LOGDIR = './logs_a1c_ensemble'
SEEDS = [2022, 2023, 2024, 2025]
EARLY_STOP_PATIENCE = 5  # matches configs/prop/crossdocked_affinity_egnn.yml train.early_stop_patience
PYTHON = os.path.expanduser('~/miniconda3/envs/drugdisc/bin/python')


def run_dir_for(seed):
    matches = sorted(glob.glob(os.path.join(LOGDIR, f'crossdocked_affinity_egnn_*_a1c_s{seed}')))
    return matches[-1] if matches else None


def ran_to_completion(run_dir):
    """True iff main()'s unconditional final log line was reached -- printed after the try/except
    block on early-stop, max_epochs, OR a caught KeyboardInterrupt. False means an abrupt kill
    (SIGTERM/SIGKILL) mid-epoch, for a run with no last.pt to resume from -- that best.pt is only
    an early improved checkpoint, not a finished answer, so it should NOT be trusted as 'done'."""
    log_path = os.path.join(run_dir, 'log.txt')
    if not os.path.exists(log_path):
        return False
    with open(log_path) as f:
        tail = f.readlines()[-5:]
    return any('Best val loss:' in line for line in tail)


def decide(seed):
    d = run_dir_for(seed)
    if d is None:
        return 'fresh', None
    best = os.path.join(d, 'checkpoints', 'best.pt')
    last = os.path.join(d, 'checkpoints', 'last.pt')
    if not os.path.exists(last):
        if os.path.exists(best) and ran_to_completion(d):
            return 'done', d  # completed (or gracefully Ctrl+C'd) under the pre-resume version of the script
        return 'fresh', None  # no usable checkpoint to resume from (never saved, or killed abruptly mid-epoch)
    ckpt = torch.load(last, map_location='cpu', weights_only=False)
    if ckpt['patience_count'] >= EARLY_STOP_PATIENCE:
        return 'done', d
    return 'resume', last


def main():
    for seed in SEEDS:
        action, path = decide(seed)
        if action == 'done':
            print(f'seed {seed}: already finished ({path}), skipping.', flush=True)
            continue
        cmd = [PYTHON, 'guidance/lp_split/train_egnn_stage0.py', CONFIG,
               '--skip_test_logging', '--logdir', LOGDIR]
        if action == 'resume':
            saved_seed = torch.load(path, map_location='cpu', weights_only=False)['seed']
            assert saved_seed == seed, (path, saved_seed, seed)
            cmd += ['--resume', path, '--seed', str(saved_seed)]
            print(f'seed {seed}: resuming from {path} (seed {saved_seed})', flush=True)
        else:
            cmd += ['--seed', str(seed), '--tag', f'a1c_s{seed}']
            print(f'seed {seed}: starting fresh', flush=True)
        ret = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': '.'}).returncode
        if ret != 0:
            print(f'seed {seed} exited with code {ret}, stopping the chain here.', flush=True)
            sys.exit(ret)
    print('All 4 seeds done.', flush=True)


if __name__ == '__main__':
    main()
