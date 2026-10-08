"""A1c: deep-ensemble calibration of uncertainty against absolute error on the LP-split held-out test set.

Written and committed to BEFORE any of the 4 new training runs finish, per this project's pre-registration
convention (A1/A1b). Only the model differs from A1/A1b: 5 independently-seeded EGNN Stage 0 checkpoints
(standard training, no dropout -- diversity comes from independent training runs, not MC sampling at
inference), instead of one checkpoint sampled stochastically.

Specification, fixed before any ensemble member but one has finished training:
  model         EGNN Stage 0 architecture (configs/prop/crossdocked_affinity_egnn.yml), 5 independently
                seeded runs: 2021 (reused -- the existing deployed Stage 0 checkpoint,
                logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt,
                test R^2 0.342), 2022, 2023, 2024, 2025 (freshly trained via
                guidance/lp_split/train_egnn_stage0.py --seed <N> --skip_test_logging, each selected by
                validation loss, the pre-declared criterion, same as the original checkpoint).
  mu, sigma     mean and sample std (ddof=1) over the 5 members' single deterministic forward pass per
                complex (eval mode, no MC sampling -- there is nothing stochastic to sample: the ensemble's
                only source of randomness is across training runs, not within one). Error e = |y - mu|.
  primary       Spearman(sigma, e) over all 11,855 test complexes.
  secondary     (i) calibration ratio R = mean(e) / (mean(sigma) * sqrt(2/pi)); 1 if sigma is a calibrated
                    Gaussian scale (same convention as A1/A1b, for comparability, even though a 5-member
                    ensemble's sigma is a much coarser scale estimate than a T=30 MC estimate).
                (ii) partial Spearman(sigma, e | n_lig), controlling for the ligand-size confound.
  CI and p      cluster bootstrap over the 127 test targets, 2,000 resamples, seed 20260925 (identical code
                to A1/A1b, imported directly so the three tests cannot drift apart).
  multiplicity  Benjamini-Hochberg across the three tests above.

This is the uncertainty-branch deep-ensemble comparison TASKS.md records as opened 2026-10-07 (user decision:
continue past A1/A1b's negative MC-Dropout result rather than close with a negative writeup). Caveat carried
over regardless of outcome: n=5 is a small ensemble (the literature norm is often 5-10); this is a real
calibration test, not a definitive ceiling on deep ensembles in general.

Usage:
  python guidance/uncertainty_a1/deep_ensemble_a1c.py --limit 200   # pilot: timing and sanity only
  python guidance/uncertainty_a1/deep_ensemble_a1c.py               # full run once all 5 checkpoints exist
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from torch_geometric.transforms import Compose
from guidance.uncertainty_a1 import mc_dropout_calibration as a1
from guidance.uncertainty_a1.mc_dropout_egnn_a1b import forward, load_model

OUT_DIR = './guidance/uncertainty_a1'
PARTIAL = os.path.join(OUT_DIR, 'a1c_partial.npz')
RESULTS = os.path.join(OUT_DIR, 'a1c_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1c_per_complex.npz')
DEVICE = 'cuda'

ORIGINAL_CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'
NEW_CKPT_GLOB = './logs_a1c_ensemble/crossdocked_affinity_egnn_*_a1c_s*/checkpoints/best.pt'
EXPECTED_SEEDS = [2021, 2022, 2023, 2024, 2025]
N_MEMBERS = 5


def find_ckpts():
    """Orders checkpoints by the seed actually recorded inside each one (not by filename/glob order,
    which depends on launch order and is not guaranteed to match seed order)."""
    candidates = [ORIGINAL_CKPT] + sorted(glob.glob(NEW_CKPT_GLOB))
    by_seed = {}
    for p in candidates:
        assert os.path.exists(p), f'missing checkpoint: {p}'
        blob = torch.load(p, map_location='cpu', weights_only=False)
        seed = blob['config'].train.seed
        assert seed not in by_seed, f'duplicate seed {seed}: {by_seed[seed]} and {p}'
        by_seed[seed] = p
    missing = set(EXPECTED_SEEDS) - set(by_seed)
    assert not missing, f'missing seeds {missing}; have {sorted(by_seed)}'
    return [by_seed[s] for s in EXPECTED_SEEDS]


def load_test():
    A = split_arrays(build_anchor_table(), 'test')
    base, _, pk_by_idx = build_lp_splits(train_subsample=None)
    transform = Compose([utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()])
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, transform)
    return A, ds


def score_member(model, ds, n):
    preds = np.full(n, np.nan)
    with torch.no_grad():
        for i in range(n):
            preds[i] = forward(model, ds[i])
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='pilot: score only the first N complexes')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_paths = find_ckpts()
    A, ds = load_test()
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f'test complexes: {n}  ensemble members: {len(ckpt_paths)}', flush=True)

    member_preds = np.full((N_MEMBERS, n), np.nan)
    for m, path in enumerate(ckpt_paths):
        model, blob = load_model(path)
        seed_used = blob['config'].train.seed
        assert seed_used == EXPECTED_SEEDS[m], (path, seed_used, EXPECTED_SEEDS[m])
        t0 = time.time()
        member_preds[m] = score_member(model, ds, n)
        print(f'member {m} (seed {seed_used}, ckpt epoch {blob["epoch"]}, val_loss {blob["val_loss"]:.4f}): '
              f'{n} complexes in {time.time() - t0:.0f}s', flush=True)
        del model

    if args.limit is not None:
        print('pilot done: no analysis run (pilot is for timing and sanity only)', flush=True)
        return

    mu = member_preds.mean(axis=0)
    sd = member_preds.std(axis=0, ddof=1)
    a1.CKPT, a1.RESULTS, a1.PER_COMPLEX = ckpt_paths, RESULTS, PER_COMPLEX
    out = a1.analyse(A, mu, sd)
    out['n_members'] = N_MEMBERS
    out['seeds'] = EXPECTED_SEEDS
    out['checkpoints'] = ckpt_paths

    y = A['pk']
    out['descriptive_test'] = dict(
        r2_of_ensemble_mean=float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        pearson_of_ensemble_mean=float(np.corrcoef(y, mu)[0, 1]))
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'tests': out['tests'], 'descriptive_test': out['descriptive_test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
