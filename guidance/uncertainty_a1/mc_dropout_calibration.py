"""A1: MC Dropout calibration of the uncertainty sigma against absolute error on the LP-split held-out test set.

Specification, fixed before any full run:
  model         GIGN-PIGNet Track A checkpoint best.pt (selected by validation loss, the pre-declared criterion).
  dropout       nn.Dropout(0.1) in the HIL layers. No retraining. BatchNorm stays in eval mode.
  T             30 stochastic forward passes per complex, seed 2021.
  mu, sigma     mean and sample std (ddof=1) over the T passes. Error e = |y - mu|.
  primary       Spearman(sigma, e) over all 11,855 test complexes.
  secondary     (i) calibration ratio R = mean(e) / (mean(sigma) * sqrt(2/pi)); 1 if sigma is a calibrated Gaussian scale.
                (ii) partial Spearman(sigma, e | n_lig), controlling for the ligand-size confound.
  CI and p      cluster bootstrap over the 127 test targets, 2,000 resamples, seed 20260925 (as in analyze_track_e.py).
                95% percentile CI. Two-sided percentile p at the null (0 for correlations, 1 for R), floored at 1/B.
  multiplicity  Benjamini-Hochberg across the three tests above.

Usage:
  python guidance/uncertainty_a1/mc_dropout_calibration.py --limit 200   # pilot: timing and sanity only, no analysis
  python guidance/uncertainty_a1/mc_dropout_calibration.py               # full run; resumes from partial.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch_geometric.transforms import Compose

import utils.transforms_prop as ut
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_a.model import GIGNPignetAffinity
from guidance.track_a.train_gign_pignet_stage2 import prepare_sample
from guidance.track_e.analyze_track_e import bh
from guidance.track_e.core import build_anchor_table, split_arrays

CKPT = './logs_track_a_full/gign_pignet_2026_09_11__18_31_46_full/checkpoints/best.pt'
REF = './guidance/track_e/cache/trackA_test_preds.npz'
OUT_DIR = './guidance/uncertainty_a1'
PARTIAL = os.path.join(OUT_DIR, 'partial.npz')
RESULTS = os.path.join(OUT_DIR, 'a1_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1_per_complex.npz')
T, MC_SEED, B, BOOT_SEED, N_SANITY = 30, 2021, 2000, 20260925, 50
N_TEST_TARGETS = 127  # matches len(test_targets) in guidance/lp_split/leakage_safe_split.json
DEVICE = 'cuda'
TESTS = ['spearman_sigma_error', 'calibration_ratio', 'partial_spearman_sigma_error_given_nlig']
NULL = np.array([0.0, 1.0, 0.0])


def load_test():
    A = split_arrays(build_anchor_table(), 'test')
    base, _, pk_by_idx = build_lp_splits(train_subsample=None)
    pf, lf, bf = ut.FeaturizeProteinAtom(), ut.FeaturizeLigandAtom(), ut.FeaturizeLigandBond()
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, Compose([pf, lf, bf]))
    return A, ds


def load_model():
    blob = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    m = GIGNPignetAffinity(blob['protein_atom_feature_dim'], blob['ligand_atom_feature_dim'],
                           hidden_dim=blob['hidden_dim']).to(DEVICE)
    m.load_state_dict(blob['model'])
    m.eval()
    return m


def enable_mc_dropout(model):
    """Dropout modules sampled at inference; BatchNorm and everything else stays in eval mode."""
    model.eval()
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in drops:
        m.train()
    return len(drops)


def partial_spearman(x, y, z):
    """Spearman of x and y after linearly removing the rank of z from both ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def metrics(idx, sd, err, nlig):
    s, e = sd[idx], err[idx]
    rho = spearmanr(s, e).correlation
    ratio = e.mean() / (s.mean() * np.sqrt(2 / np.pi))
    return np.array([rho, ratio, partial_spearman(s, e, nlig[idx])])


def boot_p(samples, null):
    lo, hi = np.mean(samples <= null), np.mean(samples >= null)
    return max(min(1.0, 2 * min(lo, hi)), 1 / B)


def run_scoring(model, ds, n, start, partial_ok, mu, sd):
    torch.manual_seed(MC_SEED + start)
    t0 = time.time()
    with torch.no_grad():
        for i in range(start, n):
            kw = prepare_sample(ds[i], DEVICE)
            passes = [model(**kw)[0].item() for _ in range(T)]
            mu[i], sd[i] = np.mean(passes), np.std(passes, ddof=1)
            if (i + 1) % 200 == 0:
                rate = (i + 1 - start) / (time.time() - t0)
                print(f'{i + 1}/{n}  {rate:.2f} complexes/s  ETA {(n - i - 1) / rate / 60:.1f} min', flush=True)
            if partial_ok and (i + 1) % 500 == 0:
                np.savez(PARTIAL, mu=mu, sd=sd, n_done=i + 1, T=T)
    if partial_ok:
        np.savez(PARTIAL, mu=mu, sd=sd, n_done=n, T=T)
    return mu, sd


def analyse(A, mu, sd):
    err = np.abs(A['pk'] - mu)
    nlig = A['n_lig'].astype(float)
    tg = A['target']

    point = metrics(np.arange(len(mu)), sd, err, nlig)

    uniq, inv = np.unique(tg, return_inverse=True)
    assert len(uniq) == N_TEST_TARGETS, len(uniq)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty((B, len(TESTS)))
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        boot[b] = metrics(np.concatenate([groups[k] for k in pick]), sd, err, nlig)

    p_raw = [boot_p(boot[:, j], NULL[j]) for j in range(len(TESTS))]
    p_bh = bh(p_raw)

    edges = np.quantile(sd, np.linspace(0, 1, 11))
    bin_id = np.clip(np.searchsorted(edges, sd, side='right') - 1, 0, 9)
    bins = [dict(bin=k, n=int((bin_id == k).sum()), mean_sigma=float(sd[bin_id == k].mean()),
                 mean_abs_error=float(err[bin_id == k].mean())) for k in range(10)]

    out = dict(n_test=int(len(mu)), n_targets=int(len(uniq)), T=T, B=B, boot_seed=BOOT_SEED, mc_seed=MC_SEED,
               checkpoint=CKPT, tests={})
    for j, name in enumerate(TESTS):
        out['tests'][name] = dict(point=float(point[j]),
                                  ci95=[float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))],
                                  null=float(NULL[j]), p_raw=float(p_raw[j]), p_bh=float(p_bh[j]))
    out['sigma_bins'] = bins
    np.savez(PER_COMPLEX, idx=A['idx'], target=A['target'], pk=A['pk'], n_lig=A['n_lig'], mu=mu, sd=sd, abs_err=err)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='pilot: score only the first N complexes, no analysis')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    A, ds = load_test()
    ref = np.load(REF, allow_pickle=True)
    assert np.allclose(ref['y_true'], A['pk'], atol=1e-4), 'test order differs from the cached Track A reference'
    model = load_model()

    # Sanity: with dropout off, the forward pass must reproduce the cached Track A prediction.
    with torch.no_grad():
        det = np.array([model(**prepare_sample(ds[i], DEVICE))[0].item() for i in range(N_SANITY)])
    max_diff = float(np.abs(det - ref['y_pred'][:N_SANITY]).max())
    print(f'sanity: eval-mode max |diff| vs cached Track A prediction over {N_SANITY} complexes = {max_diff:.2e}', flush=True)

    n_drop = enable_mc_dropout(model)
    assert n_drop == 2 * len(model.hil_layers), n_drop
    print(f'MC Dropout: {n_drop} dropout modules active; BatchNorm in eval mode', flush=True)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    mu, sd, start = np.full(len(ds), np.nan), np.full(len(ds), np.nan), 0
    partial_ok = args.limit is None
    if partial_ok and os.path.exists(PARTIAL):
        z = np.load(PARTIAL)
        if int(z['T']) == T:
            mu, sd, start = z['mu'], z['sd'], int(z['n_done'])
            print(f'resuming from {start}', flush=True)

    mu, sd = run_scoring(model, ds, n, start, partial_ok, mu, sd)
    if args.limit is not None:
        print(f'pilot done: {n} complexes; no analysis run (pilot is for timing and sanity only)', flush=True)
        return

    out = analyse(A, mu, sd)
    print(json.dumps(out['tests'], indent=2), flush=True)


if __name__ == '__main__':
    main()
