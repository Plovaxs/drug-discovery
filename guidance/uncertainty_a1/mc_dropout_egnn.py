"""A1b: the A1 protocol on the EGNN Stage 0 architecture, retrained with dropout p=0.1 in each node update.

Same specification as A1 (guidance/uncertainty_a1/mc_dropout_calibration.py): T=30 passes, seed 2021, cluster bootstrap
over the 127 test targets (B=2000, seed 20260925), BH across the three tests. Checkpoint selected by validation loss
(best epoch in the A1b training log). The test set was not evaluated during A1b training (--skip_test_logging).

Usage:
  python guidance/uncertainty_a1/mc_dropout_egnn.py --limit 200   # pilot: timing and sanity only, no analysis
  python guidance/uncertainty_a1/mc_dropout_egnn.py               # full run
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.analyze_track_e import bh
from guidance.track_e.core import build_anchor_table, split_arrays
from guidance.uncertainty_a1.mc_dropout_calibration import (
    B, BOOT_SEED, MC_SEED, N_TEST_TARGETS, NULL, T, TESTS, boot_p, metrics)
from models.property_pred.prop_model import PropPredNet

CKPT = './logs_a1b_egnn_mcdrop/crossdocked_affinity_egnn_mcdrop_2026_10_06__21_09_08_a1b/checkpoints/best.pt'
OUT_DIR = './guidance/uncertainty_a1'
RESULTS = os.path.join(OUT_DIR, 'a1b_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1b_per_complex.npz')
DEVICE = 'cuda'


def load_test():
    A = split_arrays(build_anchor_table(), 'test')
    base, _, pk_by_idx = build_lp_splits(train_subsample=None)
    transform = Compose([utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()])
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, follow_batch=['protein_element', 'ligand_element'],
                        exclude_keys=['ligand_nbh_list'])
    return A, loader


def load_model():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = PropPredNet(ck['config'].model, ck['protein_atom_feature_dim'], ck['ligand_atom_feature_dim'],
                        output_dim=1).to(DEVICE)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, ck


def forward(model, batch):
    """Same inputs as get_loss in guidance/lp_split/train_egnn_stage0.py, without position noise."""
    batch = batch.to(DEVICE)
    pred = model(
        protein_pos=batch.protein_pos,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=batch.ligand_pos,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        output_kind=None,
    )
    return pred.view(-1)[0].item()


def deterministic_pass(model, loader):
    with torch.no_grad():
        return np.array([forward(model, batch) for batch in loader])


def enable_mc_dropout(model):
    """Only the dropout modules with p > 0 are sampled; the model has no BatchNorm."""
    model.eval()
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    for m in drops:
        m.train()
    return len(drops)


def mc_pass(model, loader, n, log_every=200):
    torch.manual_seed(MC_SEED)
    mu, sd = np.empty(n), np.empty(n)
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            passes = [forward(model, batch) for _ in range(T)]
            mu[i], sd[i] = np.mean(passes), np.std(passes, ddof=1)
            if (i + 1) % log_every == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'{i + 1}/{n}  {rate:.2f} complexes/s  ETA {(n - i - 1) / rate / 60:.1f} min', flush=True)
    return mu, sd


def analyse(A, mu, sd, det):
    y = A['pk']
    err = np.abs(y - mu)
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

    ss_res = np.sum((y - det) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    deterministic = dict(r2=float(1 - ss_res / ss_tot), pearson=float(np.corrcoef(y, det)[0, 1]))

    out = dict(model='EGNN Stage 0 architecture + node-update dropout p=0.1 (A1b)', checkpoint=CKPT,
               n_test=int(len(mu)), n_targets=int(len(uniq)), T=T, B=B, boot_seed=BOOT_SEED, mc_seed=MC_SEED,
               deterministic_test=deterministic, tests={})
    for j, name in enumerate(TESTS):
        out['tests'][name] = dict(point=float(point[j]),
                                  ci95=[float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))],
                                  null=float(NULL[j]), p_raw=float(p_raw[j]), p_bh=float(p_bh[j]))
    out['sigma_bins'] = bins
    np.savez(PER_COMPLEX, idx=A['idx'], target=A['target'], pk=y, n_lig=A['n_lig'], mu=mu, sd=sd, abs_err=err, det=det)
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='pilot: score only the first N complexes, no analysis')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    A, loader = load_test()
    model, ck = load_model()
    print(f'loaded A1b checkpoint, trained epoch {ck["epoch"]}, val loss {ck["val_loss"]:.3f}', flush=True)

    if args.limit is not None:
        n = min(args.limit, len(loader.dataset))
        loader_used = DataLoader(Subset(loader.dataset, range(n)), batch_size=1, shuffle=False,
                                 follow_batch=['protein_element', 'ligand_element'], exclude_keys=['ligand_nbh_list'])
    else:
        n = len(loader.dataset)
        loader_used = loader

    det = deterministic_pass(model, loader_used)
    n_drop = enable_mc_dropout(model)
    assert n_drop == model.encoder.num_layers, (n_drop, model.encoder.num_layers)
    print(f'MC Dropout: {n_drop} dropout modules active (one per EGNN layer)', flush=True)

    t0 = time.time()
    mu, sd = mc_pass(model, loader_used, n)
    print(f'MC pass finished in {(time.time() - t0) / 60:.1f} min for {n} complexes', flush=True)

    if args.limit is not None:
        print('pilot done: no analysis run (pilot is for timing and sanity only)', flush=True)
        return

    out = analyse(A, mu, sd, det)
    print(json.dumps(out['deterministic_test'], indent=2), flush=True)
    print(json.dumps(out['tests'], indent=2), flush=True)


if __name__ == '__main__':
    main()
