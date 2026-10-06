"""A1b: the A1 calibration test, repeated on the EGNN (Stage 0 architecture) trained with node-update dropout p = 0.1.

Same specification as mc_dropout_calibration.py (A1). Only the model differs: the EGNN PropPredNet
(checkpoint from logs_a1b_egnn_mcdrop, selected by validation loss) instead of GIGN-PIGNet.
Statistics, bootstrap and BH are imported from A1 so the two tests cannot drift apart.

Usage:
  python guidance/uncertainty_a1/mc_dropout_egnn_a1b.py
"""
import glob
import json
import os
import time

import numpy as np
import torch
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from models.property_pred.prop_model import PropPredNet
from guidance.uncertainty_a1 import mc_dropout_calibration as a1

RUN_GLOB = './logs_a1b_egnn_mcdrop/crossdocked_affinity_egnn_mcdrop_*_a1b/checkpoints/best.pt'
OUT_DIR = './guidance/uncertainty_a1'
DEVICE = 'cuda'
T, MC_SEED = a1.T, a1.MC_SEED
PARTIAL = os.path.join(OUT_DIR, 'a1b_partial.npz')
RESULTS = os.path.join(OUT_DIR, 'a1b_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1b_per_complex.npz')


def find_ckpt():
    paths = sorted(glob.glob(RUN_GLOB))
    assert len(paths) == 1, f'expected one A1b run folder with best.pt, found {paths}'
    return paths[0]


def load_model(ckpt):
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m = PropPredNet(blob['config'].model, protein_atom_feature_dim=blob['protein_atom_feature_dim'],
                    ligand_atom_feature_dim=blob['ligand_atom_feature_dim'], output_dim=1).to(DEVICE)
    m.load_state_dict(blob['model'])
    m.eval()
    return m, blob


def load_test():
    """Test complexes in the same order as the anchor table (and therefore as the A1 analysis)."""
    A = split_arrays(build_anchor_table(), 'test')
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    transform = Compose([utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()])
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, transform)
    return A, ds


def forward(model, data):
    batch_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=DEVICE)
    batch_ligand = torch.zeros(data.ligand_pos.size(0), dtype=torch.long, device=DEVICE)
    pred = model(protein_pos=data.protein_pos.to(DEVICE), protein_atom_feature=data.protein_atom_feature.float().to(DEVICE),
                 ligand_pos=data.ligand_pos.to(DEVICE), ligand_atom_feature=data.ligand_atom_feature_full.float().to(DEVICE),
                 batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
    return pred.view(-1)[0].item()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = find_ckpt()
    print('checkpoint:', ckpt, flush=True)
    model, blob = load_model(ckpt)
    A, ds = load_test()
    print(f'test complexes: {len(ds)}  (checkpoint epoch {blob["epoch"]}, val loss {blob["val_loss"]:.4f})', flush=True)

    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(drops) == 6 and all(m.p == 0.1 for m in drops), [m.p for m in drops]

    # Sanity 1: eval mode is deterministic. Sanity 2: MC mode actually varies the output.
    with torch.no_grad():
        d1 = forward(model, ds[0]); d2 = forward(model, ds[0])
        assert abs(d1 - d2) < 1e-5, 'eval mode is not deterministic'
        a1.enable_mc_dropout(model)
        torch.manual_seed(MC_SEED)
        passes = [forward(model, ds[0]) for _ in range(T)]
        model.eval()
    print(f'sanity: eval deterministic; MC passes on complex 0 range {min(passes):.3f}..{max(passes):.3f}', flush=True)
    assert np.std(passes) > 0, 'dropout had no effect'

    n = len(ds)
    mu, sd, start = np.full(n, np.nan), np.full(n, np.nan), 0
    if os.path.exists(PARTIAL):
        z = np.load(PARTIAL)
        if int(z['T']) == T:
            mu, sd, start = z['mu'], z['sd'], int(z['n_done'])
            print(f'resuming from {start}', flush=True)

    a1.enable_mc_dropout(model)
    t0 = time.time()
    with torch.no_grad():
        for i in range(start, n):
            torch.manual_seed(MC_SEED + i)
            passes = [forward(model, ds[i]) for _ in range(T)]
            mu[i], sd[i] = np.mean(passes), np.std(passes, ddof=1)
            if (i + 1) % 200 == 0:
                rate = (i + 1 - start) / (time.time() - t0)
                print(f'{i + 1}/{n}  {rate:.2f} complexes/s  ETA {(n - i - 1) / rate / 60:.1f} min', flush=True)
            if (i + 1) % 500 == 0:
                np.savez(PARTIAL, mu=mu, sd=sd, n_done=i + 1, T=T)
    np.savez(PARTIAL, mu=mu, sd=sd, n_done=n, T=T)

    # Same analysis code as A1; only the output paths and the checkpoint label differ.
    a1.CKPT, a1.RESULTS, a1.PER_COMPLEX = ckpt, RESULTS, PER_COMPLEX
    out = a1.analyse(A, mu, sd)

    # Descriptive only (not part of the locked test): test performance of the mean prediction.
    y, m = A['pk'], mu
    out['descriptive_test'] = dict(r2_of_mc_mean=float(1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                                   pearson_of_mc_mean=float(np.corrcoef(y, m)[0, 1]))
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'tests': out['tests'], 'descriptive_test': out['descriptive_test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
