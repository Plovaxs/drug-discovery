"""A1e analysis: same 3 pre-registered tests as A1/A1b/A1c (Spearman(sigma,|err|), calibration ratio,
partial Spearman given n_lig), cluster bootstrap B=2,000 seed 20260925, BH correction -- imported directly
from mc_dropout_calibration so the spec cannot drift from the other three methods. Only the uncertainty
SOURCE differs: here sigma = exp(0.5*log_var), the model's own learned heteroscedastic variance head,
evaluated once (deterministic forward pass, no MC sampling and no ensembling -- this method's sigma comes
from training, not from inference-time stochasticity).

Usage:
  python guidance/uncertainty_a1/analyze_heteroscedastic.py
"""
import glob
import json
import os

import numpy as np
import torch
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from models.property_pred.prop_model import PropPredNet
from guidance.uncertainty_a1 import mc_dropout_calibration as a1

RUN_GLOB = './logs_a1e_heteroscedastic/crossdocked_affinity_egnn_*_a1e_s*/checkpoints/best.pt'
OUT_DIR = './guidance/uncertainty_a1'
RESULTS = os.path.join(OUT_DIR, 'a1e_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1e_per_complex.npz')
DEVICE = 'cpu'  # deliberate: GPU is busy training A1f (evidential); this runs in parallel on CPU
LOG_VAR_CLAMP = (-6.0, 6.0)


def find_ckpt():
    paths = sorted(glob.glob(RUN_GLOB))
    assert len(paths) == 1, f'expected one A1e run folder with best.pt, found {paths}'
    return paths[0]


def load_model(ckpt):
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m = PropPredNet(blob['config'].model, protein_atom_feature_dim=blob['protein_atom_feature_dim'],
                    ligand_atom_feature_dim=blob['ligand_atom_feature_dim'], output_dim=2).to(DEVICE)
    m.load_state_dict(blob['model'])
    m.eval()
    return m, blob


def load_test():
    A = split_arrays(build_anchor_table(), 'test')
    base, _, pk_by_idx = build_lp_splits(train_subsample=None)
    transform = Compose([utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()])
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, transform)
    return A, ds


def forward(model, data):
    bp = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=DEVICE)
    bl = torch.zeros(data.ligand_pos.size(0), dtype=torch.long, device=DEVICE)
    out = model(protein_pos=data.protein_pos.to(DEVICE), protein_atom_feature=data.protein_atom_feature.float().to(DEVICE),
               ligand_pos=data.ligand_pos.to(DEVICE), ligand_atom_feature=data.ligand_atom_feature_full.float().to(DEVICE),
               batch_protein=bp, batch_ligand=bl, output_kind=None)
    mu, log_var = out[0, 0].item(), out[0, 1].clamp(*LOG_VAR_CLAMP).item()
    return mu, np.exp(0.5 * log_var)


def main():
    ckpt = find_ckpt()
    print('checkpoint:', ckpt, flush=True)
    model, blob = load_model(ckpt)
    A, ds = load_test()
    print(f'test complexes: {len(ds)}  (checkpoint epoch {blob["epoch"]}, val NLL {blob["val_loss"]:.4f})', flush=True)

    n = len(ds)
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    with torch.no_grad():
        for i in range(n):
            mu[i], sd[i] = forward(model, ds[i])
            if (i + 1) % 2000 == 0:
                print(f'{i + 1}/{n}', flush=True)

    a1.CKPT, a1.RESULTS, a1.PER_COMPLEX = ckpt, RESULTS, PER_COMPLEX
    out = a1.analyse(A, mu, sd)
    y = A['pk']
    out['descriptive_test'] = dict(
        r2_of_mu=float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        pearson_of_mu=float(np.corrcoef(y, mu)[0, 1]))
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'tests': out['tests'], 'descriptive_test': out['descriptive_test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
