"""A1f analysis: same pre-registered 3 tests as A1/A1b/A1c/A1e, imported directly from
mc_dropout_calibration so the spec cannot drift. sigma = sqrt(beta / (nu * (alpha - 1))), the Normal-
Inverse-Gamma predictive variance (Amini et al. 2020's convention), evaluated once per test complex
(deterministic forward pass).

Usage:
  python guidance/uncertainty_a1/analyze_evidential.py
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from models.property_pred.prop_model import PropPredNet
from guidance.uncertainty_a1 import mc_dropout_calibration as a1

RUN_GLOB = './logs_a1f_evidential/crossdocked_affinity_egnn_*_a1f_s*/checkpoints/best.pt'
OUT_DIR = './guidance/uncertainty_a1'
RESULTS = os.path.join(OUT_DIR, 'a1f_results.json')
PER_COMPLEX = os.path.join(OUT_DIR, 'a1f_per_complex.npz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def find_ckpt():
    paths = sorted(glob.glob(RUN_GLOB))
    assert len(paths) == 1, f'expected one A1f run folder with best.pt, found {paths}'
    return paths[0]


def load_model(ckpt):
    blob = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    m = PropPredNet(blob['config'].model, protein_atom_feature_dim=blob['protein_atom_feature_dim'],
                    ligand_atom_feature_dim=blob['ligand_atom_feature_dim'], output_dim=4).to(DEVICE)
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
    gamma = out[0, 0].item()
    nu = F.softplus(out[0, 1]).item()
    alpha = F.softplus(out[0, 2]).item() + 1.0
    beta = F.softplus(out[0, 3]).item()
    var = beta / (nu * (alpha - 1))
    return gamma, np.sqrt(var)


def main():
    ckpt = find_ckpt()
    print('checkpoint:', ckpt, flush=True)
    model, blob = load_model(ckpt)
    A, ds = load_test()
    print(f'test complexes: {len(ds)}  (checkpoint epoch {blob["epoch"]}, val loss {blob["val_loss"]:.4f})', flush=True)

    n = len(ds)
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    with torch.no_grad():
        for i in range(n):
            mu[i], sd[i] = forward(model, ds[i])
            if (i + 1) % 2000 == 0:
                print(f'{i + 1}/{n}', flush=True)

    print(f'sigma stats: mean={sd.mean():.3f} std={sd.std():.3f} median={np.median(sd):.3f} '
          f'min={sd.min():.3f} max={sd.max():.3f}', flush=True)

    a1.CKPT, a1.RESULTS, a1.PER_COMPLEX = ckpt, RESULTS, PER_COMPLEX
    out = a1.analyse(A, mu, sd)
    y = A['pk']
    out['descriptive_test'] = dict(
        r2_of_mu=float(1 - ((y - mu) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        pearson_of_mu=float(np.corrcoef(y, mu)[0, 1]),
        sigma_mean=float(sd.mean()), sigma_std=float(sd.std()), sigma_median=float(np.median(sd)))
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'tests': out['tests'], 'descriptive_test': out['descriptive_test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
