"""A2-full's rigor pass: bootstrap 95% CI on test Pearson/R^2/Spearman,
plus the heavy-atom-count shortcut check (matching Stage 0's own
diagnostic, guidance/lp_split/stage0_checkpoint_analysis.py) -- the
original Stage 2 target was a reduction to roughly below 0.4 from Stage
0's 0.719; this reports the actual observed value regardless of whether
it meets that target.
"""
import argparse

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from guidance.lp_split.lp_split_loader import build_lp_splits
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.track_a.model import GIGNPignetAffinity
from guidance.track_a.train_gign_pignet_stage2 import get_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = GIGNPignetAffinity(
        protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
        ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
        hidden_dim=ckpt['hidden_dim'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def get_predictions(model, dataset):
    y_true, y_pred, n_heavy = [], [], []
    with torch.no_grad():
        for data in dataset:
            loss, pred, _ = get_loss(model, data, DEVICE)
            y_pred.append(pred.item())
            y_true.append(float(data.y))
            n_heavy.append(data.ligand_pos.size(0))
    return np.array(y_true), np.array(y_pred), np.array(n_heavy)


def bootstrap_ci(a, b, metric_fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(metric_fn(a[idx], b[idx]))
        except Exception:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    model, ckpt = load_model(args.ckpt)
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch")}, '
         f'val_loss={ckpt.get("val_loss")}, val_pearson={ckpt.get("val_pearson")}')

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    bond_featurizer = utils_trans.FeaturizeLigandBond()
    transform = Compose([protein_featurizer, ligand_featurizer, bond_featurizer])

    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    y_true, y_pred, n_heavy = get_predictions(model, test_set)
    print(f'n_test={len(y_true)}')

    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    pearson_lo, pearson_hi = bootstrap_ci(y_true, y_pred, lambda a, b: pearsonr(a, b)[0])
    spearman_lo, spearman_hi = bootstrap_ci(y_true, y_pred, lambda a, b: spearmanr(a, b)[0])
    r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score)

    print(f'Test Pearson:  {pearson:.4f} [{pearson_lo:.4f}, {pearson_hi:.4f}]')
    print(f'Test Spearman: {spearman:.4f} [{spearman_lo:.4f}, {spearman_hi:.4f}]')
    print(f'Test R2:       {r2:.4f} [{r2_lo:.4f}, {r2_hi:.4f}]')
    print(f'Test RMSE:     {rmse:.4f}')

    heavy_corr = pearsonr(n_heavy.astype(float), y_pred)[0]
    heavy_lo, heavy_hi = bootstrap_ci(n_heavy.astype(float), y_pred, lambda a, b: pearsonr(a, b)[0])
    print(f'Heavy-atom-count vs prediction correlation: {heavy_corr:.4f} [{heavy_lo:.4f}, {heavy_hi:.4f}] '
         f'(Stage 0 EGNN: 0.719; original Stage 2 target: <0.4)')


if __name__ == '__main__':
    main()
