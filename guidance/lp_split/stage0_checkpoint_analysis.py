"""Stage 0 mandatory checkpoint's rigor pass: bootstrap 95% CI on the new
split's test Pearson/R^2 (matching guidance/affinity_sweep_diagnostics.py's
Diagnostic 4 standard), plus the heavy-atom-count shortcut check
(Diagnostic 5a) on this new checkpoint -- to see whether the size-shortcut
signature found on the old split (Pearson 0.799) also weakens here.
"""
import json

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from models.property_pred.prop_model import PropPredNet
from guidance.lp_split.lp_split_loader import build_lp_splits
from datasets.crossdocked_affinity import CrossDockedAffinityDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'


def load_model(ckpt_path, protein_feat_dim, ligand_feat_dim):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=protein_feat_dim,
                        ligand_atom_feature_dim=ligand_feat_dim, output_dim=1).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def get_predictions(model, dataset):
    y_true, y_pred, n_heavy = [], [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            batch_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=DEVICE)
            batch_ligand = torch.zeros(data.ligand_pos.size(0), dtype=torch.long, device=DEVICE)
            pred = model(protein_pos=data.protein_pos.to(DEVICE), protein_atom_feature=data.protein_atom_feature.float().to(DEVICE),
                        ligand_pos=data.ligand_pos.to(DEVICE), ligand_atom_feature=data.ligand_atom_feature_full.float().to(DEVICE),
                        batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
            y_pred.append(pred.view(-1).item())
            y_true.append(float(data.y))
            n_heavy.append(data.ligand_pos.size(0))
    return y_true, y_pred, n_heavy


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    base, splits, pk_by_idx = build_lp_splits(train_subsample=6000)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    model = load_model(CKPT, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
    y_true, y_pred, n_heavy = get_predictions(model, test_set)

    pearson_point = pearsonr(y_true, y_pred)[0]
    r2_point = r2_score(y_true, y_pred)
    p_lo, p_hi = bootstrap_ci(y_true, y_pred, lambda a, b: pearsonr(a, b)[0])
    r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score)
    print(f'Test (n={len(y_true)}): Pearson={pearson_point:.3f} [{p_lo:.3f}, {p_hi:.3f}]  '
         f'R2={r2_point:.3f} [{r2_lo:.3f}, {r2_hi:.3f}]')

    heavy_corr, heavy_p = pearsonr(y_pred, n_heavy)
    print(f'Pearson(pred_affinity, n_heavy_atoms) = {heavy_corr:.3f} (p={heavy_p:.3g}), n={len(y_pred)}')

    results = {
        'test_n': len(y_true), 'pearson_point': pearson_point, 'pearson_ci95': [p_lo, p_hi],
        'r2_point': r2_point, 'r2_ci95': [r2_lo, r2_hi],
        'heavy_atom_corr': heavy_corr, 'heavy_atom_corr_p': heavy_p,
    }
    with open('./guidance/lp_split/stage0_checkpoint_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to ./guidance/lp_split/stage0_checkpoint_analysis_results.json')


if __name__ == '__main__':
    main()
