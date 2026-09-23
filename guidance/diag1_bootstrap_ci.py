"""Bootstrap 95% CIs for each learning-curve checkpoint's test Pearson/R2,
matching Diagnostic 4's method -- needed to judge whether the 25/50/75%
points' apparent flatness and the jump at 100% are real or within noise
given only 27 test pockets.
"""
import json

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import build_labeled_splits, CrossDockedAffinityDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINTS = {
    0.25: './logs_affinity_diag1/frac0.25/lr_3e-5_2026_09_07__16_30_00/checkpoints/best.pt',
    0.5: './logs_affinity_diag1/frac0.5/lr_3e-5_2026_09_07__17_25_27/checkpoints/best.pt',
    0.75: './logs_affinity_diag1/frac0.75/lr_3e-5_2026_09_07__19_14_34/checkpoints/best.pt',
    1.0: './logs_affinity_sweep/lr_3e-5_layers6/lr_3e-5_2026_09_06__23_57_10/checkpoints/best.pt',
}


def load_model(ckpt_path, protein_feat_dim, ligand_feat_dim):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=protein_feat_dim,
                        ligand_atom_feature_dim=ligand_feat_dim, output_dim=1).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def get_predictions(model, dataset):
    y_true, y_pred = [], []
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
    return y_true, y_pred


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
    base, splits, pk_by_idx = build_labeled_splits(train_subsample=6000)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    results = {}
    for frac, ckpt_path in CHECKPOINTS.items():
        model = load_model(ckpt_path, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
        y_true, y_pred = get_predictions(model, test_set)
        pearson_point = pearsonr(y_true, y_pred)[0]
        r2_point = r2_score(y_true, y_pred)
        p_lo, p_hi = bootstrap_ci(y_true, y_pred, lambda a, b: pearsonr(a, b)[0])
        r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_score)
        results[frac] = {
            'pearson_point': pearson_point, 'pearson_ci95': [p_lo, p_hi],
            'r2_point': r2_point, 'r2_ci95': [r2_lo, r2_hi],
        }
        print(f'fraction={frac}: Pearson={pearson_point:.3f} [{p_lo:.3f}, {p_hi:.3f}]  '
             f'R2={r2_point:.3f} [{r2_lo:.3f}, {r2_hi:.3f}]')

    with open('./guidance/diag1_bootstrap_ci_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
