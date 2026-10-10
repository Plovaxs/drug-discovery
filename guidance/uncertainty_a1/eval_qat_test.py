"""Evaluates a QAT ternary checkpoint (guidance/uncertainty_a1/train_egnn_qat_ternary.py) on the actual
held-out TEST set -- training/model selection only ever saw train+val, so this is the first time this
checkpoint's weights are scored against test, matching this project's standing test-set-until-the-end
discipline (same as A1b's --skip_test_logging).

Usage:
  python guidance/uncertainty_a1/eval_qat_test.py --seed 2021
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from models.property_pred.prop_model import PropPredNet
from guidance.tnn_qat import convert_to_ternary_qat

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def write_metrics_json(run_dir, seed, y_true, y_pred):
    """Writes test metrics to <run_dir>/metrics.json, the file guidance/aggregate_results.py looks for.

    Kept as a module-level function rather than inlined so the three seeds already evaluated before this
    was added can be backfilled from their saved .npz predictions without re-running inference on the
    GPU -- the predictions are the evidence, the metrics are just a view of them."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    metrics = dict(seed=int(seed), n=int(y_true.size), split='test',
                   R2=float(r2_score(y_true, y_pred)),
                   Pearson=float(pearsonr(y_true, y_pred)[0]),
                   Spearman=float(spearmanr(y_true, y_pred)[0]),
                   RMSE=float(np.sqrt(((y_true - y_pred) ** 2).mean())),
                   MAE=float(np.abs(y_true - y_pred).mean()),
                   source='guidance/uncertainty_a1/eval_qat_test.py')
    path = os.path.join(run_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'wrote {path}')
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(f'./logs_a1g_qat_ternary/crossdocked_affinity_egnn_*_a1g_s{args.seed}/checkpoints/best.pt'))
    assert len(paths) == 1, f'expected one run for seed {args.seed}, found {paths}'
    ckpt = torch.load(paths[0], map_location=DEVICE, weights_only=False)
    print(f'checkpoint: {paths[0]} (epoch {ckpt["epoch"]}, val_loss {ckpt["val_loss"]:.4f})')

    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(DEVICE)
    convert_to_ternary_qat(model)
    model.load_state_dict(ckpt['model'])
    model.eval()

    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, Compose([pf, lf]))

    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(test_set)):
            d = test_set[i]
            bp = torch.zeros(d.protein_pos.size(0), dtype=torch.long, device=DEVICE)
            bl = torch.zeros(d.ligand_pos.size(0), dtype=torch.long, device=DEVICE)
            pred = model(protein_pos=d.protein_pos.to(DEVICE), protein_atom_feature=d.protein_atom_feature.float().to(DEVICE),
                        ligand_pos=d.ligand_pos.to(DEVICE), ligand_atom_feature=d.ligand_atom_feature_full.float().to(DEVICE),
                        batch_protein=bp, batch_ligand=bl, output_kind=None)
            y_pred.append(pred.view(-1).item())
            y_true.append(float(d.y))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    print(f'TEST (n={len(y_true)}): R2={r2:.4f}  Pearson={pearson:.4f}  RMSE={rmse:.4f}  '
          f'(FP32 baseline: R2=0.342, Pearson=0.609)')
    np.savez(f'./guidance/uncertainty_a1/a1g_test_preds_s{args.seed}.npz', y_true=y_true, y_pred=y_pred)

    # Also persist the metrics as JSON inside the run directory. Printing them to stdout only meant the
    # multi-seed test table had to be reassembled by hand from terminal scrollback, which is how the
    # n=1 and n=2 misreadings of this very experiment happened. guidance/aggregate_results.py reads this
    # file, so the numbers it tabulates come from disk rather than from memory.
    write_metrics_json(os.path.dirname(os.path.dirname(paths[0])), args.seed, y_true, y_pred)


if __name__ == '__main__':
    main()
