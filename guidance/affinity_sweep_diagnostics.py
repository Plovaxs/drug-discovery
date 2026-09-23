"""Diagnostics 2-5 from the generalization-gap investigation addendum.
Reuses the best sweep checkpoint (lr_3e-5) and the same leakage-safe
train/test split -- no new full training run needed for any of these.

Diagnostic 2: per-target error/Pearson breakdown (with small-n caveats).
Diagnostic 3: ligand-only ECFP4 + Ridge/RF baseline (ignores pocket).
Diagnostic 4: bootstrap 95% CIs on test Pearson/R2 for best config + baseline.
Diagnostic 5: (a) predicted-affinity vs heavy-atom-count correlation
              (b) pocket-scrambling sensitivity check.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

import utils.transforms_prop as utils_trans
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import build_labeled_splits, CrossDockedAffinityDataset
from utils import reconstruct
import utils.transforms as trans

RDLogger.DisableLog('rdApp.*')

BEST_CKPT = './logs_affinity_sweep/lr_3e-5_layers6/lr_3e-5_2026_09_06__23_57_10/checkpoints/best.pt'
BASELINE_CKPT = './guidance_models/affinity_egnn.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path, protein_feat_dim, ligand_feat_dim):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PropPredNet(ckpt['config'].model,
                        protein_atom_feature_dim=protein_feat_dim,
                        ligand_atom_feature_dim=ligand_feat_dim, output_dim=1).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def get_predictions(model, dataset):
    """Returns list of dicts: idx, target, pred, true, n_heavy_atoms,
    protein_pos, protein_feat, ligand_pos, ligand_feat (last four for
    diagnostic 5's scrambling check)."""
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            protein_pos = data.protein_pos.to(DEVICE)
            protein_feat = data.protein_atom_feature.float().to(DEVICE)
            ligand_pos = data.ligand_pos.to(DEVICE)
            ligand_feat = data.ligand_atom_feature_full.float().to(DEVICE)
            batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=DEVICE)
            batch_ligand = torch.zeros(ligand_pos.size(0), dtype=torch.long, device=DEVICE)
            pred = model(protein_pos=protein_pos, protein_atom_feature=protein_feat,
                        ligand_pos=ligand_pos, ligand_atom_feature=ligand_feat,
                        batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
            idx = dataset.indices[i]
            target = dataset.base_dataset[idx].ligand_filename.split('/')[0]
            out.append({
                'idx': idx, 'target': target, 'pred': pred.view(-1).item(), 'true': float(data.y),
                'n_heavy_atoms': ligand_pos.size(0),
                'protein_pos': protein_pos, 'protein_feat': protein_feat,
                'ligand_pos': ligand_pos, 'ligand_feat': ligand_feat,
            })
    return out


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pearson_metric(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        raise ValueError('degenerate')
    return pearsonr(y_true, y_pred)[0]


def r2_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)


def diagnostic_2(preds, min_n=5):
    by_target = {}
    for p in preds:
        by_target.setdefault(p['target'], []).append(p)
    print('\n=== Diagnostic 2: per-target breakdown ===')
    print(f'{len(by_target)} unique targets across {len(preds)} test pockets '
         f'(most groups will be n=1 -- flagged per addendum instructions)')
    rows = []
    for target, items in sorted(by_target.items(), key=lambda kv: -abs(kv[1][0]['pred'] - kv[1][0]['true'])):
        errs = [abs(it['pred'] - it['true']) for it in items]
        row = {'target': target, 'n': len(items), 'mean_abs_error': float(np.mean(errs))}
        rows.append(row)
    n_singleton = sum(1 for r in rows if r['n'] < min_n)
    print(f'{n_singleton}/{len(rows)} target groups have n<{min_n} -- Pearson not meaningful for these, '
         f'reporting point-wise absolute error only')
    for r in rows[:10]:
        print(f'  {r["target"]}: n={r["n"]} mean_abs_error={r["mean_abs_error"]:.3f}')
    return rows


def diagnostic_3(train_set, test_set):
    print('\n=== Diagnostic 3: ligand-only ECFP4 baseline (ignores pocket) ===')
    def featurize(dataset):
        X, y = [], []
        for i in range(len(dataset)):
            data = dataset[i]
            idx = dataset.indices[i]
            smiles = dataset.base_dataset[idx].ligand_smiles
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                arr = np.zeros((1024,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                X.append(arr)
                y.append(float(data.y))
            except Exception:
                continue
        return np.array(X), np.array(y)

    X_train, y_train = featurize(train_set)
    X_test, y_test = featurize(test_set)
    print(f'featurized train={len(y_train)}/{len(train_set)}, test={len(y_test)}/{len(test_set)} '
         f'(some ligands fail reconstruction from raw coords, excluded)')

    results = {}
    for name, model in [('ridge', Ridge(alpha=1.0)), ('random_forest', RandomForestRegressor(n_estimators=200, random_state=0))]:
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)
        r = {
            'train_pearson': float(pearsonr(y_train, pred_train)[0]),
            'train_r2': float(r2_score(y_train, pred_train)),
            'test_pearson': float(pearsonr(y_test, pred_test)[0]) if np.std(pred_test) > 0 else None,
            'test_r2': float(r2_score(y_test, pred_test)),
        }
        results[name] = r
        print(f'  {name}: train_pearson={r["train_pearson"]:.3f} test_pearson={r["test_pearson"]} test_r2={r["test_r2"]:.3f}')
    return results, y_test, {k: None for k in results}


def diagnostic_4(preds_by_config):
    print('\n=== Diagnostic 4: bootstrap 95% CIs on test Pearson/R2 ===')
    out = {}
    for name, preds in preds_by_config.items():
        y_true = [p['true'] for p in preds]
        y_pred = [p['pred'] for p in preds]
        pearson_point = pearsonr(y_true, y_pred)[0]
        r2_point = r2_score(y_true, y_pred)
        try:
            p_lo, p_hi = bootstrap_ci(y_true, y_pred, pearson_metric)
        except Exception:
            p_lo, p_hi = None, None
        r2_lo, r2_hi = bootstrap_ci(y_true, y_pred, r2_metric)
        out[name] = {
            'pearson_point': pearson_point, 'pearson_ci95': [p_lo, p_hi],
            'r2_point': r2_point, 'r2_ci95': [r2_lo, r2_hi],
        }
        print(f'  {name}: Pearson={pearson_point:.3f} [{p_lo}, {p_hi}], R2={r2_point:.3f} [{r2_lo:.3f}, {r2_hi:.3f}]')
    return out


def diagnostic_5(model, preds):
    print('\n=== Diagnostic 5a: predicted-affinity vs heavy-atom-count correlation ===')
    pred_vals = [p['pred'] for p in preds]
    n_heavy = [p['n_heavy_atoms'] for p in preds]
    corr, pval = pearsonr(pred_vals, n_heavy)
    print(f'  Pearson(pred_affinity, n_heavy_atoms) = {corr:.3f} (p={pval:.3f}), n={len(preds)}')

    print('\n=== Diagnostic 5b: pocket-scrambling sensitivity check ===')
    rng = np.random.RandomState(0)
    n_check = min(20, len(preds))
    idxs = rng.choice(len(preds), n_check, replace=False)
    deltas = []
    with torch.no_grad():
        for i in idxs:
            p = preds[i]
            other = preds[(i + 1) % len(preds)]
            batch_ligand = torch.zeros(p['ligand_pos'].size(0), dtype=torch.long, device=DEVICE)
            batch_protein = torch.zeros(other['protein_pos'].size(0), dtype=torch.long, device=DEVICE)
            scrambled_pred = model(
                protein_pos=other['protein_pos'], protein_atom_feature=other['protein_feat'],
                ligand_pos=p['ligand_pos'], ligand_atom_feature=p['ligand_feat'],
                batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None,
            ).view(-1).item()
            deltas.append(abs(scrambled_pred - p['pred']))
    mean_delta = float(np.mean(deltas))
    pred_std = float(np.std([p['pred'] for p in preds]))
    print(f'  mean |pred(correct pocket) - pred(scrambled pocket)| = {mean_delta:.3f} '
         f'over n={n_check} pairs (for reference, std of all test predictions = {pred_std:.3f})')
    return {'affinity_vs_heavy_atoms_pearson': corr, 'affinity_vs_heavy_atoms_p': pval,
           'pocket_scramble_mean_abs_delta': mean_delta, 'pred_std_reference': pred_std, 'n_scramble_pairs': n_check}


def main():
    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    from torch_geometric.transforms import Compose
    transform = Compose([protein_featurizer, ligand_featurizer])

    base, splits, pk_by_idx = build_labeled_splits(train_subsample=6000)
    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    model_best = load_model(BEST_CKPT, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)
    model_baseline = load_model(BASELINE_CKPT, protein_featurizer.feature_dim, ligand_featurizer.feature_dim)

    preds_best = get_predictions(model_best, test_set)
    preds_baseline = get_predictions(model_baseline, test_set)

    results = {}
    results['diagnostic_2'] = diagnostic_2(preds_best)
    results['diagnostic_3'], _, _ = diagnostic_3(train_set, test_set)
    results['diagnostic_4'] = diagnostic_4({'best_lr3e-5': preds_best, 'baseline_lr1e-4': preds_baseline})
    results['diagnostic_5'] = diagnostic_5(model_best, preds_best)

    with open('./guidance/affinity_diagnostics_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nSaved diagnostics to ./guidance/affinity_diagnostics_results.json')


if __name__ == '__main__':
    main()
