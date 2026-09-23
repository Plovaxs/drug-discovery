"""Stage 0 checkpoint's second required number: the ligand-only ECFP4 +
RandomForest baseline (guidance/affinity_sweep_diagnostics.py's Diagnostic
3, re-run here on the new leakage-safe LP-PDBBind-style split instead of
the old one). If this still matches the EGNN's performance on the new
split, that would mean the generalization gap survives even a leakage-
safe, target-diverse split -- an important result either way.
"""
import json

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from datasets.pl_pair_dataset import PocketLigandPairDataset
from guidance.lp_split.lp_split_loader import build_lp_splits

RDLogger.DisableLog('rdApp.*')


def featurize(base, indices, pk_by_idx):
    X, y = [], []
    for idx in indices:
        smi = base[idx].ligand_smiles
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1024,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
        y.append(pk_by_idx[idx])
    return np.array(X), np.array(y)


def main():
    base, splits, pk_by_idx = build_lp_splits(train_subsample=6000)
    print(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  Test: {len(splits["test"])}')

    X_train, y_train = featurize(base, splits['train'], pk_by_idx)
    X_val, y_val = featurize(base, splits['val'], pk_by_idx)
    X_test, y_test = featurize(base, splits['test'], pk_by_idx)
    print(f'Featurized: train={len(y_train)} val={len(y_val)} test={len(y_test)}')

    results = {}
    for name, model in [('ridge', Ridge(alpha=1.0)),
                        ('random_forest', RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))]:
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        r = {
            'val_pearson': float(pearsonr(y_val, pred_val)[0]), 'val_r2': float(r2_score(y_val, pred_val)),
            'test_pearson': float(pearsonr(y_test, pred_test)[0]), 'test_r2': float(r2_score(y_test, pred_test)),
        }
        results[name] = r
        print(f'{name}: val_pearson={r["val_pearson"]:.3f} val_r2={r["val_r2"]:.3f} '
             f'test_pearson={r["test_pearson"]:.3f} test_r2={r["test_r2"]:.3f}')

    with open('./guidance/lp_split/ligand_only_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to ./guidance/lp_split/ligand_only_baseline_results.json')


if __name__ == '__main__':
    main()
