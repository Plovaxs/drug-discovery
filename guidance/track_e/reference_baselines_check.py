"""Track E planning-phase feasibility check (inference only, NO training).

Reference numbers for pre-registering Track E's falsification criteria, all on the leakage-safe
LP split (train 46,964 / val 6,069 / test 11,855):
  * Vina-only (raw and linearly calibrated on train), heavy-atom-only, Vina+HA linear baselines
  * Stage 0 EGNN (existing checkpoint, inference only)
  * TARGET-clustered bootstrap CIs (the honest resampling unit; Stage 0's CI resampled complexes)
  * size-neutrality of the delta target: Pearson(delta, heavy atoms)
"""
import json
import os
import pickle

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.lp_split.stage0_checkpoint_analysis import CKPT, get_predictions, load_model

VINA_TO_PK = -1.0 / 1.364
CACHE = './guidance/track_e/_ref_cache.npz'
OUT = './guidance/track_e/reference_baselines_results.json'


def collect(base, indices, pk_by_idx, vina_info):
    rows = []
    for i in indices:
        d = base[i]
        key = d.ligand_filename[:-4]
        v = vina_info.get(key, {}).get('vina', None)
        rows.append((i, d.protein_filename.split('/')[0], pk_by_idx[i], v, len(d.ligand_element)))
    return rows


def cluster_bootstrap(groups, y, yhat, fn, n_boot=2000, seed=20260925):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        vals.append(fn(y[idx], yhat[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report(name, groups, y, yhat):
    r2 = float(r2_score(y, yhat))
    pr = float(pearsonr(y, yhat)[0])
    r2ci = cluster_bootstrap(groups, y, yhat, r2_score)
    prci = cluster_bootstrap(groups, y, yhat, lambda a, b: pearsonr(a, b)[0])
    return {'name': name, 'r2': r2, 'r2_ci_target_cluster': r2ci, 'pearson': pr, 'pearson_ci_target_cluster': prci}


def main():
    info = pickle.load(open('./data/affinity_info.pkl', 'rb'))
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        tab = {k: z[k] for k in z.files}
    else:
        tab = {}
        for part in ('train', 'val', 'test'):
            rows = collect(base, splits[part], pk_by_idx, info)
            tab[part] = np.array(rows, dtype=object)
        np.savez(CACHE, **tab)

    def arr(part):
        r = tab[part]
        idx = r[:, 0].astype(int)
        tgt = r[:, 1].astype(str)
        pk = r[:, 2].astype(float)
        vina = np.array([np.nan if v is None else float(v) for v in r[:, 3]])
        ha = r[:, 4].astype(float)
        return idx, tgt, pk, vina, ha

    out = {'n': {}, 'missing_vina': {}}
    D = {p: arr(p) for p in ('train', 'val', 'test')}
    for p, (idx, tgt, pk, vina, ha) in D.items():
        out['n'][p] = int(len(idx))
        out['missing_vina'][p] = int(np.isnan(vina).sum())
        out.setdefault('n_targets', {})[p] = int(len(np.unique(tgt)))

    _, ttg, tpk, tvina, tha = D['test']
    _, _, rpk, rvina, rha = D['train']
    ok_tr = ~np.isnan(rvina)
    ok_te = ~np.isnan(tvina)
    tp_v = tvina * VINA_TO_PK
    rp_v = rvina * VINA_TO_PK

    delta_tr = rpk[ok_tr] - rp_v[ok_tr]
    out['delta_target'] = {
        'train_mean': float(delta_tr.mean()), 'train_std': float(delta_tr.std()),
        'train_pk_std': float(rpk[ok_tr].std()),
        'pearson_delta_vs_heavy_atoms_train': float(pearsonr(delta_tr, rha[ok_tr])[0]),
        'pearson_pk_vs_heavy_atoms_train': float(pearsonr(rpk[ok_tr], rha[ok_tr])[0]),
        'pearson_pkvina_vs_heavy_atoms_train': float(pearsonr(rp_v[ok_tr], rha[ok_tr])[0]),
        'pearson_pk_vs_pkvina_train': float(pearsonr(rpk[ok_tr], rp_v[ok_tr])[0]),
    }

    y, g = tpk[ok_te], ttg[ok_te]
    res = []
    res.append(report('Vina-only raw (yhat=-vina/1.364)', g, y, tp_v[ok_te]))
    lr = LinearRegression().fit(rp_v[ok_tr].reshape(-1, 1), rpk[ok_tr])
    res.append(report('Vina-only linearly calibrated on train', g, y, lr.predict(tp_v[ok_te].reshape(-1, 1))))
    lr = LinearRegression().fit(rha[ok_tr].reshape(-1, 1), rpk[ok_tr])
    res.append(report('Heavy-atom-count only (linear, train fit)', g, y, lr.predict(tha[ok_te].reshape(-1, 1))))
    X_tr = np.c_[rp_v[ok_tr], rha[ok_tr]]
    X_te = np.c_[tp_v[ok_te], tha[ok_te]]
    lr = LinearRegression().fit(X_tr, rpk[ok_tr])
    res.append(report('Vina + heavy-atom linear (train fit)', g, y, lr.predict(X_te)))
    res.append(report('Train-mean constant', g, y, np.full_like(y, rpk[ok_tr].mean())))

    model_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    test_set = CrossDockedAffinityDataset(base, list(D['test'][0]), pk_by_idx, Compose([pf, lf]))
    model = load_model(CKPT, pf.feature_dim, lf.feature_dim)
    y_true, y_pred, _ = get_predictions(model, test_set)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert np.allclose(y_true, tpk, atol=1e-4), 'test ordering mismatch'
    res.append(report('Stage 0 EGNN (existing checkpoint) [all test]', ttg, tpk, y_pred))
    res.append(report('Stage 0 EGNN (existing checkpoint) [test with Vina score]', g, y, y_pred[ok_te]))
    out['test_baselines'] = res
    out['egnn_pearson_pred_vs_heavy_atoms_test'] = float(pearsonr(y_pred, tha)[0])
    np.savez('./guidance/track_e/_stage0_test_preds.npz', y_pred=y_pred, y_true=tpk, tgt=ttg, ha=tha, vina=tvina)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
