"""Track E planning-phase check (no GPU, no training): how precisely can a PAIRED, target-clustered
comparison on the 127-target LP test set resolve a difference in R^2 between two predictors?
Uses cached Stage 0 EGNN test predictions and linear reference baselines from reference_baselines_check.py.
"""
import json

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

VINA_TO_PK = -1.0 / 1.364
z = np.load('./guidance/track_e/_ref_cache.npz', allow_pickle=True)
p = np.load('./guidance/track_e/_stage0_test_preds.npz', allow_pickle=True)
tr = z['train']
rpk = tr[:, 2].astype(float)
rha = tr[:, 4].astype(float)
rv = np.array([float(v) for v in tr[:, 3]]) * VINA_TO_PK
y, tg, ha, egnn = p['y_true'], p['tgt'].astype(str), p['ha'], p['y_pred']
tv = p['vina'] * VINA_TO_PK

lin_ha = LinearRegression().fit(rha.reshape(-1, 1), rpk).predict(ha.reshape(-1, 1))
lin_vha = LinearRegression().fit(np.c_[rv, rha], rpk).predict(np.c_[tv, ha])
lin_v = LinearRegression().fit(rv.reshape(-1, 1), rpk).predict(tv.reshape(-1, 1))

uniq = np.unique(tg)
members = {g: np.where(tg == g)[0] for g in uniq}
rng = np.random.RandomState(20260925)


def paired(a, b, n_boot=2000):
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    d = np.array(d)
    return {'delta_r2_point': float(r2_score(y, a) - r2_score(y, b)),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'sd': float(d.std())}


out = {
    'n_test_targets': int(len(uniq)),
    'pearson_pk_vs_heavy_atoms_test': float(pearsonr(y, ha)[0]),
    'pearson_egnn_pred_vs_heavy_atoms_test': float(pearsonr(egnn, ha)[0]),
    'egnn_vs_ha_only': paired(egnn, lin_ha),
    'egnn_vs_vina_calibrated': paired(egnn, lin_v),
    'egnn_vs_vina_plus_ha_linear': paired(egnn, lin_vha),
    'vina_plus_ha_vs_ha_only': paired(lin_vha, lin_ha),
}
# per-target R^2 dispersion: how many test targets contain how many complexes
sizes = np.array([len(m) for m in members.values()])
out['test_complexes_per_target'] = {'min': int(sizes.min()), 'median': float(np.median(sizes)), 'max': int(sizes.max())}
json.dump(out, open('./guidance/track_e/paired_power_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))


# ---- within-target ranking (guidance acts inside one pocket) ----
from scipy.stats import spearmanr


def within_target(pred, min_n=10):
    rs = []
    for g, m in members.items():
        if len(m) >= min_n and np.std(y[m]) > 1e-6 and np.std(pred[m]) > 1e-9:
            rs.append(spearmanr(y[m], pred[m])[0])
    rs = np.array(rs)
    boot = [np.mean(rng.choice(rs, len(rs), replace=True)) for _ in range(2000)]
    return {'n_targets': int(len(rs)), 'mean_spearman': float(rs.mean()), 'median_spearman': float(np.median(rs)),
            'ci95_mean': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


wt = {'egnn': within_target(egnn), 'vina_raw': within_target(tv), 'heavy_atoms': within_target(ha.astype(float)),
      'vina_plus_ha_linear': within_target(lin_vha)}
out['within_target_ranking_min10'] = wt
json.dump(out, open('./guidance/track_e/paired_power_results.json', 'w'), indent=1)
print(json.dumps(wt, indent=1))


# ---- reference table under the PRE-REGISTERED anchor definition: pk_vina = max(-vina, 0) / 1.364 ----
def clip_pk(v):
    return np.maximum(-np.asarray(v, dtype=float), 0.0) / 1.364


rv_c, tv_c = clip_pk(tr[:, 3].astype(float)), clip_pk(p['vina'])
lin_v_c = LinearRegression().fit(rv_c.reshape(-1, 1), rpk).predict(tv_c.reshape(-1, 1))
lin_vha_c = LinearRegression().fit(np.c_[rv_c, rha], rpk).predict(np.c_[tv_c, ha])


def cluster_ci(fn, n_boot=2000):
    v = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        v.append(fn(idx))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


ref = {}
for name, pred in [('vina_clipped_raw', tv_c), ('vina_clipped_calibrated', lin_v_c), ('heavy_atoms_only', lin_ha),
                   ('vina_clipped_plus_ha_linear', lin_vha_c), ('stage0_egnn', egnn)]:
    ref[name] = {'r2': float(r2_score(y, pred)), 'r2_ci_target_cluster': cluster_ci(lambda i: r2_score(y[i], pred[i])),
                 'pearson': float(pearsonr(y, pred)[0]),
                 'pearson_vs_heavy_atoms': float(pearsonr(pred, ha)[0]),
                 'within_target': within_target(pred)}
tr_ok = np.maximum(-tr[:, 3].astype(float), 0)
dl = rpk - clip_pk(tr[:, 3].astype(float))
ref['delta_target_clipped_train'] = {'mean': float(dl.mean()), 'std': float(dl.std()), 'pk_std': float(rpk.std()),
                                     'pearson_delta_vs_ha': float(pearsonr(dl, rha)[0]),
                                     'n_positive_vina_train': int((tr[:, 3].astype(float) > 0).sum())}
out['reference_table_clipped_anchor'] = ref
json.dump(out, open('./guidance/track_e/paired_power_results.json', 'w'), indent=1)
for k, v in ref.items():
    if 'r2' in v:
        print(f"{k:32s} R2={v['r2']:.3f} {np.round(v['r2_ci_target_cluster'],3)} r={v['pearson']:.3f} corr_HA={v['pearson_vs_heavy_atoms']:.3f} withinSp={v['within_target']['mean_spearman']:.3f} {np.round(v['within_target']['ci95_mean'],3)}")
print(ref['delta_target_clipped_train'])
