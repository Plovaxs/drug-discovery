"""Track E gate analysis (pre-registered statistics; see guidance/TRACK_E_DESIGN.md Sec. 2.4).

Resampling unit = TEST TARGET (127 clusters); paired cluster bootstrap (2,000 resamples, seed 20260925); an arm's value is the
mean over its training seeds computed inside the same resample; CIs at 90/95/99%; BH across the tests of a stage.

  python guidance/track_e/analyze_track_e.py refs      # build reference predictions (Stage 0 EGNN cache check, Track A model)
  python guidance/track_e/analyze_track_e.py e1        # E1 gate table
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import norm, pearsonr, rankdata
from sklearn.linear_model import LinearRegression

from guidance.track_e.core import CACHE_DIR, build_anchor_table, pk_vina, split_arrays

ROOT = os.environ.get('TRACK_E_ROOT', './runs_track_e')
B, SEED = 2000, 20260925
LEVELS = [0.90, 0.95, 0.99]


# ------------------------------------------------------------------------------------------------ data
def test_frame():
    t = build_anchor_table()
    A = split_arrays(t, 'test')
    return A


def load_run_preds(pattern, A):
    """[n_seeds, n_test] array of predictions in original pK units, order-checked against the anchor table."""
    out, names = [], []
    for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
        f = os.path.join(d, 'test_preds.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f)
        assert np.array_equal(z['idx'], A['idx']), f'test order mismatch in {d}'
        assert np.allclose(z['y_true'], A['pk'], atol=1e-4)
        out.append(z['y_pred']); names.append(os.path.basename(d))
    return (np.array(out) if out else None), names


def reference_preds(A):
    """Vina-only, heavy-atom-only, Vina+HA (train-fit linear), Stage 0 EGNN, Track A."""
    t = build_anchor_table()
    tr = split_arrays(t, 'train')
    ha_te = A['n_lig'].astype(float); ha_tr = tr['n_lig'].astype(float)
    pkv_te, pkv_tr = pk_vina(A['vina']), pk_vina(tr['vina'])
    refs = {
        'vina_raw': pkv_te,
        'vina_calibrated': LinearRegression().fit(pkv_tr.reshape(-1, 1), tr['pk']).predict(pkv_te.reshape(-1, 1)),
        'heavy_atoms': LinearRegression().fit(ha_tr.reshape(-1, 1), tr['pk']).predict(ha_te.reshape(-1, 1)),
        'vina_plus_ha': LinearRegression().fit(np.c_[pkv_tr, ha_tr], tr['pk']).predict(np.c_[pkv_te, ha_te]),
    }
    p = './guidance/track_e/_stage0_test_preds.npz'
    z = np.load(p, allow_pickle=True)
    assert np.allclose(z['y_true'], A['pk'], atol=1e-4)
    refs['stage0_egnn'] = z['y_pred']
    ta = os.path.join(CACHE_DIR, 'trackA_test_preds.npz')
    if os.path.exists(ta):
        refs['trackA_gign_pignet'] = np.load(ta)['y_pred']
    return refs


def make_trackA_refs(A):
    from torch_geometric.transforms import Compose
    import utils.transforms_prop as ut
    from datasets.crossdocked_affinity import CrossDockedAffinityDataset
    from guidance.lp_split.lp_split_loader import build_lp_splits
    from guidance.track_a.model import GIGNPignetAffinity
    from guidance.track_a.train_gign_pignet_stage2 import prepare_sample
    ck = './logs_track_a_full/gign_pignet_2026_09_11__18_31_46_full/checkpoints/best.pt'
    blob = torch.load(ck, map_location='cuda', weights_only=False)
    pf, lf, bf = ut.FeaturizeProteinAtom(), ut.FeaturizeLigandAtom(), ut.FeaturizeLigandBond()
    m = GIGNPignetAffinity(blob['protein_atom_feature_dim'], blob['ligand_atom_feature_dim'], hidden_dim=blob['hidden_dim']).cuda()
    m.load_state_dict(blob['model']); m.eval()
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, Compose([pf, lf, bf]))
    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            pred, _ = m(**prepare_sample(ds[i], 'cuda'))
            preds.append(pred.item())
    preds = np.array(preds)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, 'trackA_test_preds.npz'), y_pred=preds, y_true=A['pk'])
    print('Track A reference R2 = %.3f  Pearson = %.3f' % (1 - ((A['pk'] - preds) ** 2).sum() / ((A['pk'] - A['pk'].mean()) ** 2).sum(), pearsonr(A['pk'], preds)[0]))


# ------------------------------------------------------------------------------------------------ statistics
class Cluster:
    """Per-target sufficient statistics enabling fast target-level bootstrap of pooled R^2 and within-target Spearman."""

    def __init__(self, A, min_n=10):
        self.tg = A['target']; self.y = A['pk']
        self.uniq = np.unique(self.tg)
        self.members = [np.where(self.tg == g)[0] for g in self.uniq]
        self.n = np.array([len(m) for m in self.members], float)
        self.sy = np.array([self.y[m].sum() for m in self.members]); self.syy = np.array([(self.y[m] ** 2).sum() for m in self.members])
        self.ok = np.array([len(m) >= min_n and self.y[m].std() > 1e-6 for m in self.members])
        rng = np.random.RandomState(SEED)
        self.counts = rng.multinomial(len(self.uniq), np.ones(len(self.uniq)) / len(self.uniq), size=B).astype(float)

    def sse(self, pred):
        return np.array([((self.y[m] - pred[m]) ** 2).sum() for m in self.members])

    def r2_boot(self, pred):
        sse = self.sse(pred)
        N = self.counts @ self.n
        SST = self.counts @ self.syy - (self.counts @ self.sy) ** 2 / N
        return 1 - (self.counts @ sse) / SST

    def r2_point(self, pred):
        return 1 - ((self.y - pred) ** 2).sum() / ((self.y - self.y.mean()) ** 2).sum()

    def spearman_per_target(self, pred):
        out = np.full(len(self.uniq), np.nan)
        for k, m in enumerate(self.members):
            if self.ok[k] and pred[m].std() > 1e-9:
                out[k] = pearsonr(rankdata(self.y[m]), rankdata(pred[m]))[0]
        return out

    def spearman_boot(self, per_target):
        v = np.where(np.isnan(per_target), 0.0, per_target)
        w = self.counts * (~np.isnan(per_target)).astype(float)
        return (self.counts * v) .sum(1) / np.maximum(w.sum(1), 1)


def arm_stats(cl, P, HA):
    """P: [n_seeds, n_test] (or [1, n]). Returns seed-mean bootstrap arrays and point values."""
    P = np.atleast_2d(P)
    r2b = np.mean([cl.r2_boot(p) for p in P], axis=0)
    spt = np.nanmean([cl.spearman_per_target(p) for p in P], axis=0)
    return {'r2_boot': r2b, 'r2': float(np.mean([cl.r2_point(p) for p in P])),
            'sp_boot': cl.spearman_boot(spt), 'sp': float(np.nanmean(spt)), 'sp_per_target': spt,
            'r2_per_seed': [float(cl.r2_point(p)) for p in P],
            'corr_ha': float(np.mean([pearsonr(p, HA)[0] for p in P])), 'n_seeds': len(P)}


def cis(x):
    return {f'{int(l * 100)}': [float(np.percentile(x, 50 * (1 - l))), float(np.percentile(x, 100 - 50 * (1 - l)))] for l in LEVELS}


def diff(a, b, key):
    d = a[key + '_boot'] - b[key + '_boot']
    pt = a[key] - b[key]
    p_one = (1 + (d <= 0).sum()) / (B + 1)
    sd = float(d.std())
    return {'point': float(pt), 'ci': cis(d), 'p_one_sided': float(p_one), 'boot_sd': sd, 'z_effect': float(pt / sd) if sd > 0 else None,
            'power_true_0.05': float(norm.cdf(0.05 / sd - 1.96)) if sd > 0 else None}


def bh(pvals):
    p = np.array(pvals); order = np.argsort(p); m = len(p)
    adj = np.empty(m); prev = 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank); adj[i] = prev
    return adj.tolist()


def cohens_dz(cl, pred_a, pred_b):
    """Paired Cohen's d_z on per-target MSE differences (b - a > 0 means a is better)."""
    mse = lambda pr: np.array([((cl.y[m] - pr[m]) ** 2).mean() for m in cl.members])
    d = mse(pred_b) - mse(pred_a)
    return float(d.mean() / d.std(ddof=1))


# ------------------------------------------------------------------------------------------------ E1 gate
def e1_gate():
    A = test_frame(); cl = Cluster(A); HA = A['n_lig'].astype(float)
    refs = reference_preds(A)
    R = {k: arm_stats(cl, v, HA) for k, v in refs.items()}
    arms = {}
    for name, pat in [('A0prime', 'abs_phys_s*'), ('E1', 'e1_s*'), ('E1_MLP', 'e1mlp_s*'), ('EGNN_extra', 'egnn_s*')]:
        P, names = load_run_preds(pat, A)
        if P is not None:
            arms[name] = {'stats': arm_stats(cl, P, HA), 'runs': names, 'P': P}
    out = {'n_test': int(len(A['idx'])), 'n_targets': int(len(cl.uniq)), 'bootstrap': B, 'seed': SEED,
           'references': {k: {'r2': v['r2'], 'r2_ci95': cis(v['r2_boot'])['95'], 'sp': v['sp'], 'sp_ci95': cis(v['sp_boot'])['95'], 'corr_ha': v['corr_ha']} for k, v in R.items()},
           'arms': {}}
    if 'EGNN_extra' in arms:  # 3-seed EGNN baseline = Stage 0 checkpoint + the extra seeds
        P3 = np.vstack([refs['stage0_egnn'][None, :], arms['EGNN_extra']['P']])
        R['egnn_3seed_mean'] = arm_stats(cl, P3, HA)
        out['references']['egnn_3seed_mean'] = {'r2': R['egnn_3seed_mean']['r2'], 'r2_per_seed': R['egnn_3seed_mean']['r2_per_seed'],
                                                'r2_ci95': cis(R['egnn_3seed_mean']['r2_boot'])['95'], 'sp': R['egnn_3seed_mean']['sp']}
    for name, a in arms.items():
        s = a['stats']
        out['arms'][name] = {'runs': a['runs'], 'n_seeds': s['n_seeds'], 'r2': s['r2'], 'r2_per_seed': s['r2_per_seed'], 'r2_ci95': cis(s['r2_boot'])['95'],
                             'within_target_spearman': s['sp'], 'sp_ci95': cis(s['sp_boot'])['95'], 'corr_heavy_atoms': s['corr_ha']}
    crit = {}
    for head in ('E1', 'E1_MLP'):
        if head not in arms:
            continue
        E = arms[head]['stats']
        tests = {
            'C1a_vs_stage0_egnn': diff(E, R['stage0_egnn'], 'r2'),
            'C1b_vs_vina_plus_ha': diff(E, R['vina_plus_ha'], 'r2'),
            'C1c_within_target_vs_raw_vina': diff(E, R['vina_raw'], 'sp'),
        }
        if 'A0prime' in arms:
            tests['C1e_vs_A0prime'] = diff(E, arms['A0prime']['stats'], 'r2')
        if 'egnn_3seed_mean' in R:
            tests['sens_vs_egnn_3seed_mean'] = diff(E, R['egnn_3seed_mean'], 'r2')
        names = list(tests)
        adj = bh([tests[n]['p_one_sided'] for n in names])
        for n, q in zip(names, adj):
            tests[n]['p_bh'] = q
        c1d = E['corr_ha'] <= 0.605
        lo = lambda n: tests[n]['ci']['95'][0]
        t = tests
        c1a = t['C1a_vs_stage0_egnn']['point'] >= 0.05 and lo('C1a_vs_stage0_egnn') > 0
        c1b = lo('C1b_vs_vina_plus_ha') > 0
        c1c = lo('C1c_within_target_vs_raw_vina') > 0
        c1e = ('C1e_vs_A0prime' in t) and lo('C1e_vs_A0prime') > 0
        point_ok = t['C1a_vs_stage0_egnn']['point'] >= 0.05 and t['C1b_vs_vina_plus_ha']['point'] > 0
        if c1a and c1b and c1c and c1d and c1e:
            verdict = 'PASS (C1a-e)'
        elif c1a and not c1b:
            verdict = 'PARTIAL: beats Stage-0 EGNN only; adds nothing beyond Vina + heavy atoms'
        elif c1a and c1b and not c1e:
            verdict = 'gain not attributable to delta target (C1e fails)'
        elif point_ok and not (c1a and c1b):
            verdict = 'INCONCLUSIVE: point estimates meet margins, CI includes 0 (power-limited)'
        else:
            verdict = 'FAIL: C1a not met -> E1 falsified at this scale'
        crit[head] = {'tests': tests, 'C1a': c1a, 'C1b': c1b, 'C1c': c1c, 'C1d': c1d, 'C1e': c1e, 'verdict': verdict,
                      'cohens_dz_vs_vina_plus_ha': cohens_dz(cl, arms[head]['P'].mean(0), refs['vina_plus_ha'])}
    out['criteria'] = crit
    json.dump(out, open('./guidance/track_e/e1_gate_results.json', 'w'), indent=1)
    return out


def show(out):
    print('references:')
    for k, v in out['references'].items():
        print(f"  {k:22s} R2 {v['r2']:.3f} {np.round(v['r2_ci95'], 3).tolist()}  withinSp {v['sp']:.3f}" + (f"  per-seed {np.round(v['r2_per_seed'], 3).tolist()}" if 'r2_per_seed' in v else ''))
    print('arms:')
    for k, v in out['arms'].items():
        print(f"  {k:12s} seeds {v['n_seeds']}  R2 {v['r2']:.3f} {np.round(v['r2_ci95'], 3).tolist()} per-seed {np.round(v['r2_per_seed'], 3).tolist()}  withinSp {v['within_target_spearman']:.3f}  corrHA {v['corr_heavy_atoms']:.3f}")
    for h, c in out.get('criteria', {}).items():
        print(f'--- {h}: {c["verdict"]}   C1a {c["C1a"]} C1b {c["C1b"]} C1c {c["C1c"]} C1d {c["C1d"]} C1e {c["C1e"]}')
        for n, t in c['tests'].items():
            print(f"    {n:32s} diff {t['point']:+.3f}  95% {np.round(t['ci']['95'], 3).tolist()}  99% {np.round(t['ci']['99'], 3).tolist()}  p_bh {t['p_bh']:.3f}  z {t['z_effect']:.2f}  power(+0.05) {t['power_true_0.05']:.2f}")


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'refs':
        A = test_frame()
        make_trackA_refs(A)
        R = reference_preds(A)
        cl = Cluster(A)
        for k, v in R.items():
            print(k, round(cl.r2_point(v), 4))
    elif cmd == 'e1':
        show(e1_gate())
