"""THE missing control: how much of our structure-based test performance is reproducible WITHOUT the
protein?

This is the question the field puts to exactly our model class, and this project has never answered it.
Volkov et al. 2022 (J. Med. Chem., 170 cites) showed that an explicit description of protein-ligand
noncovalent interactions gives no advantage over ligand descriptors alone, and that nearest-neighbour
models over the training set already perform well -- memorisation rather than learning. Mattsson et al.
2026 measured r = 0.66 for a ligand-only model on FEP+. Graber et al. 2025 (Nat. Mach. Intell.) retrained
leading models on a leakage-cleaned split and watched their benchmark numbers collapse.

Our EGNN Stage 0 scores test R2 = 0.3420 on the LP split. That number is only evidence for
structure-based learning if models that never see the protein do WORSE. Nobody has checked. This does.

The baselines are ordered by how little they know, so the result reads as a ladder:
  1. global_mean       -- predicts the training mean. The floor; R2 = 0 by construction on train.
  2. heavy_atoms       -- ONE feature, ligand heavy-atom count, univariate linear. If this alone
                          approaches 0.34, the headline number is largely ligand size, and A1's use of
                          n_lig as a partial-correlation covariate was understating the problem.
  3. vina              -- the physics scoring function already in the anchor table. A non-ML reference.
  4. desc_*            -- 14 interpretable RDKit descriptors (no fingerprints).
  5. ecfp_ridge        -- ECFP4 (2048 bits), ridge. Classical QSAR.
  6. ecfp_desc_hgb     -- fingerprints + descriptors, gradient boosting. The strongest ligand-only model.
  7. tanimoto_1nn      -- predicts the pK of the single most similar TRAINING ligand. Volkov's
                          memorisation probe: it does no learning at all, only lookup.

Why this split makes the control sharp rather than trivial. The LP split separates by TARGET -- 779
training targets, 127 test targets, disjoint. A ligand-only model therefore cannot memorise a target's
mean affinity; whatever it achieves comes from ligand structure alone generalising across unseen
proteins. So a high ligand-only score here is not an artefact of the baseline being given something it
shouldn't have. It would mean the task, as our split poses it, is substantially solvable without the
protein.

Statistics follow this project's standing rules rather than being chosen per script: every interval is a
TARGET-CLUSTERED bootstrap (B=2,000, seed 20260925, resampling the 127 test targets, not the 11,855
complexes) because complexes sharing a target are not independent, and model selection uses val only.

CPU only -- no GPU, no torch. Safe to run while training owns the card.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/ligand_only_baseline.py
  PYTHONPATH=. python guidance/cheminformatics/ligand_only_baseline.py --b 500      # quicker CI
  PYTHONPATH=. python guidance/cheminformatics/ligand_only_baseline.py --skip tanimoto_1nn
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.cheminformatics.chem_data import build_features, split_view, target_groups
from guidance.uncertainty_a1.stats_common import cluster_bootstrap_ci

OUT = './guidance/cheminformatics/ligand_only_results.json'
PRED_OUT = './guidance/cheminformatics/ligand_only_test_preds.npz'
BOOT_SEED = 20260925      # the same seed every A1-A1g cluster bootstrap uses
EGNN_STAGE0_TEST_R2 = 0.3420     # the number this control exists to contextualise


def r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def metrics(y, p):
    from scipy.stats import pearsonr, spearmanr
    # A constant predictor (global_mean) has no defined correlation; reported as None rather than
    # letting scipy emit nan, so the table does not carry a number-shaped non-answer.
    const = float(np.std(p)) < 1e-12
    return dict(R2=r2(y, p),
                Pearson=None if const else float(pearsonr(y, p)[0]),
                Spearman=None if const else float(spearmanr(y, p)[0]),
                RMSE=float(np.sqrt(((y - p) ** 2).mean())), MAE=float(np.abs(y - p).mean()))


def tanimoto_nn(fp_query, fp_ref, y_ref, chunk=512, k=1):
    """Mean pK of the k most Tanimoto-similar reference ligands, plus the top similarity itself.

    Computed as a matrix product rather than pairwise: for binary vectors, Q @ R.T IS the intersection
    count, and |A| + |B| - |A and B| gives the union, so the full 11,855 x 46,964 similarity matrix costs
    one BLAS call per chunk. Chunked over queries because the dense matrix is ~2.2 GB in float32 and does
    not need to exist all at once."""
    Q = fp_query.astype(np.float32)
    R = fp_ref.astype(np.float32)
    q_pop = Q.sum(1, keepdims=True)
    r_pop = R.sum(1)[None, :]
    pred = np.empty(len(Q), dtype=np.float64)
    top_sim = np.empty(len(Q), dtype=np.float64)
    for s in range(0, len(Q), chunk):
        e = min(s + chunk, len(Q))
        inter = Q[s:e] @ R.T
        union = q_pop[s:e] + r_pop - inter
        sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        if k == 1:
            j = sim.argmax(1)
            pred[s:e] = y_ref[j]
            top_sim[s:e] = sim[np.arange(e - s), j]
        else:
            j = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            pred[s:e] = y_ref[j].mean(1)
            top_sim[s:e] = np.take_along_axis(sim, j, 1).max(1)
    return pred, top_sim


def fit_predict(name, tr, va, te):
    """Returns (val_pred, test_pred) for one baseline. Hyperparameters are picked on VAL only; test is
    touched once, at the end, by the caller."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    if name == 'global_mean':
        m = tr['pk'].mean()
        return np.full(len(va['pk']), m), np.full(len(te['pk']), m)

    if name == 'heavy_atoms':
        # Univariate least squares on a single integer feature. Deliberately the crudest possible model.
        X = tr['n_lig'].astype(np.float64)
        A = np.vstack([X, np.ones_like(X)]).T
        coef, *_ = np.linalg.lstsq(A, tr['pk'], rcond=None)
        f = lambda d: coef[0] * d['n_lig'].astype(np.float64) + coef[1]
        return f(va), f(te)

    if name == 'vina':
        # Vina is an energy (more negative = better), so it is rescaled onto pK by least squares on
        # TRAIN. Without that it cannot be compared on R2 at all -- a monotone score with the wrong units
        # and sign would report a large negative R2 while ranking perfectly.
        X = tr['vina'].astype(np.float64)
        A = np.vstack([X, np.ones_like(X)]).T
        coef, *_ = np.linalg.lstsq(A, tr['pk'], rcond=None)
        f = lambda d: coef[0] * d['vina'].astype(np.float64) + coef[1]
        return f(va), f(te)

    if name == 'desc_ridge':
        sc = StandardScaler().fit(np.nan_to_num(tr['desc']))
        best, best_pred = None, None
        for alpha in (0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha).fit(sc.transform(np.nan_to_num(tr['desc'])), tr['pk'])
            pv = m.predict(sc.transform(np.nan_to_num(va['desc'])))
            s = r2(va['pk'], pv)
            if best is None or s > best:
                best, best_pred = s, (pv, m.predict(sc.transform(np.nan_to_num(te['desc']))))
        return best_pred

    if name == 'desc_hgb':
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
        m.fit(np.nan_to_num(tr['desc']), tr['pk'])
        return m.predict(np.nan_to_num(va['desc'])), m.predict(np.nan_to_num(te['desc']))

    if name == 'ecfp_ridge':
        best, best_pred = None, None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            m = Ridge(alpha=alpha).fit(tr['fp'].astype(np.float32), tr['pk'])
            pv = m.predict(va['fp'].astype(np.float32))
            s = r2(va['pk'], pv)
            if best is None or s > best:
                best, best_pred = s, (pv, m.predict(te['fp'].astype(np.float32)))
        return best_pred

    if name == 'ecfp_desc_hgb':
        Xtr = np.hstack([tr['fp'].astype(np.float32), np.nan_to_num(tr['desc'])])
        Xva = np.hstack([va['fp'].astype(np.float32), np.nan_to_num(va['desc'])])
        Xte = np.hstack([te['fp'].astype(np.float32), np.nan_to_num(te['desc'])])
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0)
        m.fit(Xtr, tr['pk'])
        return m.predict(Xva), m.predict(Xte)

    if name == 'tanimoto_1nn':
        pv, _ = tanimoto_nn(va['fp'], tr['fp'], tr['pk'])
        pt, _ = tanimoto_nn(te['fp'], tr['fp'], tr['pk'])
        return pv, pt

    if name == 'tanimoto_5nn':
        pv, _ = tanimoto_nn(va['fp'], tr['fp'], tr['pk'], k=5)
        pt, _ = tanimoto_nn(te['fp'], tr['fp'], tr['pk'], k=5)
        return pv, pt

    raise ValueError(name)


ALL = ['global_mean', 'heavy_atoms', 'vina', 'desc_ridge', 'desc_hgb', 'ecfp_ridge',
       'ecfp_desc_hgb', 'tanimoto_1nn', 'tanimoto_5nn']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000, help='cluster bootstrap resamples')
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    feats = build_features()
    tr, va, te = (split_view(p, feats) for p in ('train', 'val', 'test'))
    groups = target_groups(te['target'])
    print(f"train {len(tr['pk'])} rows / {len(set(map(str, tr['target'])))} targets | "
          f"val {len(va['pk'])} / {len(set(map(str, va['target'])))} | "
          f"test {len(te['pk'])} / {len(groups)} targets", flush=True)
    print(f'test target clusters for the bootstrap: {len(groups)} (B={args.b}, seed {BOOT_SEED})\n',
          flush=True)

    names = args.only if args.only else [n for n in ALL if n not in args.skip]
    results = {}
    # Test predictions are saved per baseline so the paired comparison against the EGNN
    # (compare_vs_structure.py) can be rerun without refitting, and so the comparison is made on the
    # exact same numbers this table reports rather than on a re-fit that might differ.
    preds = dict(idx=te['idx'].astype(np.int64), target=np.asarray(te['target'], dtype=str),
                 pk=te['pk'].astype(np.float64), n_lig=te['n_lig'].astype(np.int64))
    for name in names:
        t0 = time.time()
        pv, pt = fit_predict(name, tr, va, te)
        preds[f'pred_{name}'] = np.asarray(pt, dtype=np.float64)
        val_m = metrics(va['pk'], pv)
        test_m = metrics(te['pk'], pt)
        point, ci, _ = cluster_bootstrap_ci(lambda rows: r2(te['pk'][rows], pt[rows]),
                                            groups, b=args.b, seed=BOOT_SEED)
        results[name] = dict(val=val_m, test=test_m, test_R2_cluster_ci=[ci[0], ci[1]],
                             seconds=round(time.time() - t0, 1))
        pear = 'n/a    ' if test_m['Pearson'] is None else f'{test_m["Pearson"]:+.4f}'
        spear = 'n/a' if test_m['Spearman'] is None else f'{test_m["Spearman"]:+.4f}'
        print(f'{name:16s} val R2 {val_m["R2"]:+.4f} | test R2 {test_m["R2"]:+.4f} '
              f'[{ci[0]:+.4f}, {ci[1]:+.4f}] | test Pearson {pear} '
              f'| Spearman {spear}  ({results[name]["seconds"]}s)', flush=True)

    print('\n=== CONTEXT ===')
    print(f'EGNN Stage 0 (structure-based, the model this thesis is about): test R2 = '
          f'{EGNN_STAGE0_TEST_R2:.4f}')
    best = max((v['test']['R2'], k) for k, v in results.items() if k != 'global_mean')
    print(f'best ligand-only baseline: {best[1]} at test R2 = {best[0]:.4f}')
    gap = EGNN_STAGE0_TEST_R2 - best[0]
    print(f'margin the protein buys us: {gap:+.4f} R2')
    lo, hi = results[best[1]]['test_R2_cluster_ci']
    if EGNN_STAGE0_TEST_R2 < hi:
        print(f'WARNING: the EGNN point estimate ({EGNN_STAGE0_TEST_R2:.4f}) lies INSIDE the '
              f'cluster-bootstrap CI of a ligand-only model [{lo:+.4f}, {hi:+.4f}].\n'
              f'         On this split, structure-based performance is not distinguishable from a model '
              f'that never sees the protein.\n'
              f'         This must be reported as a primary limitation, not a footnote. See Volkov et '
              f'al. 2022 and Mattsson et al. 2026.')
    else:
        print(f'the EGNN estimate lies above that baseline\'s CI upper bound ({hi:+.4f}) -- the protein '
              f'contributes measurably, though a matched-seed comparison is still required')

    payload = dict(bootstrap=dict(b=args.b, seed=BOOT_SEED, n_test_target_clusters=len(groups)),
                   egnn_stage0_test_r2=EGNN_STAGE0_TEST_R2, baselines=results,
                   note='Ligand-only controls for "is the protein used?". The LP split separates by '
                        'target, so these models cannot memorise a target mean; any performance they '
                        'reach generalises from ligand structure alone across unseen proteins.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, 'w'), indent=2)
    np.savez_compressed(PRED_OUT, **preds)
    print(f'\nSaved {OUT}')
    print(f'Saved {PRED_OUT} ({len(names)} baseline prediction vector(s), aligned to the test rows)')


if __name__ == '__main__':
    main()
