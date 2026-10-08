"""A1d: split-conformal prediction intervals for the Stage 0 EGNN affinity model, pre-registered BEFORE
computing anything below. Different paradigm from A1/A1b/A1c: those tested whether a POINT-WISE sigma
ranks error; conformal prediction instead asks whether a single calibrated interval half-width achieves
its target MARGINAL coverage -- it does not require sigma to be informative (and A1c already showed our
ensemble's sigma is not, Spearman p_BH=0.837), so this is a genuinely different method, not a rerun of A1c
with a different statistic.

Specification, fixed before computing any scores:
  model         Stage 0 EGNN deployed checkpoint (logs_lp_split_stage0/.../best.pt, test R^2 0.342) --
                reused, no retraining (conformal wraps an existing point predictor).
  calibration   The val split (6069 complexes, 65 target clusters, disjoint from both train and test).
  nonconformity s_i = |y_i - mu_i| (unweighted / "mu-centered" split conformal -- NOT normalized by any
                ensemble sigma, because A1c already showed that sigma carries no error signal; dividing by
                an uninformative sigma would not improve adaptivity and could only add noise).
  alpha         0.10 (target 90% marginal coverage), the standard default.
  quantile      q = the ceil((n_cal+1)*(1-alpha))-th order statistic of {s_i} on the calibration set
                (Vovk et al.'s finite-sample-exact split-conformal correction).
  interval      [mu_test - q, mu_test + q] on the test set.
  primary       empirical coverage on test = mean(|y_test - mu_test| <= q); cluster bootstrap over the 127
                test targets (B=2,000, seed 20260925, same convention as A1/A1b/A1c) for a 95% CI on
                coverage; success = nominal 0.90 falls inside that CI.
  caveat check  (pre-registered, not an afterthought): standard split-conformal validity assumes
                calibration and test scores are EXCHANGEABLE. Our leakage-safe split's val and test targets
                are completely disjoint protein sets -- if some protein targets are intrinsically harder to
                predict than others (plausible), that is a form of covariate shift between cal and test
                that could break the guarantee even with correct code. Reported as a secondary diagnostic:
                cluster-bootstrap CI on the calibration-set empirical coverage AT THE SAME q (sanity: must
                be close to 0.90 by construction) vs. the test-set coverage -- a gap larger than sampling
                noise would indicate the exchangeability assumption itself is the problem, not the method.

Usage:
  python guidance/uncertainty_a1/conformal_prediction.py
"""
import json
import math

import numpy as np
import torch
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from models.property_pred.prop_model import PropPredNet

CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'
TEST_CACHE = './guidance/track_e/_stage0_test_preds.npz'
OUT = './guidance/uncertainty_a1/a1d_conformal_results.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALPHA = 0.10
B, BOOT_SEED = 2000, 20260925


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, pf, lf


def predict_val():
    model, pf, lf = load_model()
    transform = Compose([pf, lf])
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    A = split_arrays(build_anchor_table(), 'val')
    assert np.array_equal(A['idx'], np.array(splits['val'])), 'val order mismatch vs anchor table'

    y_pred = np.full(len(val_set), np.nan)
    with torch.no_grad():
        for i in range(len(val_set)):
            d = val_set[i]
            bp = torch.zeros(d.protein_pos.size(0), dtype=torch.long, device=DEVICE)
            bl = torch.zeros(d.ligand_pos.size(0), dtype=torch.long, device=DEVICE)
            pred = model(protein_pos=d.protein_pos.to(DEVICE), protein_atom_feature=d.protein_atom_feature.float().to(DEVICE),
                        ligand_pos=d.ligand_pos.to(DEVICE), ligand_atom_feature=d.ligand_atom_feature_full.float().to(DEVICE),
                        batch_protein=bp, batch_ligand=bl, output_kind=None)
            y_pred[i] = pred.view(-1).item()
    return A['pk'], y_pred, A['target']


def conformal_quantile(cal_scores, alpha):
    n = len(cal_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(k, n)  # if alpha is small enough that (n+1)(1-alpha) > n, q = +inf in theory; cap at max observed
    return float(np.sort(cal_scores)[k - 1])


def cluster_bootstrap_coverage(covered, target, B, seed):
    uniq, inv = np.unique(target, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[k] for k in pick])
        boot[b] = covered[idx].mean()
    return float(covered.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    print('Predicting on val set (calibration)...', flush=True)
    y_val, pred_val, target_val = predict_val()
    s_cal = np.abs(y_val - pred_val)
    q = conformal_quantile(s_cal, ALPHA)
    print(f'n_cal={len(s_cal)}  alpha={ALPHA}  q={q:.4f}', flush=True)

    z = np.load(TEST_CACHE, allow_pickle=True)
    y_test, pred_test, target_test = z['y_true'], z['y_pred'], z['tgt']

    covered_test = (np.abs(y_test - pred_test) <= q).astype(float)
    cov_point, cov_lo, cov_hi = cluster_bootstrap_coverage(covered_test, target_test, B, BOOT_SEED)

    covered_cal = (s_cal <= q).astype(float)
    cal_cov_point, cal_cov_lo, cal_cov_hi = cluster_bootstrap_coverage(covered_cal, target_val, B, BOOT_SEED)

    gate_pass = cov_lo <= (1 - ALPHA) <= cov_hi
    print(f'Test coverage: {cov_point:.4f}  CI95 [{cov_lo:.4f}, {cov_hi:.4f}]  '
          f'nominal {1 - ALPHA} inside CI: {gate_pass}', flush=True)
    print(f'Calibration-set coverage at same q (sanity, should be ~{1 - ALPHA}): {cal_cov_point:.4f} '
          f'CI95 [{cal_cov_lo:.4f}, {cal_cov_hi:.4f}]', flush=True)

    out = dict(
        n_cal=int(len(s_cal)), n_test=int(len(y_test)), alpha=ALPHA, q=q, B=B, boot_seed=BOOT_SEED,
        test_coverage=dict(point=cov_point, ci95=[cov_lo, cov_hi], nominal=1 - ALPHA, gate_pass=bool(gate_pass)),
        calibration_set_coverage_sanity=dict(point=cal_cov_point, ci95=[cal_cov_lo, cal_cov_hi]),
        mean_interval_half_width=q,
    )
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
