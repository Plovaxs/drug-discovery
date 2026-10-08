"""A1f-corrected: A1f's raw sigma passed the ranking tests (Spearman p_BH=0.042, partial Spearman
p_BH=0.012) but failed calibration_ratio badly (sigma ~25x too large). Since Spearman rank correlation is
invariant to any monotonic rescaling of sigma, a single global multiplicative correction factor -- fit on
VAL (never test, to avoid any leakage into the number we report) -- can fix the calibration ratio to ~1
while leaving the ranking tests' p-values EXACTLY unchanged (rescaling doesn't touch rank order). This is
not re-analysis-until-significant: the ranking tests are already pre-registered and already significant on
raw sigma; this only asks whether a single, val-fit scale correction (the obvious, standard fix once a
ranking-but-not-scale result is observed) also fixes calibration, re-running calibration_ratio only (the
other two tests are provably unchanged and are not recomputed to avoid implying they could differ).

Correction factor: c = mean(|y_val - mu_val|) / (mean(sigma_val) * sqrt(2/pi)), i.e. the same calibration-
ratio statistic computed on val, inverted and applied as sigma_corrected = sigma / calibration_ratio_val.

Usage:
  python guidance/uncertainty_a1/analyze_evidential_corrected.py
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_e.core import build_anchor_table, split_arrays
from guidance.uncertainty_a1.analyze_evidential import DEVICE, find_ckpt, forward, load_model
from guidance.uncertainty_a1.mc_dropout_calibration import B, BOOT_SEED, boot_p

RESULTS_IN = './guidance/uncertainty_a1/a1f_results.json'
OUT = './guidance/uncertainty_a1/a1f_corrected_results.json'


def predict_val(model):
    A = split_arrays(build_anchor_table(), 'val')
    base, _, pk_by_idx = build_lp_splits(train_subsample=None)
    transform = Compose([utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()])
    ds = CrossDockedAffinityDataset(base, [int(i) for i in A['idx']], pk_by_idx, transform)
    mu, sd = np.full(len(ds), np.nan), np.full(len(ds), np.nan)
    with torch.no_grad():
        for i in range(len(ds)):
            mu[i], sd[i] = forward(model, ds[i])
    return A['pk'], mu, sd


def main():
    ckpt = find_ckpt()
    model, _ = load_model(ckpt)
    print('Scoring val set to fit the correction factor...', flush=True)
    y_val, mu_val, sd_val = predict_val(model)
    err_val = np.abs(y_val - mu_val)
    cal_ratio_val = err_val.mean() / (sd_val.mean() * np.sqrt(2 / np.pi))
    print(f'val calibration ratio = {cal_ratio_val:.4f} (this is the correction factor)', flush=True)

    per_complex = np.load('./guidance/uncertainty_a1/a1f_per_complex.npz')
    target = per_complex['target']
    sd_test_corrected = per_complex['sd'] * cal_ratio_val
    err_test = per_complex['abs_err']

    def cal_ratio(idx):
        return err_test[idx].mean() / (sd_test_corrected[idx].mean() * np.sqrt(2 / np.pi))

    point = cal_ratio(np.arange(len(err_test)))
    uniq, inv = np.unique(target, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        boot[b] = cal_ratio(np.concatenate([groups[k] for k in pick]))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    p_raw = boot_p(boot, 1.0)

    raw = json.load(open(RESULTS_IN))
    print(f'test calibration ratio AFTER val-fit correction: {point:.4f}  CI95 {ci}  p_raw={p_raw:.4f} '
          f'(target 1.0)', flush=True)
    print(f'Spearman / partial-Spearman UNCHANGED by construction (monotonic rescale): '
          f'{raw["tests"]["spearman_sigma_error"]["point"]:.4f} (p_BH={raw["tests"]["spearman_sigma_error"]["p_bh"]}) / '
          f'{raw["tests"]["partial_spearman_sigma_error_given_nlig"]["point"]:.4f} '
          f'(p_BH={raw["tests"]["partial_spearman_sigma_error_given_nlig"]["p_bh"]})', flush=True)
    gate_pass_all_three = ci[0] <= 1.0 <= ci[1]
    print(f'Calibration ratio CI contains 1.0 (passes on its own): {gate_pass_all_three}', flush=True)

    out = dict(correction_factor_from_val=float(cal_ratio_val),
               test_calibration_ratio_corrected=dict(point=float(point), ci95=ci, p_raw=float(p_raw),
                                                      b=B, boot_seed=BOOT_SEED, gate_pass=bool(gate_pass_all_three)),
               spearman_unchanged=raw['tests']['spearman_sigma_error'],
               partial_spearman_unchanged=raw['tests']['partial_spearman_sigma_error_given_nlig'],
               note='BH correction across all 3 tests was defined on the RAW (uncorrected) calibration '
                    'ratio in a1f_results.json; this corrected re-test is a follow-up diagnostic motivated '
                    'by that raw result, not a blind re-registration, so its p-value is reported raw/'
                    'unadjusted here rather than re-run through BH against the other two (already-reported) tests.')
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
