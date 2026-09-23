"""Track C statistical analysis: does rejection sampling (post-hoc top-k
filtering by the frozen EGNN affinity ranker's point-estimate score) select
molecules with genuinely better REAL Vina Dock outcomes than chance?

Design (per guidance/track_c_generate_pools.py docstring and
guidance/STAGE2_PLUS_EXPERIMENT_LOG.md's Track C entries): each pool was
generated ONCE unguided and docked ONCE; top-k / random-k / scrambled-top-k
are all post-hoc re-selections of that SAME already-docked pool -- no
additional generation or docking happens here.

Ranker sign convention (guidance/affinity_point_estimate.py): higher
predicted_affinity_score = predicted stronger binder (pKd/pKi/pIC50
training target). Real vina_dock: more negative = better (stronger)
binding. So the hypothesis under test is: molecules with the highest
predicted_affinity_score should have a more negative (better) real
vina_dock than an unfiltered/random selection of the same size.

Unit of analysis: one pocket = both seeds' pools combined (2x~300
molecules), since both seeds sample the same target/pocket and seed is a
consistency replicate, not a distinct condition. Per-seed sub-results are
also reported for a seed-to-seed consistency check.

Per pocket, per k-fraction, three quantities are computed on the SAME
pool:
  1. Real top-k: top k molecules by predicted_affinity_score (descending).
  2. Random-subset null: B bootstrap draws of a random k-subset (no
     resampling of the pool itself -- draws without replacement per
     iteration), giving a null distribution of "what a same-size random
     pick would achieve" with an empirical one-sided p-value for whether
     the real top-k's mean vina_dock beats it.
  3. Scrambled-filter null (mandatory negative control): B permutations of
     the predicted_affinity_score labels across the SAME molecules, then
     top-k re-selected under the permuted labels. Mathematically this is
     equivalent in expectation to random-subset (a uniform random k-subset)
     -- computed as a separate, explicit implementation-correctness check:
     if scrambled-filter shows the SAME "improvement" as the real top-k,
     that is a red flag for a leakage/ordering bug rather than a genuine
     predictive signal, not a second independent hypothesis test.

Effect size (real top-k mean vina_dock minus full-pool mean vina_dock) gets
a case-resampling bootstrap 95% CI. A two-sample KS test (real top-k vina_dock
distribution vs full-pool vina_dock distribution) is the primary per-pocket
test; its p-values are BH-corrected across the 15 pockets, matching the
dual-criterion checkpoint pattern used throughout this project
(guidance/gradient_informativeness_test.py).

Diagnostic (not hypothesis-test) checks per pocket, at the primary k:
  - Size confound: mean heavy_atoms(top-k) vs mean heavy_atoms(full pool)
    -- mirrors DIAG1's finding for the gradient-guidance tracks; bigger
    molecules mechanically tend to score better in Vina Dock regardless of
    true binding quality.
  - Vina-hacking check: pb_valid rate and ligand_efficiency, top-k vs full
    pool -- a real predictive signal should not come packaged with
    degraded structural validity.

Sensitivity analysis: the whole per-pocket procedure is repeated at
k-fractions {0.05, 0.10, 0.20, 0.30} to check whether any detected effect
is an artifact of the specific top-k threshold chosen.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from guidance.gradient_informativeness_test import benjamini_hochberg

POOL_DIR = './guidance/track_c_pools'
POCKETS_FILE = './guidance/task_f_pockets.json'
OUT_DIR = './guidance/track_c_analysis'
K_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
PRIMARY_K_FRACTION = 0.10
N_BOOTSTRAP = 2000
RNG_SEED = 20260924


def load_pocket_pools(n_pockets=15, seeds=(2021, 2022)):
    """Returns {pocket_target: {'combined': df, 'by_seed': {seed: df}}}."""
    with open(POCKETS_FILE) as f:
        pockets = json.load(f)[:n_pockets]

    out = {}
    for p in pockets:
        data_id = p['data_id']
        target = p['target']
        by_seed = {}
        for seed in seeds:
            path = os.path.join(POOL_DIR, f'pocket{data_id}', f'seed{seed}', 'honest_eval.csv')
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df = df.dropna(subset=['vina_dock', 'predicted_affinity_score']).reset_index(drop=True)
            by_seed[seed] = df
        if not by_seed:
            continue
        combined = pd.concat(by_seed.values(), ignore_index=True)
        out[target] = {'combined': combined, 'by_seed': by_seed, 'data_id': data_id}
    return out


def top_k_indices(df, k, score_col='predicted_affinity_score'):
    return df.nlargest(k, score_col).index.to_numpy()


def analyze_pool(df, k_fraction, rng, n_bootstrap=N_BOOTSTRAP):
    """Runs the full per-pool procedure at one k-fraction. Returns a dict
    of results; None if the pool is too small to support the analysis."""
    n = len(df)
    k = max(1, round(n * k_fraction))
    if n < 10 or k >= n:
        return None

    vina = df['vina_dock'].to_numpy()
    pred = df['predicted_affinity_score'].to_numpy()

    real_idx = top_k_indices(df, k)
    real_topk_vina = vina[real_idx]
    real_topk_mean = real_topk_vina.mean()
    full_mean = vina.mean()
    effect_size = real_topk_mean - full_mean  # negative = improvement

    ks_stat, ks_p = stats.ks_2samp(real_topk_vina, vina)

    # Random-subset null (bootstrap, no-replacement draws of size k).
    random_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=k, replace=False)
        random_means[b] = vina[idx].mean()
    p_random = (np.sum(random_means <= real_topk_mean) + 1) / (n_bootstrap + 1)

    # Scrambled-filter null (mandatory negative control): permute predicted
    # score labels, re-select top-k under the permutation. NOTE: a small
    # p_scrambled here (real top-k beating this null) is EXPECTED and
    # GOOD when there is a genuine effect -- it is not itself evidence of
    # a bug. The actual implementation-correctness check is
    # null_consistency_p below: scrambled-top-k and random-subset are
    # mathematically equivalent sampling processes (permuting an
    # arbitrary label and taking its top-k is a uniform random k-subset),
    # so their two null distributions should be statistically
    # indistinguishable from each other. A significant mismatch there
    # (not a significant p_scrambled) would flag a scrambling bug.
    scrambled_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        perm = rng.permutation(n)
        scrambled_pred = pred[perm]
        scrambled_top_idx = np.argpartition(-scrambled_pred, k - 1)[:k]
        scrambled_means[b] = vina[scrambled_top_idx].mean()
    p_scrambled = (np.sum(scrambled_means <= real_topk_mean) + 1) / (n_bootstrap + 1)
    null_consistency_p = float(stats.ks_2samp(random_means, scrambled_means).pvalue)

    # Case-resampling bootstrap 95% CI on the effect size.
    boot_effects = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_df_vina = vina[idx]
        boot_df_pred = pred[idx]
        boot_top_idx = np.argpartition(-boot_df_pred, k - 1)[:k]
        boot_effects[b] = boot_df_vina[boot_top_idx].mean() - boot_df_vina.mean()
    ci_lo, ci_hi = np.percentile(boot_effects, [2.5, 97.5])

    # Diagnostics.
    heavy_full = df['heavy_atoms'].to_numpy()
    heavy_topk = heavy_full[real_idx]
    size_diag = {
        'mean_heavy_atoms_topk': float(heavy_topk.mean()),
        'mean_heavy_atoms_full': float(heavy_full.mean()),
        'heavy_atoms_ks_p': float(stats.ks_2samp(heavy_topk, heavy_full).pvalue),
    }

    pb_full = df['pb_valid'].to_numpy(dtype=bool)
    pb_topk = pb_full[real_idx]
    le_full_all = df['ligand_efficiency'].to_numpy()
    le_topk_all = le_full_all[real_idx]
    le_full = le_full_all[~np.isnan(le_full_all)]
    le_topk = le_topk_all[~np.isnan(le_topk_all)]
    if len(le_topk) >= 2 and len(le_full) >= 2:
        le_ks_stat, le_ks_p = stats.ks_2samp(le_topk, le_full)
        le_effect = float(le_topk.mean() - le_full.mean())
    else:
        le_ks_p, le_effect = None, None
    vina_hack_diag = {
        'pb_valid_rate_topk': float(pb_topk.mean()),
        'pb_valid_rate_full': float(pb_full.mean()),
        'mean_ligand_efficiency_topk': float(np.nanmean(le_topk_all)),
        'mean_ligand_efficiency_full': float(np.nanmean(le_full_all)),
        'ligand_efficiency_effect': le_effect,  # negative = topk MORE efficient per atom (real improvement)
        'ligand_efficiency_ks_p': float(le_ks_p) if le_ks_p is not None else None,
    }

    return {
        'n': n, 'k': k, 'k_fraction': k_fraction,
        'real_topk_mean_vina_dock': float(real_topk_mean),
        'full_pool_mean_vina_dock': float(full_mean),
        'effect_size': float(effect_size),
        'effect_size_ci95': [float(ci_lo), float(ci_hi)],
        'ks_statistic': float(ks_stat), 'ks_pvalue': float(ks_p),
        'p_random_subset': float(p_random),
        'p_scrambled_filter': float(p_scrambled),
        'null_consistency_p': null_consistency_p,
        'size_confound': size_diag,
        'vina_hacking_check': vina_hack_diag,
    }


def verdict_band(result, alpha=0.05):
    """Matches the established verdict vocabulary
    (guidance/gradient_informativeness_test.py: no_signal /
    signal_wrong_direction / signal_but_quality_degraded / clear_signal),
    plus one Track-C-specific band for the DIAG1-style size confound.

    Primary significance test is the KS test (BH-corrected across
    pockets), same as the established pattern; p_random_subset is reported
    as a corroborating secondary test but does not itself drive the
    verdict, to stay consistent with how the rest of this project judges
    significance. null_consistency_p is a pure implementation-correctness
    sanity check (scrambled-filter and random-subset nulls should be
    statistically indistinguishable) -- a low value here means the
    scrambling code itself is suspect, independent of whether any real
    effect was found."""
    if result is None:
        return 'insufficient_data'

    if result.get('null_consistency_p', 1.0) < 0.01:
        return 'suspect_implementation_bug'

    ks_sig = result['ks_pvalue_corrected'] <= alpha
    if not ks_sig:
        return 'no_signal'

    improved = result['effect_size'] < 0  # more negative vina_dock = better
    if not improved:
        return 'signal_wrong_direction'

    # Primary disqualifier: does the improvement survive size-normalization?
    # A significantly WORSE ligand efficiency for top-k means the raw
    # vina_dock "improvement" is attributable to picking bigger molecules,
    # not better per-atom binding -- this is checked before (and is a
    # stronger falsification than) the raw heavy-atom distribution shift.
    le = result['vina_hacking_check']
    if le.get('ligand_efficiency_reject_bh') and (le.get('ligand_efficiency_effect') or 0) > 0:
        return 'signal_but_confounded_by_size'

    if result['size_confound']['heavy_atoms_ks_p'] <= alpha:
        return 'signal_but_confounded_by_size'

    if result['vina_hacking_check']['pb_valid_rate_topk'] < \
            result['vina_hacking_check']['pb_valid_rate_full'] - 0.10:
        return 'signal_but_quality_degraded'

    return 'clear_signal'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_pockets', type=int, default=15)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    pools = load_pocket_pools(n_pockets=args.n_pockets)
    print(f'Loaded {len(pools)} pockets.')

    all_k_results = {}  # k_fraction -> {pocket: result}
    for kf in K_FRACTIONS:
        per_pocket = {}
        for target, data in pools.items():
            per_pocket[target] = analyze_pool(data['combined'], kf, rng)
        all_k_results[kf] = per_pocket

    # BH correction at the primary k-fraction, across the 15 pockets.
    primary = all_k_results[PRIMARY_K_FRACTION]
    targets = [t for t, r in primary.items() if r is not None]
    ks_pvals = np.array([primary[t]['ks_pvalue'] for t in targets])
    corrected, reject = benjamini_hochberg(ks_pvals, alpha=0.05)
    for t, c, rj in zip(targets, corrected, reject):
        primary[t]['ks_pvalue_corrected'] = float(c)
        primary[t]['ks_reject_bh'] = bool(rj)

    p_random_arr = np.array([primary[t]['p_random_subset'] for t in targets])
    corrected_r, reject_r = benjamini_hochberg(p_random_arr, alpha=0.05)
    for t, c, rj in zip(targets, corrected_r, reject_r):
        primary[t]['p_random_subset_corrected'] = float(c)
        primary[t]['random_reject_bh'] = bool(rj)

    le_p_arr = np.array([primary[t]['vina_hacking_check']['ligand_efficiency_ks_p'] for t in targets])
    corrected_le, reject_le = benjamini_hochberg(le_p_arr, alpha=0.05)
    for t, c, rj in zip(targets, corrected_le, reject_le):
        primary[t]['vina_hacking_check']['ligand_efficiency_ks_p_bh'] = float(c)
        primary[t]['vina_hacking_check']['ligand_efficiency_reject_bh'] = bool(rj)

    for t in targets:
        primary[t]['verdict'] = verdict_band(primary[t])

    # Seed-consistency check at the primary k-fraction.
    seed_consistency = {}
    for target, data in pools.items():
        seeds_results = {}
        for seed, df in data['by_seed'].items():
            seeds_results[seed] = analyze_pool(df, PRIMARY_K_FRACTION, rng)
        directions = [r['effect_size'] < 0 for r in seeds_results.values() if r is not None]
        seed_consistency[target] = {
            'per_seed_effect_size': {s: (r['effect_size'] if r else None) for s, r in seeds_results.items()},
            'directions_agree': len(set(directions)) <= 1 if directions else None,
        }

    # Sensitivity summary: does the verdict direction hold across k-fractions?
    sensitivity = {}
    for target in targets:
        row = {}
        for kf in K_FRACTIONS:
            r = all_k_results[kf].get(target)
            if r is None:
                row[kf] = None
                continue
            row[kf] = {
                'effect_size': r['effect_size'],
                'ks_pvalue': r['ks_pvalue'],
                'p_random_subset': r['p_random_subset'],
                'p_scrambled_filter': r['p_scrambled_filter'],
            }
        sensitivity[target] = row

    verdict_counts = {}
    for t in targets:
        v = primary[t]['verdict']
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary = {
        'n_pockets_analyzed': len(targets),
        'primary_k_fraction': PRIMARY_K_FRACTION,
        'n_bootstrap': N_BOOTSTRAP,
        'verdict_counts': verdict_counts,
        'per_pocket_primary': primary,
        'seed_consistency': seed_consistency,
        'sensitivity_by_k_fraction': sensitivity,
    }

    out_path = os.path.join(args.out_dir, 'track_c_analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Wrote {out_path}')

    print('\n=== Verdict counts (primary k=10%) ===')
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f'  {v}: {c}/{len(targets)}')

    print('\n=== Per-pocket summary (primary k=10%) ===')
    for t in targets:
        r = primary[t]
        print(f"{t:28s} n={r['n']:4d} k={r['k']:3d} effect={r['effect_size']:+.3f} "
              f"CI=[{r['effect_size_ci95'][0]:+.3f},{r['effect_size_ci95'][1]:+.3f}] "
              f"KS_p={r['ks_pvalue']:.4f} KS_p_BH={r['ks_pvalue_corrected']:.4f} "
              f"p_rand={r['p_random_subset']:.4f} p_scram={r['p_scrambled_filter']:.4f} "
              f"verdict={r['verdict']}")


if __name__ == '__main__':
    main()
