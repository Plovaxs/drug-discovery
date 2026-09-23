"""Aggregates and statistically analyzes the Task F ablation run
(guidance/task_f_results/), per the deepened Part B rigor requirements:
mean +/- std per variant across all pockets/seeds, paired (by pocket) and
pooled significance tests comparing baseline vs affinity_only, standard +
honest success rates, and a Vina-hacking check comparing the two variants
directly (not against a single external reference, since none was
computed per-pocket in this run -- see module note below).

Validity/uniqueness/diversity are not saved by eval/honest_eval.py's CLI
(only printed), so they are recomputed here from the per-molecule CSV rows
already on disk (file, smiles, and all per-molecule metrics ARE saved).

Every claimed difference is reported with an uncertainty estimate and a
significance test -- per this project's own working rule from this point
on, a bare mean difference is not reported as evidence of anything by
itself.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

STANDARD_SUCCESS_VINA_DOCK = -8.18
STANDARD_SUCCESS_QED = 0.25
STANDARD_SUCCESS_SA = 0.59


def load_all_results(results_dir):
    rows = []
    attempted = {}  # (variant, pocket_data_id, seed) -> n_attempted
    for variant_dir in sorted(glob.glob(os.path.join(results_dir, '*'))):
        if not os.path.isdir(variant_dir):
            continue
        variant = os.path.basename(variant_dir)
        for triple_dir in sorted(glob.glob(os.path.join(variant_dir, 'pocket*_seed*'))):
            csv_path = os.path.join(triple_dir, 'honest_eval.csv')
            done_path = os.path.join(triple_dir, 'DONE')
            if not os.path.exists(csv_path) or not os.path.exists(done_path):
                continue
            df = pd.read_csv(csv_path)
            rows.append(df)
            with open(done_path) as f:
                line = f.read().strip()
            n_att = int(line.split('n_attempted=')[1].split()[0])
            basename = os.path.basename(triple_dir)
            pocket_id = int(basename.split('pocket')[1].split('_seed')[0])
            seed = int(basename.split('_seed')[1])
            attempted[(variant, pocket_id, seed)] = n_att
    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return full, attempted


def passes_standard_bar(row):
    if pd.isna(row.get('vina_dock')) or pd.isna(row.get('qed')) or pd.isna(row.get('sa')):
        return False
    return bool(row['vina_dock'] < STANDARD_SUCCESS_VINA_DOCK and
               row['qed'] > STANDARD_SUCCESS_QED and row['sa'] > STANDARD_SUCCESS_SA)


def passes_honest_bar(row):
    standard = passes_standard_bar(row)
    pb_ok = bool(row.get('pb_valid')) if pd.notna(row.get('pb_valid')) else False
    return bool(standard and pb_ok)


def mean_pairwise_diversity(smiles_list, max_n=200, seed=0):
    """Mean pairwise (1 - Tanimoto) over Morgan fingerprints. Subsamples to
    `max_n` SMILES if there are more, for tractability."""
    mols = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    if len(mols) < 2:
        return None
    if len(mols) > max_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(mols), max_n, replace=False)
        mols = [mols[i] for i in idx]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    dists = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        dists.extend([1 - s for s in sims])
    return float(np.mean(dists)) if dists else None


def compute_variant_stats(df, attempted, variant):
    sub = df[df['variant'] == variant].copy()
    n_attempted_total = sum(v for (var, _, _), v in attempted.items() if var == variant)
    n_scored = len(sub)

    smiles_list = sub['smiles'].dropna().tolist()
    n_unique = len(set(smiles_list))

    sub['standard_success'] = sub.apply(passes_standard_bar, axis=1)
    sub['honest_success'] = sub.apply(passes_honest_bar, axis=1)

    def mstd(col):
        vals = sub[col].dropna()
        return (float(vals.mean()), float(vals.std())) if len(vals) else (None, None)

    stats_dict = {
        'variant': variant,
        'n_attempted': n_attempted_total,
        'n_scored': n_scored,
        'validity_rate': n_scored / n_attempted_total if n_attempted_total else None,
        'uniqueness_rate': n_unique / n_scored if n_scored else None,
        'diversity_mean_1_minus_tanimoto': mean_pairwise_diversity(smiles_list),
        'qed_mean_std': mstd('qed'),
        'sa_mean_std': mstd('sa'),
        'ra_score_mean_std': mstd('ra_score'),
        'vina_dock_mean_std': mstd('vina_dock'),
        'ligand_efficiency_mean_std': mstd('ligand_efficiency'),
        'pb_valid_rate': float(sub['pb_valid'].fillna(False).mean()) if n_scored else None,
        'standard_success_rate': float(sub['standard_success'].mean()) if n_scored else None,
        'honest_success_rate': float(sub['honest_success'].mean()) if n_scored else None,
    }
    return stats_dict, sub


def per_pocket_means(sub, metric):
    """One mean value per pocket (averaged across its 3 seeds' molecules) --
    used for the paired-by-pocket significance test."""
    return sub.groupby('pocket_data_id')[metric].mean()


def significance_tests(sub_a, sub_b, metrics):
    results = {}
    for metric in metrics:
        a_vals = sub_a[metric].dropna()
        b_vals = sub_b[metric].dropna()
        if len(a_vals) < 2 or len(b_vals) < 2:
            results[metric] = {'mannwhitney_p': None, 'wilcoxon_paired_p': None, 'n_a': len(a_vals), 'n_b': len(b_vals)}
            continue
        try:
            mw_u, mw_p = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        except ValueError:
            mw_p = None

        pocket_a = per_pocket_means(sub_a, metric)
        pocket_b = per_pocket_means(sub_b, metric)
        common = pocket_a.index.intersection(pocket_b.index)
        wil_p = None
        if len(common) >= 6:
            try:
                _, wil_p = stats.wilcoxon(pocket_a.loc[common], pocket_b.loc[common])
            except ValueError:
                wil_p = None
        results[metric] = {
            'mannwhitney_p': float(mw_p) if mw_p is not None else None,
            'wilcoxon_paired_p': float(wil_p) if wil_p is not None else None,
            'n_pockets_paired': len(common),
            'n_a': len(a_vals), 'n_b': len(b_vals),
        }
    return results


def vina_hacking_check(stats_baseline, stats_affinity):
    findings = []
    vd_b, vd_a = stats_baseline['vina_dock_mean_std'][0], stats_affinity['vina_dock_mean_std'][0]
    le_b, le_a = stats_baseline['ligand_efficiency_mean_std'][0], stats_affinity['ligand_efficiency_mean_std'][0]
    pb_b, pb_a = stats_baseline['pb_valid_rate'], stats_affinity['pb_valid_rate']

    if vd_a is not None and vd_b is not None and vd_a < vd_b:
        findings.append(f'affinity_only Vina Dock improved vs baseline ({vd_a:.3f} < {vd_b:.3f}).')
        if le_a is not None and le_b is not None and le_a > le_b:
            findings.append(f'  ...but ligand efficiency got WORSE ({le_a:.4f} > {le_b:.4f} kcal/mol per heavy '
                            f'atom) -- consistent with Vina-hacking via size inflation.')
        else:
            findings.append(f'  ...and ligand efficiency did not worsen ({le_a:.4f} vs {le_b:.4f}) -- '
                            f'not consistent with simple size-inflation hacking.')
        if pb_a is not None and pb_b is not None and pb_a < pb_b:
            findings.append(f'  ...and PoseBusters pass rate dropped ({pb_a:.3f} < {pb_b:.3f}) -- '
                            f'poses may be less physically plausible.')
        else:
            findings.append(f'  ...PoseBusters pass rate held or improved ({pb_a:.3f} vs {pb_b:.3f}).')
    elif vd_a is not None and vd_b is not None:
        findings.append(f'affinity_only Vina Dock did NOT improve vs baseline ({vd_a:.3f} vs {vd_b:.3f}) -- '
                        f'no Vina-hacking question applies since there is no affinity gain to interrogate.')
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./guidance/task_f_results')
    parser.add_argument('--out', type=str, default='./guidance/task_f_analysis.json')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    df, attempted = load_all_results(args.results_dir)
    print(f'Loaded {len(df)} scored molecules across {len(attempted)} (variant,pocket,seed) triples.')

    variants = sorted(df['variant'].unique())
    all_stats = {}
    subs = {}
    for v in variants:
        s, sub = compute_variant_stats(df, attempted, v)
        all_stats[v] = s
        subs[v] = sub
        print(f'\n=== {v} ===')
        for k, val in s.items():
            print(f'  {k}: {val}')

    sig = None
    hacking = None
    if 'baseline' in subs and 'affinity_only' in subs:
        metrics = ['qed', 'sa', 'ra_score', 'vina_dock', 'ligand_efficiency']
        sig = significance_tests(subs['baseline'], subs['affinity_only'], metrics)
        print('\n=== Significance tests (baseline vs affinity_only) ===')
        for metric, res in sig.items():
            print(f'  {metric}: {res}')

        hacking = vina_hacking_check(all_stats['baseline'], all_stats['affinity_only'])
        print('\n=== Vina-hacking check ===')
        for f in hacking:
            print(f'  {f}')

    with open(args.out, 'w') as f:
        json.dump({'variant_stats': all_stats, 'significance': sig, 'vina_hacking': hacking}, f, indent=2)
    print(f'\nSaved full analysis to {args.out}')


if __name__ == '__main__':
    main()
