"""TODO item 5: a MoleculeACE-style activity-cliff metric -- does performance specifically collapse on
activity cliffs, the way novelty_tiers.py already shows it collapsing at high ligand similarity?

Definition and its disclosed simplification. van Tilborg, Alenicheva & Grisoni 2022 (J. Chem. Inf.
Model., 314 cites, github.com/molML/MoleculeACE) define an activity cliff as a pair of molecules with
high structural similarity but a large potency difference; their published convention
(calc_cliff_rmse(..., similarity=0.9, potency_fold=10, in_log10=True)) uses a 0.9 similarity cutoff and
a 10-fold (= 1 log10 unit) potency difference. Their original similarity criterion is a consensus across
THREE measures (substructure / scaffold / SMILES similarity, per secondary sources -- the primary
package source was not reachable to confirm the exact combination rule). This implementation uses ONE of
those three -- ECFP4 (radius=2, 2048 bit) Tanimoto, already cached by chem_data.py and used for every
other similarity-based result in this project (novelty_tiers.py's tiers, ligand_only_baseline.py's 1-NN)
-- at the SAME published thresholds (sim >= 0.9, |delta pK| >= 1.0, pK already being log10-scale so no
conversion is needed). This is disclosed as a simplification, not presented as a faithful reimplementation
of the published tool.

Cliffs are computed WITHIN target, never across targets. Two molecules binding different proteins can
differ in potency by any amount for reasons that have nothing to do with a structure-activity
discontinuity -- that comparison is meaningless, not just noisy. This project's split is target-disjoint
(no test target's ligands appear anywhere in train), so a cliff partner can only come from the SAME
target's OTHER test ligands, not from train. The per-target ligand counts here support this: median 14,
p90 232, max 1392 -- plenty of targets have enough within-target ligands to search for cliffs.

Pairwise Tanimoto is computed by matrix multiplication of the 0/1 fingerprint matrix, not a Python
double loop or repeated RDKit bulk calls: for a 0/1 matrix F, intersection = F @ F.T (the dot product of
two bit vectors counts positions where both are 1), popcount = F.sum(axis=1), union = popcount[:,None] +
popcount[None,:] - intersection, tanimoto = intersection / union. This turns an O(n^2) x O(bits) loop
into one BLAS matmul per target -- the whole reason this is tractable at the largest targets' scale
(up to ~1 million pairs for the single biggest target) on this laptop's CPU.

Why this script is QUEUED rather than run inline (see scripts/queue_steps.sh): a rough estimate put the
ten largest targets alone at ~15 billion multiply-adds, and this was written while the CPU was already
at the RQ2 run's thermal gate (literally AT the real task the user asked to prioritise) -- correctness
was verified on a tiny synthetic example instead (see guidance/cheminformatics/tests/test_chem.py),
and the real computation deferred to when the CPU is not already the bottleneck for something else.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/activity_cliffs.py
  PYTHONPATH=. python guidance/cheminformatics/activity_cliffs.py --b 500
"""
import argparse
import json

import numpy as np

from guidance.cheminformatics.chem_data import split_view, target_groups
from guidance.cheminformatics.compare_vs_structure import LIGAND_PREDS, STRUCTURE_SOURCES, load_structure
from guidance.uncertainty_a1.stats_common import bh, paired_r2_comparison

SIM_THRESHOLD = 0.9     # van Tilborg et al. 2022's published convention
DELTA_PK_THRESHOLD = 1.0   # 10-fold, already in log10 units since pK is log10-scale
BOOT_SEED = 20260925
MIN_CLUSTERS = 10
OUT = './guidance/cheminformatics/activity_cliffs_results.json'


def cliff_mask_within_target(fp, pk, groups):
    """Returns a boolean array, same length as fp/pk, True for every row that participates in at least
    one activity-cliff pair WITHIN ITS OWN TARGET GROUP. `groups` is target_groups()'s output: a list of
    row-index arrays, one per target.

    Vectorised per target via the bit-matmul identity (see module docstring) rather than per-pair RDKit
    calls, which is what makes this tractable at the largest targets (up to ~1 million pairs each)."""
    is_cliff = np.zeros(len(pk), dtype=bool)
    n_pairs_checked = 0
    n_cliff_pairs = 0
    for rows in groups:
        if len(rows) < 2:
            continue
        F = fp[rows].astype(np.int32)                  # (n, bits)
        inter = F @ F.T                                 # (n, n) -- shared-1 bit count
        popcount = F.sum(axis=1)
        union = popcount[:, None] + popcount[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        d_pk = np.abs(pk[rows][:, None] - pk[rows][None, :])
        cliff = (sim >= SIM_THRESHOLD) & (d_pk >= DELTA_PK_THRESHOLD)
        np.fill_diagonal(cliff, False)                  # a molecule is never its own cliff partner
        n = len(rows)
        n_pairs_checked += n * (n - 1) // 2
        n_cliff_pairs += int(cliff.sum()) // 2           # cliff is symmetric; each pair counted twice
        row_is_cliff = cliff.any(axis=1)
        is_cliff[rows[row_is_cliff]] = True
    return is_cliff, n_pairs_checked, n_cliff_pairs


def stratum_r2_comparison(pk, sp, lp, target, rows, B):
    """Structure-vs-ligand-only dR2 with a target-clustered bootstrap CI, restricted to `rows`.

    As of 2026-10-13: a thin wrapper around stats_common.paired_r2_comparison (the centralised version
    of exactly this pattern, written after the SAME degenerate-variance and un-paired-comparison bugs
    were found independently in this file and in protein_family_stratification.py /
    matched_molecular_pairs.py -- see that function's docstring for the full history). Kept as a
    wrapper, not inlined at call sites, so this module's field names (structure_r2/ligand_r2/
    local_pk_std) stay unchanged for main() below and for anything else that already reads this
    function's output."""
    g = [rows[gi] for gi in target_groups(target[rows])]
    r = paired_r2_comparison(pk, sp, lp, g, B, BOOT_SEED, min_clusters=MIN_CLUSTERS)
    out = dict(n=r['n'], n_target_clusters=r['n_target_clusters'], structure_r2=r['r2_a'],
              ligand_r2=r['r2_b'], d_r2=r['d_r2'], ci95=r['ci95'], boot_p=r['boot_p'],
              underpowered=r['underpowered'], degenerate=r['degenerate'])
    if r['degenerate']:
        out['local_pk_std'] = r['local_std']
    if 'note' in r:
        out['note'] = r['note']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    v = split_view('test')
    fp, pk, target = v['fp'], v['pk'].astype(np.float64), v['target'].astype(str)
    groups = target_groups(target)
    print(f'{len(pk)} test ligands, {len(groups)} targets '
         f'(n_lig median {np.median([len(g) for g in groups]):.0f}, max {max(len(g) for g in groups)})')

    is_cliff, n_pairs, n_cliff_pairs = cliff_mask_within_target(fp, pk, groups)
    print(f'checked {n_pairs} within-target pairs, found {n_cliff_pairs} activity-cliff pairs '
         f'(sim >= {SIM_THRESHOLD}, |delta pK| >= {DELTA_PK_THRESHOLD})')
    print(f'cliff compounds: {int(is_cliff.sum())} / {len(is_cliff)} test ligands '
         f'({100 * is_cliff.mean():.1f}%)')

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    assert np.array_equal(L['idx'], v['idx']), 'row order mismatch between split_view and LIGAND_PREDS'
    structure = {}
    for name, (path, ykey, pkey) in STRUCTURE_SOURCES.items():
        p, _note = load_structure(name, path, ykey, pkey, pk, target)
        if p is not None:
            structure[name] = p
    lig_names = [k[5:] for k in L.files if k.startswith('pred_') and k != 'pred_global_mean']
    champion = 'desc_ridge' if 'desc_ridge' in lig_names else lig_names[0]
    s_main = 'egnn_a1c_ensemble' if 'egnn_a1c_ensemble' in structure else next(iter(structure))
    sp, lp = structure[s_main], np.asarray(L['pred_' + champion], dtype=np.float64)
    print(f'structure model: {s_main}   ligand-only champion: {champion}')

    results = {}
    print(f'\n=== cliff vs non-cliff (target-clustered bootstrap, B={args.b}) ===')
    hdr = f'{"stratum":12s} {"n":>6s} {"tgts":>5s} {"struct_R2":>10s} {"lig_R2":>8s} {"dR2 [95% CI]":>26s}'
    print(hdr); print('-' * len(hdr))
    for label, rows in (('cliff', np.where(is_cliff)[0]), ('non_cliff', np.where(~is_cliff)[0])):
        r = stratum_r2_comparison(pk, sp, lp, target, rows, args.b)
        results[label] = r
        if r.get('degenerate'):
            print(f'{label:12s} {r["n"]:>6d} {r["n_target_clusters"]:>5d}  n/a (local pK std '
                 f'{r["local_pk_std"]:.2e} -- no real spread, not a finding)')
            continue
        ci_s = f'[{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}]' if r['ci95'] else f'(only {r["n_target_clusters"]} clusters)'
        print(f'{label:12s} {r["n"]:>6d} {r["n_target_clusters"]:>5d} {r["structure_r2"]:>10.4f} '
             f'{r["ligand_r2"]:>8.4f} {r["d_r2"]:+.4f} {ci_s:>14s}')

    if results['cliff']['boot_p'] is not None and results['non_cliff']['boot_p'] is not None:
        q = bh([results['cliff']['boot_p'], results['non_cliff']['boot_p']])
        for label, qv in zip(('cliff', 'non_cliff'), q):
            results[label]['boot_p_bh'] = float(qv)

    # Van Tilborg et al.'s own finding, stated as the comparison this script exists to check against:
    # descriptor-based models outperform deep learning SPECIFICALLY on cliffs.
    #
    # Reuses results['cliff'] rather than recomputing r2_rows independently here -- an earlier version
    # of this block did recompute it, as a bare point-estimate comparison with NO CI at all ("if
    # champ_cliff > struct_cliff"), which is a plainer version of the exact mistake caught and fixed in
    # matched_molecular_pairs.py (declaring a winner from two numbers with no significance test). The
    # table above already computed a PAIRED bootstrap CI for d_r2 = structure_r2 - ligand_r2 on the
    # cliff stratum (same resampled rows feed both terms in every draw), so that is what gets read here.
    cliff_r = results['cliff']
    if cliff_r.get('degenerate'):
        print('\nOn cliff compounds specifically: no verdict -- degenerate local pK variance.')
    elif cliff_r.get('underpowered'):
        n_tc = cliff_r['n_target_clusters']
        print(f'\nOn cliff compounds specifically: no verdict -- only {n_tc} target clusters, underpowered.')
    else:
        print(f'\nOn cliff compounds specifically: ligand-only champion R2={cliff_r["ligand_r2"]:.4f} vs '
             f'structure R2={cliff_r["structure_r2"]:.4f}  (paired dR2={cliff_r["d_r2"]:+.4f}, '
             f'CI95 {cliff_r["ci95"]}, p={cliff_r["boot_p"]:.3f}).')
        if cliff_r['ci95'][0] <= 0 <= cliff_r['ci95'][1]:
            print('CI includes zero: NOT paired-significantly different -- reporting a winner from the '
                 'point estimates alone would overclaim.')
        elif cliff_r['d_r2'] < 0:
            print('Descriptor-based model is paired-significantly AHEAD on cliffs -- consistent with '
                 'van Tilborg et al. 2022\'s own reported pattern (descriptors > deep learning on '
                 'activity cliffs).')
        else:
            print('Structure model is paired-significantly ahead on cliffs here -- NOT the pattern van '
                 'Tilborg et al. 2022 report.')

    out = dict(sim_threshold=SIM_THRESHOLD, delta_pk_threshold=DELTA_PK_THRESHOLD, B=args.b,
              boot_seed=BOOT_SEED, n_pairs_checked=n_pairs, n_cliff_pairs=n_cliff_pairs,
              n_cliff_compounds=int(is_cliff.sum()), n_test=int(len(is_cliff)),
              structure_model=s_main, ligand_champion=champion, by_stratum=results)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
