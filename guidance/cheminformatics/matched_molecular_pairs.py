"""TODO item 6: matched molecular pair (MMP) analysis -- within a series of molecules that share the
same scaffold/core and differ by ONE substituent, does a model get the DIRECTION of the affinity change
right?

Why this exists, and how it complements the class-prior decomposition. The class-prior finding
(within-target y-randomisation, R2=0.2468) says most of a ligand-only model's performance is "knowing
which target this is", not real structure-activity relationship (SAR). MMP analysis is the direct,
narrow converse: restricted to pairs that are IDENTICAL except for one substituent, is there any SAR
signal being captured at all, by either model? Kwapien et al. 2022 found additive/local chemistry is
the easy case for ANY model, deep learning included -- so this is also a check on whether this project's
EGNN is at least as good as the easy case suggests it should be.

The matched-pair definition, implemented via RDKit's rdMMPA (the standard single-cut fragmentation
algorithm, Hussain & Rea 2010's approach as shipped in RDKit -- not a bespoke substitute). For each
ligand, every single acyclic bond is cut via rdMMPA.FragmentMol(mol, maxCuts=1), producing a
(core, r-group) split at each cut; the LARGER fragment (by heavy-atom count) is called the core, the
smaller the r-group. Two ligands form a matched pair if, for some cut, they share the exact same core
SMILES and the core is at least MIN_CORE_FRACTION of the whole molecule -- the second condition is what
keeps "matched" meaningful: without it, a tiny shared ring system would group together molecules that
are mostly different, which is not "one substituent changed", it is two unrelated molecules.

Matched pairs are formed WITHIN TARGET ONLY -- the same reasoning as activity_cliffs.py: an affinity
difference between molecules binding different proteins is not evidence of anything about local SAR,
and this project's split is target-disjoint so there is no cross-target comparison available anyway.

Pose-duplicate handling, measured before assuming it away: 783 of 861 (SMILES, target) pairs in the test
set have more than one row (the same ligand docked into the same pocket more than once), and 26 of those
783 have a pK that differs slightly between rows (rounding in label harmonisation, not a different
measurement) -- deduplicated here by averaging pK per (SMILES, target) BEFORE forming any pair, so pose
multiplicity never inflates the pair count or double-counts one real compound-target relationship as
several.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/matched_molecular_pairs.py
  PYTHONPATH=. python guidance/cheminformatics/matched_molecular_pairs.py --b 500
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

from guidance.cheminformatics.chem_data import split_view, target_groups
from guidance.cheminformatics.compare_vs_structure import LIGAND_PREDS, STRUCTURE_SOURCES, load_structure
from guidance.uncertainty_a1.stats_common import cluster_bootstrap_indices

MIN_CORE_FRACTION = 0.6     # the shared core must be >= 60% of the whole molecule's heavy atoms
BOOT_SEED = 20260925
MIN_CLUSTERS = 10
OUT = './guidance/cheminformatics/matched_molecular_pairs_results.json'


def core_rgroup_cuts(smiles):
    """All (core_smiles, rgroup_smiles, core_fraction) single-cut decompositions of one molecule, core
    defined as the larger fragment. Returns [] for anything that fails to parse or has no acyclic
    single bond to cut (e.g. a single ring with no exocyclic substituent)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    total = mol.GetNumHeavyAtoms()
    if total == 0:
        return []
    out = []
    for _const, frag_pair in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = frag_pair.split('.')
        if len(parts) != 2:
            continue
        m0, m1 = Chem.MolFromSmiles(parts[0]), Chem.MolFromSmiles(parts[1])
        if m0 is None or m1 is None:
            continue
        n0, n1 = m0.GetNumHeavyAtoms(), m1.GetNumHeavyAtoms()
        if n0 == 0 or n1 == 0:
            continue
        core, rgroup = (parts[0], parts[1]) if n0 >= n1 else (parts[1], parts[0])
        core_n = max(n0, n1)
        out.append((core, rgroup, core_n / total))
    return out


def find_matched_pairs(smiles, pk, target, min_core_fraction=MIN_CORE_FRACTION):
    """Returns a list of (row_i, row_j, delta_pk) for every matched pair found WITHIN a target, plus
    diagnostic counts. `smiles`/`pk`/`target` are already deduplicated: one row per (smiles, target)."""
    groups = defaultdict(list)                       # (target, core_smiles) -> [(row, rgroup, pk)]
    for i, (s, t) in enumerate(zip(smiles, target)):
        seen_cores_this_mol = set()
        for core, rgroup, frac in core_rgroup_cuts(s):
            if frac < min_core_fraction:
                continue
            key = (t, core)
            if key in seen_cores_this_mol:          # a molecule can hit the same core via >1 cut; count once
                continue
            seen_cores_this_mol.add(key)
            groups[key].append((i, rgroup, pk[i]))

    pairs = []
    n_series = 0
    for (_t, _core), members in groups.items():
        by_rgroup = defaultdict(list)
        for row, rgroup, p in members:
            by_rgroup[rgroup].append((row, p))
        distinct = list(by_rgroup.keys())
        if len(distinct) < 2:
            continue
        n_series += 1
        # One representative row per distinct r-group within this core (first occurrence -- rows
        # sharing a core AND an r-group are the same molecule, already deduplicated by caller).
        reps = [by_rgroup[r][0] for r in distinct]
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                ra, rb = reps[a], reps[b]
                pairs.append((ra[0], rb[0], rb[1] - ra[1]))
    return pairs, n_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()
    RDLogger.DisableLog('rdApp.*')

    v = split_view('test')
    smiles_raw, pk_raw, target_raw = v['smiles'].astype(str), v['pk'].astype(np.float64), v['target'].astype(str)

    # Dedup pose-duplicate (smiles, target) rows by averaging pK, BEFORE fragmenting anything -- a
    # fragmentation cost paid once per real compound-target relationship, not once per pose.
    dedup = {}
    for s, p, t in zip(smiles_raw, pk_raw, target_raw):
        dedup.setdefault((s, t), []).append(p)
    keys = list(dedup.keys())
    smiles = np.array([k[0] for k in keys], dtype=object)
    target = np.array([k[1] for k in keys], dtype=object)
    pk = np.array([np.mean(dedup[k]) for k in keys], dtype=np.float64)
    print(f'{len(smiles_raw)} test rows -> {len(smiles)} unique (SMILES, target) after pose dedup')

    pairs, n_series = find_matched_pairs(smiles, pk, target)
    print(f'{n_series} matched-pair series found (>=2 distinct substituents sharing a core, '
         f'core >= {MIN_CORE_FRACTION:.0%} of the molecule), {len(pairs)} pairs total')
    if not pairs:
        print('No matched pairs found at this threshold -- nothing further to report.')
        return

    # Predictions, joined back onto the DEDUPLICATED rows by re-looking-up each (smiles, target) in the
    # ORIGINAL (non-deduplicated) prediction arrays -- predictions are per-row in the original cache, so
    # take the first original row matching each deduplicated key (predictions for pose-duplicate rows of
    # the SAME ligand are from a model that does not even see the ligand-only features differ across
    # poses in most cases, so any one representative row's prediction is used).
    first_row_of = {}
    for i, (s, t) in enumerate(zip(smiles_raw, target_raw)):
        first_row_of.setdefault((s, t), i)
    orig_rows = np.array([first_row_of[k] for k in keys])

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    assert np.array_equal(L['idx'], v['idx']), 'row order mismatch between split_view and LIGAND_PREDS'
    structure = {}
    for name, (path, ykey, pkey) in STRUCTURE_SOURCES.items():
        p, _note = load_structure(name, path, ykey, pkey, v['pk'].astype(np.float64), v['target'].astype(str))
        if p is not None:
            structure[name] = p
    lig_names = [k[5:] for k in L.files if k.startswith('pred_') and k != 'pred_global_mean']
    champion = 'desc_ridge' if 'desc_ridge' in lig_names else lig_names[0]
    s_main = 'egnn_a1c_ensemble' if 'egnn_a1c_ensemble' in structure else next(iter(structure))
    sp_full = structure[s_main][orig_rows]
    lp_full = np.asarray(L['pred_' + champion], dtype=np.float64)[orig_rows]
    print(f'structure model: {s_main}   ligand-only champion: {champion}')

    i_idx = np.array([p[0] for p in pairs])
    j_idx = np.array([p[1] for p in pairs])
    d_actual = np.array([p[2] for p in pairs])
    d_struct = sp_full[j_idx] - sp_full[i_idx]
    d_lig = lp_full[j_idx] - lp_full[i_idx]
    pair_target = target[i_idx]       # i and j always share a target by construction

    nonzero = np.abs(d_actual) > 1e-9
    print(f'\n{nonzero.sum()} / {len(pairs)} pairs have a nonzero actual pK change (sign agreement is '
         f'computed only over these -- a tie has no direction to get right or wrong).')

    def sign_agreement_and_corr(d_pred):
        agree = (np.sign(d_pred[nonzero]) == np.sign(d_actual[nonzero])).astype(float)
        corr = float(np.corrcoef(d_actual, d_pred)[0, 1]) if len(d_actual) > 1 else float('nan')
        return agree, corr

    results = {'n_series': n_series, 'n_pairs': len(pairs), 'n_pairs_nonzero_delta': int(nonzero.sum()),
              'min_core_fraction': MIN_CORE_FRACTION, 'structure_model': s_main, 'ligand_champion': champion}

    print(f"\n{'model':20s} {'sign agreement':>15s} {'vs chance (0.5)':>16s} {'Pearson(actual,pred)':>22s}")
    agree_struct, corr_struct = sign_agreement_and_corr(d_struct)
    agree_lig, corr_lig = sign_agreement_and_corr(d_lig)
    g = target_groups(pair_target[nonzero])
    n_clusters = len(g)
    for label, agree, corr in (('structure', agree_struct, corr_struct), ('ligand_only', agree_lig, corr_lig)):
        if n_clusters >= MIN_CLUSTERS:
            boot = np.array([agree[np.concatenate([g[k] for k in gi])].mean()
                             for gi in cluster_bootstrap_indices(n_clusters, args.b, BOOT_SEED)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ci_s, ci = f'[{lo:.3f}, {hi:.3f}]', [float(lo), float(hi)]
            vs_chance = 'contains 0.5' if lo <= 0.5 <= hi else ('ABOVE 0.5' if lo > 0.5 else 'BELOW 0.5')
        else:
            ci_s, ci, vs_chance = f'(only {n_clusters} clusters)', None, 'underpowered'
        print(f'{label:20s} {agree.mean():>8.3f} {ci_s:>15s} {vs_chance:>16s} {corr:>22.4f}')
        results[label] = dict(sign_agreement=float(agree.mean()), sign_agreement_ci95=ci,
                              n_target_clusters=int(n_clusters), pearson_actual_vs_pred=corr)

    # THE comparison that matters, done PAIRED rather than by eyeballing two overlapping marginal CIs
    # (the same discipline compare_vs_structure.py's own docstring states: "overlapping marginal
    # intervals are weak evidence either way, because both models are evaluated on the SAME" pairs).
    # agree_struct and agree_lig are computed on the IDENTICAL set of matched pairs, so their
    # per-pair difference is the right paired quantity to bootstrap.
    d_agree = agree_struct - agree_lig
    print(f'\n=== PAIRED comparison: structure sign-agreement minus ligand-only (B={args.b}) ===')
    if n_clusters >= MIN_CLUSTERS:
        boot_d = np.array([d_agree[np.concatenate([g[k] for k in gi])].mean()
                           for gi in cluster_bootstrap_indices(n_clusters, args.b, BOOT_SEED)])
        lo_d, hi_d = np.percentile(boot_d, [2.5, 97.5])
        pval = 2.0 * min((boot_d <= 0).mean(), (boot_d >= 0).mean())
        pval = min(1.0, max(pval, 1.0 / (len(boot_d) + 1)))
        print(f'delta (structure - ligand_only) = {d_agree.mean():+.4f}  CI95 [{lo_d:+.4f}, {hi_d:+.4f}]  p={pval:.3f}')
        results['paired_structure_minus_ligand'] = dict(delta=float(d_agree.mean()),
                                                         ci95=[float(lo_d), float(hi_d)], boot_p=float(pval))
        if lo_d <= 0 <= hi_d:
            print('CI includes zero: the two models are NOT distinguishably different on local-SAR sign')
            print('agreement at this sample size -- reporting "ligand-only wins" from the point estimates')
            print('alone would be exactly the overlapping-marginal-CIs mistake this project already')
            print('disciplines itself against elsewhere.')
        else:
            winner = 'structure' if d_agree.mean() > 0 else 'ligand_only'
            print(f'CI excludes zero: {winner} is paired-significantly better at getting the direction right.')
    else:
        print(f'UNDERPOWERED ({n_clusters} target clusters, need >= {MIN_CLUSTERS}) -- no paired verdict given.')
        results['paired_structure_minus_ligand'] = dict(delta=float(d_agree.mean()), ci95=None, boot_p=None)

    # Separately, and regardless of which model (if either) is paired-better than the other: BOTH
    # point estimates sit at or below 0.5 (chance), which is the headline this script exists to check
    # for, independent of which one is marginally ahead of the other.
    if agree_struct.mean() <= 0.5 and agree_lig.mean() <= 0.5:
        print('\nBoth models have sign agreement AT OR BELOW chance (0.5) on matched pairs -- neither')
        print('captures the direction of a single-substituent affinity change better than a coin flip,')
        print('independent of which of the two is (or is not) paired-significantly ahead of the other.')

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
