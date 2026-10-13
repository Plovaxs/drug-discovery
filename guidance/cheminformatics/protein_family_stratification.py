"""TODO item 7: protein-family error stratification -- where does the structure-vs-ligand-only gap
actually live, broken down by Pfam family rather than reported as one pooled number?

Why this exists. novelty_tiers.py already asks "does structure beat ligand-only among chemically novel
test molecules" and answers tier-by-tier. This asks the orthogonal question -- same comparison, broken
down by WHICH PROTEIN, not by how familiar the ligand is. A single pooled dR2 can hide a pattern where
structure helps for one target class and hurts for another; a kinase-specific result reads very
differently to a reader who works on kinases than a number averaged across 127 unrelated targets does.

Reuse, not reimplementation -- everything here is machinery this project already built and tested
elsewhere, recombined along a different grouping variable:
  * family_by_target / collapse_rare_families (guidance/uncertainty_a1/conditional_conformal_coverage.py)
    -- the SAME Pfam mapping and the SAME 'rank by distinct targets, not rows' collapsing rule already
    used for conditional conformal coverage, so a reader who has seen that table recognises this one.
  * STRUCTURE_SOURCES / load_structure / r2_rows / LIGAND_PREDS (compare_vs_structure.py) -- the exact
    champion-vs-structure comparison novelty_tiers.py already runs, just grouped differently.
  * groups_within / cluster_bootstrap_indices / bh (novelty_tiers.py / stats_common.py) -- the same
    target-clustered bootstrap and multiple-testing correction used everywhere else in this project.

Two tables, deliberately kept separate, following novelty_tiers.py's own precedent:
  1. A CONFIRMATORY table: only the top-N families by target count (collapsed, same N_TOP_FAMILIES
     convention as the conformal coverage script) get a bootstrap CI and a BH-corrected p-value --
     because running a confirmatory test on every one of ~100 mostly 1-3-target families would be
     almost entirely noise dressed up as findings.
  2. A DESCRIPTIVE table: every family's point-estimate dR2, with its target count printed alongside so
     a 2-target family's number is visibly as uncertain as it is -- a pattern to read, not evidence to
     cite, exactly how novelty_tiers.py's 'all models per tier' table is already captioned.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/protein_family_stratification.py
  PYTHONPATH=. python guidance/cheminformatics/protein_family_stratification.py --b 500
"""
import argparse
import json

import numpy as np

from guidance.cheminformatics.compare_vs_structure import (LIGAND_PREDS, STRUCTURE_SOURCES, load_structure,
                                                            r2_rows)
from guidance.cheminformatics.novelty_tiers import groups_within
from guidance.uncertainty_a1.conditional_conformal_coverage import collapse_rare_families, family_by_target
from guidance.uncertainty_a1.stats_common import bh, paired_r2_comparison

BOOT_SEED = 20260925
MIN_CLUSTERS = 10          # same threshold as novelty_tiers.py / conditional_conformal_coverage.py
N_TOP_FAMILIES = 8         # same default as conditional_conformal_coverage.py
OUT = './guidance/cheminformatics/protein_family_stratification_results.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    pk, target = L['pk'].astype(np.float64), L['target'].astype(str)
    fam = family_by_target(target)
    fam_collapsed, top_families = collapse_rare_families(fam, target, N_TOP_FAMILIES)

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

    n_annotated = len({t for f, t in zip(fam, target) if f != 'no_annotation'})
    print(f'Pfam coverage: {n_annotated} / {len(np.unique(target))} targets annotated.')
    print('Top families by target count: ' + ', '.join(f'{f}({len(ts)} tgts)' for f, ts in top_families))

    # ---- Table 1: confirmatory, top-N families (+ other_family), bootstrap CI + BH.
    print(f'\n=== CONFIRMATORY: top {N_TOP_FAMILIES} families + other_family (B={args.b}) ===')
    hdr = (f'{"family":16s} {"n":>6s} {"tgts":>5s} {"struct_R2":>10s} {"lig_R2":>8s} '
          f'{"dR2 [95% CI]":>26s}')
    print(hdr); print('-' * len(hdr))
    # As of 2026-10-13: uses stats_common.paired_r2_comparison, the centralised version of this exact
    # pattern (see its docstring for the full history -- this table's own descriptive sibling below is
    # what originally found the degenerate-variance bug this now guards against pre-emptively too,
    # though none of the top-N/other_family strata here happen to be degenerate in the real data;
    # verified byte-identical against the pre-refactor output before this replaced the hand-rolled loop).
    strata = [f for f, _ in top_families] + ['other_family']
    results_confirmatory = {}
    for label in strata:
        rows = np.where(fam_collapsed == label)[0]
        if len(rows) == 0:
            print(f'{label:16s} {0:>6d}  (empty)')
            continue
        g = groups_within(rows, target)
        r = paired_r2_comparison(pk, sp, lp, g, args.b, BOOT_SEED, min_clusters=MIN_CLUSTERS)
        s_r2, l_r2, d = r['r2_a'], r['r2_b'], r['d_r2']
        if r['degenerate']:
            ci_s = f'n/a (local pK std {r["local_std"]:.2e} -- no real spread)'
            print(f'{label:16s} {len(rows):>6d} {r["n_target_clusters"]:>5d} {ci_s:>26s}')
            results_confirmatory[label] = dict(n=r['n'], n_target_clusters=r['n_target_clusters'],
                                               structure_r2=None, ligand_r2=None, d_r2=None, ci95=None,
                                               boot_p=None, underpowered=True, degenerate=True)
            continue
        ci_s = (f'{d:+.4f} [{r["ci95"][0]:+.4f}, {r["ci95"][1]:+.4f}]' if r['ci95']
               else f'{d:+.4f} (only {r["n_target_clusters"]} clusters -- UNDERPOWERED)')
        print(f'{label:16s} {len(rows):>6d} {r["n_target_clusters"]:>5d} {s_r2:>10.4f} {l_r2:>8.4f} {ci_s:>26s}')
        results_confirmatory[label] = dict(n=r['n'], n_target_clusters=r['n_target_clusters'],
                                           structure_r2=float(s_r2), ligand_r2=float(l_r2),
                                           d_r2=float(d), ci95=r['ci95'], boot_p=r['boot_p'],
                                           underpowered=r['underpowered'], degenerate=False)

    tested = [k for k, v in results_confirmatory.items() if v['boot_p'] is not None]
    if tested:
        q = bh([results_confirmatory[k]['boot_p'] for k in tested])
        print(f'\n=== Benjamini-Hochberg over the {len(tested)} POWERED famil{"y" if len(tested)==1 else "ies"} ===')
        for k, qv in zip(tested, q):
            results_confirmatory[k]['boot_p_bh'] = float(qv)
            sig = 'SIGNIFICANT' if qv < 0.05 else 'not significant'
            print(f'  {k:16s} dR2 {results_confirmatory[k]["d_r2"]:+.4f}  '
                 f'p={results_confirmatory[k]["boot_p"]:.3f}  q={qv:.3f}  {sig}')

    # ---- Table 2: descriptive, EVERY family (point estimates only -- no CI, no test, by design).
    #
    # Guarded against a real numerical trap, found by actually reading the first unguarded run's
    # output rather than just checking it didn't crash: r2_rows divides by the family's LOCAL pK
    # variance (((yy - yy.mean())**2).sum()), and r2_rows' own `denom <= 0` guard only catches EXACT
    # zero. A small family can have near-duplicate pK values whose variance is nonzero but minuscule
    # (measured case: PF08917, 6 rows, 1 target, variance 1.13e-16) -- enough to survive that guard
    # while making the division blow up to values like +5.8e16, which is not a finding, it is
    # division by a number indistinguishable from zero at this scale. A relative threshold (local pK
    # std below 1% of the global pK std) is used rather than an absolute one, because what counts as
    # "no real spread" depends on pK's own scale, not on a constant picked by eye.
    pk_std_global = float(pk.std())
    degenerate_std = 0.01 * pk_std_global
    print(f'\n=== DESCRIPTIVE ONLY (point dR2, no CI -- a pattern to read, not evidence to cite) ===')
    print(f'{"family":16s} {"n":>6s} {"tgts":>5s} {"dR2":>10s}')
    results_descriptive = {}
    n_degenerate = 0
    # Found while verifying this file's OWN refactor (above) byte-for-byte: running this exact line
    # twice in a row produced two DIFFERENT orderings -- a pre-existing bug, not introduced by that
    # refactor. Cause: `{f for f in fam if f != 'no_annotation'}` is a set, and Python does not
    # guarantee stable iteration order for a set of strings across processes (it depends on
    # PYTHONHASHSEED); `sorted(..., key=-count)` only fixes the order BETWEEN distinct counts, so every
    # tie among same-count families -- the common case here, dozens of 1- and 2-target families --
    # was being broken arbitrarily. Fixed with an explicit secondary key (the family name itself) so
    # ties resolve the same way every run, on any machine, with any hash seed.
    all_fams = sorted({f for f in fam if f != 'no_annotation'},
                      key=lambda f: (-len({t for ff, t in zip(fam, target) if ff == f}), f))
    for label in all_fams:
        rows = np.where(fam == label)[0]
        n_tgt = len(np.unique(target[rows]))
        local_std = float(pk[rows].std())
        if local_std < degenerate_std:
            print(f'{label:16s} {len(rows):>6d} {n_tgt:>5d} {"n/a (pK std " + f"{local_std:.2e}" + ", no real spread)":>10s}')
            results_descriptive[label] = dict(n=int(len(rows)), n_targets=int(n_tgt), d_r2=None,
                                              degenerate=True, local_pk_std=local_std)
            n_degenerate += 1
            continue
        d = r2_rows(pk, sp, rows) - r2_rows(pk, lp, rows)
        print(f'{label:16s} {len(rows):>6d} {n_tgt:>5d} {d:>+10.4f}')
        results_descriptive[label] = dict(n=int(len(rows)), n_targets=int(n_tgt), d_r2=float(d),
                                          degenerate=False)
    if n_degenerate:
        print(f'\n({n_degenerate} famil{"y" if n_degenerate == 1 else "ies"} marked n/a: local pK std '
             f'below 1% of the global pK std ({degenerate_std:.4f}) -- a local R2 there is division by '
             f'a number indistinguishable from zero, not a small-sample estimate.)')

    with open(OUT, 'w') as f:
        json.dump(dict(structure_model=s_main, ligand_champion=champion,
                       confirmatory=results_confirmatory, descriptive=results_descriptive), f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
