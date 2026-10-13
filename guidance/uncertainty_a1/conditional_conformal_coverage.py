"""TODO item 3: does split-conformal's coverage hold CONDITIONALLY, not just marginally?

Why this exists. A1d is the one UQ method that passed its pre-registered gate: test coverage 0.8952
(CI95 [0.857, 0.925]) against nominal 0.90. That is a MARGINAL guarantee -- averaged over the whole test
set. Gibbs et al. 2023 give the standard framework for asking whether a marginally-valid interval is
quietly invalid within subgroups, and Jeliazkova et al. 2026 predict specifically that conformal coverage
for molecular properties DROPS in novel chemistry. If that is true here, it is the single most important
caveat on this project's one positive UQ result: an interval that is valid "on average" but systematically
under-covers exactly the novel compounds a practitioner would reach for it on.

Design. The deployed quantile q is NOT recomputed here -- it is read back from
guidance/uncertainty_a1/a1d_conformal_results.json, the exact value already reported (2.1960...). The
question is whether THAT q, used as deployed, keeps its coverage promise within two kinds of subgroup:
  * novelty tier -- the same Tanimoto-to-train bins already defined in
    guidance/cheminformatics/novelty_tiers.py (TIERS), reused rather than re-cut, so a reader who already
    knows that file's tiers does not have to learn a second set of boundaries.
  * protein family -- Pfam family via guidance/lp_split/build_family_holdout_split.py's
    _target_to_family_map() (gene_species -> first Pfam ID), reusing its exact 'gene_species = first two
    underscore-fields of the target name' convention rather than inventing a second one. Coverage here is
    981/997 targets (98.4%); targets with no annotation are kept as their own explicit bucket, never
    silently dropped or silently merged into whichever family happens to be numbered first.

Row alignment. The test predictions (guidance/track_e/_stage0_test_preds.npz, y_true/y_pred/tgt) and the
novelty-tier similarity cache (keyed to guidance/cheminformatics/ligand_only_test_preds.npz's row order)
come from two different scripts. Verified here, not assumed: both files have 11855 rows, and `tgt` vs
`target` and `y_true` vs `pk` are checked for EXACT equality before anything is joined by position --
this is the same discipline novelty_tiers.py already applies to its own similarity cache (an explicit
raise, not an assert, since assert is stripped under python -O).

Statistics: cluster_bootstrap_coverage from guidance/uncertainty_a1/conformal_prediction.py, reused
directly rather than reimplemented -- it already clusters by whatever `target` array it is given, so
calling it on a tier- or family-restricted slice produces a correctly within-subgroup-clustered bootstrap
with no new statistical code to get wrong. A stratum with fewer than MIN_CLUSTERS target clusters (the
same threshold novelty_tiers.py uses) is reported but flagged UNDERPOWERED rather than given a confident
verdict either way.

Usage:
  PYTHONPATH=. python guidance/uncertainty_a1/conditional_conformal_coverage.py
  PYTHONPATH=. python guidance/uncertainty_a1/conditional_conformal_coverage.py --b 500   # faster pilot
"""
import argparse
import json
import os

import numpy as np

from guidance.cheminformatics.novelty_tiers import TIERS, MIN_CLUSTERS, max_sim_to_train
from guidance.uncertainty_a1.conformal_prediction import cluster_bootstrap_coverage

TEST_CACHE = './guidance/track_e/_stage0_test_preds.npz'
LIGAND_PREDS = './guidance/cheminformatics/ligand_only_test_preds.npz'
Q_SOURCE = './guidance/uncertainty_a1/a1d_conformal_results.json'
UNIPROT_CACHE_PATH = './guidance/lp_split/uniprot_cache.json'
BOOT_SEED = 20260925
OUT = './guidance/uncertainty_a1/conditional_conformal_results.json'

# How many of the most frequent families get their own row; the rest are pooled into 'other_family'
# so a long tail of singleton/rare families does not produce dozens of underpowered one-line strata.
N_TOP_FAMILIES = 8


def load_test_cache_aligned():
    """Returns (y_true, y_pred, target, sim) for the test set, with the novelty-tier similarity array
    verified -- not assumed -- to be aligned to the same rows as y_true/y_pred/target."""
    tc = np.load(TEST_CACHE, allow_pickle=True)
    y_true, y_pred, target = tc['y_true'], tc['y_pred'], tc['tgt']

    lp = np.load(LIGAND_PREDS, allow_pickle=True)
    if len(lp['target']) != len(target):
        raise SystemExit(f'ABORT: {TEST_CACHE} has {len(target)} rows but {LIGAND_PREDS} has '
                         f'{len(lp["target"])}. These must be the same test set in the same order; '
                         f'rebuild whichever cache is stale before trusting any conditional result.')
    if not np.array_equal(lp['target'], target) or not np.allclose(lp['pk'], y_true):
        raise SystemExit(f'ABORT: {TEST_CACHE} and {LIGAND_PREDS} disagree on row content (target names '
                         f'or pK values differ), so they are not safely joinable by position. This check '
                         f'exists because nothing else guarantees these two independently-built caches '
                         f'share an iteration order.')

    sim = max_sim_to_train()
    if len(sim) != len(target):
        raise SystemExit(f'ABORT: novelty-tier similarity cache has {len(sim)} entries but the test set '
                         f'has {len(target)} rows. Rebuild with novelty_tiers.py --force before trusting '
                         f'any tier result here.')
    return y_true, y_pred, target, sim


def family_by_target(target):
    """Pfam family (first listed ID) per test row, via the SAME gene_species convention
    build_family_holdout_split.py uses. Returns an array of strings; unannotated targets get the
    literal string 'no_annotation' rather than None, so they show up as their own reported stratum
    instead of silently vanishing from a groupby."""
    with open(UNIPROT_CACHE_PATH) as f:
        uniprot_cache = json.load(f)
    fam_by_gene_species = {}
    for gene_species, rec in uniprot_cache.items():
        if rec.get('found') and rec.get('pfam_ids'):
            fam_by_gene_species[gene_species] = rec['pfam_ids'][0]

    fam = np.empty(len(target), dtype=object)
    for i, t in enumerate(target):
        gene_species = '_'.join(str(t).split('_')[:2])
        fam[i] = fam_by_gene_species.get(gene_species, 'no_annotation')
    return fam


def collapse_rare_families(fam, target, n_top):
    """Keeps the n_top families with the most DISTINCT TARGETS (not the most test ROWS -- a family
    with one target and many ligands should not outrank a family with many targets, since the
    bootstrap's unit of resampling is the target) as their own stratum; everything else (including
    'no_annotation') becomes 'other_family'. 'no_annotation' is deliberately never counted toward
    n_top, so it cannot crowd out a genuinely large real family."""
    targets_per_family = {}
    for f, t in zip(fam, target):
        if f == 'no_annotation':
            continue
        targets_per_family.setdefault(f, set()).add(t)
    ranked = sorted(targets_per_family.items(), key=lambda kv: -len(kv[1]))
    keep = {f for f, _ in ranked[:n_top]}
    collapsed = np.array([f if f in keep else ('no_annotation' if f == 'no_annotation' else 'other_family')
                          for f in fam], dtype=object)
    return collapsed, ranked[:n_top]


def stratum_coverage(covered, target, rows, q, B):
    """One stratum's point coverage + target-clustered bootstrap CI, or an UNDERPOWERED marker."""
    n_targets = len(np.unique(target[rows]))
    if n_targets < MIN_CLUSTERS:
        return dict(n=int(len(rows)), n_targets=int(n_targets), underpowered=True,
                   point=float(covered[rows].mean()) if len(rows) else None, ci95=None, gate_pass=None)
    point, lo, hi = cluster_bootstrap_coverage(covered[rows], target[rows], B, BOOT_SEED)
    return dict(n=int(len(rows)), n_targets=int(n_targets), underpowered=False,
               point=point, ci95=[lo, hi], gate_pass=bool(lo <= 0.90 <= hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    with open(Q_SOURCE) as f:
        qrec = json.load(f)
    q, alpha = qrec['q'], qrec['alpha']
    print(f'Using the already-deployed quantile q={q:.4f} (alpha={alpha}) from {Q_SOURCE} -- '
         f'NOT recomputed. The question here is whether THAT q keeps its coverage promise '
         f'within subgroups, not whether a different q would.')

    y_true, y_pred, target, sim = load_test_cache_aligned()
    covered = (np.abs(y_true - y_pred) <= q).astype(float)
    print(f'n_test={len(y_true)}  overall point coverage={covered.mean():.4f}  nominal={1 - alpha}')

    results = {'q': q, 'alpha': alpha, 'B': args.b, 'boot_seed': BOOT_SEED,
              'overall': stratum_coverage(covered, target, np.arange(len(target)), q, args.b)}

    print(f'\n=== PER NOVELTY TIER (target-clustered bootstrap, B={args.b}) ===')
    hdr = f'{"tier":22s} {"n":>6s} {"tgts":>5s} {"coverage":>9s} {"CI95":>20s} {"nominal in CI?":>14s}'
    print(hdr); print('-' * len(hdr))
    results['by_tier'] = {}
    for lo, hi, label in TIERS:
        rows = np.where((sim >= lo) & (sim < hi))[0]
        r = stratum_coverage(covered, target, rows, q, args.b) if len(rows) else \
            dict(n=0, n_targets=0, underpowered=True, point=None, ci95=None, gate_pass=None)
        results['by_tier'][label] = r
        ci_s = f'[{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}]' if r['ci95'] else '--'
        gate_s = 'UNDERPOWERED' if r['underpowered'] else ('YES' if r['gate_pass'] else 'NO <<<')
        pt_s = f'{r["point"]:.4f}' if r['point'] is not None else '--'
        print(f'{label:22s} {r["n"]:>6d} {r["n_targets"]:>5d} {pt_s:>9s} {ci_s:>20s} {gate_s:>14s}')

    fam = family_by_target(target)
    fam_collapsed, top_families = collapse_rare_families(fam, target, N_TOP_FAMILIES)
    n_unannotated_targets = len({t for f, t in zip(fam, target) if f == 'no_annotation'})
    print(f'\nPfam coverage: {len({t for f,t in zip(fam,target) if f!="no_annotation"})} / '
         f'{len(np.unique(target))} test targets annotated '
         f'({n_unannotated_targets} targets with no Pfam id -- their own stratum below, not dropped).')
    print(f'Top {N_TOP_FAMILIES} families by target count: ' +
         ', '.join(f'{f}({len(ts)} tgts)' for f, ts in top_families))

    print(f'\n=== PER PROTEIN FAMILY (target-clustered bootstrap, B={args.b}) ===')
    print(hdr.replace('tier', 'family')); print('-' * len(hdr))
    results['by_family'] = {}
    for label in list({f for f, _ in top_families}) + ['other_family', 'no_annotation']:
        rows = np.where(fam_collapsed == label)[0]
        r = stratum_coverage(covered, target, rows, q, args.b) if len(rows) else \
            dict(n=0, n_targets=0, underpowered=True, point=None, ci95=None, gate_pass=None)
        results['by_family'][label] = r
        ci_s = f'[{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}]' if r['ci95'] else '--'
        gate_s = 'UNDERPOWERED' if r['underpowered'] else ('YES' if r['gate_pass'] else 'NO <<<')
        pt_s = f'{r["point"]:.4f}' if r['point'] is not None else '--'
        print(f'{label:22s} {r["n"]:>6d} {r["n_targets"]:>5d} {pt_s:>9s} {ci_s:>20s} {gate_s:>14s}')

    n_fail_tier = sum(1 for r in results['by_tier'].values() if r['gate_pass'] is False)
    n_fail_fam = sum(1 for r in results['by_family'].values() if r['gate_pass'] is False)
    print(f'\n{n_fail_tier} of {len(results["by_tier"])} novelty tiers, '
         f'{n_fail_fam} of {len(results["by_family"])} protein families: '
         f'nominal 90% coverage falls OUTSIDE that stratum\'s 95% CI (not powered strata excluded).')
    if n_fail_tier or n_fail_fam:
        print('Conformal\'s one positive result is therefore MARGINAL ONLY -- it does not hold')
        print('conditionally everywhere, consistent with Jeliazkova et al. 2026\'s prediction for novel')
        print('chemistry and with Gibbs et al. 2023\'s general warning about marginal-vs-conditional')
        print('validity. This must be stated as a caveat on A1d, not omitted.')
    else:
        print('No stratum with enough power contradicts marginal coverage -- within the power this test')
        print('has, conditional coverage is not detectably different from the marginal guarantee.')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
