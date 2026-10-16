"""Audit item F051: draw the expanded ChEMBL cross-validation sample.

The original ChEMBL cross-validation (Appendix E) covered four targets, chosen because they
recurred throughout the appendix's diagnostic and docking checks, not because they were sampled.
F051 asked for a larger sample so a reader could tell whether the one disagreement found there
(CD38) was an isolated case or part of a pattern. This script performs the half of that check that
is local and reproducible: drawing a seeded random sample of 20 targets from the 127-target
leakage-safe test set and computing this project's own pK range for each.

It does not query ChEMBL. That half was performed through the ChEMBL web service (via the
bio-research ChEMBL connector) on 2026-10-16 and is not re-runnable from inside this repository as
a script; its results, including the exact ChEMBL target and assay identifiers needed to re-query
it directly, are recorded in `guidance/chembl_crossvalidation_expanded_results.json` alongside what
this script produces. Re-running this script reproduces exactly which 20 targets were drawn and
their project-side pK ranges; it does not reproduce the ChEMBL-side values, which depend on
ChEMBL's database as it stood on the retrieval date.

Run:  PYTHONPATH=. python guidance/chembl_f051_sample.py
"""
import json

import numpy as np

from guidance.cheminformatics.compare_vs_structure import LIGAND_PREDS

SEED = 20261016
N_SAMPLE = 20
OUT = 'guidance/chembl_f051_sample.json'


def main():
    L = np.load(LIGAND_PREDS, allow_pickle=True)
    target, pk = L['target'].astype(str), L['pk'].astype(float)
    targets = sorted(set(target))

    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(targets, size=N_SAMPLE, replace=False).tolist())

    rows = {}
    for t in sample:
        m = target == t
        rows[t] = dict(mnemonic=t.split('_')[0], n=int(m.sum()),
                       pk_min=float(pk[m].min()), pk_max=float(pk[m].max()))

    out = dict(seed=SEED, n_targets_in_pool=len(targets), n_sampled=N_SAMPLE, sample=rows)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'drew {N_SAMPLE} targets (seed {SEED}) from {len(targets)}; wrote {OUT}')
    for t, v in rows.items():
        print(f"  {t:36s} n={v['n']:>4d}  pK=[{v['pk_min']:.2f}, {v['pk_max']:.2f}]")


if __name__ == '__main__':
    main()
