"""Fourth and final leakage filter stage: drops BindingNet rows whose LIGAND also appears in the
CrossDocked val/test sets.

The three preceding stages are all protein-side (receptor PDB ID, ChEMBL-target -> UniProt accession,
sequence identity). compound_overlap_audit.py showed that leaves a ligand-side hole, and showed why the
hole is easy to dismiss: measured against the training pool it is 0.34% of rows, which reads as noise.
Measured against the EVALUATION set -- the direction that decides whether a test score means anything --
it is 10.12% of val+test records and 8.6% of distinct val/test ligands. The same leak, two framings,
thirtyfold apart.

The remediation is close to free, which is the whole argument for doing it: excluding every BindingNet
compound whose InChIKey connectivity skeleton matches a val/test ligand costs 423 rows out of 124,973.
Paying 0.34% of the training data to remove a known contaminant from 10% of the evaluation records is
not a trade-off worth deliberating over.

Criterion: InChIKey skeleton (first 14 characters, the connectivity block), NOT the full key. The
skeleton ignores stereochemistry and protonation, so it also catches a test ligand that appears in
training as a different tautomer, salt or stereoisomer -- all of which still leak the affinity label.
Using the full key instead would have retained 60 of the 101 colliding compound families.

Usage:
  python guidance/surrogate_data/compound_overlap_audit.py          # must be run first
  python guidance/surrogate_data/apply_compound_leakage_filter.py
"""
import argparse
import json
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RDLogger.DisableLog('rdApp.*')

FINAL_IN = './guidance/surrogate_data/bindingnet_v1_final.csv'
REPORT = './guidance/surrogate_data/compound_overlap_report.json'
OUT = './guidance/surrogate_data/bindingnet_v1_clean.csv'
SUMMARY = './guidance/surrogate_data/bindingnet_v1_clean_summary.json'

SKELETON_LEN = 14


def skeleton_of(path):
    try:
        mol = next(iter(Chem.SDMolSupplier(path, sanitize=True, removeHs=True)), None)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        return key[:SKELETON_LEN] if key and len(key) >= SKELETON_LEN else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    if not os.path.exists(REPORT):
        raise SystemExit(f'{REPORT} not found -- run compound_overlap_audit.py first')
    report = json.load(open(REPORT))
    bad_skeletons = set(report['skeleton_matches']['keys'])
    n_claimed = report['skeleton_matches']['n_keys']
    if len(bad_skeletons) < n_claimed:
        # The report truncates its key list at 500 for readability. If a future pool ever exceeds that,
        # filtering on the truncated list would silently under-filter, so refuse rather than half-apply.
        raise SystemExit(f'report lists only {len(bad_skeletons)} of {n_claimed} colliding skeletons '
                         f'(truncated); raise the keys[:500] cap in compound_overlap_audit.py and re-run')
    print(f'colliding skeletons to exclude: {len(bad_skeletons)}')

    df = pd.read_csv(FINAL_IN)
    # One parse per unique compound, not per row: 51,773 compounds back 124,973 rows, and a compound's
    # skeleton does not depend on which pocket it was posed in.
    uniq = df.drop_duplicates('chembl_compound')[['chembl_compound', 'sdf_path']]
    print(f'parsing {len(uniq)} unique compounds on {args.workers} worker(s)...', flush=True)
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        skels = pool.map(skeleton_of, uniq.sdf_path.tolist(), chunksize=64)
    skel_by_compound = dict(zip(uniq.chembl_compound, skels))

    unparsed = sum(1 for v in skel_by_compound.values() if v is None)
    leaked_compounds = {c for c, s in skel_by_compound.items() if s in bad_skeletons}
    leaked = df.chembl_compound.isin(leaked_compounds)
    clean = df.loc[~leaked].reset_index(drop=True)

    print(f'unparseable compounds (kept, cannot be judged): {unparsed}')
    print(f'compounds excluded: {len(leaked_compounds)}')
    print(f'rows: {len(df)} -> {len(clean)} (dropped {int(leaked.sum())}, '
          f'{100 * leaked.sum() / len(df):.2f}%)')
    print(f'templates: {df.pdb_template.nunique()} -> {clean.pdb_template.nunique()}')

    clean.to_csv(args.out, index=False)
    summary = dict(
        criterion='InChIKey connectivity skeleton (first 14 chars) vs CrossDocked val+test ligands',
        rows_in=len(df), rows_out=len(clean), rows_dropped=int(leaked.sum()),
        compounds_excluded=sorted(leaked_compounds), n_compounds_excluded=len(leaked_compounds),
        skeletons_excluded=sorted(bad_skeletons), unparseable_compounds_kept=unparsed,
        templates_in=int(df.pdb_template.nunique()), templates_out=int(clean.pdb_template.nunique()),
        upstream=dict(sequence_filter='bindingnet_v1_final_summary.json',
                      compound_audit='compound_overlap_report.json'),
        note='Fourth filter stage, ligand-side. The three upstream stages are protein-side only. '
             'Unparseable compounds are KEPT rather than dropped, since a parse failure is not evidence '
             'of leakage; their count is reported so the residual uncertainty is visible.')
    json.dump(summary, open(SUMMARY, 'w'), indent=2)
    print(f'Saved {args.out} and {SUMMARY}')


if __name__ == '__main__':
    main()
