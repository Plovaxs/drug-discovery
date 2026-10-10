"""Applies the sequence-identity leakage filter found by leakage_audit.py to the structure-paired
BindingNet training index, producing the final training pool.

Why a third filter stage was needed: the first two (receptor PDB ID, then ChEMBL-target -> UniProt
accession) are both ID-based, and the audit proved that leaves real leakage behind. The clearest case:
template `1fm9` chain A is **100.0% identical** to P19793 (RXRA_HUMAN), one of our 127 test targets --
1FM9 is a PPAR-gamma/RXR-alpha heterodimer crystal, so it carries the test protein, but its PDB ID is
neither 1rdt nor 3a9e (the two the ID filter caught), so it passed straight through. There turned out to be
a whole family of these nuclear-receptor heterodimer structures (1rdt, 3a9e, 1fm9, 3h0a, ...) of which
ID-matching caught only two.

Policy, stated explicitly because the gray zone is a judgement call and the thesis must disclose it:
  * >= 90% identity  -> HARD EXCLUDE. Same protein as a val/test target under a different accession/PDB
    entry. 19 unique sequences / 28 templates / 265 rows.
  * 50-90% identity  -> KEPT, and DISCLOSED as a limitation. These are same-family homologs (e.g. other
    kinases vs a kinase test target), not the same protein. Excluding them would also be defensible, but
    generalising across a protein family is the normal premise of this kind of model, and dropping 10,435
    rows (8.3%) to chase it would weaken the data without a clear leakage argument. Reported, not hidden.
  * < 50%            -> kept, no concern.

Usage:
  python guidance/surrogate_data/apply_sequence_leakage_filter.py                 # default 90% cut
  python guidance/surrogate_data/apply_sequence_leakage_filter.py --threshold 50  # strict variant
"""
import argparse
import json

import pandas as pd

PAIRED_IN = './guidance/surrogate_data/bindingnet_v1_paired.csv'
PER_SEQ = './guidance/surrogate_data/leakage_audit_report_per_sequence.csv'
OUT = './guidance/surrogate_data/bindingnet_v1_final.csv'
SUMMARY = './guidance/surrogate_data/bindingnet_v1_final_summary.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=90.0)
    ap.add_argument('--out', type=str, default=OUT)
    args = ap.parse_args()

    df = pd.read_csv(PAIRED_IN)
    audit = pd.read_csv(PER_SEQ)

    bad = audit[audit.best_identity_pct >= args.threshold]
    bad_templates = set()
    for s in bad.bn_templates:
        bad_templates.update(x.split(':')[0] for x in str(s).split(';'))

    leaked = df.pdb_template.isin(bad_templates)
    clean = df.loc[~leaked].reset_index(drop=True)

    print(f'threshold: >= {args.threshold}% identity to any val/test target')
    print(f'templates excluded: {len(bad_templates)}')
    print(f'rows: {len(df)} -> {len(clean)} (dropped {int(leaked.sum())}, '
          f'{100 * leaked.sum() / len(df):.2f}%)')
    print(f'templates: {df.pdb_template.nunique()} -> {clean.pdb_template.nunique()}')

    clean.to_csv(args.out, index=False)
    summary = dict(threshold=args.threshold, rows_in=len(df), rows_out=len(clean),
                   rows_dropped=int(leaked.sum()), templates_excluded=sorted(bad_templates),
                   templates_in=int(df.pdb_template.nunique()),
                   templates_out=int(clean.pdb_template.nunique()),
                   note='gray zone 50-90% identity is KEPT by design and must be disclosed as a '
                        'limitation; see this script docstring for the reasoning')
    json.dump(summary, open(SUMMARY, 'w'), indent=2)
    print(f'Saved {args.out} and {SUMMARY}')


if __name__ == '__main__':
    main()
