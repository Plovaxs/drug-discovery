"""Sequence-identity leakage audit for any new training data added to this project.

Why this exists: every leakage filter in the pipeline so far is ID-BASED (receptor PDB ID, ChEMBL target ->
UniProt accession). That demonstrably has a hole -- two *different* UniProt entries can be near-identical
proteins. We hit this directly: RXRA / PPARG / RARA are homologous nuclear-receptor LBD paralogs, and the
only reason the ID filter caught them at all was the coincidence that they shared a PDB entry (1rdt, 3a9e
are multi-chain heterodimer crystals). Had those paralogs appeared under separate PDB entries, an ID-based
filter would have passed them straight into training.

Sequence identity is the criterion the field actually uses -- it is exactly what LP-PDBBind's own
"LeakProof" split is built on (data/lp_pdbbind/LP_PDBBind.csv carries `seq` and `new_split`). So this audit
closes the gap between "our IDs don't collide" and "our training proteins aren't homologs of our test
proteins".

What it compares:
  * test/val side: full UniProt sequences for the 176 accessions behind our val+test target directories
    (resolved in guidance/surrogate_data/val_test_target_uniprot.json), fetched once and cached. Full
    sequences rather than pocket fragments, deliberately -- homology is a property of the protein, and
    using the full chain is the conservative choice.
  * training side: SEQRES chains of every BindingNet template actually used by
    bindingnet_v1_paired.csv (not all 17,179 in the archive -- only the 5,786 we train on).

Identity definition: a match/mismatch aligner (match=1, mismatch=0, affine gaps) is used so that the
alignment SCORE equals the count of identical aligned residues; identity = score / min(len_a, len_b).
This avoids a full traceback per pair (0.3 ms/pair measured, ~4 min for the full cross-product) while
staying a standard, interpretable definition. min-length normalisation is the conservative direction: a
short training chain that is a perfect subsequence of a long test protein still scores 100%.

Thresholds reported, not just a single pass/fail: >=90% (near-identical, treat as leakage), >=50% (same
family, a gray zone worth disclosing), plus the full distribution so the thesis can state what was found
rather than only that a check was run.

Usage:
  python guidance/surrogate_data/leakage_audit.py                 # full audit, writes JSON report
  python guidance/surrogate_data/leakage_audit.py --threshold 90  # change the leakage cut
"""
import argparse
import glob
import json
import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from Bio import Align

PAIRED_CSV = './guidance/surrogate_data/bindingnet_v1_paired.csv'
TEMPLATES = './data/bindingnet_v1/Crystal_Templates_for_BindingNet1'
VAL_TEST_UNIPROT = './guidance/surrogate_data/val_test_target_uniprot.json'
SEQ_CACHE = './guidance/surrogate_data/val_test_target_sequences.json'
OUT = './guidance/surrogate_data/leakage_audit_report.json'

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
       # AMBER protonation/disulfide variants, same residue identity (see extract_bindingnet_pockets.py)
       'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'CYM': 'C', 'CYX': 'C'}
MIN_CHAIN_LEN = 30


def parse_seqres(path):
    chains = {}
    with open(path) as f:
        for line in f:
            if line.startswith('SEQRES'):
                chains.setdefault(line[11], []).extend(line[19:].split())
    return {c: ''.join(AA3.get(r, 'X') for r in v) for c, v in chains.items()}


def fetch_uniprot_sequences(accessions):
    cache = json.load(open(SEQ_CACHE)) if os.path.exists(SEQ_CACHE) else {}
    todo = [a for a in accessions if a not in cache]
    for i, acc in enumerate(todo):
        try:
            with urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{acc}.fasta', timeout=30) as r:
                fasta = r.read().decode()
            cache[acc] = ''.join(fasta.splitlines()[1:])
        except Exception as e:
            print(f'  WARN could not fetch {acc}: {e}', flush=True)
            cache[acc] = None
        if (i + 1) % 25 == 0:
            json.dump(cache, open(SEQ_CACHE, 'w'))
            print(f'  fetched {i + 1}/{len(todo)}', flush=True)
    json.dump(cache, open(SEQ_CACHE, 'w'))
    return {a: s for a, s in cache.items() if s}


def make_aligner():
    al = Align.PairwiseAligner()
    al.match_score, al.mismatch_score = 1.0, 0.0
    al.open_gap_score, al.extend_gap_score = -1.0, -0.5
    al.mode = 'local'
    return al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=90.0, help='%% identity treated as leakage')
    ap.add_argument('--gray_threshold', type=float, default=50.0, help='%% identity treated as same-family')
    args = ap.parse_args()

    uni = json.load(open(VAL_TEST_UNIPROT))
    accessions = sorted({v for v in uni.values() if v})
    print(f'val+test target accessions: {len(accessions)}', flush=True)
    test_seqs = fetch_uniprot_sequences(accessions)
    print(f'sequences available for {len(test_seqs)}/{len(accessions)} accessions', flush=True)

    used = sorted(pd.read_csv(PAIRED_CSV, usecols=['pdb_template']).pdb_template.unique())
    print(f'BindingNet templates actually used in training data: {len(used)}', flush=True)

    bn_seqs = {}   # sequence -> list of templates carrying it
    missing_seqres = []
    for t in used:
        p = os.path.join(TEMPLATES, t, 'rec_h_opt.pdb')
        if not os.path.exists(p):
            missing_seqres.append(t)
            continue
        found = False
        for ch, s in parse_seqres(p).items():
            if len(s) >= MIN_CHAIN_LEN:
                bn_seqs.setdefault(s, []).append(f'{t}:{ch}')
                found = True
        if not found:
            missing_seqres.append(t)
    print(f'unique BN training sequences: {len(bn_seqs)} (templates without usable SEQRES: '
          f'{len(missing_seqres)})', flush=True)

    al = make_aligner()
    rows = []
    bn_list = list(bn_seqs.items())
    for i, (bseq, owners) in enumerate(bn_list):
        best_id, best_acc = 0.0, None
        for acc, tseq in test_seqs.items():
            ident = 100.0 * al.score(bseq, tseq) / min(len(bseq), len(tseq))
            if ident > best_id:
                best_id, best_acc = ident, acc
        rows.append(dict(bn_templates=';'.join(owners[:5]), n_templates=len(owners),
                         bn_len=len(bseq), best_identity_pct=best_id, best_match_accession=best_acc))
        if (i + 1) % 500 == 0:
            print(f'  aligned {i + 1}/{len(bn_list)} unique BN sequences', flush=True)

    df = pd.DataFrame(rows).sort_values('best_identity_pct', ascending=False)
    leaky = df[df.best_identity_pct >= args.threshold]
    gray = df[(df.best_identity_pct >= args.gray_threshold) & (df.best_identity_pct < args.threshold)]

    # map back to how many TRAINING ROWS are affected, which is what actually matters
    paired = pd.read_csv(PAIRED_CSV, usecols=['pdb_template'])
    def rows_for(frame):
        tmpl = set()
        for s in frame.bn_templates:
            tmpl.update(x.split(':')[0] for x in s.split(';'))
        return int(paired.pdb_template.isin(tmpl).sum()), len(tmpl)

    leak_rows, leak_tmpl = rows_for(leaky)
    gray_rows, gray_tmpl = rows_for(gray)

    print(f'\n=== LEAKAGE AUDIT ===')
    print(f'identity distribution (max vs any val/test target): '
          f'median {df.best_identity_pct.median():.1f}%  p90 {df.best_identity_pct.quantile(.9):.1f}%  '
          f'max {df.best_identity_pct.max():.1f}%')
    print(f'>= {args.threshold}% identity (LEAKAGE): {len(leaky)} unique seqs, {leak_tmpl} templates, '
          f'{leak_rows} training rows')
    print(f'{args.gray_threshold}-{args.threshold}% identity (same family, gray): {len(gray)} unique seqs, '
          f'{gray_tmpl} templates, {gray_rows} training rows')
    if len(leaky):
        print('\ntop offenders:')
        print(leaky.head(10).to_string(index=False))

    report = dict(threshold=args.threshold, gray_threshold=args.gray_threshold,
                  n_val_test_accessions=len(accessions), n_sequences_resolved=len(test_seqs),
                  n_bn_templates_used=len(used), n_unique_bn_sequences=len(bn_seqs),
                  templates_without_seqres=missing_seqres[:50],
                  identity_median=float(df.best_identity_pct.median()),
                  identity_p90=float(df.best_identity_pct.quantile(.9)),
                  identity_max=float(df.best_identity_pct.max()),
                  leakage=dict(unique_seqs=len(leaky), templates=leak_tmpl, training_rows=leak_rows),
                  gray_zone=dict(unique_seqs=len(gray), templates=gray_tmpl, training_rows=gray_rows))
    json.dump(report, open(OUT, 'w'), indent=2)
    df.to_csv(OUT.replace('.json', '_per_sequence.csv'), index=False)
    print(f'\nSaved {OUT} and *_per_sequence.csv')


if __name__ == '__main__':
    main()
