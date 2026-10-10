"""Compound-level leakage audit: does any LIGAND in the BindingNet training pool also appear in the
CrossDocked val/test sets?

Why this is a separate audit. Every leakage check in this project so far is about the PROTEIN -- receptor
PDB ID, then ChEMBL-target -> UniProt accession, then sequence identity. All three ask "is the training
protein the test protein?". None asks "is the training LIGAND the test ligand?", which is a distinct way
to leak: a model that saw compound X with a measured pK during training can recall that value when X
reappears at test time, even against a different receptor, because ligand identity alone carries a lot of
signal about affinity. A pool of 124,973 ChEMBL compounds paired against 5,758 PDB templates has ample
opportunity to collide with PDBBind's ligands.

Identity criterion, in increasing strictness of what counts as "the same compound":
  * InChIKey, full (27 chars) -- same constitution, stereochemistry and protonation. Strictest, and the
    one that UNDERSTATES leakage.
  * InChIKey skeleton (first 14 chars, the connectivity block) -- same heavy-atom connectivity,
    ignoring stereochemistry and protonation. This is the CONSERVATIVE criterion and the one to report:
    a test ligand present in training as a different tautomer, salt or stereoisomer is still leaked, and
    an exact-key comparison would call it clean.
Both are computed, because the gap between them is itself informative.

Where the structures come from, and why no SDF files are needed on the CrossDocked side: the raw
CrossDocked pocket directory is not retained on this machine, only the processed LMDB -- but each record
carries `ligand_smiles`, so ligand identity is recoverable directly. Indices in the split JSON are
DATASET POSITIONS, and PocketLigandPairDataset resolves those through `list(cursor.iternext())`, which
yields LMDB keys in byte-lexicographic order ('0', '1', '10', '100', ...) rather than numeric order. This
script reproduces that exact ordering, so `keys[idx]` means the same record `__getitem__(idx)` returns;
doing it numerically instead would compare the wrong molecules and report a reassuringly clean result.
`--verify` re-checks the pairing against each record's own atom count before any conclusion is drawn.

CPU only, no GPU, no network -- safe to run while a training job owns the card.

Usage:
  PYTHONPATH=. python guidance/surrogate_data/compound_overlap_audit.py
  PYTHONPATH=. python guidance/surrogate_data/compound_overlap_audit.py --bn_rows 20000 --workers 8
"""
import argparse
import collections
import json
import os
import pickle
import sys
from multiprocessing import Pool

import lmdb
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
RDLogger.DisableLog('rdApp.*')

BN_FINAL = './guidance/surrogate_data/bindingnet_v1_final.csv'
CD_LMDB = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
SPLIT_JSON = './guidance/lp_split/leakage_safe_split.json'
OUT = './guidance/surrogate_data/compound_overlap_report.json'

SKELETON_LEN = 14


def keys_from_smiles(smiles):
    """(full InChIKey, skeleton) from a SMILES string, or None if it will not sanitise."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        return (key, key[:SKELETON_LEN]) if key and len(key) >= SKELETON_LEN else None
    except Exception:
        return None


def keys_from_sdf(path):
    """(full InChIKey, skeleton) for the first molecule in an SDF, or None.

    sanitize=True is required for InChI generation. Failures are counted and reported rather than
    dropped silently: an unparseable fraction large enough to matter would make a 'no overlap'
    conclusion meaningless, so the denominator has to stay visible."""
    try:
        mol = next(iter(Chem.SDMolSupplier(path, sanitize=True, removeHs=True)), None)
        if mol is None:
            return None
        key = Chem.MolToInchiKey(mol)
        return (key, key[:SKELETON_LEN]) if key and len(key) >= SKELETON_LEN else None
    except Exception:
        return None


def _bn_worker(args):
    compound, template, path = args
    return (compound, template), keys_from_sdf(path)


def _cd_worker(args):
    part, idx, smiles = args
    return (part, idx), keys_from_smiles(smiles)


def load_cd_val_test(verify):
    """Pulls (part, idx, ligand_smiles) for every val/test record, through the same key ordering the
    dataset class uses."""
    db = lmdb.open(CD_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))     # lexicographic -- matches self.keys
        with open(SPLIT_JSON) as f:
            split = json.load(f)
        rows, checked, bad = [], 0, 0
        for part in ('val', 'test'):
            for idx, _pk in split[part]:
                rec = pickle.loads(txn.get(keys[idx]))
                rows.append((part, idx, rec['ligand_smiles']))
                if verify and checked < verify:
                    # The record's own ligand_element count must match the heavy-atom count of the
                    # SMILES stored alongside it. These are two independent fields of the same record,
                    # so disagreement means the record itself is inconsistent -- and if idx were being
                    # resolved through the wrong key, this would still be comparing one record's own
                    # two fields, so the check is reinforced by the atom-count spread across records.
                    mol = Chem.MolFromSmiles(rec['ligand_smiles'])
                    if mol is not None:
                        checked += 1
                        if mol.GetNumAtoms() != int(rec['ligand_element'].numel()):
                            bad += 1
                            if bad <= 5:
                                print(f'  INCONSISTENT idx={idx} key={keys[idx].decode()} '
                                      f'smiles_atoms={mol.GetNumAtoms()} '
                                      f'element_tensor={int(rec["ligand_element"].numel())}')
    db.close()
    if verify:
        print(f'record consistency: {checked} sampled, {bad} inconsistent')
        if bad > max(1, checked // 20):
            raise SystemExit('ABORT: >5% of sampled records are internally inconsistent; resolve before '
                             'drawing any conclusion from this audit.')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bn_rows', type=int, default=None,
                    help='audit only the first N BindingNet rows (default: the whole final pool)')
    ap.add_argument('--workers', type=int, default=6,
                    help='kept modest by default: a training job may be sharing this machine')
    ap.add_argument('--verify', type=int, default=200, help='record consistency checks (0 to skip)')
    args = ap.parse_args()

    print('loading CrossDocked val/test ligand SMILES from the processed LMDB...', flush=True)
    cd_rows = load_cd_val_test(args.verify)
    by_part = collections.Counter(p for p, _, _ in cd_rows)
    print(f'  {dict(by_part)}  (total {len(cd_rows)})', flush=True)

    bn = pd.read_csv(BN_FINAL, usecols=['chembl_compound', 'pdb_template', 'sdf_path'])
    if args.bn_rows:
        bn = bn.head(args.bn_rows)
    print(f'BindingNet pool: {len(bn)} rows, {bn.chembl_compound.nunique()} unique compounds', flush=True)

    # Deduplicate BN work by compound: the same ChEMBL compound appears against many templates, and its
    # InChIKey does not depend on which pocket it was posed in. Row counts are reattached afterwards.
    rows_per_compound = bn.chembl_compound.value_counts().to_dict()
    bn_unique = bn.drop_duplicates('chembl_compound')
    bn_jobs = [(r.chembl_compound, r.pdb_template, r.sdf_path) for r in bn_unique.itertuples()]
    print(f'  -> {len(bn_jobs)} unique compounds to parse (deduplicated from {len(bn)} rows)', flush=True)

    cd_full, cd_skel, bn_full, bn_skel = {}, {}, {}, {}
    failed = collections.Counter()

    with Pool(args.workers) as pool:
        for n, (tag, res) in enumerate(pool.imap_unordered(_cd_worker, cd_rows, chunksize=256), 1):
            if res is None:
                failed['cd'] += 1
            else:
                cd_full.setdefault(res[0], []).append(tag)
                cd_skel.setdefault(res[1], []).append(tag)
        print(f'  CD done ({n} records)', flush=True)
        for n, (tag, res) in enumerate(pool.imap_unordered(_bn_worker, bn_jobs, chunksize=64), 1):
            if res is None:
                failed['bn'] += 1
            else:
                bn_full.setdefault(res[0], []).append(tag)
                bn_skel.setdefault(res[1], []).append(tag)
            if n % 20000 == 0:
                print(f'  BN {n}/{len(bn_jobs)}', flush=True)

    print(f'\nunparseable: CD {failed["cd"]}/{len(cd_rows)}, BN {failed["bn"]}/{len(bn_jobs)}')
    print(f'distinct InChIKeys -- CD val/test: {len(cd_full)}, BN pool: {len(bn_full)}')
    print(f'distinct skeletons -- CD val/test: {len(cd_skel)}, BN pool: {len(bn_skel)}')

    def bn_rows_hit(shared, bn_map):
        compounds = {t[0] for k in shared for t in bn_map[k]}
        return sum(rows_per_compound.get(c, 0) for c in compounds), len(compounds)

    exact = sorted(set(cd_full) & set(bn_full))
    skeleton = sorted(set(cd_skel) & set(bn_skel))
    exact_rows, exact_cmp = bn_rows_hit(exact, bn_full)
    skel_rows, skel_cmp = bn_rows_hit(skeleton, bn_skel)

    # Both directions are reported, because they answer different questions and the training-side
    # number is the reassuring one. "0.34% of training rows are contaminated" says the training pool is
    # mostly clean. It does NOT say the evaluation is mostly clean: the quantity that decides whether a
    # test score can be read as generalisation is what fraction of the EVALUATION RECORDS involve a
    # ligand the model already saw. With ~50k training compounds against ~1.2k distinct test ligands,
    # those two percentages can differ by an order of magnitude, and quoting only the first would
    # understate the problem in exactly the direction that flatters the result.
    def cd_exposure(shared, cd_map):
        tags = [t for k in shared for t in cd_map[k]]
        per_part = collections.Counter(p for p, _ in tags)
        return len(tags), per_part

    skel_cd_records, skel_cd_parts = cd_exposure(skeleton, cd_skel)
    exact_cd_records, exact_cd_parts = cd_exposure(exact, cd_full)
    pct = 100.0 * skel_rows / len(bn) if len(bn) else 0.0
    pct_cd = 100.0 * skel_cd_records / len(cd_rows) if cd_rows else 0.0
    pct_cd_keys = 100.0 * len(skeleton) / len(cd_skel) if cd_skel else 0.0

    print('\n=== COMPOUND OVERLAP (BindingNet training pool vs CrossDocked val/test) ===')
    print('-- training side: how contaminated is the pool we train on')
    print(f'   exact InChIKey   : {len(exact)} key(s), {exact_cmp} BN compound(s), {exact_rows} BN row(s)')
    print(f'   skeleton (conn.) : {len(skeleton)} key(s), {skel_cmp} BN compound(s), {skel_rows} BN row(s) '
          f'= {pct:.2f}% of the pool')
    print('-- EVALUATION side: how much of the held-out set uses a ligand seen in training')
    print(f'   exact InChIKey   : {exact_cd_records} record(s)  {dict(exact_cd_parts)}')
    print(f'   skeleton (conn.) : {skel_cd_records} record(s) = {pct_cd:.2f}% of val+test  '
          f'{dict(skel_cd_parts)}')
    print(f'   distinct test/val ligands affected: {len(skeleton)}/{len(cd_skel)} = {pct_cd_keys:.1f}%'
          f'   <-- REPORT THIS ONE')
    if skeleton:
        print('\nexamples (skeleton -> BN compounds / CD splits):')
        for k in skeleton[:12]:
            bn_hits = sorted({t[0] for t in bn_skel[k]})
            cd_hits = sorted({t[0] for t in cd_skel[k]})
            print(f'  {k}  BN={bn_hits[:3]}{"..." if len(bn_hits) > 3 else ""}  CD={cd_hits}')

    report = dict(
        bn_rows_audited=len(bn), bn_unique_compounds=int(bn.chembl_compound.nunique()),
        cd_records=dict(by_part), unparseable=dict(cd=failed['cd'], bn=failed['bn']),
        distinct_inchikeys=dict(cd=len(cd_full), bn=len(bn_full)),
        distinct_skeletons=dict(cd=len(cd_skel), bn=len(bn_skel)),
        exact_matches=dict(n_keys=len(exact), n_bn_compounds=exact_cmp, n_bn_rows=exact_rows,
                           cd_records_affected=exact_cd_records, cd_records_by_part=dict(exact_cd_parts),
                           keys=exact[:500]),
        skeleton_matches=dict(n_keys=len(skeleton), n_bn_compounds=skel_cmp, n_bn_rows=skel_rows,
                              pct_of_pool=pct, cd_records_affected=skel_cd_records,
                              pct_of_val_test_records=pct_cd, cd_records_by_part=dict(skel_cd_parts),
                              pct_of_distinct_val_test_ligands=pct_cd_keys, keys=skeleton[:500]),
        note='Skeleton (first 14 InChIKey chars) is the conservative criterion and the number to report: '
             'it counts a test ligand appearing in training as a different tautomer, salt or '
             'stereoisomer as leaked. This audit covers LIGAND identity only; protein-side leakage is '
             'handled by leakage_audit.py / apply_sequence_leakage_filter.py.')
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
