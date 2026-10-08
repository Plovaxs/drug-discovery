"""Streams BindingDB_All.tsv (9GB uncompressed, inside data/bindingdb/BindingDB_All_202610_tsv.zip) in
chunks and filters it against this project's val+test leakage-exclusion sets, using BindingDB's OWN
UniProt accession and PDB ID columns directly -- no ChEMBL-style crossref needed here, unlike BindingNet,
because BindingDB already reports UniProt (SwissProt) Primary ID and PDB ID(s) per target chain natively.

Two independent exclusion checks per row (either one triggers a drop), applied to BOTH target chains
(BindingDB records up to several chains per target; a multi-chain complex leaks if ANY chain matches):
  1. UniProt (SwissProt) Primary ID of Target Chain {1,2,...} in our val+test accession set
     (guidance/surrogate_data/val_test_target_uniprot.json, built during the BindingNet crossref step --
     reused here so the two filters cannot define "leakage" differently).
  2. Any PDB ID in "PDB ID(s) for Ligand-Target Complex" or "PDB ID(s) of Target Chain {1,2,...}"
     in our val+test receptor PDB ID set (guidance/surrogate_data/val_test_exclusion_set.json).

Affinity columns (Ki/IC50/Kd/EC50, all nM) commonly carry inequality prefixes in BindingDB (e.g. ">10000",
"<0.5") for censored measurements -- parsed and flagged per-row (censored=True), not silently stripped to
an exact value.

This does NOT yet pair rows with 3D structures (BindingDB reports sequences/ligands/affinities, not pocket
coordinates) -- that is the next, separate step (TASKS.md "BindingDB ... belum ada struktur pocket").

Usage:
  python guidance/surrogate_data/build_bindingdb_index.py
"""
import json
import re
import zipfile

import pandas as pd

ZIP_PATH = './data/bindingdb/BindingDB_All_202610_tsv.zip'
TSV_NAME = 'BindingDB_All.tsv'
EXCLUSION = './guidance/surrogate_data/val_test_exclusion_set.json'
UNIPROT_MAP = './guidance/surrogate_data/val_test_target_uniprot.json'
OUT_FULL = './guidance/surrogate_data/bindingdb_index.csv'
OUT_FILTERED = './guidance/surrogate_data/bindingdb_index_filtered.csv'
CHUNKSIZE = 200_000
MAX_CHAINS = 4  # BindingDB's schema repeats chain-level columns up to several times; checked empirically below

AFFINITY_COLS = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
NUM_RE = re.compile(r'^\s*([<>]?)\s*([\d.eE+-]+)\s*$')


def parse_affinity(val):
    if pd.isna(val):
        return None, None
    m = NUM_RE.match(str(val))
    if not m:
        return None, None
    op, num = m.groups()
    return float(num), (op != '')


def main():
    excl = json.load(open(EXCLUSION))
    excl_rec_pdb = set(excl['val_test_rec_pdb'])
    uniprot_map = json.load(open(UNIPROT_MAP))
    excl_accessions = {acc for acc in uniprot_map.values() if acc}
    print(f'Exclusion sets: {len(excl_rec_pdb)} receptor PDB IDs, {len(excl_accessions)} UniProt accessions')

    with zipfile.ZipFile(ZIP_PATH) as zf:
        header = pd.read_csv(zf.open(TSV_NAME), sep='\t', nrows=0).columns.tolist()
    acc_cols = [c for c in header if c.startswith('UniProt (SwissProt) Primary ID of Target Chain')]
    pdb_cols = [c for c in header if c.startswith('PDB ID(s) of Target Chain')] + \
        [c for c in header if c == 'PDB ID(s) for Ligand-Target Complex']
    print(f'Found {len(acc_cols)} accession columns, {len(pdb_cols)} PDB-id columns: {acc_cols + pdb_cols}')

    usecols = ['Ligand SMILES', 'BindingDB MonomerID', 'Target Name',
               'Number of Protein Chains in Target (>1 implies a multichain complex)'] + \
        AFFINITY_COLS + acc_cols + pdb_cols

    kept_chunks = []
    n_total, n_kept = 0, 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        reader = pd.read_csv(zf.open(TSV_NAME), sep='\t', usecols=usecols, chunksize=CHUNKSIZE,
                              dtype=str, low_memory=False, on_bad_lines='warn')
        for i, chunk in enumerate(reader):
            n_total += len(chunk)
            has_affinity = chunk[AFFINITY_COLS].notna().any(axis=1)
            chunk = chunk.loc[has_affinity]

            leak_acc = pd.Series(False, index=chunk.index)
            for c in acc_cols:
                leak_acc |= chunk[c].isin(excl_accessions)
            leak_pdb = pd.Series(False, index=chunk.index)
            for c in pdb_cols:
                leak_pdb |= chunk[c].fillna('').apply(
                    lambda s: any(p.strip().lower() in excl_rec_pdb for p in s.split(',')) if s else False)

            clean = chunk.loc[~(leak_acc | leak_pdb)].copy()
            n_kept += len(clean)
            kept_chunks.append(clean)
            print(f'chunk {i}: rows so far {n_total}, with-affinity-and-clean kept so far {n_kept}', flush=True)

    df = pd.concat(kept_chunks, ignore_index=True)
    for col in AFFINITY_COLS:
        parsed = df[col].apply(parse_affinity)
        df[col.replace(' (nM)', '_nM')] = [p[0] for p in parsed]
        df[col.replace(' (nM)', '_censored')] = [p[1] for p in parsed]
    df.to_csv(OUT_FULL, index=False)
    print(f'Total rows scanned: {n_total}. With affinity + receptor/target clean: {len(df)}. Saved to {OUT_FULL}')
    df.to_csv(OUT_FILTERED, index=False)
    print(f'(full == filtered here; the filtering already happened chunk-by-chunk above) -> {OUT_FILTERED}')


if __name__ == '__main__':
    main()
