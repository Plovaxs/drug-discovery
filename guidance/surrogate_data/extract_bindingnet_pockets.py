"""Extracts pocket10-format training data from the structure-paired BindingNet v1 rows
(guidance/surrogate_data/bindingnet_v1_paired.csv, 125,238 rows already filtered clean of val+test leakage
by both receptor PDB ID and ChEMBL/UniProt target identity).

Output is deliberately the SAME on-disk layout CrossDocked's pocket10 data uses -- a directory of pocket
PDB + ligand SDF files plus an index.pkl of (pocket_fn, ligand_fn, ...) tuples -- so
datasets/pl_pair_dataset.py's PocketLigandPairDataset can process it into an LMDB with no new loader code,
and the existing, already-validated featurization path (PDBProtein.to_dict_atom, parse_sdf_file,
FeaturizeProteinAtom/FeaturizeLigandAtom) is reused verbatim rather than reimplemented.

Two conventions are matched to CrossDocked exactly, because getting either wrong would silently invalidate
any comparison against the Stage 0 baseline:

  1. HYDROGENS ARE STRIPPED. Verified empirically, not assumed: the protein atoms the Stage 0 model actually
     trains on (read straight out of the existing processed LMDB) contain ONLY Z=6,7,8,16 (C/N/O/S), zero
     Z=1, averaging 447.6 atoms/pocket. BindingNet's receptors are `rec_h_opt.pdb` -- protonated, ~6.5k
     atoms with H named like "1H"/"2H". Keeping them would roughly double the atom count and, with the
     encoder's knn=48 neighbourhood, let hydrogens crowd out the heavy atoms that carry the signal -- a
     distribution shift the Stage 0 checkpoint has never seen. H lines are dropped from the raw PDB text
     BEFORE PDBProtein parses it, so residue centers-of-mass (which drive pocket selection) are computed
     on heavy atoms too, and residues_to_pdb_block then re-emits only heavy-atom lines.
  2. radius=10 with criterion='center_of_mass', i.e. the exact defaults of
     scripts/data_preparation/extract_pockets.py, which is how CrossDocked's pocket10 set was built.

BindingNet's PDB files leave the element column (77-78) blank, so PDBProtein falls back to line[13:14] for
the element symbol. That fallback is correct for every atom here (all elements present are single-letter
C/N/O/S/H and the atom-name field is padded so line[13:14] lands on the element letter, including for
digit-prefixed hydrogens like "1H"), and all residues are the 20 standard amino acids with no HETATM
records -- both checked against the actual files before relying on it. The H filter below mirrors that same
element-inference logic so it cannot disagree with how PDBProtein would have classified an atom.

Usage:
  python guidance/surrogate_data/extract_bindingnet_pockets.py --limit 200   # pilot
  python guidance/surrogate_data/extract_bindingnet_pockets.py              # full 125,238 rows
"""
import argparse
import multiprocessing as mp
import os
import pickle
import shutil
from functools import partial

import pandas as pd
from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file

PAIRED_CSV = './guidance/surrogate_data/bindingnet_v1_paired.csv'
DEST = './data/bindingnet_pocket10'
RADIUS = 10


# AMBER-style protonation/disulfide residue variants -> their standard parent residue. BindingNet's
# receptors were prepared with AMBER naming (observed across a 300-template scan: HID, HIE, CYM; the rest
# are included because they are the same well-established convention and would otherwise KeyError in
# PDBProtein.AA_NAME_NUMBER, which only knows the 20 standard names). This is lossless for our purposes:
# the variants differ from their parent ONLY in hydrogens/protonation state, which strip_hydrogens()
# discards anyway, so the heavy-atom representation is identical -- and normalising here means the written
# pocket PDBs use standard names exactly like CrossDocked's do.
AA_VARIANTS = {
    'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS',
    'CYM': 'CYS', 'CYX': 'CYS',
    'ASH': 'ASP', 'GLH': 'GLU', 'LYN': 'LYS', 'ARN': 'ARG', 'TYM': 'TYR',
}


def element_of(line):
    """Mirrors PDBProtein._enum_formatted_atom_lines' element inference exactly."""
    symb = line[76:78].strip().capitalize()
    if len(symb) == 0:
        symb = line[13:14]
    return symb


def normalize_and_strip(pdb_text):
    """Drops hydrogens and rewrites AMBER residue-name variants to their standard parent, in one pass over
    the raw text, BEFORE PDBProtein sees it -- so residue centers-of-mass (which drive pocket selection)
    are heavy-atom-only and every res_name is one PDBProtein.AA_NAME_NUMBER knows."""
    kept = []
    unknown = set()
    for line in pdb_text.splitlines():
        if line[0:6].strip() == 'ATOM':
            if element_of(line) == 'H':
                continue
            res_name = line[17:20].strip()
            if res_name in AA_VARIANTS:
                line = line[:17] + AA_VARIANTS[res_name].ljust(3) + line[20:]
            elif res_name not in PDBProtein.AA_NAME_NUMBER:
                unknown.add(res_name)
                continue  # skip residues we cannot featurize rather than crashing the whole template
        kept.append(line)
    return '\n'.join(kept) + '\n', unknown


def process_template(task, dest, radius):
    """One receptor template and all of its ligand rows -- the receptor is parsed once and reused, since
    ~125k rows share only ~6.2k templates (parsing a 6.5k-atom PDB per row would dominate the runtime)."""
    pdb_template, receptor_path, rows = task
    out = []
    try:
        with open(receptor_path) as f:
            pdb_text, unknown_res = normalize_and_strip(f.read())
        protein = PDBProtein(pdb_text)
        if len(protein.residues) == 0:
            raise ValueError('no parseable residues left after normalisation')
    except Exception as e:
        return [(None, None, pdb_template, r['chembl_compound'], f'receptor parse failed: {e}') for r in rows]
    if unknown_res:
        print(f'  note: {pdb_template} had unrecognised residue names (skipped): {sorted(unknown_res)}', flush=True)

    for r in rows:
        sdf_path = r['sdf_path']
        try:
            ligand = parse_sdf_file(sdf_path)
            residues = protein.query_residues_ligand(ligand, radius)
            if len(residues) == 0:
                out.append((None, None, pdb_template, r['chembl_compound'], 'no residues within radius'))
                continue
            pocket_block = protein.residues_to_pdb_block(residues)

            stem = f"{pdb_template}/{pdb_template}_{r['chembl_target']}_{r['chembl_compound']}"
            pocket_fn = f'{stem}_pocket{radius}.pdb'
            ligand_fn = f'{stem}.sdf'
            os.makedirs(os.path.join(dest, pdb_template), exist_ok=True)
            with open(os.path.join(dest, pocket_fn), 'w') as f:
                f.write(pocket_block)
            shutil.copyfile(sdf_path, os.path.join(dest, ligand_fn))
            out.append((pocket_fn, ligand_fn, pdb_template, r['chembl_compound'], None))
        except Exception as e:
            out.append((None, None, pdb_template, r['chembl_compound'], f'{type(e).__name__}: {e}'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None, help='pilot: only this many rows')
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--dest', type=str, default=DEST)
    args = ap.parse_args()

    df = pd.read_csv(PAIRED_CSV)
    if args.limit:
        df = df.head(args.limit)
    print(f'rows: {len(df)}  unique templates: {df.pdb_template.nunique()}', flush=True)
    os.makedirs(args.dest, exist_ok=True)

    tasks = []
    for pdb_template, grp in df.groupby('pdb_template'):
        tasks.append((pdb_template, grp['receptor_path'].iloc[0], grp.to_dict('records')))
    print(f'tasks (one per template): {len(tasks)}', flush=True)

    results = []
    with mp.Pool(args.num_workers) as pool:
        for res in tqdm(pool.imap_unordered(partial(process_template, dest=args.dest, radius=RADIUS), tasks),
                        total=len(tasks)):
            results.extend(res)

    ok = [r for r in results if r[0] is not None]
    bad = [r for r in results if r[0] is None]
    print(f'extracted: {len(ok)}  failed: {len(bad)}', flush=True)
    if bad:
        from collections import Counter
        print('failure reasons:', Counter(r[4] for r in bad).most_common(5), flush=True)

    # index.pkl in the (pocket_fn, ligand_fn, ...) shape PocketLigandPairDataset expects; the extra
    # fields are ignored by that loader but keep the provenance for the affinity-label join below.
    index = [(r[0], r[1], r[2], r[3]) for r in ok]
    with open(os.path.join(args.dest, 'index.pkl'), 'wb') as f:
        pickle.dump(index, f)

    # Affinity labels keyed by the index POSITION, which is what the dataset's LMDB keys correspond to.
    key = df.set_index(['pdb_template', 'chembl_compound'])
    labels = []
    for i, (pocket_fn, ligand_fn, tmpl, compound) in enumerate(index):
        row = key.loc[(tmpl, compound)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        labels.append(dict(idx=i, pocket_fn=pocket_fn, ligand_fn=ligand_fn, pdb_template=tmpl,
                           chembl_compound=compound, pk=row['pk'], affinity_type=row['affinity_type'],
                           censored=row['censored'], core_rmsd=row['core_rmsd'], similarity=row['similarity']))
    pd.DataFrame(labels).to_csv(os.path.join(args.dest, 'labels.csv'), index=False)
    print(f"wrote {os.path.join(args.dest, 'index.pkl')} and labels.csv ({len(labels)} rows)", flush=True)


if __name__ == '__main__':
    main()
