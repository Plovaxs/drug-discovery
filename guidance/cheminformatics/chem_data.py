"""Ligand-side (cheminformatics) view of the LP split: SMILES, fingerprints, descriptors and scaffolds
for every CrossDocked complex, keyed by the same indices the rest of the project uses.

Why this module exists. Everything in this project so far describes a complex geometrically -- 3D
coordinates fed to an EGNN. That makes a whole class of control experiment impossible to run, and the
missing control is the one the field aims squarely at this model class:

  Volkov et al. 2022 (J. Med. Chem.) showed that for protein-ligand affinity, an explicit description of
  noncovalent interactions gives NO advantage over ligand descriptors or protein descriptors alone, and
  that simple nearest-neighbour models over the training set already perform well -- i.e. memorisation
  dominates learning. Mattsson et al. 2026 reproduced the same effect, measuring r = 0.66 for a
  ligand-only model on the FEP+ benchmark.

If a model that never sees the protein matches our EGNN on our own test split, then "structure-based
affinity prediction" is not what our number measures, and the thesis has to say so. That question cannot
be asked without ligand representations, which is what this file provides.

Design choices worth stating:
  * SMILES come from the processed LMDB's own `ligand_smiles` field, not from re-perceiving bonds from
    coordinates. The raw CrossDocked pocket directory is not retained on this machine, and re-deriving
    bond orders from geometry would introduce a second, different molecule for the same complex.
  * Targets reuse guidance/track_e/core.py's anchor table (`protein_filename.split('/')[0]`), so
    cluster-bootstrap groups here are IDENTICAL to those in A1-A1g. A ligand-only baseline evaluated with
    a different grouping than the model it is compared against would not be comparable.
  * Everything is cached to one .npz keyed by index, because fingerprinting 64,888 molecules takes
    minutes and every downstream script needs the same arrays.
  * Morgan fingerprints via rdFingerprintGenerator (the non-deprecated API in RDKit 2026.03).

Usage:
  PYTHONPATH=. python guidance/cheminformatics/chem_data.py            # build the cache
  PYTHONPATH=. python guidance/cheminformatics/chem_data.py --force    # rebuild it
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

CACHE_DIR = './guidance/cheminformatics/cache'
SMILES_CACHE = os.path.join(CACHE_DIR, 'ligand_smiles.json')
FP_CACHE = os.path.join(CACHE_DIR, 'ligand_features.npz')
CD_LMDB = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'

FP_RADIUS = 2          # ECFP4 equivalent, the cheminformatics default for SAR
FP_BITS = 2048

# A deliberately small, interpretable descriptor block rather than all ~210 RDKit descriptors. The point
# of this baseline is to show what trivially-available ligand information buys; a 200-dimensional
# descriptor dump on 46,964 training rows would start to be a competitive model in its own right and
# would muddle "is the protein being used?" with "is this a good QSAR model?".
DESCRIPTORS = ['MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
               'RingCount', 'NumAromaticRings', 'FractionCSP3', 'HeavyAtomCount',
               'NumHeteroatoms', 'MolMR', 'BalabanJ', 'BertzCT']


def _lmdb_keys_and_txn():
    import lmdb
    db = lmdb.open(CD_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    txn = db.begin()
    keys = list(txn.cursor().iternext(values=False))
    return db, txn, keys


def build_smiles_cache(force=False):
    """idx -> ligand SMILES for every index of the LP split.

    `keys` is taken in the LMDB cursor's own (byte-lexicographic) order because that is the order
    PocketLigandPairDataset exposes, so keys[idx] is the record __getitem__(idx) returns. Resolving idx
    numerically instead would silently pair each complex with a different molecule."""
    if os.path.exists(SMILES_CACHE) and not force:
        return {int(k): v for k, v in json.load(open(SMILES_CACHE)).items()}
    from guidance.track_e.core import build_anchor_table, split_arrays
    table = build_anchor_table()
    wanted = sorted({int(i) for part in ('train', 'val', 'test')
                     for i in split_arrays(table, part)['idx']})
    db, txn, keys = _lmdb_keys_and_txn()
    out = {}
    for n, idx in enumerate(wanted, 1):
        out[idx] = pickle.loads(txn.get(keys[idx]))['ligand_smiles']
        if n % 20000 == 0:
            print(f'  {n}/{len(wanted)} SMILES', flush=True)
    db.close()
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump({str(k): v for k, v in out.items()}, open(SMILES_CACHE, 'w'))
    print(f'wrote {SMILES_CACHE} ({len(out)} molecules)')
    return out


def build_features(force=False):
    """Returns a dict of aligned arrays over the union of all three splits:
        idx (int), smiles (str), fp (uint8 [n, FP_BITS]), desc (float64 [n, len(DESCRIPTORS)]),
        scaffold (str, Bemis-Murcko generic SMILES), ok (bool: molecule parsed)
    Molecules that fail to parse are KEPT as rows with ok=False rather than dropped, so every downstream
    script sees the same index set and has to decide explicitly what to do about them."""
    if os.path.exists(FP_CACHE) and not force:
        z = np.load(FP_CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    smiles_by_idx = build_smiles_cache(force=force)
    idxs = sorted(smiles_by_idx)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    desc_fns = [(name, getattr(Descriptors, name)) for name in DESCRIPTORS]

    fp = np.zeros((len(idxs), FP_BITS), dtype=np.uint8)
    desc = np.full((len(idxs), len(DESCRIPTORS)), np.nan, dtype=np.float64)
    scaffold = np.empty(len(idxs), dtype=object)
    ok = np.zeros(len(idxs), dtype=bool)
    smiles = np.empty(len(idxs), dtype=object)

    for row, idx in enumerate(idxs):
        smi = smiles_by_idx[idx]
        smiles[row] = smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold[row] = ''
            continue
        ok[row] = True
        fp[row] = np.frombuffer(gen.GetFingerprintAsNumPy(mol).astype(np.uint8).tobytes(),
                                dtype=np.uint8)
        for c, (_name, fn) in enumerate(desc_fns):
            try:
                desc[row, c] = float(fn(mol))
            except Exception:
                pass
        try:
            # Generic (topological) Bemis-Murcko framework: atom types and bond orders abstracted away.
            # The stricter choice on purpose -- it makes scaffold overlap between train and test LARGER,
            # i.e. it is the conservative direction for a leakage question.
            scaffold[row] = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)) and Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol)))
        except Exception:
            scaffold[row] = ''
        if (row + 1) % 20000 == 0:
            print(f'  featurised {row + 1}/{len(idxs)}', flush=True)

    out = dict(idx=np.array(idxs, dtype=np.int64), smiles=smiles, fp=fp, desc=desc,
               scaffold=scaffold, ok=ok, descriptor_names=np.array(DESCRIPTORS, dtype=object))
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(FP_CACHE, **out)
    print(f'wrote {FP_CACHE}: {len(idxs)} molecules, {int(ok.sum())} parsed, '
          f'{int((~ok).sum())} failed')
    return out


def split_view(part, features=None):
    """Arrays for one split, row-aligned: idx, target, pk, vina, n_lig, fp, desc, scaffold, smiles, ok.

    The join is by index against the anchor table, so target labels and pK values are exactly the ones
    A1-A1g used -- the cluster-bootstrap groups are shared, which is what makes a baseline built here
    comparable to the EGNN results already on record."""
    from guidance.track_e.core import build_anchor_table, split_arrays
    feats = features if features is not None else build_features()
    row_of = {int(i): r for r, i in enumerate(feats['idx'])}
    A = split_arrays(build_anchor_table(), part)
    rows = np.array([row_of[int(i)] for i in A['idx']])
    return dict(idx=A['idx'], target=A['target'], pk=A['pk'], vina=A['vina'], n_lig=A['n_lig'],
                fp=feats['fp'][rows], desc=feats['desc'][rows], scaffold=feats['scaffold'][rows],
                smiles=feats['smiles'][rows], ok=feats['ok'][rows])


def target_groups(target_arr):
    """Member-row indices per target, the grouping every cluster bootstrap in this project resamples."""
    groups = {}
    for r, t in enumerate(target_arr):
        groups.setdefault(str(t), []).append(r)
    return [np.array(v) for v in groups.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    feats = build_features(force=args.force)
    print(f"\ncached: {len(feats['idx'])} molecules, fp {feats['fp'].shape}, desc {feats['desc'].shape}")
    for part in ('train', 'val', 'test'):
        v = split_view(part, feats)
        print(f"  {part:5s} n={len(v['idx']):6d}  targets={len(set(map(str, v['target']))):4d}  "
              f"parsed={int(v['ok'].sum()):6d}  unique_scaffolds={len(set(v['scaffold'])):5d}  "
              f"pk mean {v['pk'].mean():.2f}")


if __name__ == '__main__':
    main()
