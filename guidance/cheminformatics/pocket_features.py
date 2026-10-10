"""Protein-side (bioinformatics) and interaction-side (biomolecular informatics) representations of each
complex, completing the baseline ladder that ligand_only_baseline.py starts.

Why this is needed to finish the argument. Volkov et al. 2022 did not only show that LIGAND descriptors
match an interaction-aware network; they showed the same for PROTEIN descriptors. Testing only the ligand
side, as this project did first, replicates half of their control and leaves the obvious rebuttal open:
maybe the network is using the protein, just not in a way a ligand model can mimic. The complete ladder
settles it:

    ligand-only        what can be predicted from the small molecule alone
    pocket-only        what can be predicted from the binding site alone, with NO ligand
    ligand + pocket    whether the two sources add anything to each other
    interaction-only   whether cross-terms between them carry signal the marginals do not
    all three          the ceiling for descriptor-level modelling

The interaction block also tests a specific published claim on our data. Brown 2025 (PNAS, CORDIAL)
argues that models should be given distance-dependent protein-ligand INTERACTION signatures while being
denied direct parameterisation of protein and ligand chemical structure, because the latter is what lets
a model learn spurious structural shortcuts; they report that this inductive bias survives
leave-superfamily-out validation where conventional models degrade. Our split is target-disjoint, so it
is a fair setting to ask whether an interaction-only representation generalises better here.

Two representations are built per complex, both from the processed LMDB (no raw PDB files needed):

POCKET descriptors (bioinformatics-style, 40 features):
  * 20-dim amino-acid composition of the pocket -- the classical sequence-composition representation,
    here over the residues lining the site rather than a whole chain
  * physicochemical groupings over those residues: hydrophobic, polar, positive, negative, aromatic,
    glycine/proline (flexibility proxies)
  * element composition (C/N/O/S) and backbone-vs-sidechain fraction
  * geometry: atom and residue counts, radius of gyration, maximum extent, and a sphericity proxy
  None of this sees the ligand.

ECIF-style INTERACTION counts (Sanchez-Cruz et al.; 72 features):
  counts of (protein element, ligand element) pairs within two distance shells, 0-4 A and 4-6 A, over
  protein elements {C,N,O,S} and ligand elements {C,N,O,F,P,S,Cl,Br,I}.

  A control that matters more than the features themselves: raw interaction counts scale with ligand
  SIZE, and ligand size alone already reaches R2 0.307 on this task. So an interaction model built on raw
  counts would look informative while mostly re-encoding heavy-atom count. Both variants are therefore
  stored -- raw counts AND counts divided by the ligand's heavy-atom count -- so "interaction density"
  can be separated from "there is a lot of ligand here". Reporting only the raw version would be a
  confound of exactly the kind the scaffold audit already got caught by.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/pocket_features.py          # build the cache
  PYTHONPATH=. python guidance/cheminformatics/pocket_features.py --force
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

CACHE = './guidance/cheminformatics/cache/pocket_interaction_features.npz'
CD_LMDB = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'

# PDBProtein's AA_NAME_NUMBER ordering, which is what protein_atom_to_aa_type indexes into.
AA_ORDER = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
            'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'CYS'}
POLAR = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'HIS'}
POSITIVE = {'LYS', 'ARG', 'HIS'}
NEGATIVE = {'ASP', 'GLU'}
AROMATIC = {'PHE', 'TRP', 'TYR', 'HIS'}
FLEXIBLE = {'GLY', 'PRO'}

PROT_ELEMS = [6, 7, 8, 16]                                   # C, N, O, S
LIG_ELEMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]                 # C N O F P S Cl Br I
SHELLS = [(0.0, 4.0), (4.0, 6.0)]

POCKET_NAMES = ([f'aa_frac_{a}' for a in AA_ORDER]
                + ['frac_hydrophobic', 'frac_polar', 'frac_positive', 'frac_negative',
                   'frac_aromatic', 'frac_flexible']
                + ['elem_frac_C', 'elem_frac_N', 'elem_frac_O', 'elem_frac_S', 'frac_backbone']
                + ['n_atoms', 'n_residues', 'radius_of_gyration', 'max_extent', 'sphericity',
                   'mean_dist_to_centroid', 'atoms_per_residue'])
ECIF_NAMES = [f'ecif_P{p}_L{l}_s{si}' for si in range(len(SHELLS))
              for p in PROT_ELEMS for l in LIG_ELEMS]


def pocket_descriptors(d):
    pos = np.asarray(d['protein_pos'], dtype=np.float64)
    aa = np.asarray(d['protein_atom_to_aa_type'], dtype=np.int64)
    elem = np.asarray(d['protein_element'], dtype=np.int64)
    bb = np.asarray(d['protein_is_backbone'], dtype=np.float64)
    n = len(pos)
    out = np.zeros(len(POCKET_NAMES), dtype=np.float64)
    if n == 0:
        return out

    # Residue-level composition, not atom-level: an atom-weighted histogram would over-count large
    # side chains and make "composition" partly a size measure. Residues are identified by contiguous
    # (aa_type, position-cluster) runs, which is as close to residue identity as the stored fields allow;
    # atoms of one residue are contiguous in these records.
    change = np.ones(n, dtype=bool)
    change[1:] = aa[1:] != aa[:-1]
    # A residue boundary also requires a spatial jump, so two identical adjacent residue types are not
    # merged into one.
    if n > 1:
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        change[1:] |= step > 4.0
    res_start = np.where(change)[0]
    res_aa = aa[res_start]
    n_res = len(res_start)

    counts = np.bincount(res_aa[(res_aa >= 0) & (res_aa < 20)], minlength=20).astype(np.float64)
    out[:20] = counts / max(1.0, counts.sum())

    names = [AA_ORDER[i] if 0 <= i < 20 else '' for i in res_aa]
    denom = max(1, len(names))
    for j, group in enumerate([HYDROPHOBIC, POLAR, POSITIVE, NEGATIVE, AROMATIC, FLEXIBLE]):
        out[20 + j] = sum(1 for nm in names if nm in group) / denom

    for j, e in enumerate(PROT_ELEMS):
        out[26 + j] = float((elem == e).mean())
    out[30] = float(bb.mean())

    centroid = pos.mean(0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    rg = float(np.sqrt((dists ** 2).mean()))
    out[31] = float(n)
    out[32] = float(n_res)
    out[33] = rg
    out[34] = float(dists.max())
    # Sphericity proxy: a compact, ball-like pocket has rg/max_extent near sqrt(3/5) ~ 0.775.
    out[35] = rg / max(1e-9, float(dists.max()))
    out[36] = float(dists.mean())
    out[37] = float(n) / max(1.0, float(n_res))
    return out


def ecif_counts(d):
    """(raw_counts, size_normalised_counts) over (protein element, ligand element, distance shell)."""
    ppos = np.asarray(d['protein_pos'], dtype=np.float64)
    lpos = np.asarray(d['ligand_pos'], dtype=np.float64)
    pel = np.asarray(d['protein_element'], dtype=np.int64)
    lel = np.asarray(d['ligand_element'], dtype=np.int64)
    raw = np.zeros(len(ECIF_NAMES), dtype=np.float64)
    if len(ppos) == 0 or len(lpos) == 0:
        return raw, raw.copy()
    # Full pairwise distances: ~334 x ~25 per complex, so the direct computation is cheaper than any
    # spatial index would be, and exact.
    dmat = np.linalg.norm(ppos[:, None, :] - lpos[None, :, :], axis=2)
    k = 0
    for lo, hi in SHELLS:
        in_shell = (dmat >= lo) & (dmat < hi)
        for p in PROT_ELEMS:
            pm = pel == p
            if not pm.any():
                k += len(LIG_ELEMS)
                continue
            sub = in_shell[pm]
            for l in LIG_ELEMS:
                lm = lel == l
                raw[k] = float(sub[:, lm].sum()) if lm.any() else 0.0
                k += 1
    n_heavy = max(1.0, float(len(lel)))
    return raw, raw / n_heavy


def build(force=False):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    import lmdb
    from guidance.track_e.core import build_anchor_table, split_arrays
    table = build_anchor_table()
    wanted = sorted({int(i) for part in ('train', 'val', 'test')
                     for i in split_arrays(table, part)['idx']})

    db = lmdb.open(CD_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        pocket = np.zeros((len(wanted), len(POCKET_NAMES)), dtype=np.float64)
        ecif_raw = np.zeros((len(wanted), len(ECIF_NAMES)), dtype=np.float64)
        ecif_norm = np.zeros_like(ecif_raw)
        for r, idx in enumerate(wanted):
            d = pickle.loads(txn.get(keys[idx]))
            pocket[r] = pocket_descriptors(d)
            ecif_raw[r], ecif_norm[r] = ecif_counts(d)
            if (r + 1) % 10000 == 0:
                print(f'  {r + 1}/{len(wanted)}', flush=True)
    db.close()

    out = dict(idx=np.array(wanted, dtype=np.int64), pocket=pocket, ecif_raw=ecif_raw,
               ecif_norm=ecif_norm, pocket_names=np.array(POCKET_NAMES, dtype=object),
               ecif_names=np.array(ECIF_NAMES, dtype=object))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f'wrote {CACHE}: {len(wanted)} complexes, pocket {pocket.shape}, ecif {ecif_raw.shape}')
    return out


def split_view_pocket(part, feats=None):
    from guidance.track_e.core import build_anchor_table, split_arrays
    f = feats if feats is not None else build()
    row_of = {int(i): r for r, i in enumerate(f['idx'])}
    A = split_arrays(build_anchor_table(), part)
    rows = np.array([row_of[int(i)] for i in A['idx']])
    return dict(pocket=f['pocket'][rows], ecif_raw=f['ecif_raw'][rows], ecif_norm=f['ecif_norm'][rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    f = build(force=args.force)
    print(f"\npocket features: {f['pocket'].shape[1]}  |  ECIF features: {f['ecif_raw'].shape[1]}")
    for part in ('train', 'val', 'test'):
        v = split_view_pocket(part, f)
        print(f"  {part:5s} pocket {v['pocket'].shape}  mean n_atoms "
              f"{v['pocket'][:, POCKET_NAMES.index('n_atoms')].mean():.1f}  "
              f"mean n_residues {v['pocket'][:, POCKET_NAMES.index('n_residues')].mean():.1f}  "
              f"mean ECIF contacts(0-4A) {v['ecif_raw'][:, :len(PROT_ELEMS) * len(LIG_ELEMS)].sum(1).mean():.1f}")


if __name__ == '__main__':
    main()
