"""New follow-up branch: tests a real generalization question raised by the
Literature Matrix PDF analysis and the 6-level research-gap taxonomy
(L3 Generalization / L5 Biological-Physical Validity) -- our entire guidance
pipeline conditions on an EXPERIMENTAL crystal/cross-docked pocket structure.
Every target we have evaluated on has one. A genuinely novel, undrugged
target usually does not. If someone wanted to apply this method to such a
target, the only structural input available would be a PREDICTED one.

This script asks two concrete, checkable questions for every target in the
LP-split test set (127 targets, all of which DO have an experimental
structure, used here as ground truth to grade the predicted alternative):
  1. How confident is AlphaFold DB's precomputed model specifically at the
     BINDING POCKET (not just globally over the whole protein)? Global
     pLDDT is the number usually reported, but it is dominated by the (much
     larger) non-pocket surface of the protein.
  2. If we superimpose the AlphaFold model onto the experimental structure
     using only the pocket residues, how far apart are they (Calpha RMSD)?

We do NOT re-run AlphaFold inference (infeasible on this 4GB GPU without
large genetic-database searches -- MSA generation alone typically takes
minutes-to-hours per target on CPU/cloud infrastructure neither available
nor justified here); we use the AlphaFold Protein Structure Database's
precomputed models (guidance/lp_split/alphafold_client.py), which is the
realistic way almost anyone without a compute cluster would obtain an
AlphaFold structure for a target of interest -- so this script measures
exactly the resource a generalization attempt would actually have.

Pocket definition: Calpha atoms of the experimental receptor within 10
angstrom of ANY docked-ligand atom (matches CrossDocked2020's own
"pocket10" naming convention already used throughout this dataset).

Residue correspondence between the experimental PDB (arbitrary author
numbering) and the AlphaFold model (strict 1..L UniProt sequence numbering)
is NOT assumed to be a trivial offset -- it is established by a real
pairwise sequence alignment of the extracted chain sequence against the
UniProt sequence (Bio.Align.PairwiseAligner, global, BLOSUM62-free identity
scoring since both sequences should be near-identical modulo missing/extra
residues at the termini or short internal gaps from disordered loops absent
in the crystal structure).

Caveats disclosed up front:
  - Targets where the experimental receptor's chain cannot be confidently
    matched to the cached UniProt accession's sequence (alignment identity
    < 0.9 over the aligned region) are skipped and reported, not forced.
  - Multi-chain receptors: only the single chain present in the
    "<pdbid>_<chain>_rec.pdb" file is used (matches how CrossDocked2020
    names these files -- one chain per file already).
  - This measures a PRECOMPUTED AlphaFold model's agreement with reality
    for already-solved, well-studied proteins (every CrossDocked2020 target
    was crystallized) -- it is a best-case proxy for the truly-novel-target
    scenario, not a direct measurement of it. Said plainly in the write-up,
    not glossed over.
"""
import argparse
import gzip
import json
import os
import re

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from rdkit import Chem

from guidance.lp_split.alphafold_client import fetch_records
from guidance.lp_split.uniprot_client import fetch_records as fetch_uniprot_records

DATA_ROOT = './data/CrossDocked2020_v1.1'  # full raw archive, see module docstring's
# 'all_local' option -- superseded ./data/test_set's 93-target demo subset
# once the full CrossDocked2020 v1.1 raw archive (43GB) was downloaded and
# extracted, giving 971/971 LP-split targets a local raw PDB rather than 46.
SPLIT_JSON = './guidance/lp_split/leakage_safe_split.json'
POCKET_RADIUS = 10.0
OUT_PATH = './guidance/alphafold_pocket_robustness_results.json'

aa3to1 = {k.upper(): v for k, v in protein_letters_3to1.items()}


def _chain_sequence_and_residues(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rec', pdb_path)
    chain = next(structure[0].get_chains())
    seq, residues = [], []
    for res in chain:
        if res.id[0] != ' ':  # skip waters/heteroatoms
            continue
        if res.resname not in aa3to1:
            continue
        if 'CA' not in res:
            continue
        seq.append(aa3to1[res.resname])
        residues.append(res)
    return ''.join(seq), residues


def _ligand_atom_coords(sdf_path):
    mol = Chem.MolFromMolFile(sdf_path, sanitize=False)
    if mol is None:
        with Chem.SDMolSupplier(sdf_path, sanitize=False) as supp:
            mol = next((m for m in supp if m is not None), None)
    if mol is None:
        return None
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _resolve_ligand_source(target_dir, sdf_basename):
    """The full raw CrossDocked2020_v1.1 archive does not ship one .sdf
    file per pose (unlike the small ./data/test_set demo subset this
    script originally targeted) -- poses are bundled into one gzipped,
    multi-record SDF per receptor-ligand pair (e.g. "..._tt_min.sdf.gz"),
    with the LP-split's ligand_filename's trailing "_<N>.sdf" naming the
    record index within that bundle, not a standalone file. Returns
    ('direct', path) or ('gz', gz_path, record_index), or None if neither
    form is found.
    """
    direct_path = os.path.join(target_dir, sdf_basename)
    if os.path.isfile(direct_path):
        return ('direct', direct_path)
    m = re.match(r'^(.*)_(\d+)\.sdf$', sdf_basename)
    if not m:
        return None
    stem, idx = m.group(1), int(m.group(2))
    gz_path = os.path.join(target_dir, stem + '.sdf.gz')
    if os.path.isfile(gz_path):
        return ('gz', gz_path, idx)
    return None


def _ligand_atom_coords_from_gz(gz_path, record_index):
    with gzip.open(gz_path, 'rb') as f:
        supp = Chem.ForwardSDMolSupplier(f, sanitize=False)
        mols = list(supp)
    if record_index >= len(mols) or mols[record_index] is None:
        return None
    mol = mols[record_index]
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def _pocket_residue_indices(residues, ligand_coords, radius):
    ca_coords = np.array([res['CA'].coord for res in residues])
    dmin = np.linalg.norm(ca_coords[:, None, :] - ligand_coords[None, :, :], axis=-1).min(axis=1)
    return np.where(dmin <= radius)[0]


def _align_to_uniprot(chain_seq, uniprot_seq):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aln = aligner.align(chain_seq, uniprot_seq)[0]
    chain_idx_to_uniprot_idx = {}
    # aln.aligned gives matched (no-gap) block ranges in each sequence
    for (c_s, c_e), (u_s, u_e) in zip(aln.aligned[0], aln.aligned[1]):
        for off in range(c_e - c_s):
            chain_idx_to_uniprot_idx[c_s + off] = u_s + off
    n_matched = sum(1 for c_i, u_i in chain_idx_to_uniprot_idx.items()
                     if chain_seq[c_i] == uniprot_seq[u_i])
    identity = n_matched / max(len(chain_idx_to_uniprot_idx), 1)
    return chain_idx_to_uniprot_idx, identity


def _af_residue_atoms(af_pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('af', af_pdb_path)
    chain = next(structure[0].get_chains())
    residues = [res for res in chain if 'CA' in res]
    return residues  # AF models are numbered 1..L in sequence order, no gaps


def _residue_plddt(af_residue):
    return af_residue['CA'].get_bfactor()  # AFDB stores per-residue pLDDT in the B-factor column


def process_target(target_name, ligand_filename, uniprot_cache, af_cache, log):
    """ligand_filename: one SPECIFIC "<TARGET>/<pdbid>_<chain>_rec_..._lig_
    ..._docked_N.sdf" entry for this target, resolved by the caller from
    the LP-split's own dataset indices (guidance/lp_split/lp_split_loader.py)
    -- not by picking "the first .sdf/.pdb in the directory". This matters
    once DATA_ROOT is the full raw CrossDocked2020_v1.1 archive rather than
    the small test_set demo subset: the full archive pools EVERY historical
    receptor conformation and ligand pose for a target into one directory
    (e.g. 111 receptor PDBs and 40 ligand SDFs in a single target's folder),
    so "first file found" would silently pair a receptor with an unrelated
    ligand pose and corrupt the pocket-residue computation below.
    """
    gene_species = '_'.join(target_name.split('_')[:2])
    urec = uniprot_cache.get(gene_species)
    if not urec or not urec.get('found'):
        return {'target': target_name, 'status': 'no_uniprot_match'}
    accession = urec['accession']
    af_rec = af_cache.get(accession)
    if not af_rec or not af_rec.get('found') or 'struct_path' not in af_rec:
        return {'target': target_name, 'status': 'no_alphafold_structure', 'accession': accession}

    target_dir = os.path.join(DATA_ROOT, target_name)
    if not os.path.isdir(target_dir):
        return {'target': target_name, 'status': 'no_local_data_dir'}
    sdf_basename = os.path.basename(ligand_filename)
    if '_rec_' not in sdf_basename:
        return {'target': target_name, 'status': 'unparseable_ligand_filename'}
    rec_basename = sdf_basename.split('_rec_')[0] + '_rec.pdb'
    rec_path = os.path.join(target_dir, rec_basename)
    if not os.path.isfile(rec_path):
        return {'target': target_name, 'status': 'missing_pdb_or_sdf'}
    ligand_source = _resolve_ligand_source(target_dir, sdf_basename)
    if ligand_source is None:
        return {'target': target_name, 'status': 'missing_pdb_or_sdf'}

    try:
        chain_seq, residues = _chain_sequence_and_residues(rec_path)
        if ligand_source[0] == 'direct':
            ligand_coords = _ligand_atom_coords(ligand_source[1])
        else:
            ligand_coords = _ligand_atom_coords_from_gz(ligand_source[1], ligand_source[2])
        if ligand_coords is None or len(chain_seq) < 10:
            return {'target': target_name, 'status': 'parse_failed'}
        pocket_idx = _pocket_residue_indices(residues, ligand_coords, POCKET_RADIUS)
        if len(pocket_idx) < 3:
            return {'target': target_name, 'status': 'pocket_too_small'}

        chain_to_uni, identity = _align_to_uniprot(chain_seq, urec['sequence'])
        if identity < 0.9:
            return {'target': target_name, 'status': 'low_alignment_identity',
                    'identity': round(identity, 4)}

        af_residues = _af_residue_atoms(af_rec['struct_path'])

        exp_ca, af_ca, af_plddt = [], [], []
        for c_i in pocket_idx:
            u_i = chain_to_uni.get(int(c_i))
            if u_i is None or u_i >= len(af_residues):
                continue
            exp_ca.append(residues[c_i]['CA'].coord)
            af_ca.append(af_residues[u_i]['CA'].coord)
            af_plddt.append(_residue_plddt(af_residues[u_i]))
        if len(exp_ca) < 3:
            return {'target': target_name, 'status': 'too_few_matched_pocket_residues',
                    'identity': round(identity, 4)}

        exp_ca = np.array(exp_ca, dtype=np.float64)
        af_ca = np.array(af_ca, dtype=np.float64)
        rmsd_pocket = _kabsch_rmsd(exp_ca, af_ca)

        return {
            'target': target_name, 'status': 'ok', 'accession': accession,
            'identity': round(identity, 4),
            'n_pocket_residues_matched': len(exp_ca),
            'pocket_rmsd_after_superposition': round(float(rmsd_pocket), 3),
            'pocket_mean_plddt': round(float(np.mean(af_plddt)), 2),
            'pocket_min_plddt': round(float(np.min(af_plddt)), 2),
            'global_plddt': round(float(af_rec['global_plddt']), 2),
        }
    except Exception as e:
        return {'target': target_name, 'status': 'error', 'error': str(e)}


def _kabsch_rmsd(P, Q):
    """Standard Kabsch superposition RMSD between two (N,3) coordinate sets."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = (R @ Pc.T).T
    return np.sqrt(np.mean(np.sum((P_rot - Qc) ** 2, axis=-1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='debug: only process first N targets')
    parser.add_argument('--targets', type=str, default='all_local',
                         choices=['test', 'all_local'],
                         help="'test': only the 127 LP-split test targets (most will be "
                              "skipped, no_local_data_dir -- only ./data/test_set's 93 raw "
                              "PDB dirs, a small demo subset, are present on this machine, "
                              "not the full CrossDocked2020 raw archive). 'all_local' "
                              "(default): pool train+val+test LP targets and keep only the "
                              "~46 that intersect ./data/test_set -- this analysis compares "
                              "AlphaFold-vs-crystal structure agreement, independent of our "
                              "LP train/test partition, so pooling introduces no leakage.")
    args = parser.parse_args()

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    if args.targets == 'test':
        test_targets = split['test_targets']
    else:
        all_t = split['train_targets'] + split['val_targets'] + split['test_targets']
        test_targets = [t for t in all_t if os.path.isdir(os.path.join(DATA_ROOT, t))]
    if args.limit:
        test_targets = test_targets[:args.limit]
    print(f'{len(test_targets)} LP-split targets selected ({args.targets})')

    print('Resolving one representative (receptor, ligand-pose) pair per target '
          'from the LP-split dataset indices (not by guessing from directory listing -- '
          'see process_target\'s docstring)...')
    from guidance.lp_split.lp_split_loader import build_lp_splits
    base, splits, _ = build_lp_splits(train_subsample=None)
    wanted = set(test_targets)
    ligand_filename_by_target = {}
    for part in ('train', 'val', 'test'):
        for idx in splits[part]:
            lf = base[idx].ligand_filename
            t = lf.split('/')[0]
            if t in wanted and t not in ligand_filename_by_target:
                ligand_filename_by_target[t] = lf
    missing = wanted - set(ligand_filename_by_target)
    if missing:
        print(f'Warning: {len(missing)} selected targets have no dataset index '
              f'(unexpected, treated as no_local_data_dir): {sorted(missing)[:5]}...')

    gene_species_tokens = sorted({'_'.join(t.split('_')[:2]) for t in test_targets})
    print(f'Fetching/caching UniProt records for {len(gene_species_tokens)} gene/species tokens...')
    uniprot_cache = fetch_uniprot_records(gene_species_tokens, verbose=False)

    accessions = sorted({r['accession'] for r in uniprot_cache.values() if r.get('found')})
    print(f'Fetching/caching AlphaFold DB records + structures for {len(accessions)} UniProt accessions...')
    af_cache = fetch_records(accessions, log=print)

    results = []
    for i, target in enumerate(test_targets, 1):
        if target not in ligand_filename_by_target:
            res = {'target': target, 'status': 'no_local_data_dir'}
        else:
            res = process_target(target, ligand_filename_by_target[target], uniprot_cache, af_cache, print)
        results.append(res)
        print(f'[{i}/{len(test_targets)}] {target}: {res["status"]}')

    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r['status'] == 'ok']
    print(f'\n{len(ok)}/{len(results)} targets fully processed.')
    if ok:
        rmsd = np.array([r['pocket_rmsd_after_superposition'] for r in ok])
        pocket_plddt = np.array([r['pocket_mean_plddt'] for r in ok])
        global_plddt = np.array([r['global_plddt'] for r in ok])
        print(f'Pocket RMSD (AF vs crystal, after local superposition): '
              f'mean={rmsd.mean():.3f} median={np.median(rmsd):.3f} max={rmsd.max():.3f} A')
        print(f'Pocket-local mean pLDDT:  mean={pocket_plddt.mean():.2f}  median={np.median(pocket_plddt):.2f}')
        print(f'Whole-protein global pLDDT: mean={global_plddt.mean():.2f}  median={np.median(global_plddt):.2f}')
        print(f'Pocket pLDDT - global pLDDT (mean diff): {(pocket_plddt - global_plddt).mean():.2f}')
    from collections import Counter
    print('Status breakdown:', Counter(r['status'] for r in results))


if __name__ == '__main__':
    main()
