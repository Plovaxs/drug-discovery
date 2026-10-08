"""Pairs the fully-filtered BindingNet v1 ligand index (bindingnet_v1_index_filtered2.csv -- already
excludes val+test leakage by both receptor-PDB-ID and ChEMBL/UniProt target identity) with the receptor
POCKET structures from Crystal_Templates_for_BindingNet1.tar.gz (now extracted to
data/bindingnet_v1/Crystal_Templates_for_BindingNet1/{pdb_template}/rec_h_opt.pdb).

The main BindingNetv1 archive's .pdb/.sdf files per entry are the MODELED LIGAND POSE only (verified when
that archive was first indexed) -- the actual protein receptor was never in that archive. Crystal_Templates
provides it: one rec_h_opt.pdb per template PDB ID, in the same coordinate frame the ligand pose was modeled
into (same template-based docking procedure), paired with the template's own native crystal ligand
(cry_lig_opt_converted.sdf, NOT used here -- that's the template's original co-crystallized ligand, a
different compound from the one in each row of the filtered index).

Sanity-checks the coordinate-frame assumption directly (not just assumed): for a sample of rows, computes
the minimum ligand-atom-to-receptor-atom distance and flags any pair where it's absurdly large (e.g. >50A,
which would mean the ligand pose and "paired" receptor are NOT in the same frame -- a real possibility if
the template-ID-to-directory mapping has edge cases, so this is checked, not assumed).

Usage:
  python guidance/surrogate_data/pair_bindingnet_structures.py
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem

IN = './guidance/surrogate_data/bindingnet_v1_index_filtered2.csv'
TEMPLATES_ROOT = './data/bindingnet_v1/Crystal_Templates_for_BindingNet1'
OUT = './guidance/surrogate_data/bindingnet_v1_paired.csv'
N_SANITY = 100


def receptor_path(pdb_template):
    p = os.path.join(TEMPLATES_ROOT, pdb_template, 'rec_h_opt.pdb')
    return p if os.path.exists(p) else None


def min_ligand_receptor_distance(sdf_path, receptor_path):
    mol = Chem.MolFromMolFile(sdf_path, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    lig_xyz = mol.GetConformer().GetPositions()
    rec_xyz = []
    with open(receptor_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                rec_xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not rec_xyz:
        return None
    rec_xyz = np.array(rec_xyz)
    d = np.sqrt(((lig_xyz[:, None, :] - rec_xyz[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def main():
    df = pd.read_csv(IN)
    df['receptor_path'] = df['pdb_template'].apply(receptor_path)
    has_structure = df['receptor_path'].notna()
    print(f'Rows with a receptor structure available: {has_structure.sum()} / {len(df)}')
    print(f'Missing templates (in filtered index but not in Crystal_Templates): '
          f'{df.loc[~has_structure, "pdb_template"].nunique()} distinct')

    paired = df.loc[has_structure].reset_index(drop=True)

    print(f'Sanity-checking coordinate frame on {N_SANITY} random rows...', flush=True)
    rng = np.random.RandomState(0)
    sample = paired.sample(n=min(N_SANITY, len(paired)), random_state=rng)
    dists = []
    for _, row in sample.iterrows():
        d = min_ligand_receptor_distance(row['sdf_path'], row['receptor_path'])
        dists.append(d)
    dists = np.array([d for d in dists if d is not None])
    n_far = int((dists > 50).sum())
    print(f'min ligand-receptor atom distance over {len(dists)} sampled rows: '
          f'mean={dists.mean():.2f}A, max={dists.max():.2f}A, n>50A (likely frame mismatch)={n_far}')
    if n_far > 0:
        print('WARNING: some sampled rows have ligand/receptor absurdly far apart -- '
              'coordinate-frame assumption may not hold for all rows; investigate before training.')

    paired.to_csv(OUT, index=False)
    print(f'Saved {len(paired)} structure-paired rows to {OUT}')


if __name__ == '__main__':
    main()
