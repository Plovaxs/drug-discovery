"""One-off visualization script for thesis illustration purposes.

Re-docks a single already-selected molecule (the best PoseBusters-valid
Vina Dock score across all Track C pools) to recover its actual docked
3D pose (honest_eval.py discards the pose object and keeps only the
score), then renders a PyMOL figure of it inside the protein pocket.

Not part of the statistical pipeline -- purely for a qualitative example
figure. Uses the same VinaDockingTask class as eval/honest_eval.py so the
docking call (mode='dock', exhaustiveness=8, seed=0 default) is identical
to what actually produced the recorded score.
"""
import os
import subprocess

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.evaluation.docking_vina import VinaDockingTask

PROTEIN_ROOT = './data/test_set'
POCKET_DATA_ID = 70
PROTEIN_PDB = 'SQHC_ALIAD_1_631_0/1h36_A_rec.pdb'
LIGAND_SDF = './guidance/track_c_pools/pocket70/seed2021/sdf/259.sdf'
OUT_DIR = './guidance/example_render'
os.makedirs(OUT_DIR, exist_ok=True)

protein_path = os.path.join(PROTEIN_ROOT, PROTEIN_PDB)
mol = next(iter(Chem.SDMolSupplier(LIGAND_SDF, sanitize=True)))
assert mol is not None, 'failed to load ligand'

print(f'Re-docking {LIGAND_SDF} into {protein_path} ...')
task = VinaDockingTask(protein_path, mol, tmp_dir='./tmp')
result = task.run(mode='dock', exhaustiveness=8)
score = result[0]['affinity']
pose_pdbqt_str = result[0]['pose']
print(f'Re-docked score: {score:.3f} (expected ~ -15.786)')

pose_pdbqt_path = os.path.join(OUT_DIR, 'best_pose.pdbqt')
with open(pose_pdbqt_path, 'w') as f:
    f.write(pose_pdbqt_str)

pose_pdb_path = os.path.join(OUT_DIR, 'best_pose.pdb')
subprocess.run(['obabel', pose_pdbqt_path, '-O', pose_pdb_path], check=True)

print(f'Wrote docked pose to {pose_pdb_path}')
print(f'Protein path: {protein_path}')

with open(os.path.join(OUT_DIR, 'render_info.txt'), 'w') as f:
    f.write(f'protein_path={protein_path}\n')
    f.write(f'pose_pdb_path={pose_pdb_path}\n')
    f.write(f'score={score}\n')
