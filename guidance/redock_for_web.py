"""Redock a specific already-generated molecule to recover its docked pose
as a PDB file, for the interactive 3D web viewer (Candidate Explorer
artifact). Mirrors guidance/render_examples.py's redock() but skips the
PyMOL rendering step -- only the pose PDB is needed here."""
import os
import subprocess
import sys

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.evaluation.docking_vina import VinaDockingTask

PROTEIN_ROOT = './data/test_set'
OUT_DIR = './guidance/example_render'
os.makedirs(OUT_DIR, exist_ok=True)


def redock(name, protein_pdb, ligand_sdf):
    protein_path = os.path.join(PROTEIN_ROOT, protein_pdb)
    mol = next(iter(Chem.SDMolSupplier(ligand_sdf, sanitize=True)))
    assert mol is not None, f'failed to load {ligand_sdf}'

    print(f'Re-docking {name} ...')
    task = VinaDockingTask(protein_path, mol, tmp_dir='./tmp')
    result = task.run(mode='dock', exhaustiveness=8)
    score = result[0]['affinity']
    print(f'  score={score:.3f}')

    pdbqt_path = os.path.join(OUT_DIR, f'{name}_pose.pdbqt')
    with open(pdbqt_path, 'w') as f:
        f.write(result[0]['pose'])
    pdb_path = os.path.join(OUT_DIR, f'{name}_pose.pdb')
    subprocess.run(['obabel', pdbqt_path, '-O', pdb_path], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'  wrote {pdb_path}')
    return protein_path, pdb_path, score


if __name__ == '__main__':
    redock('HDAC8', 'HDAC8_HUMAN_1_377_0/4rn0_B_rec.pdb',
           './guidance/generated_candidates/HDAC8_HUMAN_1_377_0/sdf/003.sdf')
    redock('P2Y12', 'P2Y12_HUMAN_1_342_0/4pxz_A_rec.pdb',
           './guidance/generated_candidates/P2Y12_HUMAN_1_342_0/sdf/008.sdf')
