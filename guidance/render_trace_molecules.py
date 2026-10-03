"""Renders the two molecules from guidance/trace_guidance_trajectory.py's
guided-vs-unguided comparison in 3D, in their real pocket context, reusing
guidance/render_examples.py's PyMOL rendering function. Re-runs the exact
same trace (same seed, same checkpoint, same lambda) to deterministically
reproduce the identical two molecules -- their SMILES already matched
GUIDANCE_MECHANISM_TRACE_FINDING.md's reported values when this was
checked, confirming determinism holds.
"""
import os

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

import utils.misc as misc
from guidance.lambda_sweep import load_everything, EXAMPLE_PDB
from guidance.affinity_guidance import AffinityGuidance
from guidance.trace_guidance_trajectory import run_one_trajectory
from utils.evaluation import atom_num
from utils.transforms import get_atomic_number_from_index, is_aromatic_from_index
from models.molopt_score_model import log_sample_categorical
from utils import reconstruct
from guidance.render_examples import render

OUT_DIR = './guidance/example_render'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data, model = load_everything(device)
    affinity_model = AffinityGuidance(
        './logs_lp_split_stage0_gradalign/crossdocked_affinity_egnn_2026_10_01__16_28_09_t500_a1_fixed/'
        'checkpoints/best.pt', device=device)

    seed = 7
    misc.seed_all(seed)
    protein_pos = data.protein_pos.to(device)
    protein_v = data.protein_atom_feature.float().to(device)
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)
    pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
    n_atoms = int(atom_num.sample_atom_num(pocket_size))
    batch_ligand = torch.zeros(n_atoms, dtype=torch.long, device=device)
    center = protein_pos.mean(dim=0, keepdim=True)
    init_ligand_pos = center + torch.randn(n_atoms, 3, device=device)
    uniform_logits = torch.zeros(n_atoms, model.num_classes, device=device)
    init_ligand_v = log_sample_categorical(uniform_logits)

    print('Regenerating unguided trajectory...')
    out_u, _ = run_one_trajectory(model, protein_pos, protein_v, batch_protein,
                                  init_ligand_pos, init_ligand_v, batch_ligand,
                                  1000, device, 0.0, affinity_model, seed)
    print('Regenerating guided trajectory (lambda=1.0)...')
    out_g, _ = run_one_trajectory(model, protein_pos, protein_v, batch_protein,
                                  init_ligand_pos, init_ligand_v, batch_ligand,
                                  1000, device, 1.0, affinity_model, seed)

    for name, out in [('unguided', out_u), ('guided', out_g)]:
        pos = out['pos'].cpu()
        v = out['v'].cpu()
        elem = get_atomic_number_from_index(v, mode='add_aromatic')
        aro = is_aromatic_from_index(v, mode='add_aromatic')
        mol = reconstruct.reconstruct_from_generated(
            pos.tolist(), [int(x) for x in elem], [bool(x) for x in aro])
        print(f'{name}: {Chem.MolToSmiles(mol)}')
        pdb_path = os.path.join(OUT_DIR, f'trace_{name}.pdb')
        Chem.MolToPDBFile(mol, pdb_path)
        png_path = os.path.join(OUT_DIR, f'trace_{name}.png')
        render(EXAMPLE_PDB, pdb_path, png_path, label_title=f'{name} (lambda={"0.0" if name=="unguided" else "1.0"})')
        print(f'  rendered -> {png_path}')


if __name__ == '__main__':
    main()
