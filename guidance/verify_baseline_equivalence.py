"""Verifies that sample_diffusion_ligand_guided with lambda_affinity=0 and
lambda_synth=0 reproduces ScorePosNet3D.sample_diffusion (the original,
unmodified baseline).

Tolerance, not bit-exactness: on GPU, even calling the *unmodified baseline*
twice with the same seed does not give bit-identical output (verified
directly: two back-to-back calls to model.sample_diffusion with seed=12345
differ by max abs diff 3.814697265625e-06, which is exactly the diff this
script also sees between baseline and lambda=0-guided). This is standard
CUDA scatter/atomic-add reduction-order nondeterminism, not a guidance bug
-- so the pass criterion here is "guided-vs-baseline diff is of the same
order as baseline-vs-baseline diff", checked via torch.allclose with a
tolerance derived from that measurement, not torch.equal.

Uses the bundled example pocket (examples/1h36_..._pocket10.pdb) so this
runs without needing the CrossDocked2020 dataset. Run directly:

    python guidance/verify_baseline_equivalence.py

Exits nonzero and prints a diff summary if any mismatch is found.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from scripts.sample_for_pocket import pdb_to_pocket_data
from guidance.guided_sampling import sample_diffusion_ligand_guided

EXAMPLE_PDB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'examples', '1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb')
CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'pretrained_models', 'pretrained_diffusion.pt')


def build_init_state(model, data, device, seed, n_atoms=20):
    misc.seed_all(seed)
    batch_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.long, device=device)
    batch_ligand = torch.zeros(n_atoms, dtype=torch.long, device=device)
    center = data.protein_pos.mean(dim=0, keepdim=True).to(device)
    init_ligand_pos = center + torch.randn(n_atoms, 3, device=device)
    uniform_logits = torch.zeros(n_atoms, model.num_classes, device=device)
    init_ligand_v = log_sample_categorical(uniform_logits)
    return batch_protein, batch_ligand, init_ligand_pos, init_ligand_v


# Measured directly: two back-to-back calls to the unmodified
# model.sample_diffusion with the same seed differ by this much on GPU,
# purely from nondeterministic scatter/atomic-add reduction order. Guided
# (lambda=0) vs baseline is held to the same tolerance.
GPU_NONDETERMINISM_ATOL = 1e-5


def main(num_steps=20, seed=12345, n_atoms=20, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer])

    data = pdb_to_pocket_data(EXAMPLE_PDB)
    data = transform(data)
    data = data.to(device)

    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    protein_v = data.protein_atom_feature.float()

    # --- baseline: unmodified ScorePosNet3D.sample_diffusion ---
    batch_protein, batch_ligand, init_pos, init_v = build_init_state(model, data, device, seed, n_atoms)
    with torch.no_grad():
        baseline = model.sample_diffusion(
            protein_pos=data.protein_pos, protein_v=protein_v, batch_protein=batch_protein,
            init_ligand_pos=init_pos, init_ligand_v=init_v, batch_ligand=batch_ligand,
            num_steps=num_steps, center_pos_mode='protein', pos_only=False,
        )

    # --- guided, lambda=0 everywhere: must match exactly ---
    batch_protein, batch_ligand, init_pos, init_v = build_init_state(model, data, device, seed, n_atoms)
    guided = sample_diffusion_ligand_guided(
        model, protein_pos=data.protein_pos, protein_v=protein_v, batch_protein=batch_protein,
        init_ligand_pos=init_pos, init_ligand_v=init_v, batch_ligand=batch_ligand,
        num_steps=num_steps, center_pos_mode='protein', pos_only=False,
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
    )

    pos_max_diff = (baseline['pos'].cpu() - guided['pos'].cpu()).abs().max().item()
    pos_match = pos_max_diff <= GPU_NONDETERMINISM_ATOL
    v_match = torch.equal(baseline['v'].cpu(), guided['v'].cpu())

    print(f'pos max abs diff: {pos_max_diff:.3e} (tolerance: {GPU_NONDETERMINISM_ATOL:.0e})')
    print(f'v exact match:    {v_match}')

    if pos_match and v_match:
        print('PASS: lambda=0 guided sampling reproduces the unguided baseline '
              '(within GPU floating-point nondeterminism tolerance).')
        return 0
    else:
        print('FAIL: guided sampling with lambda=0 diverged from the baseline '
              'by more than expected GPU nondeterminism.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
