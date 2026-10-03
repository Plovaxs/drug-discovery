"""Direct test of the "guidance-mechanism-itself" hypothesis (the best-
supported explanation by elimination so far, per guidance/
FOLLOWUP_PHASE_POOLED_CORRECTION.md and the per-checkpoint finding docs:
every predictor-side fix -- size-bias, train/inference mismatch,
predictor accuracy, all four combined -- was tried and none rescued
guidance). This asks the mechanistic question directly: when guidance IS
applied, step by step, what actually happens to the trajectory?

Method: samples ONE molecule TWICE from the identical starting noise and
identical per-step stochastic draws -- once unguided (lambda=0), once
guided (lambda>0) -- by resetting torch's RNG to the same seed
immediately before each call. Since the diffusion core's forward pass is
deterministic given its input, and guidance is a deterministic nudge
(not a source of new randomness), the two trajectories are
BIT-IDENTICAL up to the first guided step; any divergence after that is
attributable ONLY to the guidance perturbation, not to independent
sampling noise. This is the clean counterfactual the hypothesis needs:
"what would this exact trajectory have looked like without guidance."

Tracked per guided step (via guided_sampling.py's new grad_capture_list
hook): the clean-data estimate (pos0-hat) BEFORE and AFTER the guidance
nudge, and the raw nudge itself. Tracked at every step via the function's
existing pos_traj return value: the ligand's actual position, in both
the guided and unguided runs.

What would CONFIRM the "injection mechanism absorbs/erases the nudge"
hypothesis: the guided-vs-unguided POSITION gap stays small/flat or
oscillates despite a nonzero nudge being re-applied every step (the
diffusion model's own posterior re-estimate pulls the state back toward
where the unguided trajectory already was).

What would REJECT it: the gap grows smoothly/monotonically (the nudge
compounds rather than getting erased) while the molecule stays valid
(single-fragment, no structural collapse) -- in that case the bottleneck
would have to be elsewhere (e.g. the gradient's direction being locally
correct but not pointing toward anything Vina's scoring function
rewards).
"""
import argparse
import json

import numpy as np
import torch
from rdkit import Chem, RDLogger

import utils.misc as misc
from guidance.lambda_sweep import load_everything, EXAMPLE_PDB
from guidance.affinity_guidance import AffinityGuidance
from guidance.guided_sampling import sample_diffusion_ligand_guided
from utils.evaluation import atom_num
from models.molopt_score_model import log_sample_categorical
from utils import reconstruct


def run_one_trajectory(model, protein_pos, protein_v, batch_protein,
                       init_ligand_pos, init_ligand_v, batch_ligand,
                       num_steps, device, lambda_affinity, affinity_model, seed):
    torch.manual_seed(seed)
    grad_capture = [] if lambda_affinity != 0.0 else None
    out = sample_diffusion_ligand_guided(
        model, protein_pos=protein_pos, protein_v=protein_v, batch_protein=batch_protein,
        init_ligand_pos=init_ligand_pos.clone(), init_ligand_v=init_ligand_v.clone(),
        batch_ligand=batch_ligand, num_steps=num_steps, center_pos_mode='protein', pos_only=False,
        affinity_model=affinity_model, synth_model=None,
        lambda_affinity=lambda_affinity, lambda_synth=0.0,
        grad_capture_list=grad_capture,
    )
    return out, grad_capture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--affinity_ckpt', type=str,
                        default='./logs_lp_split_stage0_gradalign/'
                                 'crossdocked_affinity_egnn_2026_10_01__16_28_09_t500_a1_fixed/'
                                 'checkpoints/best.pt',
                        help='default: gradient-alignment, the checkpoint whose gradient is most '
                             'directly trained to point toward the pocket -- if any checkpoint '
                             "should show a coherent, persistent guidance effect, it's this one")
    parser.add_argument('--lambda_affinity', type=float, default=1.0,
                        help='1.0: inside the stable (non-degrading) range per '
                             'GRADALIGN_GUIDANCE_FINDING.md\'s prelambda check')
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=str, default='./guidance/trace_guidance_trajectory_results.json')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data, model = load_everything(device)
    affinity_model = AffinityGuidance(args.affinity_ckpt, device=device)

    misc.seed_all(args.seed)
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

    print(f'n_atoms={n_atoms}')
    print('=== Running UNGUIDED trajectory ===')
    out_unguided, _ = run_one_trajectory(
        model, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
        args.num_steps, device, 0.0, affinity_model, args.seed)

    print(f'=== Running GUIDED trajectory (lambda={args.lambda_affinity}) ===')
    out_guided, grad_capture = run_one_trajectory(
        model, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
        args.num_steps, device, args.lambda_affinity, affinity_model, args.seed)

    pos_traj_u = out_unguided['pos_traj']  # list of (n_atoms, 3) tensors, one per step, step 0 = first (noisiest)
    pos_traj_g = out_guided['pos_traj']
    n_steps = len(pos_traj_u)
    assert n_steps == len(pos_traj_g) == len(grad_capture)

    # Find the atom with the largest cumulative |guidance nudge| across the whole trajectory
    nudge_mags_per_step_per_atom = torch.stack([g[2].norm(dim=-1) for g in grad_capture])  # (n_steps, n_atoms)
    cum_nudge = nudge_mags_per_step_per_atom.sum(dim=0)
    tracked_atom = int(cum_nudge.argmax().item())
    print(f'Tracking atom {tracked_atom} (largest cumulative guidance nudge = {cum_nudge[tracked_atom]:.4f})')

    first_divergence_step = None
    records = []
    for step_idx in range(n_steps):
        raw_i = grad_capture[step_idx][0]
        pos_u = pos_traj_u[step_idx][tracked_atom]
        pos_g = pos_traj_g[step_idx][tracked_atom]
        gap = float((pos_g - pos_u).norm())
        nudge_mag = float(grad_capture[step_idx][2][tracked_atom].norm())
        if first_divergence_step is None and gap > 1e-4:
            first_divergence_step = raw_i
        records.append({'raw_timestep': raw_i, 'guided_minus_unguided_gap': gap, 'nudge_magnitude': nudge_mag})

    print(f'First step with nonzero guided-vs-unguided gap: raw_timestep={first_divergence_step}')
    print(f'Final gap (last sampled step): {records[-1]["guided_minus_unguided_gap"]:.4f}')
    print(f'Max gap anywhere in trajectory: {max(r["guided_minus_unguided_gap"] for r in records):.4f}')

    dist_u_final = torch.cdist(pos_traj_u[-1][tracked_atom:tracked_atom+1], protein_pos.cpu()).min().item()
    dist_g_final = torch.cdist(pos_traj_g[-1][tracked_atom:tracked_atom+1], protein_pos.cpu()).min().item()
    print(f'Tracked atom final distance to nearest pocket atom: unguided={dist_u_final:.3f} A, '
          f'guided={dist_g_final:.3f} A')

    # All-atom check: the single most-nudged atom's own position gap staying
    # near zero does not by itself mean the WHOLE molecule stayed close --
    # a tiny position difference can flip a discrete atom-type sample
    # (log_sample_categorical is a Gumbel-max argmax, a decision-boundary-
    # sensitive operation) at an intermediate step, after which the two
    # trajectories' joint pos+type forward passes can diverge in ways not
    # visible by watching one atom's position alone. Checked explicitly
    # rather than assumed.
    all_atom_gap_final = (pos_traj_g[-1] - pos_traj_u[-1]).norm(dim=-1)  # (n_atoms,)
    print(f'All-atom final-step gap: mean={all_atom_gap_final.mean():.4f} A, '
          f'max={all_atom_gap_final.max():.4f} A (atom {int(all_atom_gap_final.argmax())}), '
          f'min={all_atom_gap_final.min():.4f} A')
    all_atom_gap_over_time = torch.stack([(pos_traj_g[s] - pos_traj_u[s]).norm(dim=-1) for s in range(n_steps)])
    mean_gap_per_step = all_atom_gap_over_time.mean(dim=1)  # (n_steps,) averaged over atoms
    print('Mean all-atom gap at 10 evenly-spaced steps through the trajectory:')
    for s in range(0, n_steps, max(1, n_steps // 10)):
        print(f'  step idx {s} (raw_timestep={grad_capture[s][0]}): mean_gap={mean_gap_per_step[s]:.4f} A')

    def try_reconstruct(pos, atom_type_idx):
        try:
            from utils.transforms import get_atomic_number_from_index, is_aromatic_from_index
            elem = get_atomic_number_from_index(atom_type_idx, mode='add_aromatic')
            aro = is_aromatic_from_index(atom_type_idx, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(
                pos.tolist(), [int(x) for x in elem], [bool(x) for x in aro])
            return Chem.MolToSmiles(mol), len(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
        except Exception as e:
            return None, str(e)

    smiles_u, frags_u = try_reconstruct(out_unguided['pos'].cpu(), out_unguided['v'].cpu())
    smiles_g, frags_g = try_reconstruct(out_guided['pos'].cpu(), out_guided['v'].cpu())
    print(f'Unguided final molecule: {smiles_u} (fragments={frags_u})')
    print(f'Guided final molecule:   {smiles_g} (fragments={frags_g})')

    result = {
        'affinity_ckpt': args.affinity_ckpt, 'lambda_affinity': args.lambda_affinity, 'seed': args.seed,
        'n_atoms': n_atoms, 'tracked_atom': tracked_atom,
        'first_divergence_raw_timestep': first_divergence_step,
        'final_gap': records[-1]['guided_minus_unguided_gap'],
        'max_gap': max(r['guided_minus_unguided_gap'] for r in records),
        'tracked_atom_dist_to_pocket_unguided': dist_u_final,
        'tracked_atom_dist_to_pocket_guided': dist_g_final,
        'all_atom_final_gap_mean': float(all_atom_gap_final.mean()),
        'all_atom_final_gap_max': float(all_atom_gap_final.max()),
        'all_atom_final_gap_max_atom': int(all_atom_gap_final.argmax()),
        'mean_all_atom_gap_per_step': [float(x) for x in mean_gap_per_step],
        'unguided_smiles': smiles_u, 'unguided_n_fragments': frags_u,
        'guided_smiles': smiles_g, 'guided_n_fragments': frags_g,
        'per_step': records,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
