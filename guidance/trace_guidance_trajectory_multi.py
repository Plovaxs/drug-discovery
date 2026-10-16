"""Audit item F060: repeat the paired per-step trajectory trace over many molecules, not one.

Appendix E.5 reports a single molecule sampled twice from bit-identical starting noise, once
unguided and once guided, and finds a final all-atom position gap of about 0.05 A, roughly 1% of
the raw cumulative nudge. One molecule cannot distinguish a general property of the mechanism from
an accident of one trajectory, which is what the audit objected to. This script runs the same
paired trace over many molecules and reports the distribution.

Two design points matter for the cost and for the validity.

The model, the dataset and the affinity checkpoint are loaded once and reused across every
molecule. The single-molecule script reloads all three per invocation, and on this hardware that
overhead is a large fraction of a four-minute run; amortising it is what makes a few dozen
molecules affordable at all on one laptop GPU.

Each molecule's guided and unguided trajectories are run from the *same* starting noise and the
*same* per-step stochastic draws, by resetting the seed immediately before each of the two calls,
exactly as the single-molecule version does. Without that, the two trajectories would differ
because of sampling noise and the measured gap would be meaningless. The seed varies between
molecules and is held fixed within a molecule.

Run:  PYTHONPATH=. python guidance/trace_guidance_trajectory_multi.py --n 24
"""
import argparse
import json
import time

import numpy as np
import torch
from rdkit import RDLogger

from guidance.trace_guidance_trajectory import run_one_trajectory
from guidance.lambda_sweep import load_everything
from guidance.affinity_guidance import AffinityGuidance
from models.molopt_score_model import log_sample_categorical
from utils.evaluation import atom_num
import utils.misc as misc

OUT = 'guidance/trace_guidance_trajectory_multi_results.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24, help='number of molecules to trace')
    ap.add_argument('--affinity_ckpt', type=str,
                    default='./logs_lp_split_stage0_gradalign/'
                            'crossdocked_affinity_egnn_2026_10_01__16_28_09_t500_a1_fixed/checkpoints/best.pt')
    ap.add_argument('--lambda_affinity', type=float, default=1.0)
    ap.add_argument('--num_steps', type=int, default=1000)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--out', type=str, default=OUT)
    args = ap.parse_args()
    RDLogger.DisableLog('rdApp.*')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data, model = load_everything(device)
    affinity_model = AffinityGuidance(args.affinity_ckpt, device=device)
    protein_pos = data.protein_pos.to(device)
    protein_v = data.protein_atom_feature.float().to(device)
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)
    pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
    center = protein_pos.mean(dim=0, keepdim=True)

    mols, t_start = [], time.time()
    for i in range(args.n):
        seed = args.seed0 + i
        misc.seed_all(seed)
        n_atoms = int(atom_num.sample_atom_num(pocket_size))
        batch_ligand = torch.zeros(n_atoms, dtype=torch.long, device=device)
        init_pos = center + torch.randn(n_atoms, 3, device=device)
        init_v = log_sample_categorical(torch.zeros(n_atoms, model.num_classes, device=device))

        out_u, _ = run_one_trajectory(model, protein_pos, protein_v, batch_protein,
                                      init_pos, init_v, batch_ligand, args.num_steps,
                                      device, 0.0, affinity_model, seed)
        out_g, grad = run_one_trajectory(model, protein_pos, protein_v, batch_protein,
                                         init_pos, init_v, batch_ligand, args.num_steps,
                                         device, args.lambda_affinity, affinity_model, seed)

        gaps = (out_g['pos_traj'][-1] - out_u['pos_traj'][-1]).norm(dim=-1)
        nudge = torch.stack([g[2].norm(dim=-1) for g in grad])      # (steps, atoms)
        cum_nudge = float(nudge.sum().item())
        final_gap_mean = float(gaps.mean().item())
        rec = dict(seed=seed, n_atoms=n_atoms,
                   final_gap_mean=final_gap_mean,
                   final_gap_max=float(gaps.max().item()),
                   cumulative_nudge_total=cum_nudge,
                   cumulative_nudge_per_atom=cum_nudge / max(n_atoms, 1),
                   gap_over_nudge=final_gap_mean / (cum_nudge / max(n_atoms, 1))
                   if cum_nudge > 0 else float('nan'))
        mols.append(rec)
        el = time.time() - t_start
        print(f'[{i + 1}/{args.n}] seed={seed} n_atoms={n_atoms} '
              f'final_gap_mean={final_gap_mean:.4f} A  '
              f'gap/nudge={rec["gap_over_nudge"] * 100:.2f}%  '
              f'({el / 60:.1f} min elapsed, {el / (i + 1) / 60:.1f} min/mol)', flush=True)
        with open(args.out, 'w') as f:
            json.dump(dict(config=vars(args), n_done=len(mols), molecules=mols), f, indent=1)

    g = np.array([m['final_gap_mean'] for m in mols])
    r = np.array([m['gap_over_nudge'] for m in mols]) * 100
    summary = dict(n_molecules=len(mols),
                   final_gap_mean_mean=float(g.mean()), final_gap_mean_median=float(np.median(g)),
                   final_gap_mean_sd=float(g.std(ddof=1)) if len(g) > 1 else None,
                   final_gap_mean_min=float(g.min()), final_gap_mean_max=float(g.max()),
                   gap_over_nudge_pct_median=float(np.median(r)),
                   gap_over_nudge_pct_min=float(r.min()), gap_over_nudge_pct_max=float(r.max()))
    print('\n=== F060 summary over %d molecules ===' % len(mols))
    for k, v in summary.items():
        print(f'  {k:32s} {v}')
    with open(args.out, 'w') as f:
        json.dump(dict(config=vars(args), n_done=len(mols), summary=summary, molecules=mols),
                  f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
