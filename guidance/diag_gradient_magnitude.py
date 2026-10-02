"""Follow-up to DIAG1: compares raw gradient MAGNITUDE (not just direction)
of the affinity guidance model across checkpoints, on the exact same
captured unguided x0-hat states (captured once, reused for every
checkpoint, so differences are attributable only to the checkpoint).

Motivated by the noise-matched + Vina-target checkpoint's lambda sweep
showing its own predicted score barely moving (6.9327 -> 6.9320) across a
100x range of lambda, unlike every earlier checkpoint -- this asks whether
that is a genuine near-zero gradient magnitude, not just a misdirected one.
"""
import argparse
import json

import numpy as np
import torch

from guidance.affinity_guidance import AffinityGuidance
from guidance.diag_size_confound import load_model_and_dataset, CAPTURE_TIMESTEPS, NUM_TIMESTEPS
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
import utils.misc as misc


def capture_states(model, test_set, pocket, n_samples, device):
    misc.seed_all(2021)
    data = test_set[pocket['data_id']]
    captures = []
    sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=n_samples, device=device,
        num_steps=NUM_TIMESTEPS, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        capture_timesteps=CAPTURE_TIMESTEPS, capture_list=captures,
    )
    n_protein_atoms = data.protein_pos.size(0)
    protein_v_single = data.protein_atom_feature.float().to(device)
    per_graph = []
    for i, pos0, v0, batch_ligand, protein_pos_all in captures:
        n_graphs = int(batch_ligand.max().item()) + 1
        for g in range(n_graphs):
            mask = batch_ligand == g
            pos0_g = pos0[mask]
            v0_g = v0[mask]
            n_l = pos0_g.size(0)
            if n_l < 3:
                continue
            protein_pos_g = protein_pos_all[g * n_protein_atoms:(g + 1) * n_protein_atoms]
            per_graph.append((pos0_g, v0_g, protein_pos_g))
    return per_graph, protein_v_single


def grad_mags_for_ckpt(ckpt_path, per_graph, protein_v_single, device):
    # NOTE: grad_log_score's second return value is grad_v (all-zero, since
    # guidance is position-only in this project -- see
    # guidance/affinity_guidance.py), not the model's own predicted score.
    # The predicted score is recomputed here directly from the same forward
    # pass logic for reporting purposes only (not used for the gradient).
    guidance_model = AffinityGuidance(ckpt_path, device=device, normalize_gradient=False)
    all_mags = []
    all_own_scores = []
    for pos0_g, v0_g, protein_pos_g in per_graph:
        n_l = pos0_g.size(0)
        batch_ligand_g = torch.zeros(n_l, dtype=torch.long, device=device)
        batch_protein_g = torch.zeros(protein_pos_g.size(0), dtype=torch.long, device=device)
        grad_pos, _ = guidance_model.grad_log_score(
            pos0_g, v0_g, batch_ligand_g, protein_pos_g, protein_v_single, batch_protein_g)
        grad_mag = grad_pos.norm(dim=-1).detach().cpu().numpy()
        all_mags.extend(grad_mag.tolist())
        with torch.no_grad():
            from guidance.atom_features import v0_to_ligand_feature
            ligand_feat = v0_to_ligand_feature(v0_g, guidance_model._ligand_atom_feature_dim,
                                                pos=pos0_g, use_bond_aware=guidance_model.use_bond_aware)
            pred = guidance_model.model(
                protein_pos=protein_pos_g, protein_atom_feature=protein_v_single,
                ligand_pos=pos0_g, ligand_atom_feature=ligand_feat,
                batch_protein=batch_protein_g, batch_ligand=batch_ligand_g, output_kind=None)
        all_own_scores.append(float(pred.sum().cpu().item()))
    return np.array(all_mags), np.array(all_own_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--ckpts', type=str, nargs='+', required=True,
                         help='label=path pairs, e.g. original=./guidance_models/affinity_egnn_lpsplit.pt')
    parser.add_argument('--out', type=str, default='./guidance/diag_gradient_magnitude_results.json')
    args = parser.parse_args()

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)

    all_per_graph = []
    for pocket in pockets:
        print(f'=== capturing pocket {pocket["target"]} (checkpoint-independent, done once) ===')
        per_graph, protein_v_single = capture_states(model, test_set, pocket, args.n_samples, device)
        all_per_graph.append((per_graph, protein_v_single))
        print(f'  {len(per_graph)} captured (atom, pocket) states')

    result = {}
    for spec in args.ckpts:
        label, path = spec.split('=', 1)
        mags_all, own_all = [], []
        for per_graph, protein_v_single in all_per_graph:
            mags, own = grad_mags_for_ckpt(path, per_graph, protein_v_single, device)
            mags_all.append(mags)
            own_all.append(own)
        mags_all = np.concatenate(mags_all)
        own_all = np.concatenate(own_all)
        result[label] = {
            'ckpt': path,
            'grad_mag_mean': float(mags_all.mean()), 'grad_mag_median': float(np.median(mags_all)),
            'grad_mag_std': float(mags_all.std()), 'grad_mag_max': float(mags_all.max()),
            'own_score_mean': float(own_all.mean()), 'own_score_std': float(own_all.std()),
            'n_atoms': int(mags_all.size),
        }
        print(f'{label}: grad_mag mean={mags_all.mean():.6f} median={np.median(mags_all):.6f} '
              f'max={mags_all.max():.6f}  own_score mean={own_all.mean():.4f} std={own_all.std():.4f}')

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
