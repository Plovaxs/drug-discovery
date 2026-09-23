"""DIAG1-size-direction: standalone size-confound diagnostic (addendum
Part 2), NOT a guidance variant -- no pass/fail gate, not tested through
the honest-eval/gradient-informativeness checkpoint pipeline. Answers a
narrower, interpretability-focused question: is the Stage 0 EGNN affinity
model's raw gradient DIRECTION (not just its trained point-prediction, per
Stage 0's own 0.719 heavy-atom-count correlation) dominated by a
size/heaviness-related signal, or does it look more plausibly
interaction-relevant (oriented toward nearby pocket contacts)?

Method: captures genuine mid-trajectory (pos0-hat, v0-hat) states from the
UNGUIDED sampling path (guidance/guided_sampling.py's capture_timesteps
hook -- read-only, zero effect on the sampled trajectory) at 3
representative timesteps (early/middle/late thirds, matching Track B3's
convention), computes the raw (unnormalized) affinity-model gradient at
each captured state post-hoc, then correlates per-atom gradient magnitude
against (a) that atom's expected heaviness under its current (soft, not
yet collapsed) type distribution, and (b) that atom's distance to the
nearest protein/pocket atom.

Per-sample Pearson correlations are aggregated with a bootstrap 95% CI,
consistent with this project's standard statistical-rigor requirement --
being a diagnostic does not exempt it from that.
"""
import argparse
import json

import numpy as np
import torch
from torch_geometric.transforms import Compose
from scipy import stats

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
from guidance.affinity_guidance import AffinityGuidance
from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
AFFINITY_CHECKPOINT = './guidance_models/affinity_egnn_lpsplit.pt'
NUM_TIMESTEPS = 1000
# Representative points within early/middle/late thirds (Track B3's convention:
# i=0 least noisy/final, i=999 most noisy/first).
CAPTURE_TIMESTEPS = {150, 500, 850}


def load_model_and_dataset(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    dataset, subsets = get_dataset(config=ckpt['config'].data, transform=transform)
    test_set = subsets['test']
    model = ScorePosNet3D(
        ckpt['config'].model, protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, test_set


def expected_heaviness(v0):
    """v0: (n_atoms, 13) 'add_aromatic'-mode logits. Returns (n_atoms,)
    expected atomic number under the current softmax distribution -- a
    continuous heaviness proxy reflecting what the model is currently
    leaning toward, not a hard, prematurely-collapsed argmax pick."""
    probs = torch.softmax(v0, dim=-1)
    atomic_numbers = torch.tensor(
        [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in range(v0.size(-1))],
        device=v0.device, dtype=probs.dtype)
    return (probs * atomic_numbers.view(1, -1)).sum(dim=-1)


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def run_pocket(model, test_set, guidance_model, pocket, n_samples, device):
    misc.seed_all(2021)
    data = test_set[pocket['data_id']]
    captures = []
    # batch_size=n_samples: forces exactly one internal batch, so
    # capture_list's batch_ligand indexing stays valid (see
    # guided_sampling.py's capture-hook docstring) without needing to track
    # per-internal-batch offsets.
    sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=n_samples, device=device,
        num_steps=NUM_TIMESTEPS, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        capture_timesteps=CAPTURE_TIMESTEPS, capture_list=captures,
    )

    n_protein_atoms = data.protein_pos.size(0)
    protein_v_single = data.protein_atom_feature.float().to(device)

    size_corrs, dist_corrs = [], []
    for i, pos0, v0, batch_ligand, protein_pos_all in captures:
        # protein_pos_all is the FULL BATCH's protein positions -- n_samples
        # concatenated copies of the same pocket (each independently
        # recentered by center_pos_mode='protein', so not byte-identical
        # across graphs, but each graph's own n_protein_atoms-sized block is
        # exactly what that graph's ligand atoms should be compared
        # against). Reconstruct the per-graph slicing explicitly rather than
        # assuming it's a single pocket's worth.
        n_graphs = int(batch_ligand.max().item()) + 1
        for g in range(n_graphs):
            mask = batch_ligand == g
            # grad_log_score does its own detach().clone().requires_grad_(True)
            # internally (see guidance/affinity_guidance.py), so pos0_g is
            # passed in plain, matching how every other caller uses it.
            pos0_g = pos0[mask]
            v0_g = v0[mask]
            n_l = pos0_g.size(0)
            if n_l < 3:
                continue
            protein_pos_g = protein_pos_all[g * n_protein_atoms:(g + 1) * n_protein_atoms]
            batch_ligand_g = torch.zeros(n_l, dtype=torch.long, device=device)
            batch_protein_g = torch.zeros(n_protein_atoms, dtype=torch.long, device=device)

            grad_pos, _ = guidance_model.grad_log_score(
                pos0_g, v0_g, batch_ligand_g, protein_pos_g, protein_v_single, batch_protein_g)
            grad_mag = grad_pos.norm(dim=-1).detach().cpu().numpy()

            heaviness = expected_heaviness(v0_g).detach().cpu().numpy()
            dist_to_pocket = torch.cdist(pos0_g.detach(), protein_pos_g).min(dim=-1).values.cpu().numpy()

            if np.std(grad_mag) > 1e-12 and np.std(heaviness) > 1e-12:
                size_corrs.append(float(stats.pearsonr(grad_mag, heaviness)[0]))
            if np.std(grad_mag) > 1e-12 and np.std(dist_to_pocket) > 1e-12:
                dist_corrs.append(float(stats.pearsonr(grad_mag, dist_to_pocket)[0]))

    return size_corrs, dist_corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--out', type=str, default='./guidance/diag_size_confound_results.json')
    args = parser.parse_args()

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    guidance_model = AffinityGuidance(AFFINITY_CHECKPOINT, device=device, normalize_gradient=False)

    all_size_corrs, all_dist_corrs = [], []
    per_pocket = {}
    for pocket in pockets:
        print(f'=== pocket {pocket["target"]} ===')
        size_corrs, dist_corrs = run_pocket(model, test_set, guidance_model, pocket, args.n_samples, device)
        print(f'  n_size_corrs={len(size_corrs)}, n_dist_corrs={len(dist_corrs)}')
        per_pocket[pocket['target']] = {'size_corrs': size_corrs, 'dist_corrs': dist_corrs}
        all_size_corrs += size_corrs
        all_dist_corrs += dist_corrs

    size_mean, size_lo, size_hi = bootstrap_ci(all_size_corrs)
    dist_mean, dist_lo, dist_hi = bootstrap_ci(all_dist_corrs)
    print(f'\nSize/heaviness correlation: mean={size_mean:.3f} 95% CI=[{size_lo:.3f}, {size_hi:.3f}] (n={len(all_size_corrs)})')
    print(f'Distance-to-pocket correlation: mean={dist_mean:.3f} 95% CI=[{dist_lo:.3f}, {dist_hi:.3f}] (n={len(all_dist_corrs)})')

    result = {
        'per_pocket': per_pocket,
        'size_correlation': {'mean': size_mean, 'ci_lo': size_lo, 'ci_hi': size_hi, 'n': len(all_size_corrs)},
        'distance_correlation': {'mean': dist_mean, 'ci_lo': dist_lo, 'ci_hi': dist_hi, 'n': len(all_dist_corrs)},
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
