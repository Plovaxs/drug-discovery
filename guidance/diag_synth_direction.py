"""D.5: DIAG1-equivalent mechanistic diagnostic for synth-guidance
(mirrors guidance/diag_size_confound.py's method and rigor, adapted since
SynthGuidance is ligand-only -- no protein/pocket input at all, so the
original "distance-to-pocket" axis doesn't apply here).

Question: does the synth-guidance gradient's direction correlate with
something plausibly related to real synthesizability, or with something
else? Two proxies tested, both cheap and directly computable from the
captured diffusion state:
- Size/heaviness (same construct as DIAG1's affinity-side check): expected
  atomic number under the current soft type distribution.
- Aromaticity fraction: expected P(aromatic) under the current soft type
  distribution, a cheap structural-complexity proxy (RA-score is a
  fingerprint-based classifier sensitive to ring/aromatic substructure
  complexity).

Method: capture genuine mid-trajectory (pos0-hat, v0-hat) states from the
UNGUIDED sampling path (read-only hook, zero effect on the trajectory),
compute the raw (unnormalized) synth-guidance gradient at each captured
state post-hoc, correlate per-atom gradient magnitude against each proxy.
Per-sample Pearson correlations aggregated with bootstrap 95% CI.
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
from guidance.synth_guidance import SynthGuidance
from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
SYNTH_CHECKPOINT = './logs_synth_lp/synth_ra_egnn_lp_2026_09_12__15_38_21_full/checkpoints/best.pt'
NUM_TIMESTEPS = 1000
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


def expected_heaviness_and_aromaticity(v0):
    """v0: (n_atoms, 13) 'add_aromatic'-mode logits. Returns (heaviness,
    aromaticity), each (n_atoms,), expectations under the current softmax."""
    probs = torch.softmax(v0, dim=-1)
    atomic_numbers = torch.tensor(
        [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in range(v0.size(-1))],
        device=v0.device, dtype=probs.dtype)
    is_aromatic = torch.tensor(
        [float(MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1]) for i in range(v0.size(-1))],
        device=v0.device, dtype=probs.dtype)
    heaviness = (probs * atomic_numbers.view(1, -1)).sum(dim=-1)
    aromaticity = (probs * is_aromatic.view(1, -1)).sum(dim=-1)
    return heaviness, aromaticity


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(lo), float(hi)


def run_pocket(model, test_set, synth_model, pocket, n_samples, device):
    misc.seed_all(2021)
    data = test_set[pocket['data_id']]
    captures = []
    sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=n_samples, device=device,
        num_steps=NUM_TIMESTEPS, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        capture_timesteps=CAPTURE_TIMESTEPS, capture_list=captures,
    )

    size_corrs, aromatic_corrs = [], []
    for i, pos0, v0, batch_ligand, protein_pos_all in captures:
        n_graphs = int(batch_ligand.max().item()) + 1
        for g in range(n_graphs):
            mask = batch_ligand == g
            pos0_g = pos0[mask]
            v0_g = v0[mask]
            n_l = pos0_g.size(0)
            if n_l < 3:
                continue
            batch_ligand_g = torch.zeros(n_l, dtype=torch.long, device=device)

            grad_pos, _ = synth_model.grad_log_score(pos0_g, v0_g, batch_ligand_g, None, None, None)
            grad_mag = grad_pos.norm(dim=-1).detach().cpu().numpy()

            heaviness, aromaticity = expected_heaviness_and_aromaticity(v0_g)
            heaviness = heaviness.detach().cpu().numpy()
            aromaticity = aromaticity.detach().cpu().numpy()

            if np.std(grad_mag) > 1e-12 and np.std(heaviness) > 1e-12:
                size_corrs.append(float(stats.pearsonr(grad_mag, heaviness)[0]))
            if np.std(grad_mag) > 1e-12 and np.std(aromaticity) > 1e-12:
                aromatic_corrs.append(float(stats.pearsonr(grad_mag, aromaticity)[0]))

    return size_corrs, aromatic_corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--out', type=str, default='./guidance/diag_synth_direction_results.json')
    args = parser.parse_args()

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    synth_model = SynthGuidance(SYNTH_CHECKPOINT, device=device, use_bond_aware=True)

    all_size_corrs, all_aromatic_corrs = [], []
    per_pocket = {}
    for pocket in pockets:
        print(f'=== pocket {pocket["target"]} ===')
        size_corrs, aromatic_corrs = run_pocket(model, test_set, synth_model, pocket, args.n_samples, device)
        print(f'  n_size_corrs={len(size_corrs)}, n_aromatic_corrs={len(aromatic_corrs)}')
        per_pocket[pocket['target']] = {'size_corrs': size_corrs, 'aromatic_corrs': aromatic_corrs}
        all_size_corrs += size_corrs
        all_aromatic_corrs += aromatic_corrs

    size_mean, size_lo, size_hi = bootstrap_ci(all_size_corrs)
    arom_mean, arom_lo, arom_hi = bootstrap_ci(all_aromatic_corrs)
    print(f'\nSize/heaviness correlation: mean={size_mean:.3f} 95% CI=[{size_lo:.3f}, {size_hi:.3f}] (n={len(all_size_corrs)})')
    print(f'Aromaticity correlation: mean={arom_mean:.3f} 95% CI=[{arom_lo:.3f}, {arom_hi:.3f}] (n={len(all_aromatic_corrs)})')

    result = {
        'per_pocket': per_pocket,
        'size_correlation': {'mean': size_mean, 'ci_lo': size_lo, 'ci_hi': size_hi, 'n': len(all_size_corrs)},
        'aromaticity_correlation': {'mean': arom_mean, 'ci_lo': arom_lo, 'ci_hi': arom_hi, 'n': len(all_aromatic_corrs)},
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
