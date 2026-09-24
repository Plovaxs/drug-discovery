"""D.5 follow-up: a third candidate proxy for the synth-guidance
gradient's misdirection, requested after D.5 ruled out size/heaviness and
aromaticity (both CI include 0, DIAG_SYNTH_DIRECTION_FINDING.md).

New proxy: heteroatom fraction (P(atom is N/O/S/P/halogen), i.e. NOT
carbon or hydrogen) under the current soft type distribution -- distinct
from D.5's "heaviness" proxy (which weights by raw atomic number, so O
counts more than N which counts more than C) because RA-score-style
synthesizability classifiers are known to be sensitive to heteroatom-
dense functional groups (esters, amides, sulfonamides, etc.) in a way a
smooth atomic-number-weighted average does not specifically capture.

Same method, same scale as D.5 (3 pockets x n=16, captured at the same 3
timesteps from the unguided path) for direct comparability -- this is a
fresh generation run (the original D.5 script does not cache captures to
disk), not a re-analysis of stored data.
"""
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
POCKETS_FILE = './guidance/lpsplit_confirmation_pockets.json'
N_POCKETS = 3
N_SAMPLES = 16

HETEROATOM_INDICES = [i for i, (z, _) in MAP_INDEX_TO_ATOM_TYPE_AROMATIC.items() if z not in (1, 6)]


def load_model_and_dataset(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt['config'].data.transform.ligand_atom_mode)
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


def expected_heteroatom_fraction(v0):
    probs = torch.softmax(v0, dim=-1)
    het_mask = torch.zeros(v0.size(-1), device=v0.device, dtype=probs.dtype)
    het_mask[HETEROATOM_INDICES] = 1.0
    return (probs * het_mask.view(1, -1)).sum(dim=-1)


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

    het_corrs = []
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

            het_frac = expected_heteroatom_fraction(v0_g).detach().cpu().numpy()

            if np.std(grad_mag) > 1e-12 and np.std(het_frac) > 1e-12:
                het_corrs.append(float(stats.pearsonr(grad_mag, het_frac)[0]))

    return het_corrs


def main():
    with open(POCKETS_FILE) as f:
        pockets = json.load(f)[:N_POCKETS]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    synth_model = SynthGuidance(SYNTH_CHECKPOINT, device=device, use_bond_aware=True)

    all_het_corrs = []
    per_pocket = {}
    for pocket in pockets:
        print(f'=== pocket {pocket["target"]} ===')
        het_corrs = run_pocket(model, test_set, synth_model, pocket, N_SAMPLES, device)
        print(f'  n_het_corrs={len(het_corrs)}')
        per_pocket[pocket['target']] = {'heteroatom_corrs': het_corrs}
        all_het_corrs += het_corrs

    het_mean, het_lo, het_hi = bootstrap_ci(all_het_corrs)
    print(f'\nHeteroatom-fraction correlation: mean={het_mean:.3f} '
          f'95% CI=[{het_lo:.3f}, {het_hi:.3f}] (n={len(all_het_corrs)})')

    result = {
        'per_pocket': per_pocket,
        'heteroatom_correlation': {'mean': het_mean, 'ci_lo': het_lo, 'ci_hi': het_hi, 'n': len(all_het_corrs)},
    }
    with open('./guidance/diag_synth_direction_followup_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('Saved to ./guidance/diag_synth_direction_followup_results.json')


if __name__ == '__main__':
    main()
