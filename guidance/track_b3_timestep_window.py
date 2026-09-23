"""Track B3 (cheapest of the Stage 2+ Track B variants, run first per the
addendum's explicit prioritization): timestep-windowed guidance. Tests
whether the affinity gradient is more informative in some phase of the
diffusion trajectory (coarse-structure/early, middle, fine-detail/late)
than applied uniformly -- a pure sampling-time change, no new training,
using the existing Stage 0 model (guidance_models/affinity_egnn_lpsplit.pt).

Conditions per pocket: unguided baseline, uniform guidance (all
timesteps -- the condition used throughout every prior stage of this
project, as a reference point), and three windows (early/middle/late
thirds of the 1000-step trajectory), all at a single fixed,
previously-validated-safe lambda (10.0 -- confirmed non-fragmenting and
already used in the Stage 0 multi-pocket confirmation, so this reuses a
known-safe operating point rather than re-deriving one).

Checkpointed per (pocket, condition), matching this project's established
resumable-run pattern.
"""
import argparse
import json
import os

import torch
from torch_geometric.transforms import Compose
from rdkit import Chem, RDLogger

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
from guidance.affinity_guidance import AffinityGuidance
from utils import reconstruct
from eval.honest_eval import evaluate_molecules, load_sdf_mols

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
AFFINITY_CHECKPOINT = './guidance_models/affinity_egnn_lpsplit.pt'
PROTEIN_ROOT = './data/test_set'
FIXED_LAMBDA = 10.0
NUM_TIMESTEPS = 1000
WINDOWS = {
    'unguided': None,  # lambda forced to 0 regardless
    'uniform': (0, NUM_TIMESTEPS - 1),
    'late_third': (0, NUM_TIMESTEPS // 3 - 1),
    'middle_third': (NUM_TIMESTEPS // 3, 2 * NUM_TIMESTEPS // 3 - 1),
    'early_third': (2 * NUM_TIMESTEPS // 3, NUM_TIMESTEPS - 1),
}


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


def run_condition(model, test_set, affinity_model, condition, pocket, samples_per_pocket,
                  num_steps, batch_size, device, out_dir, dock_exhaustiveness, verbose=False):
    result_dir = os.path.join(out_dir, condition, f'pocket{pocket["data_id"]}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir

    sdf_dir = os.path.join(result_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    misc.seed_all(2021)
    data = test_set[pocket['data_id']]

    lam = 0.0 if condition == 'unguided' else FIXED_LAMBDA
    am = None if condition == 'unguided' else affinity_model
    window = WINDOWS[condition]

    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, samples_per_pocket, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=am, synth_model=None, lambda_affinity=lam, lambda_synth=0.0,
        guidance_timestep_window=window,
    )

    n_recon = 0
    for i, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception:
            continue
        w = Chem.SDWriter(os.path.join(sdf_dir, f'{i:03d}.sdf'))
        w.write(mol)
        w.close()
        n_recon += 1

    protein_path = os.path.join(PROTEIN_ROOT, pocket['expected_protein_pdb'])
    mols = load_sdf_mols(sdf_dir)
    df = evaluate_molecules(mols, protein_path, docking_mode='vina_dock', exhaustiveness=dock_exhaustiveness,
                            reference_sdf=None, run_posebusters=True, verbose=verbose)
    df['condition'] = condition
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df.to_csv(os.path.join(result_dir, 'honest_eval.csv'), index=False)

    with open(done_marker, 'w') as f:
        f.write(f'n_attempted={samples_per_pocket} n_reconstructed={n_recon} n_scored={len(df)}\n')

    return 'completed', result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=2, help='exploratory tier per Sec 1.2: 2-3 pockets')
    parser.add_argument('--samples_per_pocket', type=int, default=20, help='exploratory tier per Sec 1.2: 20-30')
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8,
                        help='lighter than the Task-F/full-tier standard of 16, since this is an '
                             'exploratory early-warning check per Sec 0.2, not the final evidentiary test')
    parser.add_argument('--out_dir', type=str, default='./guidance/track_b3_results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    affinity_model = AffinityGuidance(AFFINITY_CHECKPOINT, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')

    for pocket in pockets:
        for condition in WINDOWS:
            try:
                status, result_dir = run_condition(
                    model, test_set, affinity_model, condition, pocket, args.samples_per_pocket,
                    args.num_steps, args.batch_size, device, args.out_dir, args.dock_exhaustiveness,
                    verbose=args.verbose)
                entry = {'condition': condition, 'pocket': pocket['target'], 'status': status}
                print(f'{status.upper()}: condition={condition} pocket={pocket["target"]}')
            except Exception as e:
                import traceback
                entry = {'condition': condition, 'pocket': pocket['target'], 'status': 'failed',
                        'error': str(e), 'traceback': traceback.format_exc()}
                print(f'FAILED: condition={condition} pocket={pocket["target"]}: {e}')
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    print('Track B3 run complete (or all remaining conditions processed).')


if __name__ == '__main__':
    main()
