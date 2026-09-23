"""Track C, Part 1: generates unguided sampling pools and scores/docks
them. Reuses Task F's exact 20-pocket set (guidance/task_f_pockets.json)
for direct comparability with this project's other rigorous ablations
(user-approved scope decision, see guidance/STAGE2_PLUS_EXPERIMENT_LOG.md's
C1-scope entry).

Design: dock the FULL pool once per (pocket, seed) -- top-k / random-k /
scrambled-top-k are all post-hoc sub-selections of the SAME already-docked
pool (guidance/analyze_track_c.py), not separately generated/docked
conditions. This is the key cost saving relative to a naive 3x-the-work
design: real Vina Dock + PoseBusters only needs to run once per pool.

No gradient, no guidance, no training -- this reuses the plain unguided
sampling path (lambda_affinity=0.0) exactly as used for every "unguided"
baseline elsewhere in this project, plus a frozen forward-pass-only
ranker (guidance/affinity_point_estimate.py).
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
from models.property_pred.prop_model import PropPredNet
from guidance.affinity_point_estimate import score_finished_molecule
from utils import reconstruct
from eval.honest_eval import evaluate_molecules, load_sdf_mols
from eval.honest_eval_parallel import evaluate_molecules_parallel

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
AFFINITY_CHECKPOINT = './guidance_models/affinity_egnn_lpsplit.pt'
PROTEIN_ROOT = './data/test_set'


def load_diffusion_model_and_dataset(device):
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


def load_affinity_ranker(device):
    ckpt = torch.load(AFFINITY_CHECKPOINT, map_location=device, weights_only=False)
    model = PropPredNet(
        ckpt['config'].model,
        protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
        ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
        output_dim=1,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_pool(diffusion_model, test_set, ranker, pocket, seed, pool_size, num_steps,
            batch_size, device, out_dir, dock_exhaustiveness, verbose=False,
            use_amp=False, docking_n_workers=1):
    result_dir = os.path.join(out_dir, f'pocket{pocket["data_id"]}', f'seed{seed}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir

    sdf_dir = os.path.join(result_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    misc.seed_all(seed)
    data = test_set[pocket['data_id']]
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        diffusion_model, data, pool_size, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        use_amp=use_amp,
    )

    protein_pos = data.protein_pos.to(device)
    protein_atom_feature = data.protein_atom_feature.float().to(device)

    filename_to_score = {}
    n_written = 0
    for pos, v in zip(pred_pos, pred_v):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception:
            continue
        try:
            score = score_finished_molecule(ranker, mol, protein_pos, protein_atom_feature, device)
        except Exception as e:
            if verbose:
                print(f'  scoring failed: {e}')
            continue
        fname = f'{n_written:03d}.sdf'
        w = Chem.SDWriter(os.path.join(sdf_dir, fname))
        w.write(mol)
        w.close()
        filename_to_score[fname] = score
        n_written += 1

    protein_path = os.path.join(PROTEIN_ROOT, pocket['expected_protein_pdb'])
    mols_loaded = load_sdf_mols(sdf_dir)
    if docking_n_workers > 1:
        df = evaluate_molecules_parallel(mols_loaded, protein_path, docking_mode='vina_dock',
                                         exhaustiveness=dock_exhaustiveness, reference_sdf=None,
                                         run_posebusters=True, verbose=verbose,
                                         skip_score_and_minimize=True, n_workers=docking_n_workers)
    else:
        df = evaluate_molecules(mols_loaded, protein_path, docking_mode='vina_dock',
                                exhaustiveness=dock_exhaustiveness, reference_sdf=None,
                                run_posebusters=True, verbose=verbose,
                                skip_score_and_minimize=True)  # vina_score/vina_min unused by Track C's analysis
    # Join by the 'file' column evaluate_molecules itself reports, not
    # positional order -- load_sdf_mols silently drops any sdf that fails
    # to re-parse, which would otherwise silently misalign a positional
    # merge without raising any error.
    missing = set(df['file']) - set(filename_to_score)
    assert not missing, f'honest_eval reported files with no matching score: {missing}'
    df['predicted_affinity_score'] = df['file'].map(filename_to_score)
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df['seed'] = seed
    df.to_csv(os.path.join(result_dir, 'honest_eval.csv'), index=False)

    with open(done_marker, 'w') as f:
        f.write(f'n_attempted={pool_size} n_reconstructed_and_scored={n_written} n_docked={len(df)}\n')
    return 'completed', result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/task_f_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    parser.add_argument('--pool_size', type=int, default=300)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='./guidance/track_c_pools')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_amp', action='store_true',
                        help='bf16 autocast for the diffusion core forward pass; verify quality before trusting')
    parser.add_argument('--docking_n_workers', type=int, default=1,
                        help='parallelize Vina docking across this many worker processes (item 4 optimization)')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    diffusion_model, test_set = load_diffusion_model_and_dataset(device)
    ranker = load_affinity_ranker(device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')
    for pocket in pockets:
        for seed in args.seeds:
            try:
                status, result_dir = run_pool(
                    diffusion_model, test_set, ranker, pocket, seed, args.pool_size,
                    args.num_steps, args.batch_size, device, args.out_dir, args.dock_exhaustiveness,
                    verbose=args.verbose, use_amp=args.use_amp, docking_n_workers=args.docking_n_workers)
                entry = {'pocket': pocket['target'], 'seed': seed, 'status': status}
                print(f'{status.upper()}: pocket={pocket["target"]} seed={seed}')
            except Exception as e:
                import traceback
                entry = {'pocket': pocket['target'], 'seed': seed, 'status': 'failed',
                        'error': str(e), 'traceback': traceback.format_exc()}
                print(f'FAILED: pocket={pocket["target"]} seed={seed}: {e}')
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    print('Track C pool generation complete (or all remaining conditions processed).')


if __name__ == '__main__':
    main()
