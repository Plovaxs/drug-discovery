"""Track A exploratory-tier dual-criterion checkpoint (Stage 2+ addendum
Part 3): the partially-trained GIGN+PIGNet2 model
(guidance/track_a/train_gign_pignet_stage2.py's checkpoint) is wired into
the guided sampling loop via guidance/track_a/gign_pignet_guidance.py, then
tested for gradient-informativeness (KS test on real Vina Dock, guided vs.
unguided, BH-corrected) on 2-3 pockets -- mirroring
guidance/track_b3_timestep_window.py's checkpointed per-(pocket,condition)
pattern, with "condition" here meaning a lambda value rather than a
timestep window.

Two phases, run via --phase:
  prelambda -- quick single-pocket fragmentation/stability check (n=8,
    cheap, no docking) to find THIS model's own stable lambda range,
    per the addendum's explicit instruction not to assume Stage 0's
    lambda range transfers unchanged to a different guidance model.
  main -- the actual checkpoint: 2-3 pockets x (unguided + N stable
    lambdas), N=20-30 molecules/condition, real Vina Dock + PoseBusters.
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
from guidance.track_a.gign_pignet_guidance import GIGNPignetGuidance
from utils import reconstruct
from eval.honest_eval import evaluate_molecules, load_sdf_mols

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
PROTEIN_ROOT = './data/test_set'


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


def sample_and_reconstruct(model, test_set, guidance_model, lam, pocket, n_samples, num_steps,
                           batch_size, device, seed=2021):
    misc.seed_all(seed)
    data = test_set[pocket['data_id']]
    am = None if lam == 0.0 else guidance_model
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=am, synth_model=None, lambda_affinity=lam, lambda_synth=0.0,
    )
    mols = []
    n_frag1 = 0
    for pos, v in zip(pred_pos, pred_v):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception:
            continue
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) == 1:
            n_frag1 += 1
        mols.append(mol)
    return mols, n_frag1


def run_prelambda_check(model, test_set, guidance_model, pocket, grid, n_samples, num_steps,
                        batch_size, device):
    print(f'=== Pre-lambda fragmentation check on pocket {pocket["target"]} ===')
    results = []
    for lam in grid:
        mols, n_frag1 = sample_and_reconstruct(
            model, test_set, guidance_model, lam, pocket, n_samples, num_steps, batch_size, device)
        rate = n_frag1 / n_samples
        print(f'  lambda={lam}: n_reconstructed={len(mols)}/{n_samples}, single_fragment_rate={rate:.2f}')
        results.append({'lambda': lam, 'n_reconstructed': len(mols),
                        'n_samples': n_samples, 'single_fragment_rate': rate})
    return results


def run_condition(model, test_set, guidance_model, lam, pocket, samples_per_pocket, num_steps,
                  batch_size, device, out_dir, dock_exhaustiveness, verbose=False):
    label = 'unguided' if lam == 0.0 else f'lambda_{lam}'
    result_dir = os.path.join(out_dir, label, f'pocket{pocket["data_id"]}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir

    sdf_dir = os.path.join(result_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    mols, _ = sample_and_reconstruct(
        model, test_set, guidance_model, lam, pocket, samples_per_pocket, num_steps, batch_size, device)
    for i, mol in enumerate(mols):
        w = Chem.SDWriter(os.path.join(sdf_dir, f'{i:03d}.sdf'))
        w.write(mol)
        w.close()

    protein_path = os.path.join(PROTEIN_ROOT, pocket['expected_protein_pdb'])
    mols_loaded = load_sdf_mols(sdf_dir)
    df = evaluate_molecules(mols_loaded, protein_path, docking_mode='vina_dock',
                            exhaustiveness=dock_exhaustiveness, reference_sdf=None,
                            run_posebusters=True, verbose=verbose)
    df['condition'] = label
    df['lambda'] = lam
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df.to_csv(os.path.join(result_dir, 'honest_eval.csv'), index=False)

    with open(done_marker, 'w') as f:
        f.write(f'n_attempted={samples_per_pocket} n_reconstructed={len(mols)} n_scored={len(df)}\n')
    return 'completed', result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prelambda', 'main'], required=True)
    parser.add_argument('--gign_ckpt', type=str, required=True)
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--prelambda_grid', type=float, nargs='+', default=[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    parser.add_argument('--prelambda_n_samples', type=int, default=8)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.3, 1.0, 3.0],
                        help='for phase=main: stable lambda values chosen from the prelambda check')
    parser.add_argument('--samples_per_pocket', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='./guidance/track_a_exploratory_results')
    parser.add_argument('--prelambda_out', type=str, default='./guidance/track_a_prelambda_check.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    guidance_model = GIGNPignetGuidance(args.gign_ckpt, device=device)

    if args.phase == 'prelambda':
        results = run_prelambda_check(
            model, test_set, guidance_model, pockets[0], args.prelambda_grid,
            args.prelambda_n_samples, args.num_steps, args.batch_size, device)
        with open(args.prelambda_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved prelambda check to {args.prelambda_out}')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')
    lambdas_to_run = [0.0] + list(args.lambdas)
    for pocket in pockets:
        for lam in lambdas_to_run:
            try:
                status, result_dir = run_condition(
                    model, test_set, guidance_model, lam, pocket, args.samples_per_pocket,
                    args.num_steps, args.batch_size, device, args.out_dir, args.dock_exhaustiveness,
                    verbose=args.verbose)
                entry = {'lambda': lam, 'pocket': pocket['target'], 'status': status}
                print(f'{status.upper()}: lambda={lam} pocket={pocket["target"]}')
            except Exception as e:
                import traceback
                entry = {'lambda': lam, 'pocket': pocket['target'], 'status': 'failed',
                        'error': str(e), 'traceback': traceback.format_exc()}
                print(f'FAILED: lambda={lam} pocket={pocket["target"]}: {e}')
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    print('Track A exploratory checkpoint run complete (or all remaining conditions processed).')


if __name__ == '__main__':
    main()
