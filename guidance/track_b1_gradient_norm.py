"""Track B1 (redefined as gradient-norm normalization -- see
guidance/gradient_normalization.py's docstring for the full motivation and
the council-consultation addendum that superseded the original
reference-affinity-subtraction framing).

Tested on the Stage 0 EGNN model (guidance_models/affinity_egnn_lpsplit.pt),
the same base predictor Track B2/B3 use, per the addendum's original
framing of Track B as "three cheaper mechanism variants testable on the
EXISTING Stage 0 model."

Three-way design (mandatory per the addendum's Sec 1.6-1.7 -- a two-way
unguided-vs-normalized comparison would not actually test the hypothesis):
  (a) unguided (lambda=0)
  (b) raw-gradient guidance at Stage 0's own previously-established stable
      lambda (10.0, reused from guidance/LPSPLIT_LAMBDA_RESWEEP_FINDING.md's
      multi-pocket confirmation and guidance/track_b3_timestep_window.py's
      FIXED_LAMBDA)
  (c) normalized-gradient guidance at a freshly-derived stable lambda
      (found via --phase prelambda; raw per-atom gradient norm measured at
      ~0.0012 in a smoke test, so normalization to unit-per-atom before
      lambda is applied means lambda now has units of "Angstrom nudge per
      atom per step" -- an entirely different, much smaller safe range is
      expected, and must not be assumed to be anywhere near 10.0)

The decision criterion (addendum Sec 1.7) hinges on whether (c) differs
materially from (b) under the same statistical protocol, not on either
condition's raw comparison to (a) alone.
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
RAW_STABLE_LAMBDA = 10.0


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
    print(f'=== Pre-lambda check (normalized gradient) on pocket {pocket["target"]} ===')
    results = []
    for lam in grid:
        mols, n_frag1 = sample_and_reconstruct(
            model, test_set, guidance_model, lam, pocket, n_samples, num_steps, batch_size, device)
        rate = n_frag1 / n_samples
        print(f'  lambda={lam}: n_reconstructed={len(mols)}/{n_samples}, single_fragment_rate={rate:.2f}')
        results.append({'lambda': lam, 'n_reconstructed': len(mols),
                        'n_samples': n_samples, 'single_fragment_rate': rate})
    return results


def run_condition(model, test_set, guidance_model, label, lam, pocket, samples_per_pocket, num_steps,
                  batch_size, device, out_dir, dock_exhaustiveness, verbose=False):
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
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--prelambda_grid', type=float, nargs='+',
                        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument('--prelambda_n_samples', type=int, default=4)
    parser.add_argument('--normalized_lambda', type=float, default=None,
                        help='for phase=main: the stable normalized-gradient lambda chosen from the prelambda check')
    parser.add_argument('--samples_per_pocket', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='./guidance/track_b1_results')
    parser.add_argument('--prelambda_out', type=str, default='./guidance/track_b1_prelambda_check.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)

    if args.phase == 'prelambda':
        guidance_model = AffinityGuidance(AFFINITY_CHECKPOINT, device=device, normalize_gradient=True)
        results = run_prelambda_check(
            model, test_set, guidance_model, pockets[0], args.prelambda_grid,
            args.prelambda_n_samples, args.num_steps, args.batch_size, device)
        with open(args.prelambda_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved prelambda check to {args.prelambda_out}')
        return

    assert args.normalized_lambda is not None, '--phase main requires --normalized_lambda'
    guidance_raw = AffinityGuidance(AFFINITY_CHECKPOINT, device=device, normalize_gradient=False)
    guidance_norm = AffinityGuidance(AFFINITY_CHECKPOINT, device=device, normalize_gradient=True)

    conditions = [
        ('unguided', 0.0, None),
        ('raw', RAW_STABLE_LAMBDA, guidance_raw),
        ('normalized', args.normalized_lambda, guidance_norm),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')
    for pocket in pockets:
        for label, lam, gm in conditions:
            try:
                status, result_dir = run_condition(
                    model, test_set, gm, label, lam, pocket, args.samples_per_pocket,
                    args.num_steps, args.batch_size, device, args.out_dir, args.dock_exhaustiveness,
                    verbose=args.verbose)
                entry = {'condition': label, 'lambda': lam, 'pocket': pocket['target'], 'status': status}
                print(f'{status.upper()}: condition={label} lambda={lam} pocket={pocket["target"]}')
            except Exception as e:
                import traceback
                entry = {'condition': label, 'lambda': lam, 'pocket': pocket['target'], 'status': 'failed',
                        'error': str(e), 'traceback': traceback.format_exc()}
                print(f'FAILED: condition={label} lambda={lam} pocket={pocket["target"]}: {e}')
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    print('Track B1 checkpoint run complete (or all remaining conditions processed).')


if __name__ == '__main__':
    main()
