"""Checkpointed, resumable runner for Part B's rigorous Task F ablation.

Structure: an outer loop over (variant, pocket, seed) triples. Each triple
samples `--samples_per_pocket` molecules, evaluates them through
eval/honest_eval.py's real harness (incl. Vina docking + PoseBusters)
against that pocket's actual receptor PDB, and writes its result to disk
immediately -- nothing is held in memory for a single end-of-run write.

Resumability: before running a triple, checks for a DONE marker file under
its result directory and skips if present. A run interrupted at any point
resumes by re-invoking this same command -- already-completed triples are
skipped, not re-run or duplicated.

On-disk layout (all under --out_dir, default guidance/task_f_results/):
  {variant}/pocket{data_id}_seed{seed}/
    sdf/*.sdf            -- generated molecules for this triple
    honest_eval.csv       -- per-molecule metrics from eval.honest_eval
    DONE                  -- marker file, written only on success

Coarse-grained progress log: --out_dir/run_log.jsonl, one JSON line per
completed (or failed) triple, so overall progress and any per-pocket
failures are visible without reading every individual result directory.
A failed triple is logged with its error and the run continues to the next
triple -- it is never silently dropped from the final count.
"""
import argparse
import json
import os
import time
import traceback

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
from eval.honest_eval import evaluate_molecules

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
AFFINITY_CHECKPOINT = './guidance_models/affinity_egnn.pt'
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


def run_triple(model, test_set, affinity_model, variant, pocket, seed, samples_per_pocket,
               num_steps, batch_size, device, out_dir, lambda_affinity, verbose=False):
    result_dir = os.path.join(out_dir, variant, f'pocket{pocket["data_id"]}_seed{seed}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir

    sdf_dir = os.path.join(result_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    misc.seed_all(seed)
    data = test_set[pocket['data_id']]

    la = lambda_affinity if variant == 'affinity_only' else 0.0
    am = affinity_model if variant == 'affinity_only' else None

    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, samples_per_pocket, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=am, synth_model=None, lambda_affinity=la, lambda_synth=0.0,
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
    from eval.honest_eval import load_sdf_mols
    mols = load_sdf_mols(sdf_dir)
    df = evaluate_molecules(mols, protein_path, docking_mode='vina_dock', exhaustiveness=16,
                            reference_sdf=None, run_posebusters=True, verbose=verbose)
    df['variant'] = variant
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df['seed'] = seed
    df.to_csv(os.path.join(result_dir, 'honest_eval.csv'), index=False)

    with open(done_marker, 'w') as f:
        f.write(f'n_attempted={samples_per_pocket} n_reconstructed={n_recon} n_scored={len(df)}\n')

    return 'completed', result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=['baseline', 'affinity_only'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--samples_per_pocket', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lambda_affinity', type=float, default=1.0)
    parser.add_argument('--pockets_file', type=str, default='./guidance/task_f_pockets.json')
    parser.add_argument('--out_dir', type=str, default='./guidance/task_f_results')
    parser.add_argument('--limit_pockets', type=int, default=None,
                        help='for dry runs: only use the first N pockets')
    parser.add_argument('--affinity_ckpt', type=str, default=AFFINITY_CHECKPOINT,
                        help='affinity guidance checkpoint -- pass '
                             './guidance_models/affinity_egnn_lpsplit.pt to use the Stage 0 '
                             'leakage-safe-split-retrained EGNN instead of the original deployed one')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)
    if args.limit_pockets:
        pockets = pockets[:args.limit_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    affinity_model = None
    if 'affinity_only' in args.variants:
        affinity_model = AffinityGuidance(args.affinity_ckpt, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')

    triples = [(variant, pocket, seed)
              for variant in args.variants for pocket in pockets for seed in args.seeds]
    print(f'{len(triples)} total (variant, pocket, seed) triples to process '
         f'({len(args.variants)} variants x {len(pockets)} pockets x {len(args.seeds)} seeds)')

    for variant, pocket, seed in triples:
        t0 = time.time()
        try:
            status, result_dir = run_triple(
                model, test_set, affinity_model, variant, pocket, seed, args.samples_per_pocket,
                args.num_steps, args.batch_size, device, args.out_dir, args.lambda_affinity,
                verbose=args.verbose)
            elapsed = time.time() - t0
            entry = {'variant': variant, 'pocket_data_id': pocket['data_id'], 'pocket_target': pocket['target'],
                     'seed': seed, 'status': status, 'elapsed_sec': round(elapsed, 1),
                     'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            print(f'[{entry["timestamp"]}] {status.upper()}: variant={variant} '
                 f'pocket={pocket["target"]}(id={pocket["data_id"]}) seed={seed} ({elapsed:.0f}s)')
        except Exception as e:
            elapsed = time.time() - t0
            entry = {'variant': variant, 'pocket_data_id': pocket['data_id'], 'pocket_target': pocket['target'],
                     'seed': seed, 'status': 'failed', 'error': str(e), 'traceback': traceback.format_exc(),
                     'elapsed_sec': round(elapsed, 1), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
            print(f'[{entry["timestamp"]}] FAILED: variant={variant} '
                 f'pocket={pocket["target"]}(id={pocket["data_id"]}) seed={seed}: {e}')

        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    print('Run complete (or all remaining triples processed for this invocation).')


if __name__ == '__main__':
    main()
