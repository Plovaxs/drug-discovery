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

RQ2 positive control (variant 'vina_rerank', added 2026-10-12): see
guidance/RQ2_POSITIVE_CONTROL_PREREGISTRATION.md for the full design and
decision criteria, written BEFORE this ran. In one line: 'baseline' and
'affinity_only' test whether gradient guidance from our trained EGNN
surrogate helps; neither test whether ANY signal can -- so a null result
there cannot distinguish "our surrogate is uninformative" (M1) from "this
harness cannot register improvement no matter what". 'vina_rerank'
answers that by using real AutoDock Vina score (never trained on our pK
labels, not a gradient at all -- Vina is not differentiable) to select the
best-of-K*samples_per_pocket from an otherwise-unguided pool, then scores
the KEPT set with an INDEPENDENT Vina pass (full 'dock' mode search, not
the cheap 'score_only' used to select) so the comparison is not circular.
run_triple() above is untouched by this addition -- 'baseline' and
'affinity_only' results are not re-run and not at risk of regression.
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
from utils.evaluation.docking_vina import VinaDockingTask

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


def run_triple_vina_rerank(model, test_set, pocket, seed, samples_per_pocket, oversample_k,
                           num_steps, batch_size, device, out_dir, verbose=False):
    """The RQ2 positive control. See the module docstring and
    guidance/RQ2_POSITIVE_CONTROL_PREREGISTRATION.md for the design rationale --
    this function is the pre-registered plan's step-by-step implementation,
    nothing here should be read as a post-hoc choice.

    Sibling of run_triple(), not a modification of it: 'baseline' and
    'affinity_only' keep calling the original function unchanged.
    """
    variant = 'vina_rerank'
    result_dir = os.path.join(out_dir, variant, f'pocket{pocket["data_id"]}_seed{seed}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir

    pool_dir = os.path.join(result_dir, 'pool_sdf')
    sdf_dir = os.path.join(result_dir, 'sdf')
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)

    misc.seed_all(seed)
    data = test_set[pocket['data_id']]
    pool_size = oversample_k * samples_per_pocket

    # Step 1 (pre-reg Sec.3.1): an UNGUIDED pool, oversample_k times larger than the final set.
    # Exactly the baseline sampling path -- affinity_model=None, lambda_affinity=0.0 -- so any
    # difference from 'baseline' downstream comes from the selection step, not from a different
    # generator.
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, pool_size, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
    )

    pool_mols = []    # (pool_index, mol, sdf_path)
    for i, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception:
            continue
        sdf_path = os.path.join(pool_dir, f'{i:03d}.sdf')
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()
        pool_mols.append((i, mol, sdf_path))

    protein_path = os.path.join(PROTEIN_ROOT, pocket['expected_protein_pdb'])

    # Step 2 (pre-reg Sec.3.2): cheap selection pass. mode='score_only' evaluates the pose Vina is
    # handed with NO internal search (confirmed in utils/evaluation/docking_vina.py: it calls
    # v.score() directly, and exhaustiveness is accepted but unused for this mode) -- this is what
    # keeps oversampling affordable. A molecule whose Vina scoring fails (bad valence, degenerate
    # pose, etc.) is excluded from the ranking rather than crashing the triple; how many were
    # excluded is recorded so a high exclusion rate is visible rather than silently absorbed.
    scored = []
    n_select_failed = 0
    for i, mol, sdf_path in pool_mols:
        try:
            select_score = VinaDockingTask(protein_path, mol).run(mode='score_only')[0]['affinity']
        except Exception as e:
            n_select_failed += 1
            if verbose:
                print(f'[vina_rerank] selection scoring failed for pool member {i}: {e}')
            continue
        scored.append((select_score, i, mol, sdf_path))

    if not scored:
        # Every pool member failed selection scoring -- cannot proceed, but this must be visible
        # as a FAILURE for this triple, not silently produce an empty, misleadingly-"completed" result.
        raise RuntimeError(
            f'vina_rerank pocket={pocket["target"]} seed={seed}: all {len(pool_mols)} reconstructed '
            f'pool molecules failed Vina score_only; nothing to select from.')

    # Lower (more negative) Vina affinity = better predicted binding -- the same convention
    # eval/honest_eval.py's vina_dock column already uses, so sorting ascending keeps the "best"
    # molecules, consistent with how 'affinity_only'/'baseline' are already read and compared.
    scored.sort(key=lambda t: t[0])
    kept = scored[:samples_per_pocket]

    # Step 3: the kept set, written to the SAME sdf/ layout run_triple() uses, so
    # eval/honest_eval.load_sdf_mols and every downstream analysis script works unmodified.
    select_scores = {}
    for rank, (select_score, i, mol, src_path) in enumerate(kept):
        dst_path = os.path.join(sdf_dir, f'{rank:03d}.sdf')
        w = Chem.SDWriter(dst_path)
        w.write(mol)
        w.close()
        select_scores[f'{rank:03d}.sdf'] = select_score

    # Step 4 (pre-reg Sec.3.4): INDEPENDENT evaluation, same mode/exhaustiveness as baseline and
    # affinity_only ('vina_dock', exhaustiveness=16 -- a full pose search), so the three-way
    # comparison is apples-to-apples. This is deliberately a DIFFERENT Vina pass from the
    # score_only selection above -- that separation is what makes the comparison non-circular
    # (see the pre-registration's Sec.3 for why identical selection/evaluation scores would prove
    # nothing).
    from eval.honest_eval import load_sdf_mols
    mols = load_sdf_mols(sdf_dir)
    df = evaluate_molecules(mols, protein_path, docking_mode='vina_dock', exhaustiveness=16,
                            reference_sdf=None, run_posebusters=True, verbose=verbose)
    df['variant'] = variant
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df['seed'] = seed
    df['pool_size'] = pool_size
    df['n_select_failed'] = n_select_failed
    # file names in df come from load_sdf_mols reading sdf_dir -- same '{rank:03d}.sdf' keys used above.
    df['vina_select_score'] = df['file'].map(select_scores)
    df.to_csv(os.path.join(result_dir, 'honest_eval.csv'), index=False)

    with open(done_marker, 'w') as f:
        # n_attempted is REQUIRED, not optional decoration: guidance/analyze_task_f.py's
        # load_all_results parses exactly that key out of every variant's DONE file (it was written
        # against run_triple()'s marker format, baseline/affinity_only's n_attempted=samples_per_pocket,
        # and this file originally used a different, incompatible set of keys with no n_attempted at
        # all -- IndexError on 'n_attempted='.split()[1], caught only when analyze ran against the real
        # 60-triple result set, not against the earlier synthetic/dummy verifications). Semantically it
        # must equal samples_per_pocket here too, matching what the OTHER variants mean by it (molecules
        # the variant was asked to deliver, not an internal pool size) -- it is NOT n_pool, which is this
        # variant's own oversampling detail and would make validity/success RATES incomparable across
        # variants if used instead. The diagnostic fields are kept alongside it, not replaced.
        f.write(f'n_attempted={samples_per_pocket} n_pool={len(pool_mols)} '
               f'n_select_failed={n_select_failed} n_kept={len(kept)} n_scored={len(df)}\n')

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
    parser.add_argument('--oversample_k', type=int, default=2,
                        help="for variant 'vina_rerank' only: pool size = "
                             'oversample_k * samples_per_pocket (see '
                             'guidance/RQ2_POSITIVE_CONTROL_PREREGISTRATION.md Sec.6 for the cost '
                             'this trades off -- K=2 is the pre-registered default)')
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
            # 'vina_rerank' goes through its own sibling function (pre-registered, see the module
            # docstring); 'baseline' and 'affinity_only' keep calling the original run_triple()
            # completely unchanged, so this branch carries zero regression risk for either.
            if variant == 'vina_rerank':
                status, result_dir = run_triple_vina_rerank(
                    model, test_set, pocket, seed, args.samples_per_pocket, args.oversample_k,
                    args.num_steps, args.batch_size, device, args.out_dir, verbose=args.verbose)
            else:
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
