"""Track D, Step D.3: full lambda re-sweep of the bond-aware-feature
synth-guidance mechanism against the NEW leakage-safe checkpoint
(guidance/TRACK_D_LEAKAGE_SAFE_RETRAIN_FINDING.md), at the same sample
size/statistical-rigor standard as the affinity investigation's full-tier
checkpoints -- not the original SYNTH_GUIDANCE_FINDING.md's n=8,
single-pocket, no-statistical-test scale.

Single pocket first (--n_pockets 1, the default), matching D.3 point 1;
multi-pocket confirmation (--n_pockets 5-10) only if the single-pocket
result shows the real-RA-score/own-score divergence resolved or
substantially improved (per D.4's explicit gating rule -- do not spend
multi-pocket effort on a single-pocket result that hasn't resolved the
core problem).

At every lambda point, reports (matching the original finding table's
columns, at proper N): fragmentation rate, reconstructable %, guidance's
own predicted score, AND the real, independently-computed RA-score
(eval/honest_eval.py's RAScorer) -- this is the exact comparison that
revealed the original divergence and must be re-run at full power, not
assumed fixed.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.transforms import Compose
from rdkit import Chem, RDLogger

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
from guidance.synth_guidance import SynthGuidance
from guidance.atom_features import v0_to_ligand_feature
from utils import reconstruct
from eval.honest_eval import RAScorer

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
SYNTH_CHECKPOINT = './logs_synth_lp/synth_ra_egnn_lp_2026_09_12__15_38_21_full/checkpoints/best.pt'
DEFAULT_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


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


def synth_own_score(synth_model, pos, v_discrete, device):
    """The guidance model's own predicted score (sigmoid probability) on
    one FINAL generated molecule -- not a gradient. v_discrete: the
    finalized (argmax) per-atom type indices. Mirrors
    guidance/lambda_sweep.py's synth_own_score helper."""
    n = pos.size(0)
    batch_ligand = torch.zeros(n, dtype=torch.long, device=device)
    fake_logits = F.one_hot(v_discrete, 13).float()
    with torch.no_grad():
        ligand_feat = v0_to_ligand_feature(
            fake_logits, synth_model._ligand_atom_feature_dim, pos=pos,
            use_bond_aware=synth_model.use_bond_aware)
        logit = synth_model.model(ligand_pos=pos, ligand_atom_feature=ligand_feat, batch_ligand=batch_ligand)
    return torch.sigmoid(logit).view(-1).item()


def run_condition(model, test_set, synth_model, lam, pocket, n_samples, num_steps,
                  batch_size, device, out_dir, seed=2021, verbose=False):
    label = f'lambda_{lam}'
    result_dir = os.path.join(out_dir, label, f'pocket{pocket["data_id"]}')
    done_marker = os.path.join(result_dir, 'DONE')
    if os.path.exists(done_marker):
        return 'skipped', result_dir
    os.makedirs(result_dir, exist_ok=True)

    misc.seed_all(seed)
    data = test_set[pocket['data_id']]
    sm = None if lam == 0.0 else synth_model
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=batch_size, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=sm, lambda_affinity=0.0, lambda_synth=lam,
    )

    ra_scorer = RAScorer()
    rows = []
    n_single_frag, n_reconstructed = 0, 0
    for pos, v in zip(pred_pos, pred_v):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        row = {}
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception:
            row['reconstructed'] = False
            rows.append(row)
            continue
        row['reconstructed'] = True
        n_reconstructed += 1
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        row['n_fragments'] = len(frags)
        if len(frags) == 1:
            n_single_frag += 1

        smiles = Chem.MolToSmiles(mol)
        row['smiles'] = smiles
        if '.' not in smiles:
            pos_t = torch.from_numpy(pos).float().to(device)
            v_t = torch.from_numpy(v).long().to(device)
            try:
                row['synth_own_score'] = synth_own_score(synth_model, pos_t, v_t, device)
            except Exception as e:
                if verbose:
                    print(f'  own-score forward pass failed: {e}')
            try:
                row['real_ra_score'] = ra_scorer.predict(mol)
            except Exception as e:
                if verbose:
                    print(f'  RA-score failed: {e}')
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    df['lambda'] = lam
    df['pocket_data_id'] = pocket['data_id']
    df['pocket_target'] = pocket['target']
    df['n_attempted'] = n_samples
    df['n_reconstructed'] = n_reconstructed
    df['n_single_fragment'] = n_single_frag
    df.to_csv(os.path.join(result_dir, 'results.csv'), index=False)

    with open(done_marker, 'w') as f:
        f.write(f'n_attempted={n_samples} n_reconstructed={n_reconstructed} n_single_fragment={n_single_frag}\n')
    return 'completed', result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/task_f_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=1)
    parser.add_argument('--grid', type=float, nargs='+', default=DEFAULT_GRID)
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_bond_aware', action='store_true', default=True)
    parser.add_argument('--out_dir', type=str, default='./guidance/track_d_resweep_results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    synth_model = SynthGuidance(SYNTH_CHECKPOINT, device=device, use_bond_aware=args.use_bond_aware)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'run_log.jsonl')
    for pocket in pockets:
        for lam in args.grid:
            try:
                status, result_dir = run_condition(
                    model, test_set, synth_model, lam, pocket, args.n_samples,
                    args.num_steps, args.batch_size, device, args.out_dir, verbose=args.verbose)
                entry = {'lambda': lam, 'pocket': pocket['target'], 'status': status}
                print(f'{status.upper()}: lambda={lam} pocket={pocket["target"]}')
            except Exception as e:
                import traceback
                entry = {'lambda': lam, 'pocket': pocket['target'], 'status': 'failed',
                        'error': str(e), 'traceback': traceback.format_exc()}
                print(f'FAILED: lambda={lam} pocket={pocket["target"]}: {e}')
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    print('Track D lambda re-sweep complete (or all remaining conditions processed).')


if __name__ == '__main__':
    main()
