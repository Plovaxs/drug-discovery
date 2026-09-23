"""Part 2: lambda sweep for guidance/guided_sampling.py.

Grid choice, justified by Part 1 (guidance/diagnose_lambda.py):
the addendum's working hypothesis was "guidance gradients are orders of
magnitude larger than the base score, so try much smaller lambdas
(0.001-0.1)." Measured evidence contradicts this: at lambda=1.0, step 500
of 1000, affinity grad_pos.norm() / base pos0.norm() = 0.0002 and
synth's = 0.0046 -- guidance is two to four orders of magnitude SMALLER
than the base signal, not larger. Yet a same-seed/same-init control
(unguided vs. lambda_affinity=1.0, lambda_synth=1.0) showed unguided
produces one connected 22-atom molecule while guided produces 2 fragments
(a 3-atom piece split off a 19-atom body) -- so guidance does cause this,
just not through gradient magnitude dominating the base score. Diffusion
sampling can be sensitive to small, consistent perturbations compounded
over 1000 steps, especially ones applied throughout the whole trajectory
including near the end when connectivity is being finalized.

Given that, this sweep centers its grid ON and AROUND 1.0 (not near 0),
to find where the fragmentation rate actually starts increasing as a
function of scale -- using n=8 samples per point (a real rate, not a
single anecdote), at the model's full 1000 steps, on the bundled example
pocket.

Scope: QED/SA/RA-score are computed per point (cheap, no GPU/no docking).
Vina Dock is NOT run in the sweep itself -- exhaustive Vina docking at
every (lambda, sample) point across ~15-20 sweep points x 8 samples would
be by far the most expensive part of this sweep for information that Part
3's actual ablation already needs to compute per-pocket anyway; the sweep
uses "does the guidance model's own predicted score on the final molecule
move the way we intend, without breaking validity" as its selection
signal, and Part 3 confirms with the full honest-eval harness (incl. Vina)
on the chosen operating point.
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_for_pocket import pdb_to_pocket_data
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
from guidance.affinity_guidance import AffinityGuidance
from guidance.synth_guidance import SynthGuidance
from guidance.atom_features import v0_to_ligand_feature
from utils import reconstruct
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask
from eval.honest_eval import RAScorer
from rdkit import Chem, RDLogger

EXAMPLE_PDB = 'examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb'
CHECKPOINT = 'pretrained_models/pretrained_diffusion.pt'


def load_everything(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer])
    data = pdb_to_pocket_data(EXAMPLE_PDB)
    data = transform(data).to(device)
    model = ScorePosNet3D(
        ckpt['config'].model, protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return data, model


def affinity_own_score(affinity_model, pos, v_discrete, protein_pos, protein_v, device):
    """affinity_model's own forward prediction (pKd/pKi/pIC50) on one
    FINAL generated molecule -- not a gradient. Confirms guidance is
    moving the actual outcome, not just avoiding breakage.
    """
    n = pos.size(0)
    batch_ligand = torch.zeros(n, dtype=torch.long, device=device)
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long, device=device)
    fake_logits = F.one_hot(v_discrete, 13).float()
    # Use whichever feature mode the guidance model itself is configured
    # with, so this "own score" check reflects what guidance actually saw.
    with torch.no_grad():
        ligand_feat = v0_to_ligand_feature(
            fake_logits, affinity_model._ligand_atom_feature_dim, pos=pos,
            use_bond_aware=getattr(affinity_model, 'use_bond_aware', False))
        pred = affinity_model.model(
            protein_pos=protein_pos, protein_atom_feature=protein_v,
            ligand_pos=pos, ligand_atom_feature=ligand_feat,
            batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
    return pred.view(-1).item()


def synth_own_score(synth_model, pos, v_discrete, device):
    """synth_model's own forward prediction (logit; higher = more
    synthesizable) on one FINAL generated molecule -- not a gradient.
    """
    n = pos.size(0)
    batch_ligand = torch.zeros(n, dtype=torch.long, device=device)
    fake_logits = F.one_hot(v_discrete, 13).float()
    with torch.no_grad():
        ligand_feat = v0_to_ligand_feature(
            fake_logits, synth_model._ligand_atom_feature_dim, pos=pos,
            use_bond_aware=getattr(synth_model, 'use_bond_aware', False))
        logit = synth_model.model(ligand_pos=pos, ligand_atom_feature=ligand_feat, batch_ligand=batch_ligand)
    return torch.sigmoid(logit).view(-1).item()


def run_one_point(data, model, affinity_model, synth_model, lambda_affinity, lambda_synth,
                  n_samples, num_steps, seed, device, batch_size=None, verbose=False,
                  dock=False, dock_exhaustiveness=8):
    misc.seed_all(seed)
    # Guidance holds extra memory (the guidance model's own forward+backward
    # graph, refreshed every step) on top of the diffusion core's inference
    # pass -- batch_size=8 OOM'd on the RTX 3050's 4GB the moment guidance
    # was active (lambda_affinity=0.1), even though the same batch size ran
    # fine unguided (lambda=0). Chunking via batch_size keeps peak memory
    # bounded regardless of lambda.
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=batch_size or n_samples, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=affinity_model, synth_model=synth_model,
        lambda_affinity=lambda_affinity, lambda_synth=lambda_synth,
    )

    n_single_frag, n_valid = 0, 0
    qed_list, sa_list, ra_list, vina_dock_list = [], [], [], []
    affinity_own_list, synth_own_list = [], []
    per_molecule = []  # one dict per sample, for fine-grained (non-averaged) inspection
    ra_scorer = RAScorer()
    mols = []
    for sample_idx, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        row = {'sample_idx': sample_idx}
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
            smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            if verbose:
                print(f'  reconstruction failed: {e}')
            row['reconstructed'] = False
            per_molecule.append(row)
            continue
        row['reconstructed'] = True
        n_valid += 1
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        row['n_fragments'] = len(frags)
        if len(frags) == 1:
            n_single_frag += 1

        pos_t = torch.from_numpy(pos).float().to(device)
        v_t = torch.from_numpy(v).long().to(device)
        try:
            aff_own = affinity_own_score(affinity_model, pos_t, v_t, data.protein_pos,
                                         data.protein_atom_feature.float(), device)
            syn_own = synth_own_score(synth_model, pos_t, v_t, device)
            affinity_own_list.append(aff_own)
            synth_own_list.append(syn_own)
            row['affinity_own_score'] = aff_own
            row['synth_own_score'] = syn_own
        except Exception as e:
            if verbose:
                print(f'  own-score forward pass failed: {e}')

        if '.' not in smiles:
            try:
                chem = scoring_func.get_chem(mol)
                ra = ra_scorer.predict(mol)
                qed_list.append(chem['qed'])
                sa_list.append(chem['sa'])
                ra_list.append(ra)
                mols.append(mol)
                row['qed'] = chem['qed']
                row['sa'] = chem['sa']
                row['real_ra_score'] = ra
            except Exception:
                pass

            if dock:
                try:
                    task = VinaDockingTask(EXAMPLE_PDB, mol)
                    dock_result = task.run(mode='dock', exhaustiveness=dock_exhaustiveness)
                    vina_dock = dock_result[0]['affinity']
                    vina_dock_list.append(vina_dock)
                    row['vina_dock'] = vina_dock
                except Exception as e:
                    if verbose:
                        print(f'  docking failed: {e}')
        per_molecule.append(row)

    return {
        'lambda_affinity': lambda_affinity, 'lambda_synth': lambda_synth,
        'n_samples': n_samples, 'n_valid': n_valid, 'n_single_fragment': n_single_frag,
        'single_fragment_rate': n_single_frag / n_samples,
        'validity_rate': n_valid / n_samples,
        'per_molecule': per_molecule,
        'mean_affinity_model_own_score': float(sum(affinity_own_list) / len(affinity_own_list)) if affinity_own_list else None,
        'mean_synth_model_own_score': float(sum(synth_own_list) / len(synth_own_list)) if synth_own_list else None,
        'mean_qed': float(sum(qed_list) / len(qed_list)) if qed_list else None,
        'mean_sa': float(sum(sa_list) / len(sa_list)) if sa_list else None,
        'mean_ra_score': float(sum(ra_list) / len(ra_list)) if ra_list else None,
        'n_complete_for_chem': len(qed_list),
        'mean_vina_dock': float(sum(vina_dock_list) / len(vina_dock_list)) if vina_dock_list else None,
        'std_vina_dock': float(np.std(vina_dock_list)) if len(vina_dock_list) > 1 else None,
        'n_docked': len(vina_dock_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='chunk size for sampling -- kept smaller than n_samples to avoid '
                             'OOM once guidance is active (RTX 3050, 4GB)')
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='./guidance/lambda_sweep_results.json')
    parser.add_argument('--stage', type=str, choices=['affinity', 'synth', 'dual'], required=True)
    parser.add_argument('--grid', type=float, nargs='+', default=None,
                        help='for stage=affinity/synth: values for the swept lambda')
    parser.add_argument('--pairs', type=str, nargs='+', default=None,
                        help='for stage=dual: "la:ls" pairs, e.g. --pairs 0.3:0.1 1.0:0.3 0.3:0.3')
    parser.add_argument('--fixed_other', type=float, default=0.0,
                        help='value for the other lambda when sweeping one signal independently')
    parser.add_argument('--use_bond_aware', action='store_true',
                        help='Phase 1 of the architectural-upgrade addendum: use geometry-derived '
                             'Degree/NumHs/Hybridization (guidance/bond_estimator.py) instead of '
                             'the fixed placeholder features')
    parser.add_argument('--dock', action='store_true',
                        help='also compute a real Vina Dock score per valid generated molecule '
                             '(against the bundled example pocket) -- slower, but Task F\'s null '
                             'result was about real downstream metrics, not the guidance model\'s '
                             'own self-assessment, so this closes that gap for the lambda sweep too')
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--affinity_ckpt', type=str, default='./guidance_models/affinity_egnn.pt',
                        help='affinity guidance checkpoint -- pass '
                             './guidance_models/affinity_egnn_lpsplit.pt to use the Stage 0 '
                             'leakage-safe-split-retrained EGNN instead of the original deployed one')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data, model = load_everything(device)
    affinity_model = AffinityGuidance(args.affinity_ckpt, device=device,
                                      use_bond_aware=args.use_bond_aware)
    synth_model = SynthGuidance('./guidance_models/synth_ra_score.pt', device=device,
                                use_bond_aware=args.use_bond_aware)

    if args.stage == 'dual':
        if not args.pairs:
            raise ValueError('--stage dual requires --pairs "la:ls" ...')
        points = [tuple(float(x) for x in p.split(':')) for p in args.pairs]
    else:
        default_grid = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        grid = args.grid or default_grid
        if args.stage == 'affinity':
            points = [(lam, args.fixed_other) for lam in grid]
        else:
            points = [(args.fixed_other, lam) for lam in grid]

    results = []
    for la, ls in points:
        print(f'=== stage={args.stage} lambda_affinity={la} lambda_synth={ls} '
             f'(n={args.n_samples}, steps={args.num_steps}) ===')
        res = run_one_point(data, model, affinity_model, synth_model, la, ls,
                            args.n_samples, args.num_steps, args.seed, device,
                            batch_size=args.batch_size, verbose=args.verbose,
                            dock=args.dock, dock_exhaustiveness=args.dock_exhaustiveness)
        print(f'  {res}')
        results.append(res)

        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nSaved sweep results to {args.out}')


if __name__ == '__main__':
    main()
