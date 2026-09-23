"""Interactive-style molecule generation: pick a target protein pocket
(from this project's existing 100-pocket CrossDocked2020 test set), the
model generates N candidate molecules for it, and this script produces a
plain reference report (SMILES + computed properties) a researcher can
use as a starting point for further study/synthesis.

Deliberately reuses PLAIN UNGUIDED generation (lambda_affinity=0,
lambda_synth=0) -- Tracks A, B, C, and D all independently found that
this project's affinity/synthesizability guidance mechanisms do not
produce a genuine improvement (see guidance/DUAL_FALSIFICATION_CONCLUSION.md
and guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md). Reporting a raw
predicted_affinity_score or vina_dock as if it reliably ranks "better
binders" would misrepresent that finding, so this tool reports those
numbers as rough, unreliable priors and leads with structural validity
and basic medicinal-chemistry properties (QED/SA/RA-score) instead --
see the caveat printed in the report itself.

Usage:
    python guidance/generate_molecules_for_target.py --list
    python guidance/generate_molecules_for_target.py --target CD38_HUMAN_44_300_0 --n_molecules 10
    python guidance/generate_molecules_for_target.py --data_id 10 --n_molecules 10
"""
import argparse
import os

import pandas as pd
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

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
AFFINITY_CHECKPOINT = './guidance_models/affinity_egnn_lpsplit.pt'
PROTEIN_ROOT = './data/test_set'

CAVEAT = """
============================================================================
IMPORTANT -- how to read this report
============================================================================
This project's own rigorous testing (Tracks A-D, see
guidance/DUAL_FALSIFICATION_CONCLUSION.md and
guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md) found that this affinity
model's score -- and even the raw Vina Dock score -- is substantially
confounded by molecule size and does NOT reliably distinguish genuinely
better binders. Treat `predicted_affinity_score` and `vina_dock` below as
weak, unreliable priors, NOT a validated ranking of binding quality.

The more trustworthy signals here are:
  - `pb_valid`: did the pose pass basic structural sanity checks
    (bond lengths/angles, no steric clashes, connected atoms)? Molecules
    with pb_valid=False may be geometrically implausible.
  - `qed` / `sa`: standard drug-likeness and synthetic-accessibility
    heuristics (0-1, higher = more drug-like / easier to synthesize).
  - `ra_score`: a learned synthesizability estimate (higher = more likely
    synthesizable by a real retrosynthesis-aware model).

Use this list as a rough SET OF STARTING IDEAS to inspect manually
(structure, functional groups, drug-likeness) -- not as a pre-ranked
"top candidate" list. Real triage should involve visual inspection,
expert chemistry judgment, and independent re-docking/re-scoring before
any synthesis decision.
============================================================================
"""


def list_targets():
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt['config'].data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    dataset, subsets = get_dataset(config=ckpt['config'].data, transform=transform)
    test_set = subsets['test']
    print(f'{len(test_set)} targets available in the test set:\n')
    for i in range(len(test_set)):
        name = test_set[i].ligand_filename.split('/')[0]
        print(f'  data_id={i:3d}  {name}')


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


def find_data_id(test_set, target_name):
    for i in range(len(test_set)):
        if test_set[i].ligand_filename.split('/')[0] == target_name:
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='list all available targets and exit')
    parser.add_argument('--target', type=str, default=None, help='target name, e.g. CD38_HUMAN_44_300_0')
    parser.add_argument('--data_id', type=int, default=None, help='alternative to --target: direct test-set index')
    parser.add_argument('--n_molecules', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dock_exhaustiveness', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--out_dir', type=str, default='./guidance/generated_candidates')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--skip_docking', action='store_true',
                         help='faster -- skip real Vina docking, report only QED/SA/RA-score/predicted_affinity')
    args = parser.parse_args()

    RDLogger.DisableLog('rdApp.*')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.list:
        list_targets()
        return

    diffusion_model, test_set = load_diffusion_model_and_dataset(device)

    if args.data_id is not None:
        data_id = args.data_id
    elif args.target is not None:
        data_id = find_data_id(test_set, args.target)
        if data_id is None:
            print(f"Target '{args.target}' not found. Run with --list to see available targets.")
            return
    else:
        print('Specify --target NAME or --data_id N (or use --list to see options).')
        return

    data = test_set[data_id]
    target_name = data.ligand_filename.split('/')[0]
    print(f'Generating {args.n_molecules} candidate molecules for {target_name} (data_id={data_id}) ...')

    ranker = load_affinity_ranker(device)

    misc.seed_all(args.seed)
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        diffusion_model, data, args.n_molecules, batch_size=args.batch_size, device=device,
        num_steps=args.num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        use_amp=args.use_amp,
    )

    protein_pos = data.protein_pos.to(device)
    protein_atom_feature = data.protein_atom_feature.float().to(device)

    out_dir = os.path.join(args.out_dir, target_name)
    sdf_dir = os.path.join(out_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

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
        except Exception:
            continue
        fname = f'{n_written:03d}.sdf'
        w = Chem.SDWriter(os.path.join(sdf_dir, fname))
        w.write(mol)
        w.close()
        filename_to_score[fname] = score
        n_written += 1

    print(f'Reconstructed and scored {n_written}/{args.n_molecules} molecules.')

    protein_path = os.path.join(PROTEIN_ROOT, f'{target_name}/{sorted(os.listdir(os.path.join(PROTEIN_ROOT, target_name)))[0]}')
    for fn in os.listdir(os.path.join(PROTEIN_ROOT, target_name)):
        if fn.endswith('_rec.pdb'):
            protein_path = os.path.join(PROTEIN_ROOT, target_name, fn)
            break

    mols_loaded = load_sdf_mols(sdf_dir)
    df = evaluate_molecules(mols_loaded, protein_path,
                             docking_mode='vina_dock' if not args.skip_docking else 'vina_score',
                             exhaustiveness=args.dock_exhaustiveness, reference_sdf=None,
                             run_posebusters=True, verbose=False,
                             skip_score_and_minimize=True)
    if args.skip_docking:
        df['vina_dock'] = None
    missing = set(df['file']) - set(filename_to_score)
    assert not missing, f'files with no matching score: {missing}'
    df['predicted_affinity_score'] = df['file'].map(filename_to_score)
    df['target'] = target_name

    cols = ['file', 'smiles', 'heavy_atoms', 'qed', 'sa', 'ra_score', 'pb_valid',
            'predicted_affinity_score', 'vina_dock', 'ligand_efficiency']
    df_out = df[[c for c in cols if c in df.columns]].sort_values('qed', ascending=False)
    csv_path = os.path.join(out_dir, 'candidates.csv')
    df_out.to_csv(csv_path, index=False)

    report_path = os.path.join(out_dir, 'REPORT.txt')
    with open(report_path, 'w') as f:
        f.write(f'Candidate molecules generated for target: {target_name}\n')
        f.write(f'(test-set data_id={data_id}, {n_written} valid molecules out of {args.n_molecules} attempted)\n')
        f.write(CAVEAT)
        f.write('\n' + df_out.to_string(index=False) + '\n')

    print(CAVEAT)
    print(df_out.to_string(index=False))
    print(f'\nWrote {csv_path}')
    print(f'Wrote {report_path}')
    print(f'SDF files (3D structures) in {sdf_dir}/')


if __name__ == '__main__':
    main()
