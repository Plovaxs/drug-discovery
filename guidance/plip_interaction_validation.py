"""New follow-up branch, Phase 2 ("1-2 closely-connected new branches"):
every real-docking test run in this investigation so far asks only "did
Vina Dock improve" -- a single scalar that rewards ANY tighter-fitting
pose, including ones that improve by brute surface contact rather than by
forming the kind of specific, named interaction (hydrogen bond,
hydrophobic contact, pi-stacking, salt bridge) medicinal chemists actually
care about. This asks a sharper, complementary question: when guidance
changes the Vina score, does it change the NUMBER OF REAL INTERACTIONS
the docked pose forms, per PLIP (Protein-Ligand Interaction Profiler)?

This is NOT Track E (`guidance/TRACK_E_DESIGN.md`), which is a gated,
~38 GPU-hour plan to TRAIN a new delta-learning + PLIP-multi-task-
supervised model -- that gate (no Track E training without a fresh,
explicit go-ahead on its Sec. 1 decisions D1-D6) is untouched by this
script. This is a pure ANALYSIS of molecules sampled from checkpoints
ALREADY trained and already real-docking-tested in this investigation,
reusing Track E's planning-phase PLIP utility
(guidance/track_e/plip_label_extraction.py) for the label vocabulary
(hbond, hydrophobic, pi_stack, salt_bridge) but at complex-level interaction
COUNTS, not Track E's per-ligand-atom label matrix (which was built for
the LMDB's synthetic-reconstructed receptor; this script analyzes the
real Vina-DOCKED pose against the real receptor PDB file lambda_sweep.py
already uses, via PLIP's own PDB-complex auto-detection instead of
explicit atom-serial bookkeeping).

Method: reuses guidance/lambda_sweep.py's exact sampling/guidance/docking
pipeline (same fixed pocket, EXAMPLE_PDB) so results are directly
comparable to every real-docking finding already reported. For each
generated molecule, VinaDockingTask.run() already returns the best-
scoring DOCKED pose as a PDBQT string; this is converted to PDB (via
OpenBabel/pybel, not RDKit, which has no native PDBQT reader), merged
with the receptor PDB text into one complex, and handed to PLIP exactly
as `plip_label_extraction.py` does for its own reconstructed complexes.
"""
import argparse
import contextlib
import io
import json
import os
import re
import warnings

import torch
from openbabel import pybel
from rdkit import Chem, RDLogger

warnings.filterwarnings('ignore')
from plip.structure.preparation import PDBComplex

import utils.misc as misc
import utils.transforms as trans
from guidance.lambda_sweep import load_everything, EXAMPLE_PDB
from guidance.affinity_guidance import AffinityGuidance
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
from utils import reconstruct
from utils.evaluation.docking_vina import VinaDockingTask

CLASSES = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']
TMP_COMPLEX_PATH = './tmp/plip_interaction_validation_complex.pdb'


def count_interactions_for_pose(receptor_pdb_text, docked_pose_pdbqt_text):
    """Returns {class: count} for one docked pose against the receptor,
    or None if PLIP finds no ligand interaction set at all (e.g. the pose
    ended up outside the pocket -- reported as all-zero, not silently
    skipped, by the caller)."""
    ob_mol = pybel.readstring('pdbqt', docked_pose_pdbqt_text)
    ligand_pdb_text = ob_mol.write('pdb')
    ligand_lines = [l for l in ligand_pdb_text.splitlines()
                    if l.startswith('HETATM') or l.startswith('ATOM')]
    # Force HETATM + a non-amino-acid resname, AND an explicit chain/resSeq
    # (OpenBabel's PDB writer leaves both blank) -- PLIP's ligand
    # auto-detection needs a chain distinct from the receptor's (here 'A')
    # to treat this as a separate binding partner, matching the convention
    # guidance/track_e/plip_label_extraction.py already uses successfully
    # (chain 'L', resnum 9999) for its own synthetic complexes.
    fixed_lines = []
    for l in ligand_lines:
        l = l.ljust(80)
        l = 'HETATM' + l[6:17] + 'LIG' + ' Z' + '9999' + l[26:]
        fixed_lines.append(l)

    # The receptor PDB already ends in its own "END" record -- most PDB
    # parsers (OpenBabel included) stop reading there, so appending the
    # ligand's HETATM lines after it would silently make them invisible
    # (this was found the hard way: the exact proven-working ligand-line
    # format from plip_label_extraction.py still gave zero ligands until
    # this was stripped). Strip it before appending, then write a single
    # END of our own at the very end of the merged complex.
    receptor_pdb_text = re.sub(r'(?m)^END\s*\n?', '', receptor_pdb_text)
    os.makedirs(os.path.dirname(TMP_COMPLEX_PATH), exist_ok=True)
    with open(TMP_COMPLEX_PATH, 'w') as f:
        f.write(receptor_pdb_text.rstrip('\n') + '\n')
        f.write('\n'.join(fixed_lines) + '\n')
        f.write('END\n')

    m = PDBComplex()
    m.load_pdb(TMP_COMPLEX_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        m.analyze()

    counts = {c: 0 for c in CLASSES}
    found_any_set = False
    for key, s in m.interaction_sets.items():
        if 'LIG' not in key:
            continue
        found_any_set = True
        counts['hbond'] += len(s.hbonds_ldon) + len(s.hbonds_pdon)
        counts['hydrophobic'] += len(s.hydrophobic_contacts)
        counts['pi_stack'] += len(s.pistacking) + len(s.pication_laro) + len(s.pication_paro)
        counts['salt_bridge'] += len(s.saltbridge_lneg) + len(s.saltbridge_pneg)
    return counts if found_any_set else {c: 0 for c in CLASSES}


def sample_and_score(data, model, affinity_model, n_samples, lambda_affinity, num_steps, seed, device,
                      batch_size=None, dock_exhaustiveness=8):
    """Samples n_samples molecules under the given guidance strength,
    docks each, and returns a list of per-molecule interaction counts
    (plus Vina affinity for cross-reference). Mirrors lambda_sweep.py's
    run_one_point's sampling/reconstruction/docking steps exactly (same
    function call, same per-sample reconstruction path via pred_pos/
    pred_v as parallel per-sample lists, not a single batched tensor +
    batch_ligand index -- that was this script's first-draft mistake,
    caught by checking run_one_point directly rather than assuming the
    sampling function's return shape), adding PLIP interaction counting
    on the Vina-docked pose as the one new step.
    """
    misc.seed_all(seed)
    pred_pos, pred_v, *_ = sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=batch_size or n_samples, device=device,
        num_steps=num_steps, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=affinity_model, synth_model=None,
        lambda_affinity=lambda_affinity, lambda_synth=0.0,
    )

    receptor_pdb_text = open(EXAMPLE_PDB).read()
    results = []
    for sample_idx, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(torch.from_numpy(v), mode='add_aromatic')
        pred_aromatic = trans.is_aromatic_from_index(torch.from_numpy(v), mode='add_aromatic')
        try:
            mol = reconstruct.reconstruct_from_generated(
                pos.astype(float).tolist(), [int(x) for x in pred_atom_type], [bool(x) for x in pred_aromatic])
        except Exception as e:
            results.append({'sample_idx': sample_idx, 'status': 'reconstruct_failed', 'error': str(e)})
            continue
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) != 1:
            results.append({'sample_idx': sample_idx, 'status': 'fragmented', 'n_fragments': len(frags)})
            continue
        try:
            task = VinaDockingTask(EXAMPLE_PDB, mol)
            dock_result = task.run(mode='dock', exhaustiveness=dock_exhaustiveness)[0]
        except Exception as e:
            results.append({'sample_idx': sample_idx, 'status': 'dock_failed', 'error': str(e)})
            continue
        try:
            counts = count_interactions_for_pose(receptor_pdb_text, dock_result['pose'])
        except Exception as e:
            results.append({'sample_idx': sample_idx, 'status': 'plip_failed', 'error': str(e)})
            continue
        results.append({'sample_idx': sample_idx, 'status': 'ok',
                        'vina_dock': dock_result['affinity'], **counts})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--affinity_ckpt', type=str, required=True)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 1.0])
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='guidance holds extra memory per internal batch -- 4 was the '
                             'largest safe value found for n_samples=8+ on this 4GB GPU '
                             '(see lambda_sweep.py\'s own docstring note on the same OOM)')
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--use_esm2', action='store_true',
                        help='same meaning as lambda_sweep.py --use_esm2: use AffinityGuidanceESM2 '
                             'for an ESM2-trained checkpoint, pocket fixed to SQHC_ALIAD_1_631_0.')
    parser.add_argument('--resume', action='store_true',
                        help='skip lambdas already present in --out and append to it, same '
                             'intent as lambda_sweep.py --resume')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data, model = load_everything(device)
    if args.use_esm2:
        from guidance.affinity_guidance_esm2 import AffinityGuidanceESM2
        affinity_model = AffinityGuidanceESM2(args.affinity_ckpt, device=device)
        affinity_model.set_pocket('SQHC_ALIAD_1_631_0')
    else:
        affinity_model = AffinityGuidance(args.affinity_ckpt, device=device)

    all_results = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            all_results = json.load(f)
        print(f'--resume: {list(all_results.keys())} already in {args.out}')

    for lam in args.lambdas:
        if str(lam) in all_results:
            print(f'=== lambda_affinity={lam} -- already done, skipping ===')
            continue
        print(f'=== lambda_affinity={lam} ===')
        res = sample_and_score(data, model, affinity_model, args.n_samples, lam,
                               args.num_steps, args.seed, device, batch_size=args.batch_size)
        ok = [r for r in res if r['status'] == 'ok']
        print(f'  {len(ok)}/{len(res)} ok; statuses: {[r["status"] for r in res]}')
        for c in CLASSES:
            if ok:
                print(f'  mean {c}: {sum(r[c] for r in ok)/len(ok):.2f}')
        all_results[str(lam)] = res
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
