"""Builds the training data for Task D's synthesizability guidance model.

RA score (the real reymond-group model wired up in eval/honest_eval.py) is
itself non-differentiable -- an XGBoost classifier over Morgan-fingerprint
counts of a SMILES string, with no notion of 3D geometry at all. So it
cannot be used directly as an in-loop diffusion guidance signal, which
needs a gradient with respect to continuous ligand position. This script
distills it into a differentiable surrogate's training set: for a sample of
CrossDocked2020 ligands, reconstruct each into an RDKit mol from its real
(crystal/docked) 3D geometry via utils.reconstruct.reconstruct_from_generated
-- the same geometry-to-molecule bond-perception routine already used to
turn diffusion samples into molecules -- canonicalize to SMILES, score it
with the real RA-score model, and cache (index -> ra_score). A 3D EGNN
(guidance/synth_model.py) is then trained to regress this label directly
from 3D structure, giving Task D's grad_log_synth something to
differentiate through.

This uses the CrossDocked2020 LMDB already downloaded for Task A/C -- no
extra data source needed. Unlike Task C's affinity labels, this is not
restricted to entries with real experimental labels since RA score is
computed on the fly, so the full base dataset (166,500 entries) is
available to sample from.
"""
import argparse
import pickle
import random

from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from datasets.pl_pair_dataset import PocketLigandPairDataset
from utils.reconstruct import reconstruct_from_generated, MolReconsError
from eval.honest_eval import RAScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--n_samples', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--out', type=str, default='./data/synth_ra_labels.pkl')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    ds = PocketLigandPairDataset(args.root)
    n_total = len(ds)
    rng = random.Random(args.seed)
    candidate_indices = list(range(n_total))
    rng.shuffle(candidate_indices)

    scorer = RAScorer()
    print(f'RAScorer is_proxy={scorer.is_proxy} (must be False for real RA-score labels)')
    assert not scorer.is_proxy, 'Refusing to build a distillation dataset from the QED/SA proxy -- fix RAscore install first.'

    labels = {}
    n_fail = 0
    pbar = tqdm(total=args.n_samples, desc='Labeling ligands with RA score')
    for idx in candidate_indices:
        if len(labels) >= args.n_samples:
            break
        d = ds[idx]
        if d.ligand_pos.size(0) < 3:
            continue
        try:
            mol = reconstruct_from_generated(
                d.ligand_pos.numpy().tolist(), d.ligand_element.numpy().tolist())
            smiles = Chem.MolToSmiles(mol)
            if '.' in smiles:  # disconnected fragments; skip
                n_fail += 1
                continue
            ra = scorer.predict(mol)
        except (MolReconsError, Exception) as e:
            n_fail += 1
            if args.verbose:
                print(f'idx {idx} failed: {e}')
            continue
        labels[idx] = float(ra)
        pbar.update(1)
    pbar.close()

    print(f'Labeled {len(labels)} ligands, {n_fail} reconstruction/scoring failures.')
    with open(args.out, 'wb') as f:
        pickle.dump(labels, f)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
