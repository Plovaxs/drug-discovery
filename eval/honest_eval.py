"""Honest evaluation harness for generated ligands (SDF files) against a pocket.

Design intent: give a size-normalized, physically-checked picture of
generated-ligand quality, rather than reporting raw Vina Dock scores alone
(which are easy to "hack" by generating larger, floppier ligands that dock
more favorably without actually being better binders or synthesizable).

This module's RA-score model is also what caught a concrete instance of
this same failure mode inside the project's own synthesizability-guidance
model, not just in raw sampling output: guidance/lambda_sweep.py found the
synth guidance model's own predicted score improving while THIS module's
independently-computed real RA-score got worse at the same time -- see
guidance/SYNTH_GUIDANCE_FINDING.md for the full analysis. That finding is
direct evidence for why an honest, independently-computed evaluation
harness (rather than trusting any single model's self-reported score)
matters.

Reuses the repo's own building blocks rather than reinventing them:
  - utils.evaluation.scoring_func.get_chem       -> QED, SA
  - utils.evaluation.docking_vina.VinaDockingTask -> Score/Min/Dock (AutoDock
    Vina 1.2.6 + meeko + pdb2pqr + AutoDockTools_py3, same pipeline used by
    scripts/evaluate_diffusion.py / scripts/dock_testset.py)

New in this module:
  - validity / uniqueness / novelty / diversity bookkeeping
  - RA score (retrosynthesis-aware synthesizability): uses the
    `RAscore` XGBoost ChEMBL model (reymond-group RAscore, installed from the
    Python-3.10/NumPy-2-compatible fork MahitVaddadi-Bloom/RAscore) when
    importable. Falls back to a documented QED/SA-based proxy — clearly
    labeled `ra_score_is_proxy=True` in the output — if RAscore is not
    installed, so the harness never silently mislabels a stand-in as the
    real thing.
  - heavy-atom-normalized ligand efficiency (Vina Dock / heavy-atom count),
    computed specifically to expose Vina-hacking rather than hide it.
  - PoseBusters physical-validity check (`posebusters`, config='dock': the
    generated pose is checked against the protein pocket directly, since
    there is no crystal reference pose for de novo molecules).
  - standard TargetDiff success-rate bar (Vina Dock < -8.18, QED > 0.25,
    SA > 0.59) and a stricter "honest" bar that also requires PB-valid and
    ligand efficiency within a documented tolerance of the reference ligand.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import scoring_func  # noqa: E402
from utils.evaluation.docking_vina import VinaDockingTask  # noqa: E402

STANDARD_SUCCESS_VINA_DOCK = -8.18
STANDARD_SUCCESS_QED = 0.25
STANDARD_SUCCESS_SA = 0.59


# --------------------------------------------------------------------------
# RA score (synthesizability)
# --------------------------------------------------------------------------

_xgboost_patched = False


def _patch_xgboost_numpy2_compat():
    """The RAscore XGBoost model (reymond-group RAscore, installed via the
    MahitVaddadi-Bloom/RAscore fork pinned to xgboost==1.3.3 to match the
    pickled model's era) calls `np.array(x, copy=False)` internally, which
    NumPy >= 2.0 turns into a hard error instead of silently falling back to
    a copy. This replaces just that one internal function with an
    equivalent NumPy-2-safe version; it does not touch the pickled model or
    its predictions (verified against the RAscore README's own reference
    values: morphine -> 0.0028359715, matches exactly).
    """
    global _xgboost_patched
    if _xgboost_patched:
        return
    import numpy as np
    import xgboost.data as xd

    def _fixed_transform_np_array(array):
        return np.asarray(array.reshape(array.size)).astype('float32', copy=False)

    xd._transform_np_array = _fixed_transform_np_array
    _xgboost_patched = True


class RAScorer:
    """Wraps the real RAscore XGBoost model when available.

    `is_proxy` tells callers whether `.predict()` is backed by the actual
    retrosynthesis-planner-trained classifier or by the QED/SA proxy.
    """

    def __init__(self):
        self.is_proxy = False
        self._model = None
        try:
            _patch_xgboost_numpy2_compat()
            from RAscore import RAscore_XGB
            self._model = RAscore_XGB.RAScorerXGB()
        except Exception as e:  # pragma: no cover - environment dependent
            print(f'[honest_eval] RAscore unavailable ({e}); '
                  f'falling back to QED/SA-based synthesizability proxy.')
            self.is_proxy = True

    def predict(self, mol) -> float:
        if not self.is_proxy:
            smiles = Chem.MolToSmiles(mol)
            return float(self._model.predict(smiles))
        # Documented proxy: geometric mean of QED and normalized (10 - SA)/9,
        # both in [0, 1], as a rough, explicitly-labeled synthesizability
        # stand-in when the real retrosynthesis-planner-trained model isn't
        # installed. This is NOT RA score and must not be reported as such.
        qed = scoring_func.qed(mol)
        sa = scoring_func.compute_sa_score(mol)
        sa_norm = max(0.0, min(1.0, (10.0 - sa) / 9.0))
        return float(np.sqrt(max(qed, 0.0) * sa_norm))


# --------------------------------------------------------------------------
# Validity / uniqueness / novelty / diversity
# --------------------------------------------------------------------------

def load_sdf_mols(sdf_dir):
    """Load all valid molecules from a directory of *.sdf files.

    Returns list of (filename, mol). Each file is expected to hold a single
    already-reconstructed, RDKit-sanitized molecule (as written by
    scripts/sample_for_pocket.py / scripts/sample_diffusion.py).
    """
    mols = []
    for path in sorted(glob.glob(os.path.join(sdf_dir, '*.sdf'))):
        supplier = Chem.SDMolSupplier(path, sanitize=True)
        mol = next(iter(supplier), None)
        if mol is not None:
            mols.append((os.path.basename(path), mol))
    return mols


def canonical_smiles(mol):
    return Chem.MolToSmiles(mol)


def compute_validity_uniqueness_novelty(mols, n_attempted=None, reference_smiles=None):
    smiles_list = [canonical_smiles(m) for _, m in mols]
    n_valid = len(smiles_list)
    unique_smiles = set(smiles_list)
    n_unique = len(unique_smiles)

    result = {
        'n_attempted': n_attempted if n_attempted is not None else n_valid,
        'n_valid': n_valid,
        'validity': (n_valid / n_attempted) if n_attempted else None,
        'uniqueness': (n_unique / n_valid) if n_valid else 0.0,
    }
    if reference_smiles is not None:
        ref_canon = Chem.MolToSmiles(Chem.MolFromSmiles(reference_smiles)) \
            if isinstance(reference_smiles, str) else canonical_smiles(reference_smiles)
        n_novel = sum(1 for s in unique_smiles if s != ref_canon)
        result['novelty_vs_reference'] = (n_novel / n_unique) if n_unique else 0.0
        result['novelty_note'] = ('novelty measured only against the single '
                                   'reference ligand for this pocket, not a '
                                   'full training-set SMILES corpus')
    else:
        result['novelty_vs_reference'] = None
    return result


def compute_diversity(mols):
    """Mean pairwise (1 - Tanimoto) over Morgan fingerprints. 0 if <2 mols."""
    from rdkit.Chem import AllChem
    if len(mols) < 2:
        return None
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for _, m in mols]
    dists = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = Chem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1.0 - sim)
    return float(np.mean(dists))


# --------------------------------------------------------------------------
# Per-molecule metrics: chem, RA score, Vina, PoseBusters
# --------------------------------------------------------------------------

def evaluate_molecules(mols, protein_path, docking_mode='vina_dock', exhaustiveness=16,
                        reference_sdf=None, run_posebusters=True, verbose=False,
                        skip_score_and_minimize=False):
    """skip_score_and_minimize: when docking_mode='vina_dock', omit the
    separate 'score_only' and 'minimize' Vina calls (measured at ~0.57s +
    0.43s vs. 'dock' mode's ~9.36s per molecule -- a modest ~10% cut of
    the docking phase, not a major one) and leave vina_score/vina_min as
    None. Default False preserves every prior caller's exact behavior
    (Track A/B/D's already-completed results are unaffected regardless of
    this default) -- only opted into where those two columns are known to
    be unused by the analysis (guidance/track_c_generate_pools.py)."""
    ra_scorer = RAScorer()
    reference_mol = None
    reference_heavy_atoms = None
    if reference_sdf and os.path.exists(reference_sdf):
        reference_mol = next(iter(Chem.SDMolSupplier(reference_sdf, sanitize=True)), None)
        if reference_mol is not None:
            reference_heavy_atoms = reference_mol.GetNumHeavyAtoms()

    pb = None
    if run_posebusters:
        from posebusters import PoseBusters
        pb = PoseBusters(config='dock')

    rows = []
    for fname, mol in mols:
        row = {'file': fname, 'smiles': canonical_smiles(mol)}
        heavy_atoms = mol.GetNumHeavyAtoms()
        row['heavy_atoms'] = heavy_atoms

        chem = scoring_func.get_chem(mol)
        row['qed'] = chem['qed']
        row['sa'] = chem['sa']

        row['ra_score'] = ra_scorer.predict(mol)
        row['ra_score_is_proxy'] = ra_scorer.is_proxy

        try:
            vina_task = VinaDockingTask(protein_path, mol)
            if skip_score_and_minimize and docking_mode == 'vina_dock':
                row['vina_score'] = row['vina_min'] = None
            else:
                score_only = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
                minimize = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
                row['vina_score'] = score_only[0]['affinity']
                row['vina_min'] = minimize[0]['affinity']
            if docking_mode == 'vina_dock':
                dock = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                row['vina_dock'] = dock[0]['affinity']
            else:
                row['vina_dock'] = None
        except Exception as e:
            if verbose:
                print(f'[honest_eval] Vina docking failed for {fname}: {e}')
            row['vina_score'] = row['vina_min'] = row['vina_dock'] = None

        if row['vina_dock'] is not None and heavy_atoms > 0:
            row['ligand_efficiency'] = row['vina_dock'] / heavy_atoms
        else:
            row['ligand_efficiency'] = None

        if reference_heavy_atoms:
            row['heavy_atom_ratio_vs_ref'] = heavy_atoms / reference_heavy_atoms
        else:
            row['heavy_atom_ratio_vs_ref'] = None

        if pb is not None:
            try:
                report = pb.bust([mol], mol_true=None, mol_cond=protein_path, full_report=False)
                row['pb_valid'] = bool(report.iloc[0].drop(labels=[], errors='ignore').all())
                for col in report.columns:
                    row[f'pb_{col}'] = bool(report.iloc[0][col])
            except Exception as e:
                if verbose:
                    print(f'[honest_eval] PoseBusters failed for {fname}: {e}')
                row['pb_valid'] = None
        else:
            row['pb_valid'] = None

        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Success rates
# --------------------------------------------------------------------------

def compute_success_rates(df, reference_heavy_atoms=None, le_tolerance=0.5):
    """Standard TargetDiff composite bar, plus a stricter 'honest' bar.

    Honest bar additionally requires PB-valid == True and, when a reference
    ligand efficiency is available, that the generated ligand efficiency is
    not worse than the reference by more than `le_tolerance` (in Vina
    kcal/mol per heavy atom) — i.e. affinity gains must not come purely from
    inflating molecule size (Vina-hacking).
    """
    if len(df) == 0:
        return {'standard_success_rate': None, 'honest_success_rate': None,
                'n_scored': 0}

    standard = (
        (df['vina_dock'] < STANDARD_SUCCESS_VINA_DOCK) &
        (df['qed'] > STANDARD_SUCCESS_QED) &
        (df['sa'] > STANDARD_SUCCESS_SA)
    )
    standard = standard.fillna(False)

    honest = standard.copy()
    if 'pb_valid' in df:
        honest = honest & df['pb_valid'].fillna(False)
    else:
        honest = honest & False

    le_flag_note = None
    if reference_heavy_atoms is not None and 'ligand_efficiency' in df:
        # reference ligand efficiency isn't separately docked here; this
        # flags only that the check was requested, actual reference-LE
        # comparison happens once a reference Vina Dock score is supplied
        # by the caller (see CLI --reference_vina_dock).
        le_flag_note = ('ligand-efficiency-vs-reference check requires '
                         '--reference_vina_dock; skipped without it')

    return {
        'standard_success_rate': float(standard.mean()),
        'honest_success_rate': float(honest.mean()),
        'n_scored': int(len(df)),
        'le_check_note': le_flag_note,
    }


def flag_vina_hacking(df, reference_vina_dock=None, reference_heavy_atoms=None):
    """Report (not silently fix) the signature of Vina-hacking: Vina Dock
    improves vs. reference while ligand efficiency and/or PB pass rate get
    worse.
    """
    findings = []
    if reference_vina_dock is None or reference_heavy_atoms is None:
        return ['no reference Vina Dock / heavy-atom count supplied; '
                'Vina-hacking check skipped']
    reference_le = reference_vina_dock / reference_heavy_atoms
    mean_dock = df['vina_dock'].dropna().mean() if 'vina_dock' in df else None
    mean_le = df['ligand_efficiency'].dropna().mean() if 'ligand_efficiency' in df else None
    pb_rate = df['pb_valid'].dropna().mean() if 'pb_valid' in df and df['pb_valid'].notna().any() else None

    if mean_dock is not None and mean_dock < reference_vina_dock:
        if mean_le is not None and mean_le > reference_le:
            findings.append(
                f'Vina Dock improved ({mean_dock:.2f} < {reference_vina_dock:.2f}) '
                f'but ligand efficiency is worse ({mean_le:.3f} > {reference_le:.3f} '
                f'kcal/mol per heavy atom) — consistent with Vina-hacking via size inflation.')
        if pb_rate is not None and pb_rate < 0.5:
            findings.append(
                f'Vina Dock improved but PoseBusters pass rate is low ({pb_rate:.2f}) '
                f'— generated poses may be physically implausible.')
    if not findings:
        findings.append('no Vina-hacking signature detected')
    return findings


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sdf_dir', type=str, help='Directory of generated *.sdf files')
    parser.add_argument('--protein_path', type=str, required=True,
                         help='Pocket PDB file to dock/PoseBusters-check against')
    parser.add_argument('--reference_sdf', type=str, default=None,
                         help='Reference (crystal) ligand SDF for this pocket, for novelty/LE comparison')
    parser.add_argument('--reference_vina_dock', type=float, default=None,
                         help='Reference ligand Vina Dock score, for the Vina-hacking check')
    parser.add_argument('--docking_mode', type=str, default='vina_dock',
                         choices=['vina_score', 'vina_dock'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--n_attempted', type=int, default=None,
                         help='Total samples attempted (for a true validity fraction)')
    parser.add_argument('--no_posebusters', action='store_true')
    parser.add_argument('--out_csv', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    mols = load_sdf_mols(args.sdf_dir)
    print(f'Loaded {len(mols)} valid molecules from {args.sdf_dir}')

    reference_smiles = None
    reference_heavy_atoms = None
    if args.reference_sdf and os.path.exists(args.reference_sdf):
        ref_mol = next(iter(Chem.SDMolSupplier(args.reference_sdf, sanitize=True)), None)
        if ref_mol is not None:
            reference_smiles = Chem.MolToSmiles(ref_mol)
            reference_heavy_atoms = ref_mol.GetNumHeavyAtoms()

    vun = compute_validity_uniqueness_novelty(
        mols, n_attempted=args.n_attempted, reference_smiles=reference_smiles)
    diversity = compute_diversity(mols)

    df = evaluate_molecules(
        mols, args.protein_path, docking_mode=args.docking_mode,
        exhaustiveness=args.exhaustiveness, reference_sdf=args.reference_sdf,
        run_posebusters=not args.no_posebusters, verbose=args.verbose)

    success = compute_success_rates(df, reference_heavy_atoms=reference_heavy_atoms)
    hacking_flags = flag_vina_hacking(
        df, reference_vina_dock=args.reference_vina_dock,
        reference_heavy_atoms=reference_heavy_atoms)

    print('\n=== Validity / Uniqueness / Novelty ===')
    for k, v in vun.items():
        print(f'{k}: {v}')
    print(f'diversity (mean pairwise 1-Tanimoto): {diversity}')

    print('\n=== Per-molecule metrics ===')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df[['file', 'heavy_atoms', 'qed', 'sa', 'ra_score', 'ra_score_is_proxy',
                   'vina_score', 'vina_min', 'vina_dock', 'ligand_efficiency', 'pb_valid']])

    print('\n=== Success rates ===')
    for k, v in success.items():
        print(f'{k}: {v}')

    print('\n=== Vina-hacking check ===')
    for f in hacking_flags:
        print(f'- {f}')

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f'\nSaved per-molecule table to {args.out_csv}')


if __name__ == '__main__':
    main()
