"""Task G: Interaction & Synthesis Blueprint Report.

For every generated molecule that passes the *honest* success bar (Task B:
standard composite bar AND PoseBusters-valid), produces a human-readable
Markdown report plus a 2D structure PNG, combining:
  G.1 molecular identity (formula, SMILES, 2D structure, MW/LogP/QED/SA/RA)
  G.2 protein-ligand interaction analysis (eval/interaction_analysis.py) --
      a geometric read of the docked pose, never phrased as physiological
      effect
  G.3 synthesis blueprint -- RA-score (Task B's real reymond-group model)
      plus an SA-score fragment-complexity breakdown, since AiZynthFinder
      (a concrete retrosynthetic-route generator) was checked and found
      incompatible with this environment (see module docstring below) --
      and the two allowed structural flags, Lipinski Ro5 and PAINS,
      always named and never rephrased as toxicity/safety predictions.

Markdown was chosen over HTML for the per-molecule report because it
renders the inline structure image with a single `![...](...)` line and
stays readable as plain text in a terminal or diff, without needing any
templating; the image file sits next to the .md file and is referenced by
relative path.

Molecules that do NOT pass the honest bar get no report at all -- this
report is meant to represent "what we'd actually consider taking toward
synthesis," not a report on every raw sample (see Task G addendum).

--------------------------------------------------------------------------
AiZynthFinder was checked, not installed: its PyPI metadata pins
`numpy<2.0.0` and `rdkit<2024.0.0`, both hard-incompatible with this
environment's validated stack (numpy 2.2.6, rdkit 2026.3.5 -- the versions
the rest of this pipeline, including the diffusion core and both guidance
models, were built and tested against). Installing it would force a
downgrade that risks breaking that already-validated stack, which is
exactly the kind of invasive change the working rules ask to check before
doing. So G.3 uses the documented fallback: RA-score (real model, not the
proxy) plus the SA-score fragment-complexity breakdown, with no concrete
retrosynthetic route. This is reported plainly in every generated report,
not silently substituted.
--------------------------------------------------------------------------
"""
import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors, Descriptors, Crippen, Lipinski

from eval.honest_eval import (
    load_sdf_mols, STANDARD_SUCCESS_VINA_DOCK, STANDARD_SUCCESS_QED, STANDARD_SUCCESS_SA,
)
from eval.interaction_analysis import analyze_interactions
from utils.evaluation.scoring_func import is_pains
from utils.evaluation.sascorer import readFragmentScores, iteritems, numBridgeheadsAndSpiro
import utils.evaluation.sascorer as sascorer_mod

DISCLAIMER = (
    "> **Disclaimer:** This is a computational structural and synthesizability "
    "estimate. No in-body efficacy, toxicity, or physiological behavior is "
    "predicted or implied anywhere in this report. Wet-lab / biomolecular "
    "specialist review is required before any synthesis is attempted."
)


# --------------------------------------------------------------------------
# G.1 -- identity block
# --------------------------------------------------------------------------

def identity_block(mol):
    return {
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'smiles': Chem.MolToSmiles(mol),
        'mol_weight': round(Descriptors.ExactMolWt(mol), 2),
        'logp': round(Crippen.MolLogP(mol), 2),
    }


def save_structure_image(mol, out_path):
    Draw.MolToFile(mol, out_path, size=(500, 400))


# --------------------------------------------------------------------------
# G.3 -- synthesis blueprint: SA-score fragment breakdown + Ro5/PAINS
# --------------------------------------------------------------------------

def sa_score_breakdown(mol):
    """Replicates utils/evaluation/sascorer.calculateScore's internal
    components (it only returns the final blended score) so the fallback
    report can show *why* a molecule scores as complex/simple, not just
    the number.
    """
    if sascorer_mod._fscores is None:
        readFragmentScores()
    fscores = sascorer_mod._fscores

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1, nf = 0., 0
    for bitId, v in iteritems(fps):
        nf += v
        score1 += fscores.get(bitId, -4) * v
    score1 /= nf

    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridge, n_spiro = numBridgeheadsAndSpiro(mol, ri)
    n_macrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

    from utils.evaluation.sascorer import compute_sa_score
    return {
        'fragment_familiarity_score': round(score1, 3),
        'num_atoms': n_atoms,
        'num_chiral_centers': n_chiral,
        'num_spiro_atoms': n_spiro,
        'num_bridgehead_atoms': n_bridge,
        'num_macrocycles': n_macrocycles,
        'sa_score_final': round(compute_sa_score(mol), 3),
    }


LIPINSKI_RULES = [
    ('Molecular weight < 500 Da', lambda m: Descriptors.ExactMolWt(m) < 500),
    ('H-bond donors <= 5', lambda m: Lipinski.NumHDonors(m) <= 5),
    ('H-bond acceptors <= 10', lambda m: Lipinski.NumHAcceptors(m) <= 10),
    ('LogP between -2 and 5', lambda m: -2 <= Crippen.MolLogP(m) <= 5),
    ('Rotatable bonds <= 10', lambda m: rdMolDescriptors.CalcNumRotatableBonds(m) <= 10),
]


def lipinski_ro5(mol):
    """Lipinski's Rule of Five (Lipinski et al., Adv. Drug Deliv. Rev.
    1997) -- a structural drug-likeness heuristic, not a toxicity or
    efficacy prediction.
    """
    results = [(name, bool(fn(mol))) for name, fn in LIPINSKI_RULES]
    n_violations = sum(1 for _, ok in results if not ok)
    return {'n_violations': n_violations, 'rules': results}


def pains_filter(mol):
    """RDKit's built-in PAINS_A substructure catalog (Baell & Holloway,
    J. Med. Chem. 2010) -- flags known assay-interference substructures.
    A structural alert, not a toxicity prediction.
    """
    return bool(is_pains(mol))


# --------------------------------------------------------------------------
# Honest success bar (per-row; mirrors honest_eval.compute_success_rates)
# --------------------------------------------------------------------------

def passes_honest_bar(row):
    if pd.isna(row.get('vina_dock')) or pd.isna(row.get('qed')) or pd.isna(row.get('sa')):
        return False
    standard = (row['vina_dock'] < STANDARD_SUCCESS_VINA_DOCK and
               row['qed'] > STANDARD_SUCCESS_QED and row['sa'] > STANDARD_SUCCESS_SA)
    pb_ok = bool(row.get('pb_valid')) if pd.notna(row.get('pb_valid')) else False
    return bool(standard and pb_ok)


# --------------------------------------------------------------------------
# G.4 -- report assembly
# --------------------------------------------------------------------------

def generate_report(row, mol, protein_path, out_dir, verbose=False):
    stem = os.path.splitext(row['file'])[0]
    img_path = os.path.join(out_dir, f'{stem}_structure.png')
    save_structure_image(mol, img_path)

    ident = identity_block(mol)
    contacts, contact_method = analyze_interactions(protein_path, mol, verbose=verbose)
    sa_breakdown = sa_score_breakdown(mol)
    ro5 = lipinski_ro5(mol)
    pains = pains_filter(mol)

    lines = []
    lines.append(f'# Blueprint Report -- {row["file"]}')
    lines.append('')
    lines.append(DISCLAIMER)
    lines.append('')
    lines.append('## G.1 Molecular Identity')
    lines.append('')
    lines.append(f'![2D structure]({os.path.basename(img_path)})')
    lines.append('')
    lines.append(f"- **Molecular formula:** {ident['formula']}")
    lines.append(f"- **Canonical SMILES:** `{ident['smiles']}`")
    lines.append(f"- **Molecular weight:** {ident['mol_weight']} Da")
    lines.append(f"- **LogP:** {ident['logp']}")
    lines.append(f"- **QED:** {row['qed']:.3f}")
    lines.append(f"- **SA score:** {row['sa']:.3f} (1=easy, 10=hard)")
    ra_label = 'real RA-score model' if not row.get('ra_score_is_proxy', True) else 'QED/SA proxy (NOT real RA-score)'
    lines.append(f"- **RA score:** {row['ra_score']:.3f} ({ra_label})")
    lines.append(f"- **Vina Dock:** {row['vina_dock']:.2f} kcal/mol "
                 f"(ligand efficiency: {row.get('ligand_efficiency', float('nan')):.3f} kcal/mol per heavy atom)")
    lines.append('')

    lines.append('## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)')
    lines.append('')
    lines.append(f'*Method: {"PLIP (typed interactions)" if contact_method == "plip" else "distance-only contact map (no interaction typing; PLIP unavailable/failed for this complex)"}.*')
    lines.append('')
    if contacts:
        lines.append('| Residue | Chain | Interaction type | Closest distance (A) |')
        lines.append('|---|---|---|---|')
        for c in contacts:
            itype = c['interaction_type'] or '(untyped contact)'
            lines.append(f"| {c['resname']}{c['resnum']} | {c['chain']} | {itype} | {c['distance_A']} |")
    else:
        lines.append('No contacts detected within the analysis cutoff.')
    lines.append('')
    lines.append('*All statements above describe the geometry of the docked/generated pose only '
                 '(e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") '
                 '-- no claim of biological/physiological effect is made or implied.*')
    lines.append('')

    lines.append('## G.3 Synthesis Blueprint')
    lines.append('')
    lines.append('**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and '
                 'found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both '
                 'conflicting with the validated diffusion/guidance stack); see blueprint_report.py '
                 'module docstring. The figures below are feasibility estimates only.')
    lines.append('')
    lines.append(f"- **RA score:** {row['ra_score']:.3f} ({ra_label})")
    lines.append(f"- **SA score:** {sa_breakdown['sa_score_final']} "
                 f"(fragment familiarity: {sa_breakdown['fragment_familiarity_score']}, "
                 f"{sa_breakdown['num_atoms']} atoms, "
                 f"{sa_breakdown['num_chiral_centers']} chiral centers, "
                 f"{sa_breakdown['num_spiro_atoms']} spiro atoms, "
                 f"{sa_breakdown['num_bridgehead_atoms']} bridgeheads, "
                 f"{sa_breakdown['num_macrocycles']} macrocycles)")
    lines.append('')
    lines.append(f"- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** "
                 f"{ro5['n_violations']} violation(s)")
    for name, ok in ro5['rules']:
        lines.append(f"  - {'✓' if ok else '✗'} {name}")
    lines.append(f"- **PAINS filter (structural flag, Baell & Holloway 2010):** "
                 f"{'ALERT -- matches a known assay-interference substructure' if pains else 'no alerts'}")
    lines.append('')

    report_path = os.path.join(out_dir, f'{stem}_blueprint.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return {
        'file': row['file'],
        'report_path': report_path,
        'image_path': img_path,
        'formula': ident['formula'],
        'smiles': ident['smiles'],
        'mol_weight': ident['mol_weight'],
        'qed': row['qed'],
        'sa': row['sa'],
        'ra_score': row['ra_score'],
        'vina_dock': row['vina_dock'],
        'ligand_efficiency': row.get('ligand_efficiency'),
        'n_contacts': len(contacts),
        'contact_method': contact_method,
        'lipinski_violations': ro5['n_violations'],
        'pains_alert': pains,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('honest_eval_csv', type=str)
    parser.add_argument('sdf_dir', type=str)
    parser.add_argument('--protein_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.honest_eval_csv), 'blueprint_reports')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.honest_eval_csv)
    mols = dict(load_sdf_mols(args.sdf_dir))

    summary_rows = []
    for _, row in df.iterrows():
        if row['file'] not in mols:
            continue
        if not passes_honest_bar(row):
            continue
        mol = mols[row['file']]
        print(f'Generating blueprint report for {row["file"]} (passes honest bar)...')
        summary_rows.append(generate_report(row, mol, args.protein_path, out_dir, verbose=args.verbose))

    print(f'\n{len(summary_rows)}/{len(df)} molecules passed the honest success bar and got a blueprint report.')

    summary_path = os.path.join(out_dir, 'blueprint_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f'Summary table saved to {summary_path}')


if __name__ == '__main__':
    main()
