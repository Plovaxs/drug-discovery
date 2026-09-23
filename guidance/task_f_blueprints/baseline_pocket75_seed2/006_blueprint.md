# Blueprint Report -- 006.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](006_structure.png)

- **Molecular formula:** C17H18N2O3
- **Canonical SMILES:** `NC(=O)CC1=N[C@@]2(O)CC=C(c3ccccc3)C(=O)[C@H]2CC1`
- **Molecular weight:** 298.13 Da
- **LogP:** 1.46
- **QED:** 0.884
- **SA score:** 0.680 (1=easy, 10=hard)
- **RA score:** 0.248 (real RA-score model)
- **Vina Dock:** -10.15 kcal/mol (ligand efficiency: -0.461 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| THR138 | A | hydrogen bond | 3.0 |
| THR140 | A | hydrogen bond | 3.71 |
| GLN86 | A | hydrogen bond | 2.71 |
| GLN68 | A | hydrogen bond | 3.79 |
| GLN86 | A | hydrophobic contact | 3.56 |
| THR140 | A | hydrophobic contact | 3.99 |
| TYR65 | A | hydrophobic contact | 3.49 |
| TYR65 | A | hydrophobic contact | 3.8 |
| GLN68 | A | hydrophobic contact | 3.67 |
| ILE84 | A | hydrophobic contact | 3.74 |
| LEU142 | A | hydrophobic contact | 3.71 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.248 (real RA-score model)
- **SA score:** 0.68 (fragment familiarity: 0.939, 22 atoms, 2 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 0 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
