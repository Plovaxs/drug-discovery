# Blueprint Report -- 009.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](009_structure.png)

- **Molecular formula:** C14H15NO4S
- **Canonical SMILES:** `COC1=C(C)C2=C3[C@H]1NS(O)(O)[C@@H]3c1ccccc1O2`
- **Molecular weight:** 293.07 Da
- **LogP:** 2.95
- **QED:** 0.742
- **SA score:** 0.620 (1=easy, 10=hard)
- **RA score:** 0.593 (real RA-score model)
- **Vina Dock:** -8.44 kcal/mol (ligand efficiency: -0.422 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| ARG313 | A | hydrogen bond | 3.04 |
| ARG313 | A | hydrogen bond | 3.83 |
| TRP148 | A | hydrophobic contact | 3.82 |
| TYR255 | A | hydrophobic contact | 3.74 |
| TRP148 | A | pi-stacking | 3.67 |
| TRP148 | A | pi-stacking | 3.81 |
| ARG309 | A | pi-cation interaction | 5.02 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.593 (real RA-score model)
- **SA score:** 0.62 (fragment familiarity: 0.517, 20 atoms, 2 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 0 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
