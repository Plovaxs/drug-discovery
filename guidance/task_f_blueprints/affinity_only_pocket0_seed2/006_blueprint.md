# Blueprint Report -- 006.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](006_structure.png)

- **Molecular formula:** C12H14FN5O3S
- **Canonical SMILES:** `CC(=O)N(O)CC(=C1CNC(F)S1)c1cnnc(C(N)=O)c1`
- **Molecular weight:** 327.08 Da
- **LogP:** 0.11
- **QED:** 0.408
- **SA score:** 0.610 (1=easy, 10=hard)
- **RA score:** 0.512 (real RA-score model)
- **Vina Dock:** -9.07 kcal/mol (ligand efficiency: -0.412 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| SER28 | A | hydrogen bond | 2.33 |
| VAL46 | A | hydrogen bond | 3.79 |
| SER86 | A | hydrogen bond | 2.94 |
| ARG82 | A | hydrogen bond | 3.4 |
| ARG82 | A | hydrogen bond | 2.74 |
| CYS88 | A | hydrogen bond | 2.84 |
| ASN45 | A | hydrogen bond | 3.68 |
| ALA55 | A | hydrogen bond | 2.8 |
| VAL29 | A | hydrophobic contact | 4.0 |
| TYR126 | B | hydrophobic contact | 3.71 |
| TRP128 | B | hydrophobic contact | 3.96 |
| HIS48 | A | halogen bond | 3.66 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.512 (real RA-score model)
- **SA score:** 0.61 (fragment familiarity: 0.302, 22 atoms, 1 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 0 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
