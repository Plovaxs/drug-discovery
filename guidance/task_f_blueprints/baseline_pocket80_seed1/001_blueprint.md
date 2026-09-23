# Blueprint Report -- 001.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](001_structure.png)

- **Molecular formula:** C24H30N2O4
- **Canonical SMILES:** `COCOc1ccc([C@@H](COC(=O)[C@H]2CCNc3ncccc32)C2CCCC2)cc1`
- **Molecular weight:** 410.22 Da
- **LogP:** 4.48
- **QED:** 0.511
- **SA score:** 0.710 (1=easy, 10=hard)
- **RA score:** 0.853 (real RA-score model)
- **Vina Dock:** -9.38 kcal/mol (ligand efficiency: -0.313 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| THR82 | A | hydrogen bond | 3.1 |
| LEU264 | A | hydrophobic contact | 3.2 |
| ARG273 | A | hydrophobic contact | 3.98 |
| GLN79 | A | hydrophobic contact | 3.84 |
| VAL270 | A | hydrophobic contact | 3.3 |
| TYR272 | A | hydrophobic contact | 3.12 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.853 (real RA-score model)
- **SA score:** 0.71 (fragment familiarity: 1.35, 30 atoms, 2 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 0 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
