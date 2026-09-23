# Blueprint Report -- 004.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](004_structure.png)

- **Molecular formula:** C24H39NO4
- **Canonical SMILES:** `CCC[C@@H]1CCCNC[C@@H]1[C@@H](O)[C@@H](C)c1cccc(CCC[C@H](O)CC(=O)O)c1`
- **Molecular weight:** 405.29 Da
- **LogP:** 3.73
- **QED:** 0.450
- **SA score:** 0.660 (1=easy, 10=hard)
- **RA score:** 0.095 (real RA-score model)
- **Vina Dock:** -8.67 kcal/mol (ligand efficiency: -0.299 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| SER150 | A | hydrogen bond | 4.07 |
| ARG152 | A | hydrogen bond | 2.67 |
| SER150 | A | hydrogen bond | 3.16 |
| TYR22 | A | hydrophobic contact | 3.19 |
| TYR22 | A | hydrophobic contact | 3.73 |
| TYR90 | A | hydrophobic contact | 3.78 |
| TYR90 | A | hydrophobic contact | 3.89 |
| LEU149 | A | hydrophobic contact | 3.99 |
| ARG152 | A | salt bridge | 3.91 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.095 (real RA-score model)
- **SA score:** 0.66 (fragment familiarity: 1.303, 29 atoms, 5 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 1 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✗ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
