# Blueprint Report -- 007.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](007_structure.png)

- **Molecular formula:** C18H27N3O3
- **Canonical SMILES:** `CC[C@@H](CCCCCO)[C@@H]1c2cnncc2C2=C(O)C[C@@H](OC)N21`
- **Molecular weight:** 333.21 Da
- **LogP:** 3.02
- **QED:** 0.712
- **SA score:** 0.600 (1=easy, 10=hard)
- **RA score:** 0.374 (real RA-score model)
- **Vina Dock:** -8.21 kcal/mol (ligand efficiency: -0.342 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| ILE113 | A | hydrogen bond | 3.14 |
| SER150 | A | hydrogen bond | 2.61 |
| GLU64 | A | hydrogen bond | 3.92 |
| ASN92 | A | hydrogen bond | 3.57 |
| THR91 | A | hydrogen bond | 3.67 |
| THR115 | A | hydrogen bond | 3.57 |
| THR115 | A | hydrogen bond | 2.79 |
| SER150 | A | hydrogen bond | 3.76 |
| TYR62 | A | hydrophobic contact | 3.46 |
| TYR90 | A | hydrophobic contact | 3.37 |
| LEU147 | A | hydrophobic contact | 3.93 |
| TYR22 | A | hydrophobic contact | 3.57 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.374 (real RA-score model)
- **SA score:** 0.6 (fragment familiarity: 0.624, 24 atoms, 3 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 0 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✓ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
