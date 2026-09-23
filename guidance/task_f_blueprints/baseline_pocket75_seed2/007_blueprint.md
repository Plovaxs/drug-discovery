# Blueprint Report -- 007.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](007_structure.png)

- **Molecular formula:** C17H17Cl2NO
- **Canonical SMILES:** `C[C@@H]1C(=N)CCC2=C1C=C[C@H](Oc1ccc(Cl)c(Cl)c1)C2`
- **Molecular weight:** 321.07 Da
- **LogP:** 5.45
- **QED:** 0.772
- **SA score:** 0.650 (1=easy, 10=hard)
- **RA score:** 0.371 (real RA-score model)
- **Vina Dock:** -10.28 kcal/mol (ligand efficiency: -0.490 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| GLN89 | A | hydrogen bond | 3.88 |
| ASN125 | A | hydrogen bond | 3.52 |
| THR138 | A | hydrogen bond | 2.96 |
| THR140 | A | hydrogen bond | 3.8 |
| ILE60 | A | hydrophobic contact | 3.94 |
| TYR65 | A | hydrophobic contact | 3.52 |
| GLN86 | A | hydrophobic contact | 3.8 |
| LEU91 | A | hydrophobic contact | 3.86 |
| LEU142 | A | hydrophobic contact | 3.87 |
| LEU142 | A | hydrophobic contact | 3.67 |
| ILE84 | A | halogen bond | 3.29 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.371 (real RA-score model)
- **SA score:** 0.65 (fragment familiarity: 0.772, 21 atoms, 2 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 1 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✗ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
