# Blueprint Report -- 004.sdf

> **Disclaimer:** This is a computational structural and synthesizability estimate. No in-body efficacy, toxicity, or physiological behavior is predicted or implied anywhere in this report. Wet-lab / biomolecular specialist review is required before any synthesis is attempted.

## G.1 Molecular Identity

![2D structure](004_structure.png)

- **Molecular formula:** C16H19N3O8
- **Canonical SMILES:** `CN(C(=O)c1ccncc1)[C@H]1C(=O)O[C@H]2[C@H](N)[C@@H](O)[C@@H](O)[C@H](O)C(=O)O[C@H]21`
- **Molecular weight:** 381.12 Da
- **LogP:** -3.22
- **QED:** 0.382
- **SA score:** 0.600 (1=easy, 10=hard)
- **RA score:** 0.058 (real RA-score model)
- **Vina Dock:** -8.61 kcal/mol (ligand efficiency: -0.319 kcal/mol per heavy atom)

## G.2 Protein-Ligand Interaction Analysis (docked-pose geometry)

*Method: PLIP (typed interactions).*

| Residue | Chain | Interaction type | Closest distance (A) |
|---|---|---|---|
| GLU310 | A | hydrogen bond | 2.91 |
| SER258 | A | hydrogen bond | 3.52 |
| GLU310 | A | hydrogen bond | 2.94 |
| ASN424 | A | hydrogen bond | 3.38 |
| ARG313 | A | hydrogen bond | 3.17 |
| ARG313 | A | hydrogen bond | 3.35 |
| SER258 | A | hydrogen bond | 3.76 |
| ARG309 | A | salt bridge | 4.14 |
| ARG313 | A | salt bridge | 4.39 |

*All statements above describe the geometry of the docked/generated pose only (e.g. "residue X sits within Y A of the ligand, consistent with a hydrogen bond") -- no claim of biological/physiological effect is made or implied.*

## G.3 Synthesis Blueprint

**No concrete retrosynthetic route is included** -- AiZynthFinder was checked and found incompatible with this environment (pins numpy<2.0 and rdkit<2024, both conflicting with the validated diffusion/guidance stack); see blueprint_report.py module docstring. The figures below are feasibility estimates only.

- **RA score:** 0.058 (real RA-score model)
- **SA score:** 0.6 (fragment familiarity: 0.982, 27 atoms, 7 chiral centers, 0 spiro atoms, 0 bridgeheads, 0 macrocycles)

- **Lipinski Ro5 (structural flag, Lipinski et al. 1997):** 1 violation(s)
  - ✓ Molecular weight < 500 Da
  - ✓ H-bond donors <= 5
  - ✓ H-bond acceptors <= 10
  - ✗ LogP between -2 and 5
  - ✓ Rotatable bonds <= 10
- **PAINS filter (structural flag, Baell & Holloway 2010):** no alerts
