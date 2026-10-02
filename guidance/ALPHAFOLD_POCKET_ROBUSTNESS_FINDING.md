# AlphaFold-vs-Crystal Pocket Robustness: Full-Scale Result (n=835)

Supersedes the preliminary n=40 estimate (limited to the ~46 LP-split
targets that happened to have a raw PDB in `./data/test_set`'s small demo
subset) now that the full raw CrossDocked2020 v1.1 archive (43GB,
`https://bits.csb.pitt.edu/files/crossdock2020/v1.1/CrossDocked2020_v1.1.tgz`)
has been downloaded and extracted, giving 971/971 LP-split targets a local
raw receptor PDB. See `guidance/alphafold_pocket_robustness.py`'s module
docstring for the full method and disclosed caveats (precomputed AlphaFold
DB models, not fresh inference; pocket defined as receptor Calpha atoms
within 10A of any docked-ligand atom; residue correspondence established
by real pairwise sequence alignment, not assumed numbering).

Getting from n=40 to n=835 required two real fixes to the original script,
not just pointing it at more data: (1) the full raw archive pools every
historical receptor conformation and ligand pose for a target into one
directory (e.g. 111 receptor PDBs, 40 ligand SDFs for a single target) --
picking "the first .pdb/.sdf found" would have silently paired mismatched
receptor/ligand files, so the exact (receptor, ligand-pose) pair now comes
from the LP-split's own dataset index (`ligand_filename`), not directory
guessing; (2) the archive bundles multiple docked poses into one gzipped
multi-record SDF per receptor-ligand pair rather than one file per pose,
requiring the specific pose index to be extracted from the bundle rather
than read as a standalone file.

## Coverage

835/971 targets (86%) fully processed. Failure breakdown: 62 pocket too
small (<3 residues within 10A), 35 no AlphaFold DB model for that
UniProt accession, 14 no UniProt match at all, 10 low sequence-alignment
identity (<0.9, receptor chain doesn't confidently match the cached
UniProt sequence), 10 too few matched pocket residues after alignment, 4
parsing errors, 1 unparseable ligand record.

## Result: strong overall agreement, with a real and now well-characterized minority tail

| | n=40 (preliminary) | n=835 (full) |
|---|---|---|
| Median pocket RMSD | 0.42 A | **0.473 A** |
| Mean pocket RMSD | 1.10 A | 0.88 A |
| Pearson r(RMSD, pocket pLDDT) | -0.75 (p=2.5e-8) | **-0.628 (p=1.2e-92)** |
| Spearman r(RMSD, pocket pLDDT) | -0.80 (p=6.8e-10) | **-0.686 (p=5.5e-117)** |
| Pocket pLDDT vs global pLDDT | +5.4 | **+7.4** |
| Targets with RMSD > 2A | 4/40 (10%) | **71/835 (8.5%)** |

The headline numbers are remarkably stable going from n=40 to n=835 (median
RMSD 0.42->0.47 A, the >2A tail 10%->8.5%), which is itself informative:
the n=40 estimate, despite its small size, was not a fluke of a biased
small sample -- it is confirmed at a dramatically larger, astronomically
more significant scale (p-values now in the e-92 to e-117 range).

## The overconfidence failure mode, now large enough to characterize

At n=835, 18 targets exceed 5A pocket RMSD. Of those, **13/835 (1.6%)**
have a pocket-local pLDDT above 80 -- AlphaFold's own "confident"
threshold -- while still being substantially wrong (5.3-12.5 A) at the
binding site specifically:

| Target | Pocket RMSD (A) | Pocket pLDDT |
|---|---|---|
| RNAS2_HUMAN_27_161_0 | 7.9 | **96.4** (AlphaFold's own "very high" tier) |
| KDM5A_HUMAN_175_588_0 | 7.3 | 93.4 |
| KS6B1_HUMAN_76_373_0 | 5.3 | 91.4 |
| GCKR_HUMAN_1_625_0 | 12.3 | 91.7 |
| MDM2_HUMAN_16_114_0 | 6.6 | 90.6 |
| OPRM_MOUSE_52_354_0 | 7.5 | 89.5 |
| PDE4B_HUMAN_324_663_0 | 8.0 | 87.8 |
| MK09_HUMAN_4_363_0 | 5.4 | 87.5 |
| AKT1_HUMAN_1_137_0 | 9.2 | 85.8 |
| ... (4 more, 81-83 pLDDT) | 5.3-8.7 | 81-83 |

This is not a small-sample artifact (AKT1 was the single such case flagged
at n=40; RNAS2's 96.4 pLDDT / 7.9A RMSD is, if anything, a more striking
example of the same failure mode). A pragmatic takeaway for anyone
considering using an AlphaFold-predicted structure as a substitute pocket
input (e.g. for a target with no experimental structure): the overall
correlation between pLDDT and accuracy is strong and highly significant,
but roughly 1 in 64 "confident" predictions in this sample is
substantially wrong specifically at the binding site -- pLDDT is a good
average-case guide, not a guarantee, and this failure mode cannot be
detected from the pLDDT score alone without an independent check (e.g. the
kind of experimental-structure comparison this analysis itself performs,
which is of course unavailable for a genuinely novel target).

## Scope note unchanged from the n=40 version

This measures a precomputed AlphaFold model's agreement with reality for
already-solved, well-studied proteins (every CrossDocked2020 target was
crystallized) -- a best-case proxy for the truly-novel-target scenario a
real generalization application would face, not a direct measurement of
it.

## Artifacts

- `guidance/alphafold_pocket_robustness.py` -- analysis script (now fixed
  for the full-archive file layout).
- `guidance/alphafold_pocket_robustness_results.json` -- full per-target
  results, 971 entries.
- `guidance/lp_split/alphafold_cache.json`, `alphafold_structures/` --
  cached AlphaFold DB records and downloaded structures (920 UniProt
  accessions).
