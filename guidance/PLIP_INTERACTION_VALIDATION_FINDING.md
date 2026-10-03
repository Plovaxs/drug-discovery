# PLIP Interaction-Type Validation: No Confirmed Effect, Including a Promising-Looking Signal That Did Not Replicate

Phase 2's first closely-connected new branch (`guidance/plip_interaction_validation.py`):
rather than asking only "did Vina Dock improve," this asks whether guidance
changes the number of real, PLIP-detected interactions (hydrogen bonds,
hydrophobic contacts, pi-stacking, salt bridges) a docked pose actually
forms. Not Track E (`guidance/TRACK_E_DESIGN.md`, a separately-gated
~38 GPU-hour training plan) — this is analysis only, on the ESM2
checkpoint already trained and real-docking-tested
(`ESM2_GUIDANCE_FINDING.md`).

## Method

Same fixed pocket (SQHC_ALIAD_1_631_0 / PDB 1h36) and sampling/docking
pipeline as every other real-docking test in this investigation
(`guidance/lambda_sweep.py`'s `load_everything`/`VinaDockingTask`), with
one new step: each Vina-docked pose is converted (OpenBabel PDBQT->PDB)
and merged with the receptor into a PLIP complex, and interaction counts
are tallied per molecule. Getting this to work required finding and
fixing a real bug: the receptor PDB file already ends in its own "END"
record, which silently truncated every ligand HETATM line appended after
it, making PLIP report zero ligands for every pose until fixed (see the
script's own comment at the fix site).

## Exploratory tier (n=8): a seemingly interesting dissociation

At lambda=3.0, Vina Dock improved by a mean of -2.27 kcal/mol (raw
p=0.031) while PLIP-detected interactions *decreased* by a mean of -3.71
(p=0.094, all four interaction types falling together, not driven by one
category) -- correlation between per-molecule delta-Vina and
delta-interactions was near zero (-0.02 to -0.13). The surface reading:
guidance might be improving the raw score without forming more real
interactions, i.e. a "chemically hollow" score gain.

Per this project's standing protocol -- demonstrated repeatedly, most
directly with the soft-v signal (`guidance/lambda_sweep_vinatarget_softv_highlam_results.json`
vs `..._confirm_results.json`) -- no encouraging small-n signal in this
investigation has ever been reported as a finding without a confirmatory
run at larger n first. This one was not either.

## Confirmatory tier (n=16, new seed): the signal reversed, not just weakened

| | n=8 (exploratory) | n=16 (confirmatory) |
|---|---|---|
| Interactions, mean diff | -3.71 | **+1.00** (reversed sign) |
| Interactions, Wilcoxon p | 0.094 | 0.131 |
| Vina Dock, mean diff | -2.27 | -1.02 (weaker) |
| Vina Dock, Wilcoxon p | 0.031 | 0.077 (no longer <0.05) |
| corr(delta-Vina, delta-interactions) | -0.02 to -0.13 | -0.32 (still negative, still n.s.) |

The interaction-count effect did not merely shrink (the usual pattern for
a noise-inflated small-n estimate) -- it flipped sign, going from "fewer
interactions under guidance" to "slightly more." This is strong evidence
the n=8 result was noise, not a real chemically-hollow-score-gain
phenomenon, and it is reported as such rather than as a qualified
"direction worth watching."

## Comparison checkpoint: gradient-alignment (opposite diagnostic profile, same null)

Repeated on the gradient-alignment checkpoint (the strongest DIAG1 fix,
and the flattest real-docking result of any checkpoint tested --
`GRADALIGN_GUIDANCE_FINDING.md`), n=8, lambda in {1.0, 10.0}
(`guidance/plip_validation_gradalign_results.json`): both the raw score
and the interaction counts are flat and non-significant at both lambdas
(interactions: mean diff +0.38/p=0.594 at lambda=1, -0.88/p=0.281 at
lambda=10; Vina Dock: mean diff -0.099/p=0.844 at lambda=1, +0.427/p=0.078
at lambda=10, the latter in the wrong direction). No encouraging signal
at either lambda, so no confirmatory tier was needed here (unlike ESM2's
exploratory tier, which did look encouraging before failing to
replicate). This is the same clean-null pattern this checkpoint showed
on raw Vina Dock, now also holding at the chemistry level -- the two
checkpoints with opposite diagnostic profiles (best gradient-distance
fix vs. best predictive quality) agree on this outcome.

## Conclusion

No confirmed effect, in either direction, on PLIP-detected interaction
counts under guidance, on either checkpoint tested. Combined with each
checkpoint's own real-docking result, this specifically rules out one
plausible alternative explanation for the overall null pattern across
this investigation: it is not the case that Vina Dock is a misleading
proxy that is hiding a real chemistry-level improvement PLIP would
catch. Neither the raw score nor the underlying chemistry shows a
defensible guidance effect, on the checkpoint that came closest on raw
score (ESM2) or the one with the strongest diagnostic fix
(gradient-alignment).

This result is not yet pooled into `FOLLOWUP_PHASE_POOLED_CORRECTION.md`
(a different outcome variable -- interaction counts, not Vina Dock --
pooling it with the Vina Dock-based tests there would mix two different
measured quantities under one correction; it is reported on its own
terms here instead).

## Artifacts

- `guidance/plip_interaction_validation.py` -- analysis script (contains the END-record-truncation bug fix).
- `guidance/plip_validation_esm2_results.json` -- exploratory tier (n=8).
- `guidance/plip_validation_esm2_confirm_results.json` -- confirmatory tier (n=16).
