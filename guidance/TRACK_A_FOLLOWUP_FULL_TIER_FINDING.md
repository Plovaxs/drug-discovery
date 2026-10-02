# Track A Follow-up Full-Tier Finding: Noise-Matching + Vina-Target on GIGN+PIGNet2 — Still a Clean Null

This closes the loop on `guidance/track_a/train_gign_pignet_full_followup.py`
(trained during this investigation's follow-up phase, see
`guidance/STAGE2_PLUS_EXPERIMENT_LOG.md` context): the two interventions
that gave the best diagnostic profile on the EGNN backbone (noise-matched
ligand-position curriculum + Vina-derived pK anchor, see
`guidance/lp_split/train_egnn_stage0_noisematched.py` and
`train_egnn_stage0_vinatarget.py`'s module docstrings) were applied
together to Track A's physics-anchored GIGN+PIGNet2 architecture, to test
whether the null-guidance pattern established for the original Track A
checkpoint (`TRACK_A_FULL_TIER_FINDING.md`, 0/24 significant after BH
correction) is specific to how that checkpoint was trained, or general
across training recipes on the same architecture.

The raw per-pocket statistics (`guidance/track_a_followup_full_results/
gradient_informativeness_results.csv`) were computed at the time but never
written up into a finding document — this fills that gap.

## Setup

Model: GIGN+PIGNet2, trained with noise-matched ligand-position curriculum
(t ~ Uniform(0, 500), same diffusion noise schedule as TargetDiff) and a
Vina-derived pK anchor instead of experimental pK (epoch 43, test
R²=0.158, Pearson=0.801 — see checkpoint at
`logs_track_a_followup/gign_pignet_2026_09_30__07_24_48_noisematched_vinatarget/checkpoints/best.pt`).

Checkpoint test: 8 pockets (`guidance/lpsplit_confirmation_pockets.json`,
the same diverse set used throughout this phase), unguided baseline + 2
lambdas (0.1, 0.3), **n=30 molecules/condition**, real Vina Dock
(exhaustiveness=8) + PoseBusters — same protocol and same pocket set as
the original Track A full-tier test, so this is a direct like-for-like
comparison of training recipe, not architecture or evaluation protocol.

## Result: 0/16 comparisons significant after BH correction

8 pockets × 2 lambdas = 16 KS tests, BH-corrected together.
**Minimum raw p-value: 0.031** (CD38_HUMAN:λ=0.3); **minimum BH-corrected
p-value: 0.489** — nothing survives correction, same conclusion as the
original checkpoint.

| | Original Track A (full-tier) | Follow-up (noise-matched + vina-target) |
|---|---|---|
| Comparisons | 24 (8 pockets × 3 lambdas) | 16 (8 pockets × 2 lambdas) |
| Significant after BH | 0/24 | 0/16 |
| Min raw p | 0.0065 | 0.031 |
| Min BH-corrected p | 0.157 | 0.489 |
| Correct-direction rate | 16/24 (67%) | **14/16 (87.5%)** |
| PoseBusters degradation | 100% at λ=1.0 | 9/16 (56%) at λ∈{0.1,0.3} |

### The direction-consistency number is numerically the best of anything tried on this architecture — and it still does not matter

14/16 comparisons (87.5%) point the "correct" way (more negative/better
Vina Dock under guidance) — a higher rate than the original Track A
checkpoint's 67%, and higher than most other follow-up branches reported
in `guidance/followup_figures/meta_summary_all_routes.png`. Reported
plainly rather than either hidden or oversold: **this is the strongest
raw directional signal found anywhere in this investigation's Track A
lineage, and it still produces zero significant effects after correction,
with real PoseBusters degradation in over half the tested conditions.**
This is exactly the pattern the meta-analysis across all follow-up
branches already flagged as the single most important finding: fixing a
diagnosed mechanism issue (here: two issues stacked) moves surface-level
numbers (direction-consistency, diagnostic correlations, predictive R²)
without the real, independently-measured docking outcome ever clearing
the bar a defensible claim requires.

### PoseBusters degradation is less universal than the original checkpoint, but still real

9/16 (56%) of conditions show PoseBusters validity degradation, concentrated
more at λ=0.3 (6/8 pockets) than λ=0.1 (3/8 pockets) — a weaker dose-response
than the original checkpoint's clean 12.5%→37.5%→100% progression (that
checkpoint was tested up to λ=1.0; this one was not, since the module's
own prelambda check found this recipe's stable range narrower — see the
follow-up training script's docstring). The comparison is therefore
partial: this result cannot rule out the original checkpoint's
λ=1.0-equivalent catastrophic degradation also occurring here at a lambda
that was not tested.

## Why this strengthens rather than merely repeats the existing conclusion

`DUAL_FALSIFICATION_CONCLUSION.md` and `THESIS_FINDINGS_SYNTHESIS.md`
establish that Track A's architecture, trained the original way, fails to
show a guidance effect. A live, open question after that synthesis was
whether a *different, diagnostically-motivated training recipe* on the
*same* architecture might behave differently — since Track A had not yet
been retrained with the fixes that helped (on paper) the EGNN backbone's
diagnostic profile. This result closes that specific gap: **no, applying
the two best-performing diagnostic fixes to Track A's architecture does
not produce a detectable guidance effect either.** Combined with
`guidance/lp_split/train_egnn_stage0_gradalign.py`'s result (gradient-
alignment loss directly minimizing DIAG1's diagnosed correlation on the
EGNN backbone, align_corr -0.04→-0.75, yet — pending its own real-docking
test, see Pending Work below — showing the same pattern of diagnostic-only
improvement in every prior case) and the noise-matched/vina-target EGNN
checkpoints' own real-docking nulls (computed in this same session, see
below), the pattern now holds across: 2 architectures (EGNN, GIGN+PIGNet2)
× up to 4 distinct mechanism-level interventions (noise-matching,
Vina-target, gradient-alignment, soft-v relaxation), independently.

## Companion result: EGNN noise-matched and vina-target checkpoints, real-docking re-check

The EGNN backbone's noise-matched and vina-target checkpoints
(`guidance/lambda_sweep_noisematched_results.json`,
`guidance/lambda_sweep_vinatarget_results.json`, n=8 molecules/condition,
3 nonzero lambdas each, paired-by-seed across lambda) had never been
statistically tested beyond the exploratory raw numbers. A paired
Wilcoxon signed-rank test (appropriate here since the same 8 seeds are
reused across lambda conditions) between each nonzero lambda and λ=0
gives, for both checkpoints, **no p-value below 0.10** (noise-matched:
p=0.156/0.250/0.312 at λ=0.1/1.0/10.0; vina-target: p=0.312/0.102/0.945 at
λ=0.1/1.0/10.0), with mean Vina Dock shifts of at most 0.14 kcal/mol in
either direction — noise, not signal, at this sample size. This is
additional, not previously reported, confirmation of the same pattern on
the EGNN backbone specifically, independent of the GIGN+PIGNet2 result
above.

## Pending work this finding does NOT cover

- Gradient-alignment's own real-docking test has not yet been run (the
  checkpoint only just finished training in this session — best epoch 11,
  test R²=0.345, Pearson=0.626). Per this document's own logic, a
  diagnostic improvement (align_corr -0.75, the strongest yet) predicting
  a real effect would be the exception, not the rule, established by every
  other checkpoint tested so far — but it has not been directly checked
  and should not be assumed null by analogy alone.
- The ESM2-augmented EGNN checkpoint (training in progress at the time of
  writing) is a different kind of intervention (adds a new global feature
  rather than changing the training curriculum/target/loss) and is
  likewise untested for real docking effect.

## Conclusion

Adding this checkpoint, the total count of independently-trained
guidance checkpoints showing zero significant, correctly-directed,
non-degraded real-docking effect after multiple-comparison correction is
now (at minimum): Stage 0 EGNN original, Track A original GIGN+PIGNet2,
B1/B2/B3 mechanism variants, Track C ranker, Track D synthesizability,
EGNN noise-matched, EGNN vina-target, GIGN+PIGNet2 noise-matched+vina-
target — **9 independently-trained models, 3 distinct architectures, 2
guidance mechanisms (gradient steering, rejection sampling), all null.**
