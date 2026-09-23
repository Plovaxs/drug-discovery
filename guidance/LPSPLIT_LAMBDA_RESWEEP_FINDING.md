# Re-Testing Guidance with the Stage 0-Corrected EGNN: Lambda Re-Sweep

Following the user's decision after Stage 0's checkpoint (see
`FLAGSHIP_ARCHITECTURE_RESULTS.md`), the 5-stage architecture build was
paused to first ask the question the whole investigation ultimately
serves: **does affinity guidance actually help, now that the affinity
model itself is demonstrably much better** (test R^2 0.342 vs the old
split's -0.537, now clearly beating a ligand-only baseline)?

## Method

Same single-pocket lambda-sweep protocol used throughout this project
(`guidance/lambda_sweep.py`, bundled example pocket, n=8 samples/point,
full 1000 diffusion steps, real Vina Dock scoring at exhaustiveness=8),
with `--affinity_ckpt` pointed at the new
`guidance_models/affinity_egnn_lpsplit.pt` (Stage 0's leakage-safe-split-
retrained EGNN) instead of the original deployed checkpoint. Per the
addendum's own instruction, the old model's chosen operating point
(lambda=1.0) and lambda-scale intuitions were **not** assumed to carry
over -- a wide grid spanning three orders of magnitude was swept from
scratch: `[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]`.

## Results

| lambda | validity | single-fragment rate | mean own-score | mean Vina Dock (kcal/mol) | std | n docked |
|---|---|---|---|---|---|---|
| 0.0 (unguided) | 1.0 | 1.0 | 7.891 | -10.500 | 1.411 | 8 |
| 0.1  | 1.0 | 1.0 | 7.904 | -10.370 | 1.562 | 8 |
| 0.3  | 1.0 | 1.0 | 7.904 | -10.427 | 1.490 | 8 |
| 1.0  | 1.0 | 1.0 | 7.904 | -10.503 | 1.353 | 8 |
| 3.0  | 1.0 | 1.0 | 7.901 | -10.689 | 1.438 | 8 |
| 10.0 | 1.0 | 1.0 | 7.893 | -10.495 | 1.115 | 8 |
| 30.0 | 1.0 | 1.0 | 7.914 | -10.503 | 1.393 | 8 |
| 100.0 | 1.0 | **0.875** | 7.993 | -10.715 | 0.834 | 7 |

**Structural robustness is markedly better than the old model**:
fragmentation barely appears even at lambda=100 (1 of 8 samples, vs the
old model's 50% fragmentation already at lambda=30 -- see
`AFFINITY_LAMBDA_RESWEEP_FINDING.md`). This alone is a meaningful,
positive side-effect of Stage 0's split fix, independent of whether
guidance actually improves affinity.

Significance test (Mann-Whitney U, two-sided, unpaired, n=8 per group,
lambda=0.0 unguided as reference):

| Comparison | mean diff (kcal/mol) | p-value |
|---|---|---|
| 0.0 vs 0.1  | +0.130 | 0.798 |
| 0.0 vs 0.3  | +0.072 | 0.878 |
| 0.0 vs 1.0  | -0.003 | 0.959 |
| 0.0 vs 3.0  | -0.189 | 0.878 |
| 0.0 vs 10.0 | +0.005 | 0.878 |
| 0.0 vs 30.0 | -0.003 | 1.000 |
| 0.0 vs 100.0 | -0.215 | 0.694 |

The guidance model's own predicted score creeps up very slightly across
the grid (7.89 at lambda=0.1 to 7.99 at lambda=100, ~1.3% over a 1000x
range of guidance strength) -- an even smaller, less directional drift
than the old model showed. Real Vina Dock score shows **no dose-response
relationship whatsoever**: every comparison against the unguided
baseline has p > 0.69, with differences bouncing around zero and no
consistent sign across three orders of magnitude of guidance strength.

## Conclusion

**A substantially improved, leakage-safe-split-retrained affinity model
(test R^2 0.342, tight CI, clearly outperforming a ligand-only baseline
as a standalone predictor) still produces no statistically detectable
effect on real Vina Dock scores when used as a diffusion guidance
signal, at any tested lambda spanning three orders of magnitude.**

This is a clean, well-powered null result at the single-pocket level,
directly answering Stage 5's final checkpoint question ("does the new
guidance model produce a statistically detectable effect on real Vina
Dock scores?") in the negative -- reported plainly, per this project's
standing no-overstatement rule, rather than reframed to fit an
expectation that guidance should now work.

**What this changes and doesn't change:**
- It does NOT contradict Stage 0's positive finding about the affinity
  model's *standalone* predictive quality -- that result (test R^2=0.342
  on 11,855 held-out entries) stands on its own and remains a genuine,
  statistically robust improvement over the old split's negative-R^2
  model.
- It DOES mean that fixing the affinity model's generalization gap was
  **necessary but not sufficient** for guidance to work. A model that
  predicts affinity well in a static, non-adversarial evaluation setting
  does not automatically produce a useful *gradient direction* for
  steering a diffusion sampler -- these are related but distinct
  properties, and this project's guidance design
  (`guidance/affinity_guidance.py`, position-only gradient through a
  frozen predictor, no composition gradient -- see that file's own
  documented scope-narrowing) may simply not be well-suited to
  translating a good static predictor into a useful sampling-time
  signal, independent of how good the predictor itself is.
- The improved structural robustness (fragmentation delayed from
  lambda=30 to lambda=100) suggests the new model's gradient field is
  smoother/smaller-scale than the old one's, consistent with it having
  learned less noisy, better-regularized representations -- but "safer to
  apply" and "useful to apply" are different claims, and this sweep shows
  the latter still doesn't hold.

## Multi-Pocket Confirmation (8 pockets, 1 seed, 3 representative lambdas)

Per the decision after this single-pocket result, a scoped confirmation
was run using the project's established Task-F infrastructure
(`guidance/run_task_f.py`, real per-pocket Vina docking at
exhaustiveness=16, not the lighter single-pocket-sweep setting): 8
diverse pockets (distinct protein families, drawn from the same 20-pocket
Task F pool), 1 seed, n=8 samples/pocket/lambda, at 3 representative
lambda values spanning the original range (0.3, 10.0, 100.0) plus the
shared lambda=0 baseline. 32 (pocket, variant) triples total, zero
failures.

### Per-pocket results (Mann-Whitney U vs. that pocket's own baseline)

| pocket | baseline mean | λ=0.3 mean (p) | λ=10 mean (p) | λ=100 mean (p) |
|---|---|---|---|---|
| BSD_ASPTE | -8.410 | -8.362 (0.798) | -8.256 (1.000) | -8.694 (0.281) |
| HDAC8_HUMAN | -6.164 | -6.152 (0.959) | -6.101 (1.000) | -6.465 (0.798) |
| CD38_HUMAN | -8.376 | -8.335 (0.878) | -8.358 (0.959) | -8.118 (0.382) |
| MENE_BACSU | -8.143 | -8.103 (0.959) | -8.170 (0.645) | -7.963 (0.959) |
| NQO1_HUMAN | -7.057 | -7.017 (0.721) | -6.999 (0.878) | -7.136 (0.645) |
| PAK4_HUMAN | -6.889 | -6.834 (1.000) | -6.878 (0.878) | -6.909 (0.878) |
| PNTM_STRAE | -5.282 | -5.308 (0.959) | -5.590 (0.645) | -5.552 (0.721) |
| RIBB_VIBCH | -6.545 | -6.517 (1.000) | -6.550 (0.721) | -6.227 (0.105) |

**All 24 per-pocket comparisons are non-significant** (p ranging
0.105-1.000) -- no pocket shows a real, isolated guidance effect at any
tested lambda.

### Pooled and paired aggregate tests

| lambda | pooled (n=64v64) mean diff | Mann-Whitney p | paired-by-pocket (n=8) Wilcoxon p |
|---|---|---|---|
| 0.3 | +0.030 | 0.952 | **0.023** |
| 10.0 | -0.005 | 0.826 | 0.547 |
| 100.0 | -0.000 | 0.921 | 0.742 |

**One nuance worth reporting honestly, not smoothing over**: the
paired-by-pocket Wilcoxon test at lambda=0.3 reaches nominal significance
(p=0.023) -- but this is driven by a *consistent, tiny-magnitude* effect
(7 of 8 pockets shift by +0.01 to +0.055 kcal/mol, i.e. very slightly
*worse* predicted binding under guidance, not better), not a real,
practically meaningful shift. The pooled mean difference (+0.030
kcal/mol) is an order of magnitude below Vina Dock's typical
per-molecule noise (std ~1.1-1.6 kcal/mol throughout this project's
sweeps) and in the wrong direction to support "guidance helps." With 6
primary significance tests run across this confirmation (3 lambdas x 2
test types), encountering one nominal p<0.05 by chance is not surprising
and does not clear the bar for a real finding on its own -- reported here
for completeness and honesty, not as evidence guidance works, and
explicitly NOT swept aside just because it complicates the clean-null
narrative. lambda=10 and lambda=100 show no such pattern at all (p=0.547,
0.742), reinforcing that this is very unlikely to reflect a stable,
lambda-dependent phenomenon.

### Conclusion of the confirmation

**The single-pocket null result holds consistently across a diverse
8-pocket sample.** No pocket, and no tested lambda, shows a real,
practically meaningful guidance effect on real Vina Dock score. The one
nominally-significant paired test (lambda=0.3) reflects a negligible,
wrong-directioned effect, not a genuine positive signal, and does not
change this conclusion.

## Final Conclusion

The user chose to run the scoped multi-pocket confirmation rather than
close the question on the single-pocket result alone. It confirms that
result cleanly: **across 8 diverse pockets and 3 representative lambda
values (0.3, 10.0, 100.0), a substantially improved, leakage-safe-split-
retrained affinity model still produces no statistically meaningful
guidance effect on real Vina Dock score.** This answers Stage 5's final
checkpoint question definitively, not just at the single-pocket level:
fixing the affinity model's generalization gap (Stage 0) was necessary
but not sufficient for affinity guidance to work in this project's
current guidance design (position-only gradient through a frozen
predictor -- see `guidance/affinity_guidance.py`'s documented scope).

This closes the "re-test guidance first" branch chosen after Stage 0's
checkpoint. Per the addendum's own routing logic, since the null result
is now confirmed rather than merely single-pocket-suggestive, the
remaining strategic choice (continue to Stage 2+ of the flagship
architecture, or treat this as the thesis's final honest answer on
affinity guidance) is a decision for the user, informed by this
confirmed result plus Stage 0's real, positive, independently valuable
finding about split-construction artifacts in affinity model evaluation.

## Reproducibility

- Sweep code (checkpoint-selectable via `--affinity_ckpt`): `guidance/lambda_sweep.py`
- Raw results (incl. per-molecule Vina Dock scores): `guidance/lpsplit_lambda_resweep.json`
- Full run log: `guidance/lpsplit_lambda_resweep.log`
- Multi-pocket confirmation pockets: `guidance/lpsplit_confirmation_pockets.json`
- Multi-pocket confirmation raw results (per-pocket honest_eval.csv, checkpointed): `guidance/lpsplit_confirmation_results/{lambda_0.3,lambda_10.0,lambda_100.0}/{baseline,affinity_only}/pocket<id>_seed1/`
- Multi-pocket confirmation run log: `guidance/lpsplit_confirmation.log`
- Guidance checkpoint used: `guidance_models/affinity_egnn_lpsplit.pt` (copy of `logs_lp_split_stage0/.../checkpoints/best.pt`)
- Old-model comparison sweep: `guidance/AFFINITY_LAMBDA_RESWEEP_FINDING.md`
