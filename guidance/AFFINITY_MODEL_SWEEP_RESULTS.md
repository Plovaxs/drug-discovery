# Affinity Model Hyperparameter Sweep (Part 3.3, Full Epoch Budget)

Follows `AFFINITY_MODEL_LEAKAGE_CHECK.md` (no leakage found) and
`AFFINITY_MODEL_STANDALONE_QUALITY.md` (deployed model's test R^2 is
negative at every epoch). This sweep asks: is that generalization gap
fixable with a light LR/architecture search, or does it persist across
reasonable hyperparameter choices?

## Setup

4 configs, full 20-epoch budget each (early-stop patience 5), identical
data/split/seed to the deployed model -- only LR and one architecture knob
(EGNN depth) varied:

| Config | LR | EGNN layers | Wall time |
|---|---|---|---|
| baseline (deployed) | 1e-4 | 6 | (already trained) |
| lr_3e-5 | 3e-5 | 6 | 215.7 min |
| lr_3e-4 | 3e-4 | 6 | 211.1 min |
| arch_4layers | 1e-4 | 4 | 148.0 min |

All other hyperparameters (batch_size=1, pos_noise_std=0.1, Adam
beta1/beta2, plateau scheduler) held fixed at the deployed model's values.
Checkpoint selection is by best *validation* loss in all four runs,
identical to the deployed model's own selection procedure.

## Results

| Config | Val Pearson | Val R^2 | **Test Pearson** | **Test R^2** | Test RMSE |
|---|---|---|---|---|---|
| baseline (lr=1e-4, 6L) | 0.788 | 0.616 | 0.233 | -0.537 | 2.250 |
| **lr=3e-5, 6L** | 0.803 | 0.628 | **0.538** | **0.046** | **1.772** |
| lr=3e-4, 6L | 0.727 | 0.523 | 0.272 | -0.442 | 2.179 |
| lr=1e-4, 4L (arch) | 0.806 | 0.641 | 0.119 | -0.671 | 2.346 |

(Val n=4,061; Test n=27 in every row -- see the small-n caveat in
Diagnostic 4 below.)

## Interpretation

**Lowering the learning rate to 3e-5 produces the only config with
positive test R^2** (0.046) and by far the best test Pearson (0.538 vs.
0.119-0.272 for every other config) -- a genuine, meaningful improvement
over the deployed model's -0.537 R^2 / 0.233 Pearson. This is a real,
fixable-by-cheap-tuning component of the original gap: the deployed
model's LR (1e-4) was too aggressive for this small, thin (n=6,000,
~12 examples/target) dataset, likely overfitting to per-target train-set
idiosyncrasies faster than it could learn transferable structure.

**However, the gap is not closed, only narrowed.** R^2=0.046 is barely
above zero -- the best config explains almost none of the test-set
variance in real terms, even though it is a clear ranking improvement over
every other config tested. Pearson=0.538 is a moderate, not strong,
correlation for a quantity (binding affinity) with real decimal-place
consequences downstream.

**Architecture depth does not help, and by this one data point actively
hurts**: the 4-layer variant has the *best* validation performance of all
four configs (Pearson 0.806, R^2 0.641) but the *worst* test performance
(Pearson 0.119, R^2 -0.671). This is a striking, concrete illustration of
the val/test disconnect already flagged in
`AFFINITY_MODEL_STANDALONE_QUALITY.md`: validation loss, computed on
data sharing targets with train, is not a reliable signal for selecting
a model that will generalize to genuinely new targets -- optimizing
against it can select instead for a model that has additionally
overfit to shared-target patterns.

## Diagnostic 2: per-target error breakdown

The 27-pocket test set spans 26 unique targets (one target has 2 pockets)
-- **every group is a singleton**, so no per-family or per-target Pearson
correlation is computable at all; only point-wise absolute error is
reportable per target. Per the addendum's own instruction to flag
groups too small for correlation (<5), that applies to **all 26/26
groups here** -- this diagnostic is genuinely inconclusive at this
granularity given the test set's size, not by any choice made in this
analysis. Errors range from 1.7 to 3.4+ pK units across targets with no
extractable structure at n=1 per group (top absolute errors:
`F16P1_HUMAN` 3.44, `BGL07_ORYSJ` 3.37, `CD38_HUMAN` 3.29 -- see
`guidance/affinity_diagnostics_results.json` for the full list).

## Diagnostic 3: ligand-only baseline (ignores the protein pocket entirely)

An ECFP4 (Morgan, r=2, 1024 bits) fingerprint of the ligand ALONE -- no
protein/pocket information at all -- fed to a plain Random Forest
regressor, trained and tested on the exact same leakage-safe split:

| Model | Test Pearson | Test R^2 |
|---|---|---|
| Ridge (ligand-only) | 0.427 | -0.769 |
| **Random Forest (ligand-only)** | **0.493** | **0.047** |
| EGNN best config (lr=3e-5, uses full pocket) | 0.538 | 0.046 |

**A model that never sees the protein pocket at all performs
statistically indistinguishably from the best full pocket-aware EGNN
config** (R^2 0.047 vs 0.046; Pearson 0.493 vs 0.538, well within each
other's bootstrap CI per Diagnostic 4 below). This is the single most
important diagnostic result: it indicates the generalization gap is
**not specific to the EGNN's 3D pocket representation** -- a
representation that discards the pocket entirely reaches the same
ballpark of test performance on this split. The bottleneck is much more
consistent with data scarcity/diversity in the labeled training set
(507 targets, ~12 examples/target) than with anything wrong in the EGNN's
architecture specifically.

## Diagnostic 4: bootstrap 95% confidence intervals (n=27 test pockets)

| Config | Pearson (95% CI) | R^2 (95% CI) |
|---|---|---|
| Best (lr=3e-5) | 0.538 (0.230, 0.775) | 0.046 (-0.601, 0.445) |
| Deployed baseline (lr=1e-4) | 0.233 (-0.145, 0.595) | -0.537 (-1.769, 0.183) |

**Every R^2 confidence interval crosses zero substantially** -- with only
27 test pockets, "R^2 is positive" for the best config is not a claim
that survives resampling uncertainty; the honest statement is "R^2 is
not distinguishable from zero, and is somewhat more likely positive than
the deployed baseline's, whose CI leans further negative." The Pearson
CIs are more informative: the best config's CI (0.23-0.78) excludes zero,
while the deployed baseline's CI (-0.14-0.60) does not -- so the *ranking*
improvement from lowering the learning rate is better supported than the
absolute R^2 value, but neither should be reported as a precise number
given n=27.

## Diagnostic 5: what is the model actually keying on?

**5a -- superficial-feature shortcut check:** predicted affinity
correlates strongly with ligand heavy-atom count alone: **Pearson =
0.799 (p<0.001, n=27)**. This is a large, hard-to-ignore effect --
structurally the same pattern as the Vina-hacking concern documented
elsewhere in this project (`eval/honest_eval.py`'s ligand-efficiency
check, `guidance/TASK_F_RESULTS.md`'s Vina-hacking section): "bigger
ligand -> higher predicted affinity" as a shortcut, rather than
pocket-specific binding discrimination. Combined with Diagnostic 3, this
paints a consistent picture: much of what the model predicts is
explainable by ligand-side properties alone, not by genuine
protein-ligand interaction modeling.

**5b -- pocket-scrambling sensitivity check:** feeding the model a
ligand paired with the *wrong* (mismatched) pocket, across 20 sampled
pairs, changes the prediction by a mean of 1.49 pK units -- comparable
to the overall standard deviation of predictions across the real test
set (1.77). This means the model is **not literally ignoring the
pocket** (a pocket-blind model would show ~zero change under scrambling)
-- it is sensitive to which pocket is given. But this sensitivity, per
Diagnostic 3, does not translate into pocket information adding
predictive value beyond what the ligand alone already provides on this
test set -- consistent with a model that reacts to the pocket input
without having learned a reliable, generalizable pocket-affinity
relationship from only ~12 examples per target.

## Diagnostic 1: training-size learning curve (does more target diversity help?)

The direct, causal test of the "data scarcity" hypothesis raised by
Diagnostic 3: retrain the best config (lr=3e-5, 6 layers) at 25%, 50%,
and 75% of the 507-target training pool (127/254/380 targets
respectively, examples capped proportionally so target count is the sole
manipulated variable -- see `guidance/target_subsample_splits.py`), full
20-epoch budget each, same fixed 27-pocket test set throughout. The 100%
point reuses the main sweep's already-completed lr=3e-5 run.

| Fraction | n targets | Test Pearson (95% CI) | Test R^2 (95% CI) |
|---|---|---|---|
| 25% | 127 | 0.298 (-0.044, 0.638) | -0.519 (-1.594, 0.068) |
| 50% | 254 | 0.305 (-0.041, 0.634) | -0.348 (-1.252, 0.203) |
| 75% | 380 | 0.296 (-0.067, 0.649) | -0.573 (-1.696, 0.060) |
| 100% | 507 | 0.538 (0.230, 0.775) | 0.046 (-0.601, 0.445) |

**This is a genuinely ambiguous result, reported honestly as such rather
than forced into either of the two clean patterns anticipated.** The
point estimates show no climb at all across 25%-75% (0.296-0.305,
essentially flat/noisy) and then an apparent jump at 100% (0.538). If
taken at face value, this would look like "no benefit until the full
dataset," an unusual, hard-to-explain pattern on its own. But the
bootstrap CIs tell a more cautious story: **every one of the four CIs
overlaps substantially with its neighbors** -- the 25/50/75% points' CI
upper bounds (0.63-0.65) comfortably overlap the 100% point's CI lower
bound (0.23). With only 27 test pockets, **the apparent jump at 100% is
not statistically distinguishable from the same noise band the 25-75%
points already occupy.** This diagnostic cannot confidently confirm
"still climbing" (no CI-supported monotonic trend exists across the
four points) nor "plateaued at a low level" (the 100% point's higher
point estimate is still compatible with real signal, just not provably
so at this sample size). The honest conclusion is: **this test set is too
small to resolve whether target diversity helps within the range tested**
-- a limitation of test-set size, not evidence against the data-scarcity
hypothesis, but also not evidence for it beyond what Diagnostic 3 already
established correlationally.

## Synthesis: what explains the generalization gap?

Putting all five diagnostics together, the most defensible account is
**data scarcity/diversity relative to the target-transfer task is the
leading candidate explanation, correlationally well-supported but not
confirmed causally at this test-set size -- and the gap is demonstrably
not an EGNN-architecture-specific limitation.**

Each claim traced to its diagnostic: (1) the main sweep showed
architecture depth doesn't help -- the 4-layer variant had the *best*
validation performance and the *worst* test performance of all four
configs -- and the LR effect that does help (3e-5) is a real ranking
improvement but its absolute R^2 is not distinguishable from zero under
bootstrap resampling (Diagnostic 4); (2) a ligand-only baseline that
discards the pocket entirely matches the best EGNN's test performance
(Diagnostic 3, R^2 0.047 vs 0.046) -- direct evidence the gap is not
specific to the EGNN's 3D pocket representation; (3) the model's
predictions correlate strongly with a superficial ligand property,
heavy-atom count (Diagnostic 5a, Pearson 0.799), suggesting much of its
apparent signal is a size-based shortcut rather than learned
protein-ligand interaction chemistry; (4) the model is measurably
sensitive to pocket identity (Diagnostic 5b) without that sensitivity
producing a demonstrated predictive edge over ligand-only information
(Diagnostic 3); (5) Diagnostic 1's direct causal test of "more target
diversity would help" was **inconclusive** -- test Pearson was flat
across 25-75% of the target pool and only rose at 100%, but every
confidence interval across all four points overlaps substantially, so
this cannot be reported as a confirmed climbing trend, only as
compatible with one. Diagnostic 2 could not add further resolution (all
26 test targets are singletons).

**Honest bottom line**: the correlational evidence (Diagnostics 3 and
5a) points at data scarcity/superficial-feature reliance over an
architecture-specific flaw, and the sweep (Diagnostic 4-annotated) rules
out architecture depth as a fix. But the one diagnostic designed to
causally confirm "more target-diverse data would close the gap"
(Diagnostic 1) could not resolve the question either way at n=27 test
pockets -- this project cannot claim, with the evidence gathered, that
more data is a *proven* fix, only that it remains the most plausible
one, and that fixing it via architecture or learning-rate search on the
current small labeled set has already been tried and found insufficient.
**Concrete future work**: acquiring substantially more labeled affinity
examples (real experimental pKd/pKi/pIC50, not more of the same thin
per-target density) and re-running Diagnostic 1's exact learning-curve
protocol is the natural next step to actually settle this -- the
current investigation identifies the right experiment but does not have
enough held-out data to make it conclusive.

This also connects back to Task F's original null result
(`guidance/TASK_F_RESULTS.md`) and the lambda re-sweep
(`guidance/AFFINITY_LAMBDA_RESWEEP_FINDING.md`): a guidance signal built
on a predictor whose test-set behavior is substantially explainable by
ligand size alone, with no demonstrated net benefit from pocket
information over a ligand-only baseline, offers little reason to expect
it would reliably steer diffusion sampling toward pocket-specific higher
real binding affinity -- consistent with, and now mechanistically
supported by, the observed null/flat results in both of those
experiments.

## Investigation status: complete

This closes the generalization-gap investigation as scoped (Parts 1-4 of
the diagnostics addendum). Per the routing decision established
alongside Diagnostic 1: since the learning curve did not confirm a
closeable gap (Diagnostic 1's result is inconclusive, not positive), and
since even the best config found here (lr=3e-5) was never itself
re-tested as a guidance signal in a full Task-F-style ablation (that
would be a separate, substantial undertaking beyond this investigation's
scope), Task F's original null-result conclusion stands. No further
diagnostic branches are opened from this investigation; the natural
future-work directions are (a) more labeled affinity examples per
target, re-running Diagnostic 1's protocol to actually resolve the
climbing-vs-plateaued question, and (b) a guidance mechanism that does
not depend on point-prediction accuracy from a data-scarce affinity
model the way this one does -- alongside the flexible-pocket and
synth-guidance open questions already documented elsewhere in this
project.

**UPDATE (2026-09-12):** direction (b) was pursued to completion in the
Stage 2+ phase -- a physics-anchored architecture (GIGN+PIGNet2) and three
independent guidance-mechanism variants (gradient-norm normalization,
classifier reformulation, timestep-windowing) were all tested, all null.
A standalone diagnostic (`guidance/DIAG1_SIZE_CONFOUND_FINDING.md`) found
the underlying gradient's direction, not its dependence on point-
prediction accuracy per se, is the more likely limiting factor (gradient
magnitude correlates positively with distance-to-pocket, i.e. it is
strongest on atoms least likely to matter for binding). See
`guidance/DUAL_FALSIFICATION_CONCLUSION.md` for the full synthesis.

## Reproducibility

- Driver (checkpointed at config granularity): `guidance/run_affinity_sweep.py`
- Configs: `configs/prop/crossdocked_affinity_egnn.yml` (baseline), `configs/prop/affinity_sweep/{lr_3e-5,lr_3e-4,arch_4layers}.yml`
- Raw sweep results: `guidance/affinity_sweep_results.json`
- Per-config training logs: `logs_affinity_sweep/<name>/*/log.txt`
- Diagnostics 2-5 code and raw results: `guidance/affinity_sweep_diagnostics.py`, `guidance/affinity_diagnostics_results.json`
- Diagnostic 1 (target-fraction split builder, training script, checkpointed driver): `guidance/target_subsample_splits.py`, `guidance/train_affinity_diag1.py`, `guidance/run_diag1_learning_curve.py`
- Diagnostic 1 raw results and bootstrap CIs: `guidance/diag1_learning_curve_results.json`, `guidance/diag1_bootstrap_ci.py`, `guidance/diag1_bootstrap_ci_results.json`
- Diagnostic 1 per-fraction training logs: `logs_affinity_diag1/frac{0.25,0.5,0.75}/*/log.txt`
