# Affinity Guidance Model: Standalone Predictive Quality (Part 3.4)

Uses the training run that produced the deployed checkpoint
(`guidance_models/affinity_egnn.pt`, epoch 20, `logs_affinity/crossdocked_affinity_egnn_2026_09_04__01_23_11/log.txt`)
-- no retraining was needed for this report; `train_affinity_model.py`
already evaluates on the true held-out test split (27 pockets, disjoint
targets from train, see `AFFINITY_MODEL_LEAKAGE_CHECK.md`) every time val
loss improves, and logs Pearson/Spearman/RMSE/MAE/R^2 via
`utils/misc_prop.py`'s `get_eval_scores`. This report surfaces that
existing, honestly-logged signal, which had not been examined before this
addendum.

## Important structural note on the val split

`datasets/crossdocked_affinity.py`'s `build_labeled_splits` carves `val`
as a random 90/10 split of the **pocket/pose indices** in the official
train split -- not by protein target. Since CrossDocked2020 contains many
poses per target, this means `val` and `train` (507 vs 456 unique targets
in this run, drawn from the same ~1,905-target train pool) substantially
overlap in target identity, even though individual pocket/pose instances
don't repeat. `val` therefore measures **within-target-distribution**
generalization (interpolating across poses of targets partially seen in
training), not the true target-transfer setting guidance actually needs
at test time (an unseen pocket in Task F, or any real deployment). `test`
(27 pockets, targets fully disjoint from train, confirmed in
`AFFINITY_MODEL_LEAKAGE_CHECK.md`) is the metric that actually matches
that setting.

## Results across training (every checkpoint where val improved)

| Epoch | Val Pearson | Val R^2 | **Test Pearson** | **Test Spearman** | **Test R^2** | Test RMSE (pK) |
|---|---|---|---|---|---|---|
| 1  | -- | -- | 0.212 | 0.170 | -0.996 | 2.564 |
| 2  | -- | -- | 0.231 | 0.208 | -0.952 | 2.535 |
| 3  | -- | -- | 0.386 | 0.321 | -0.454 | 2.188 |
| 5  | 0.682 | 0.452 | 0.315 | 0.237 | -0.401 | 2.148 |
| 9  | 0.724 | 0.505 | 0.302 | 0.236 | -0.195 | 1.984 |
| 10 | 0.745 | 0.548 | 0.381 | 0.313 | -0.161 | 1.955 |
| 12 | 0.766 | 0.572 | 0.319 | 0.273 | -0.283 | 2.055 |
| 15 | 0.768 | 0.590 | 0.121 | 0.150 | -0.797 | 2.433 |
| 16 | 0.784 | 0.593 | 0.266 | 0.245 | -0.442 | 2.179 |
| 19 | 0.795 | 0.616 | 0.327 | 0.304 | -0.275 | 2.049 |
| **20 (deployed)** | **0.788** | **0.616** | **0.233** | **0.141** | **-0.537** | **2.250** |

(Val n=4,061; Test n=27 throughout.)

## Honest interpretation

**On the true held-out target-transfer test set, the affinity model's
predictions are not usefully correlated with real experimental binding
affinity, and are worse than simply always predicting the training mean**
(R^2 is negative at every single logged checkpoint across all 20 epochs --
this is not an early-training artifact that later resolves). Test Pearson
fluctuates noisily in the 0.12-0.39 range with no consistent improvement
over training, while validation Pearson climbs steadily to ~0.79-0.80 --
a gap fully explained by the val/train target overlap noted above, not by
the model actually learning to predict affinity for new targets.

The deployed checkpoint (selected by best *val* loss, per
`train_affinity_model.py`'s standard early-stopping criterion) is not even
the best-performing checkpoint on test: epoch 3's test Pearson (0.386) is
the highest logged, but epoch 3 was never selected because it wasn't a
val-loss improvement large enough relative to other epochs to be the
running-best at that point in training -- illustrating that model
selection by val loss here does not track true test/target-transfer
quality at all, since val and test are measuring different things (as
established above).

## What this explains

This is a plausible root cause, independent of and complementary to the
lambda re-sweep finding (`AFFINITY_LAMBDA_RESWEEP_FINDING.md`): if the
affinity model's predictions carry near-zero usable signal about real
binding affinity on unseen targets (R^2 < 0, Pearson ~0.2-0.4), then no
amount of gradient-guidance strength during sampling can be expected to
reliably steer generation toward higher real affinity on a new pocket --
the guidance direction itself is not well-aligned with truth in this
regime. This is consistent with, and gives a mechanistic explanation for,
both Task F's null result and the flat (no dose-response) real Vina Dock
scores found across the entire safe lambda range in the re-sweep.

## Citable standalone DTI result (as requested)

For thesis reporting purposes, the honest, apples-to-target-transfer
number to cite is the **test** split, not validation:

- **Test Pearson r = 0.233 (deployed checkpoint) / peak observed 0.386 (epoch 3) -- both weak**
- **Test R^2 = -0.537 (deployed) -- negative, i.e. worse than a mean-only baseline**
- **Test RMSE = 2.25 pK units (deployed)**, on labels with std ~1.6-1.9 pK -- error exceeds a large fraction of the natural label spread
- n=27 pockets, small enough that these numbers carry real sampling
  uncertainty themselves and should be reported with that caveat, not as
  a precise population estimate.

This is a genuine, reportable negative DTI-transfer result, not a
methodology error -- consistent with a known-hard problem (small
n=6,000 labeled training examples spread thin across 507 targets,
~11.8 examples/target on average, per `AFFINITY_MODEL_LEAKAGE_CHECK.md`'s
coverage numbers) rather than any leakage or engineering bug.

## Reproducibility

- Source log: `logs_affinity/crossdocked_affinity_egnn_2026_09_04__01_23_11/log.txt`
- Deployed checkpoint: `guidance_models/affinity_egnn.pt` (epoch 20, val_loss=1.375)
- Eval code: `utils/misc_prop.py`'s `get_eval_scores`
- Training/eval loop: `guidance/train_affinity_model.py`
