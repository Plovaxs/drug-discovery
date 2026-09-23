# Track D, Step D.2a: Leakage-Safe Retrain — A Major Correction to the Synth Model's Reported Quality

## What was found (recap of `TRACK_D_STATUS_D2.md`)

The original synthesizability-guidance model (`guidance_models/synth_ra_score.pt`,
trained 2026-09-04) was evaluated on a plain random 90/10 train/val split
of its 15,000 RA-score-labeled entries, with **no held-out test set** and
**no leakage-safety check at all** — the same category of problem Stage 0
was built to fix on the affinity side. Its reported quality (val R²=0.769,
Pearson=0.878) rested entirely on this un-audited split.

## Fix

Reused the existing leakage-safe target-level assignment
(`guidance/lp_split/leakage_safe_split.json`) to re-partition the same
15,000 RA-score-labeled entries by target identity — no new RA-score
computation, no new similarity analysis, since leakage-safety is a
property of which targets land in which split
(`guidance/lp_split/build_synth_lp_splits.py`). Result: **8,835 train /
1,015 val / 1,940 test** (3,210 entries discarded — their target wasn't
covered by the leakage-safe assignment).

Retrained the identical architecture (`SynthPredNet`, unchanged
hyperparameters) to genuine convergence: early-stopped at epoch 20
(patience=15, no artificial epoch cap — the original run was capped at a
fixed `max_epochs=40` with patience only at 1/6, i.e. never actually
converged). Best checkpoint: epoch 5.

## Result: real quality is much lower than previously reported

| | Original (leakage-unchecked, val only) | New (leakage-safe, held-out test) |
|---|---|---|
| R² | 0.769 | **0.414** |
| Pearson | 0.878 | **0.647** |
| Spearman | 0.871 | **0.649** |
| RMSE | 0.192 | **0.287** |

**The genuine, held-out predictive quality is substantially lower than
what was previously reported** — R² drops by nearly half. This is not a
training regression (the new model trained to real convergence, unlike
the original); it is the leakage-inflation being removed. The true
picture: this model still predicts RA-score meaningfully better than
chance (R²=0.41, Pearson=0.65 is a real, useful signal — notably still
stronger correlation than either affinity model ever achieved, ~0.58-0.61),
just not as well as the un-audited number suggested.

## Implication for Track D going forward

All of `SYNTH_GUIDANCE_FINDING.md`'s prior guidance-effect results
(placeholder-feature and Phase 1 bond-aware) were built on the OLD,
leakage-inflated model. This new, honestly-evaluated checkpoint
(`logs_synth_lp/synth_ra_egnn_lp_.../checkpoints/best.pt`) is what D.3's
full lambda re-sweep should now be run against — both because it is the
methodologically correct backbone, and because a synth-guidance divergence
result tested against a properly-validated model is far more defensible
than one built on an inflated number, in either direction (if the
divergence persists, that's still meaningful evidence against a
now-more-trustworthy model; if it resolves, that finding is far more
credible than it would be against the leakage-contaminated original).

Logged as part of the D.2a step in `guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`
(or the dedicated Track D log, per the addendum's documentation choice —
see that log for which was used).
