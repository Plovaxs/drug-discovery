# Track B2 Finding: Classifier-Guidance Reformulation — No Signal, Wrong-Direction Leaning

Classifier guidance (`guidance/classifier_guidance.py`): a small linear
classification head (best val AUROC=0.783, test AUROC=0.677 — a
reasonably informative binary "good binder" classifier) trained on
Stage 0's frozen EGNN backbone's pooled embedding, binarized at the
training set's median pk (7.03). Guidance target is the classic
classifier-guidance gradient, `grad_x log sigmoid(logit)`, not a raw
regression output's gradient.

## Prelambda check

Raw gradient norm ~0.00057/atom (similar order of magnitude to Stage 0's
regression AffinityGuidance, ~0.0012/atom). Notably more robust to high
lambda than either prior guidance mechanism tested this phase:

| Lambda | Single-fragment rate |
|---|---|
| 1 – 100 | 1.00 (fully stable) |
| 300 | 0.00 (fully fragmented) |

Lambda=100 was carried into the main checkpoint — an order of magnitude
higher than Stage 0's own stable point (10) and three orders above Track
A's physics model (0.1-1 before normalization; fragmenting by 3).

## Main checkpoint (reusing Track B1's unguided/raw conditions directly)

3 pockets, n=20/condition, real Vina Dock + PoseBusters. Only the new
`classifier` condition was generated; `unguided` and `raw` conditions were
reused byte-for-byte from Track B1 (same seed/pockets/settings, valid
since unguided sampling doesn't depend on any guidance model and `raw` is
literally the same Stage 0 EGNN guidance at the same lambda=10.0).

| Comparison | KS p (BH-corr.) | Direction (mean) | Median shift | PB degraded? |
|---|---|---|---|---|
| BSD_ASPTE: classifier vs unguided | 0.524 | wrong | +0.79 (worse) | no |
| BSD_ASPTE: classifier vs raw | 0.263 | "correct" (raw's known 286.58 outlier artifact — see below) | **+0.73 (worse)** | no |
| HDAC8: classifier vs unguided | 0.857 | wrong | +0.36 (worse) | **yes, severe (0.60->0.05)** |
| HDAC8: classifier vs raw | 0.671 | wrong | +0.76 (worse) | **yes, severe (0.60->0.05)** |
| CD38: classifier vs unguided | 0.893 | "correct" (tiny, mean-only) | +0.13 (worse) | yes (0.40->0.22) |
| CD38: classifier vs raw | 0.893 | wrong | +0.26 (worse) | yes (0.45->0.22) |

(Full table: `guidance/track_b2_results/gradient_informativeness_results.csv`.)

**None of the 6 tests survive BH correction** (min corrected p=0.263).
The BSD_ASPTE:classifier_vs_raw row's "correct" mean-direction label is an
artifact of `raw`'s single already-documented outlier
(`guidance/TRACK_B1_FINDING.md`'s +286.58 pathological pose) — the
**median** shift for that same comparison is +0.73, i.e. worse, matching
every other row's direction.

## Interpretation

**This is the cleanest wrong-direction result of this entire phase.** By
median (the robust statistic), classifier guidance made Vina Dock scores
*worse* in 5 of 6 comparisons, and caused severe PoseBusters degradation
at HDAC8 (0.60->0.05 — the sharpest validity collapse seen in any Track
A/B checkpoint so far, worse than Track A's λ=1.0 PB collapses). None of
this reaches statistical significance after correction (n=20/pocket is
modest relative to real docking-score variance), so the formal verdict is
still `no_signal`, not a confirmed harmful effect — but there is no
comparison here that looks encouraging even before considering
significance, unlike Track A's exploratory checkpoint (which had a
genuinely promising, if confounded, subset of results) or Track B1
(which was merely indistinguishable from raw).

The classifier reformulation's higher stable-lambda tolerance (up to 100
without fragmenting) did NOT translate into a more effective or safer
guidance signal in practice — it just meant a larger nudge was applied
before generation broke down structurally, without that larger nudge
producing better real docking outcomes. If anything, this is consistent
with DIAG1's diagnostic finding: the underlying issue is gradient
*direction*, and reformulating the training objective (regression to
classification) while keeping the same frozen backbone representation
does not change what that representation's gradient is pointing toward.

## Routing

This was the last item in the Track B queue (B1 gradient-norm
normalization, B2 classifier reformulation, B3 timestep-windowing — all
three now tested). Logged as `B2-exp` in
`guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`.
