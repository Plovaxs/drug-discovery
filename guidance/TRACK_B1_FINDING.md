# Track B1 Finding: Gradient-Norm Normalization — Hypothesis Not Supported

Per the council-consultation addendum, B1 was redefined as gradient-norm
normalization (rescaling the raw guidance gradient to a fixed per-sample,
per-atom norm before applying lambda), tested against the Stage 0 EGNN
model via the mandatory three-way design: (a) unguided, (b) raw gradient
at its previously-established stable lambda (10.0), (c) normalized
gradient at its freshly-derived stable lambda (0.1, from a dedicated
prelambda check — see below). The decision criterion hinges specifically
on (c) vs. (b), not on either condition's comparison to (a) alone.

## Prelambda check (normalized gradient)

Raw gradient norm measured at ~0.0012/atom in a smoke test; after
unit-per-atom normalization, lambda has units of roughly "Angstrom nudge
per atom per step." Single-pocket check (n=4, full 1000 steps):

| Lambda | Single-fragment rate |
|---|---|
| 0.001 – 0.1 | 1.00 (fully stable) |
| 0.3 | 0.25 (degraded) |
| 1.0 | 0.00 (fully fragmented) |

Lambda=0.1 (the largest fully-stable value) was carried into the main
checkpoint.

## Main checkpoint: three-way comparison, 3 pockets, n=20/condition

Full table: `guidance/track_b1_results/gradient_informativeness_results.csv`.
**None of the 9 tests (3 pockets x {raw-vs-unguided, normalized-vs-
unguided, normalized-vs-raw}) survive BH correction** (min corrected
p=0.192).

| Pocket | raw vs. unguided | normalized vs. unguided | **normalized vs. raw (decisive)** |
|---|---|---|---|
| BSD_ASPTE | no_signal (median shift ~0, KS p=0.83) | no_signal, **wrong direction** (median +0.52), LE degraded | no_signal, **wrong direction** (median +0.46), LE degraded |
| HDAC8 | no_signal, correct dir. (median -0.40) | no_signal, **wrong direction** (median +0.28), LE + PB degraded (PB 0.60->0.30) | no_signal, **wrong direction** (median +0.68), LE + PB degraded |
| CD38 | no_signal, correct dir. (median -0.13) | no_signal, correct dir. (median -0.74), PB degraded (0.40->0.25) | no_signal, correct dir. (median -0.61), PB degraded |

**Decisive comparison (normalized vs. raw): 2/3 pockets show normalized
performing WORSE than raw (wrong direction), 1/3 shows numerically better
but confounded by PoseBusters degradation, and none reach significance.**

## A data-quality note, investigated and resolved (not a bug)

BSD_ASPTE's raw condition contains one extreme outlier: a Vina Dock score
of **+286.58** (vs. a normal range of -12 to -5), which single-handedly
inflates that condition's *mean* shift to a nonsensical +14.3 in the raw
comparison table. Verified this is a genuine (if pathological) docking
outcome from a severely malformed generated pose under raw guidance, not
a software bug or unfiltered error sentinel (`utils/evaluation/
docking_vina.py` has no such sentinel-value path; Vina's own scoring
function is not strictly bounded and can produce large positive scores
for catastrophically clashing geometries). **This is why this project's
gradient-informativeness test uses the KS test (robust to a single
extreme point) and reports both mean and median shift, rather than mean
alone** — the KS test and median-based reading for this comparison are
unaffected (correctly reporting "no significant difference"), while a
naive mean-based read would have been badly misleading. Worth flagging
as its own small, practical point in favor of normalization's motivation:
raw, unbounded-magnitude guidance can occasionally produce such
pathological outputs; normalized guidance's controlled per-atom magnitude
structurally cannot (its own vina_dock range, -8.8 to -2.6, has no such
outlier).

## A separate methodological lesson

Lambda=0.1 passed the prelambda check's fragmentation-rate criterion
(single_fragment_rate=1.00) but still showed real PoseBusters degradation
in the full checkpoint (HDAC8: 0.60->0.30; CD38: 0.40->0.25). **Whole-
molecule connectivity (fragmentation) and fine-grained physical validity
(PoseBusters) are different failure modes** — a lambda that looks safe
by the cheap fragmentation-only prelambda check is not guaranteed to be
safe by the fuller PoseBusters criterion. Future prelambda checks in this
project should consider a quick PoseBusters pass too, not fragmentation
rate alone, if this pattern recurs.

## Interpretation

**The gradient-norm-normalization hypothesis is not supported.** Per the
addendum's own decision framing: since (b) and (c) are statistically
indistinguishable (and where they do differ numerically, normalized is
more often worse than raw, not better), this strengthens the case that
the persistent lack of guidance effect across this project (Stage 0,
Track A, Track B3, and now B1) reflects an uninformative gradient
*direction*, not merely a badly-scaled *magnitude*. This is a genuinely
useful negative result: it rules out one of the simplest, cheapest
possible fixes (raw scale mismatch) as the explanation.

**This converges with DIAG1's independent finding** (`guidance/
DIAG1_SIZE_CONFOUND_FINDING.md`): the Stage 0 model's gradient magnitude
correlates *positively* with distance-to-pocket (+0.240), i.e. it pushes
hardest on atoms farthest from any binding contact — a direction-level
problem, not a scale problem. Two independent lines of evidence (B1's
empirical null on rescaled magnitude, and DIAG1's diagnostic on gradient
direction) now point the same way.

## Routing

Per the addendum's priority order, proceed to **Track B2** (classifier-
guidance reformulation) next — the last item in the Track B queue.
Logged as `B1-exp` in `guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`.
