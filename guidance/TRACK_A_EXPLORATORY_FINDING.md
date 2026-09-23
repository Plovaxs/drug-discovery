# Track A Exploratory-Tier Finding: GIGN+PIGNet2 Physics-Anchored Model

Per the Stage 2+ addendum's priority order (B3 → A-exploratory → B1/B2 →
A-full) and the follow-up addendum authorizing full implementation now,
this is the exploratory-tier dual-criterion checkpoint for Track A: a
from-scratch GIGN heterogeneous backbone + PIGNet2 physics-decomposition
head (`guidance/track_a/`), partially trained, then tested for
gradient-informativeness exactly like every other checkpoint this phase.

## Implementation time (addendum Part 1, point 5)

- Architecture + featurization + training script + guidance wrapper
  (`guidance/track_a/physics.py`, `hil.py`, `features.py`, `model.py`,
  `train_gign_pignet_stage2.py`, `gign_pignet_guidance.py`,
  `track_a_exploratory_checkpoint.py`): ~25-40 minutes of active
  implementation + smoke-testing (first file `physics.py` created
  2026-09-10 15:39:52; training launched 15:42:32, restarted once at
  15:46:56 after the follow-up addendum specified hidden_dim=256 instead
  of the original 64).
- Partial training: 632s (~10.5 min) for 5000 iterations, batch_size=1.
- Prelambda fragmentation check: ~40 min (one run killed/restarted after
  an output-buffering false alarm, then completed cleanly with a tighter
  n=4/5-lambda-point scope).
- **Main exploratory checkpoint (3 pockets x 4 conditions x 20 molecules,
  real Vina Dock + PoseBusters): ~4h19m** (launched 16:36, completed
  20:55) -- by far the dominant cost, driven by real docking at
  exhaustiveness=8 across 240 molecules, not by implementation or
  training. This matches this project's established per-molecule docking
  cost (consistent with Track B3's and Stage 0's confirmation run
  timings), not scope creep in the "exploratory" sense the addendum
  warned against -- the dual-criterion protocol's real-Vina-Dock
  requirement is inherently this expensive regardless of which guidance
  mechanism is being tested.
- **Total wall-clock, first line of code to final result: ~5h15m.**

## Partial-training predictive quality

5000 iterations (~1.67 "epochs" over a 3000-sample subsample), test set
(11,855 labeled entries, same LP-PDBBind-style leakage-safe split as
Stage 0): **RMSE=1.737, MAE=1.263, R²=-0.110, Pearson=0.583,
Spearman=0.623.** Comparable to Stage 0's fully-trained EGNN
(Pearson=0.609, R²=0.342) despite far less training -- correlation is
already close to Stage 0's ceiling, though R² remains negative. This is
an expected, disclosed architectural gap, not a training-insufficiency
signal: the physics-summed output has no learned bias/scale head (unlike
EGNN's full output MLP), and PIGNet2's own "scoring" task convention
(regression directly against raw summed energies) does not correct for
this either -- see `guidance/track_a/model.py`'s docstring. Training and
validation loss were both still improving at the iter-5000 cutoff (val
loss 3.724 -> 3.509 -> 3.412 -> [3000->4000 no improvement] -> 3.237),
i.e. this partial checkpoint was cut off while still decreasing, not
plateaued -- a real basis for optimism that a full-tier training run
would do better on both Pearson and (with a bias-correction head) R².

## Prelambda stability check (addendum Part 3, point 2)

Single-pocket (`BSD_ASPTE_1_130_0`), n=4, full 1000 steps, no docking
(reconstruction/fragmentation-rate signal only):

| Lambda | N reconstructed | Single-fragment rate |
|---|---|---|
| 0.1 | 4/4 | 1.00 |
| 0.3 | 4/4 | 0.75 |
| 1.0 | 3/4 | 0.75 |
| 3.0 | 4/4 | **0.00** |
| 10.0 | 4/4 | **0.00** |

**This model's stable lambda range is much narrower than Stage 0's
EGNN** (which stayed non-fragmenting through lambda=10-30, only breaking
down at lambda=100 -- see `guidance/LPSPLIT_LAMBDA_RESWEEP_FINDING.md`).
Track A's physics-based gradient causes complete fragmentation by
lambda=3.0, an order of magnitude lower. This is a real, reportable
finding on its own, independent of the main checkpoint's result: **different
guidance mechanisms can have very different safe operating ranges**, and
Stage 0's lambda range must not be assumed to transfer to a new
architecture -- exactly the caution the addendum's Sec 1.2 anticipated.
Plausible reason: the physics head's energy terms are unbounded sums over
all ligand-protein atom pairs (no learned per-pair saturation the way
EGNN's fully-learned scalar output has), so the same nominal lambda value
translates to a much larger raw gradient magnitude on positions.
Lambda=0.1, 0.3, 1.0 were carried into the main checkpoint as the stable
range; lambda=3.0+ was excluded as unusable regardless of any predictive
benefit, since a guidance mechanism that reliably breaks the molecule
apart cannot be a viable guidance mechanism at that strength.

## Main exploratory checkpoint (addendum Part 3, points 3-6)

3 pockets (`BSD_ASPTE_1_130_0`, `HDAC8_HUMAN_1_377_0`,
`CD38_HUMAN_44_300_0`), unguided baseline + lambda in {0.1, 0.3, 1.0},
n=20/condition, real Vina Dock (exhaustiveness=8) + PoseBusters, same-seed
control per pocket (isolates the guidance intervention). Gradient-
informativeness test: KS test per (pocket, lambda) vs. that pocket's own
unguided baseline, BH-corrected across all 9 tests in this checkpoint.

| Comparison | KS p (raw) | KS p (BH-corr.) | Mean shift | Direction | PB degraded? | Verdict |
|---|---|---|---|---|---|---|
| BSD_ASPTE:0.1 | 0.336 | 0.431 | -0.70 | correct | no | no_signal |
| BSD_ASPTE:0.3 | **0.012** | 0.111 | -1.41 | correct | no | no_signal |
| BSD_ASPTE:1.0 | 0.079 | 0.182 | -1.48 | correct | **yes** (pb 0.25→0.06) | no_signal |
| HDAC8:0.1 | 0.175 | 0.262 | -0.32 | correct | no | no_signal |
| HDAC8:0.3 | 0.081 | 0.182 | -0.54 | correct | no | no_signal |
| HDAC8:1.0 | **0.037** | 0.168 | -0.54 | correct | **yes** (pb 0.60→0.21) | no_signal |
| CD38:0.1 | 0.571 | 0.630 | -0.46 | correct | no | no_signal |
| CD38:0.3 | 0.175 | 0.262 | -0.88 | correct | **yes** (pb 0.40→0.35) | no_signal |
| CD38:1.0 | 0.630 | 0.630 | +0.15 | wrong | **yes** (pb 0.40→0.00) | no_signal |

(Full table: `guidance/track_a_exploratory_results/gradient_informativeness_results.csv`.)

**Formal verdict per the addendum's decision bands: No signal.** No
comparison survives BH correction (min corrected p = 0.111). Per §4.1/
Part 4: this is a yellow flag, not a stage failure, and the established
priority order proceeds to Track B1/B2 next.

**However, this is a qualitatively different null than Track B3's.**
Worth stating plainly rather than burying in the "no_signal" label:

- 8/9 comparisons show the "correct" direction (more negative/better Vina
  Dock), and the raw mean shifts are large relative to baseline spread --
  e.g. BSD_ASPTE at lambda=0.3/1.0 shows a ~1.4-1.5 mean shift, roughly an
  order of magnitude bigger than anything Track B3 or the Stage 0
  lambda-resweep ever produced (those topped out around 0.1-0.2). Two raw
  p-values (0.012, 0.037) were nominally significant before correction --
  Track B3 never got below 0.57.
- The reason nothing survives correction is a combination of modest
  per-pocket sample size (n=20) relative to real docking-score variance,
  and the fact that the largest, cleanest-looking effect (lambda=0.3 at
  BSD_ASPTE) is a single result among 9 simultaneous tests, not the
  effect size itself being negligible.
- **The apparent effect is confounded with physical-validity degradation
  at the higher end of the stable range**: PoseBusters pass rate collapses
  at lambda=1.0 in all three pockets (most severely at CD38: 40%→0%), and
  even partially at lambda=0.3 for CD38. This is exactly the
  Vina-hacking-pattern check the addendum's gate exists to catch: at
  lambda=1.0, part of whatever "improvement" appears may be coming from
  the guidance pushing structures into physically implausible poses that
  Vina still scores favorably, not genuine binding-relevant improvement.
  Ligand efficiency was NOT degraded anywhere, which argues against pure
  "make it bigger" hacking specifically, but does not rule out the
  PoseBusters-flagged validity failures.
- Lambda=0.1 is the only condition with zero PoseBusters degradation
  across all 3 pockets, and also has the smallest (most conservative)
  effect sizes -- consistent with a real but small, currently
  underpowered signal that a properly-powered full-tier run (more
  pockets, more molecules/pocket, matching Stage 0's confirmation-tier
  protocol) could plausibly resolve either way.

## Interpretation and routing

Per the addendum's Part 4 "No signal" band: log as a yellow flag (not a
failure), proceed to Track B1/B2 next as already scheduled, and revisit
whether Track A's full training run is still worth prioritizing once
B1/B2's results are in hand. Given the qualitative differences described
above (real-magnitude, consistently-directional shifts vs. Track B3's
near-zero noise), Track A is a stronger candidate for revisiting than a
mechanism that showed literally nothing -- but per the addendum's
explicit requirement, this exploratory-tier result alone is not grounds
to elevate Track A's priority ahead of B1/B2; it only argues for not
writing off physics-anchored guidance entirely based on this single,
underpowered pass.

**Next**: Track B1 (relative/reference-normalized guidance) and B2
(classifier-guidance reformulation) on the existing Stage 0 model, per
the established priority order.
