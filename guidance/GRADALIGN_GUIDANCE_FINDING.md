# Gradient-Alignment Guidance: Strongest Diagnostic Fix, Flattest Real-Docking Result Yet

Closes the loop on `guidance/lp_split/train_egnn_stage0_gradalign.py`
(the auxiliary-loss checkpoint that directly minimizes DIAG1's diagnosed
gradient/distance-to-pocket correlation during training, not just as a
side effect of some other change). See `DIAG1_SIZE_CONFOUND_FINDING.md`
context and this investigation's other checkpoint-specific finding docs
for the comparison set.

## Diagnostic: by far the strongest signed correlation of any checkpoint -- at a cost

| | size/heaviness corr | distance-to-pocket corr |
|---|---|---|
| Original | -0.127 | +0.240 |
| Noise-matched | +0.125 | +0.062 |
| Vina-target | -0.084 | +0.047 |
| ESM2 | -0.016 | -0.072 |
| **Gradient-alignment** | **+0.331 (worst of all 5)** | **-0.784 (best of all 5, by far)** |

Directly optimizing the distance correlation during training worked
dramatically better than any indirect intervention (3x the magnitude of
ESM2's incidental fix, correctly signed, 95% CI nowhere near zero:
[-0.799, -0.770]). It also produced the single worst size/heaviness
correlation of any checkpoint -- an unmonitored trade-off, not something
the training loss touched at all. Worth flagging on its own terms: fixing
one diagnosed symptom precisely can make a different, related symptom
worse, if nothing is explicitly holding it in place.

## Real-docking test: flat, slightly wrong-signed, no exploratory signal worth confirming

Prelambda check (`guidance/prelambda_gradalign_results.json`, n=8,
standard 7-point grid, fixed pocket SQHC_ALIAD_1_631_0):

| Lambda | Mean Vina Dock | Mean diff vs. lambda=0 | Wilcoxon p | single_fragment_rate |
|---|---|---|---|---|
| 0.0 | -10.554 | - | - | 1.00 |
| 0.1 | -10.414 | +0.140 (worse) | 0.078 | 1.00 |
| 0.3 | -10.511 | +0.042 (worse) | 0.109 | 1.00 |
| 1.0 | -10.528 | +0.026 (worse) | 0.547 | 1.00 |
| 3.0 | -10.348 | +0.206 (worse) | 0.313 | 1.00 |
| 10.0 | -10.003 | +0.550 (worse) | 0.078 | 1.00 |
| 30.0 | -10.721 | - | - | 0.88 |

Unlike every other checkpoint's exploratory tier (which typically shows
*some* encouraging direction at *some* lambda, later weakening on
confirmation -- see the soft-v and ESM2 precedents), gradient-alignment
shows **no encouraging direction at any lambda tested**: every mean diff
points the wrong way (worse, not better), though none reach significance.
Structural validity is the best-behaved of any checkpoint (single-fragment
rate still 0.88 at lambda=30, where most other checkpoints have already
collapsed to near 0).

Per this project's protocol, a confirmatory n=24+ run is reserved for
signals that look *encouraging* at small n (the whole point is to guard
against a promising-looking fluke). There is nothing encouraging here to
confirm -- the exploratory tier itself is already a clean, flat null, so
no further sampling budget was spent chasing it.

## This is, if anything, the cleanest illustration of the investigation's central pattern

Gradient-alignment is the checkpoint where the diagnosed mechanism
(gradient/distance correlation) was fixed most directly and most
strongly of anything tried -- a 3x larger, correctly-signed effect with a
95% CI nowhere near zero. If "fixing the diagnosed mechanism" were
sufficient to produce a real guidance effect, this is the checkpoint most
likely to show it. It shows the flattest, least encouraging real-docking
result of the entire investigation instead.

## Updated pooled correction

Adding these 5 tests (the main-grid lambdas excluding the 0.0 baseline
and the already-collapsed lambda=30) to the running pool: **32 tests
total, 0/32 significant after BH correction** (min raw p unchanged at
0.0306; min BH-corrected p=0.500).

## Artifacts

- `guidance/lp_split/train_egnn_stage0_gradalign.py` -- training (see its
  own module docstring for the OOM bug found and fixed mid-training, and
  the non-determinism discovered on restart).
- `guidance/diag_size_confound_gradalign_results.json` -- DIAG1 diagnostic.
- `guidance/prelambda_gradalign_results.json` -- real-docking exploratory tier.
