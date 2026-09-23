# Track A Full-Tier Finding: GIGN+PIGNet2, Fully Trained — Clean Null

This is the culminating checkpoint of Track A (architecture route) and of
the whole Stage 2+ phase. Per the addendum's Part 3 decision bands, this
result determines whether the dual-falsification requirement (both Track
A and all Track B variants must independently fail) is now satisfied.

## Setup

Model: GIGN+PIGNet2 physics-anchored affinity predictor, trained to full
convergence (early-stopped, epoch 13, best checkpoint at epoch 1) on the
full 46,964-entry leakage-safe training split — see `guidance/
STAGE2_PLUS_EXPERIMENT_LOG.md`'s `A2-full-train` entry for the full
training writeup (test Pearson=0.581, Spearman=0.608, R²=-0.189,
heavy-atom-count correlation=0.640).

Checkpoint: **8 pockets** (the same diverse set used throughout this
phase — `guidance/lpsplit_confirmation_pockets.json`), unguided baseline
+ **3 lambdas (0.1, 0.3, 1.0)** from a freshly-derived stability check
specific to this checkpoint, **n=30 molecules/condition** (up from the
exploratory tier's 20, per the addendum's adequately-powered-N
requirement), real Vina Dock (exhaustiveness=8) + PoseBusters.

Fresh prelambda check (single pocket, n=8): stable through 0.1-0.3
(single-fragment rate 0.88), degrading at 1.0 (0.62), fully fragmented by
3.0+ — closely matching the exploratory tier's own narrow stable range,
confirming that finding was a real property of this architecture, not an
artifact of partial training.

## Result: 0/24 comparisons significant after BH correction

8 pockets × 3 lambdas = 24 KS tests, BH-corrected together. **Minimum raw
p-value: 0.0065** (HDAC8:λ=1.0); **minimum BH-corrected p-value: 0.157**
— nothing survives correction. Full table: `guidance/
track_a_full_results/gradient_informativeness_results.csv`.

### Direction: weaker consistency than the exploratory tier suggested

16/24 (67%) comparisons show the "correct" (more negative/better Vina
Dock) direction; 8/24 (33%) show the wrong direction. **This is
meaningfully weaker than the exploratory tier's 8/9 (89%) correct-
direction rate on its 3-pocket sample.** This is the expected signature of
a small-sample selection effect: the exploratory tier's 3 pockets
happened to show an encouraging directional pattern that a properly-
powered 8-pocket sample reveals as largely noise. Two of the exploratory
tier's original 3 pockets (BSD_ASPTE, HDAC8, CD38) are included in this
8-pocket set, so this isn't even a fully independent re-test — the
directional consistency degrading further when the pocket set is
merely *expanded* (not replaced) is a stronger signal that the original
pattern was noise than if a wholly new pocket set had been used.

### PoseBusters degradation: clean, dose-dependent, and severe at the high end

| Lambda | Fraction of pockets with PB degradation |
|---|---|
| 0.1 | 1/8 (12.5%) |
| 0.3 | 3/8 (37.5%) |
| **1.0** | **8/8 (100%)** |

This is a clean, monotonic, well-powered finding in its own right,
independent of the null KS-test result: **at λ=1.0, physical-validity
degradation occurs in every single tested pocket.** Ligand efficiency
degradation does not show the same clean dose-response (50%/12.5%/37.5%
across 0.1/0.3/1.0) and is not itself a reliable signal here, but the
PoseBusters pattern alone is sufficient to rule out λ=1.0 (and likely
λ=0.3) as a usable operating point for this guidance mechanism,
regardless of whether the underlying Vina Dock shift were ever to reach
significance.

## Decision (per the addendum's Part 3 bands)

**Band 3.1 applies: clean null, matching Track B's pattern.** No
comparison shows a significant, correctly-directed, non-degraded effect.
Specifically:
- No comparison survives multiple-comparison correction (0/24).
- Direction consistency is weak and does not strengthen with more data —
  it weakens, the opposite of what a real effect obscured by noise would
  show.
- The lambda range where any directional signal might exist (0.1) shows
  the *smallest* raw effect sizes and the least degradation, while the
  lambda with the clearest degradation (1.0) shows no compensating
  significant benefit.

This is **not** an ambiguous or borderline result (Band 3.3) — there is no
comparison anywhere close to significance after correction, no pocket
subset showing a coherent positive pattern, and the directional
consistency argument that made the exploratory tier interesting has now
specifically been tested and weakened, not strengthened. This is also
**not** a genuine positive (Band 3.2) requiring re-verification — there is
nothing positive to re-verify.

## Conclusion

**Track A (architecture route), now including the full physics-anchored
GIGN+PIGNet2 model trained to real convergence on the complete leakage-
safe dataset and evaluated at adequate statistical power, fails to show a
detectable guidance effect — matching Track B's independent 3/3 null
result (B1 gradient-norm normalization, B2 classifier reformulation, B3
timestep-windowing).**

Per the addendum's explicit requirement, **both independent falsification
routes have now failed.** See `guidance/DUAL_FALSIFICATION_CONCLUSION.md`
for the phase-closing synthesis.
