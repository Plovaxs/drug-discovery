# Track D, Step D.3: Full Lambda Re-Sweep — Divergence Persists, Not Resolved

Single-pocket sweep (BSD_ASPTE_1_130_0), bond-aware features, using the
NEW leakage-safe-retrained synth model (`guidance/
TRACK_D_LEAKAGE_SAFE_RETRAIN_FINDING.md`, real test R²=0.414), n=30
samples/lambda (up from the original finding's n=8), full 1000 diffusion
steps, grid 0.0–30.0.

## Result table

| λ_synth | N reconstructed | Single-fragment rate | N scoreable | Own score (mean) | Real RA-score (mean) |
|---|---|---|---|---|---|
| 0.0 (baseline) | 30/30 | 0.90 | 27 | 0.5605 | 0.2581 |
| 0.01 | 30/30 | 0.90 | 27 | 0.5668 | 0.2515 |
| 0.03 | 29/30 | 0.90 | 26 | 0.5922 | 0.1916 |
| 0.1 | 30/30 | 0.77 | 23 | 0.5784 | 0.1863 |
| 0.3 | 29/30 | 0.21 | 6 | 0.5577 | 0.0659 |
| 1.0 | 28/30 | 0.07 | 2 | 0.7925 | 0.2391 |
| 3.0 | 28/30 | 0.00 | 0 | — | — |
| 10.0 | 30/30 | 0.00 | 0 | — | — |
| 30.0 | 26/30 | 0.00 | 0 | — | — |

## Statistical test: KS test (real RA-score, guided vs. unguided), BH-corrected across the 5 testable lambdas

| λ | N | Raw p | BH-corrected p | Mean shift vs. baseline | 95% bootstrap CI |
|---|---|---|---|---|---|
| 0.01 | 27 | 0.997 | 0.997 | -0.009 | [-0.161, 0.140] |
| 0.03 | 26 | 0.886 | 0.997 | -0.066 | [-0.207, 0.079] |
| 0.1 | 23 | 0.812 | 0.997 | -0.071 | [-0.212, 0.072] |
| 0.3 | 6 | 0.226 | 0.997 | **-0.192** | [-0.315, -0.078] |
| 1.0 | 2 | 0.448 | 0.997 | -0.018 | [-0.194, 0.148] |

**Formally: no signal survives BH correction** (min corrected p=0.997) —
matching the "no significant effect after correction" pattern seen across
every guidance mechanism tested in this entire project (Track A, B1, B2,
B3).

## But: the direction is not ambiguous, and the divergence pattern itself persists

Unlike a genuinely random null, **every single tested lambda shows the
same-direction shift**: real RA-score mean is lower than baseline at
every non-zero lambda (0.01 through 1.0), never once higher. At λ=0.03
(N=26, the largest well-powered non-trivial-lambda point), **the classic
divergence signature reappears**: guidance's own score *rises*
(0.5605→0.5922, +5.7%) while the real, independently-computed RA-score
*falls* (0.2581→0.1916, -25.8%) — the same pattern
`SYNTH_GUIDANCE_FINDING.md` originally documented with the old,
leakage-inflated model, now reproduced with the properly-retrained,
leakage-safe model.

The λ=0.3 point shows the largest raw effect (-0.192, and its own
un-corrected 95% CI excludes zero) but rests on only N=6 scoreable
molecules — heavy fragmentation at that lambda (single-fragment rate
0.21) sharply cuts the number of moleculaes that survive to be scored,
not because the effect vanishes but because the sample size collapses.
This is exactly the kind of result this project's rigor standard exists
to catch: a large point estimate on a tiny, underpowered sample must not
be reported as a finding on its own.

## Decision (per D.4's explicit gating rule)

> "If Step 2 still shows the divergence persisting (even partially): this
> confirms Phase 1's fix was insufficient, and Track D should proceed
> directly to D.5 rather than spending multi-pocket confirmation effort
> on a single-pocket result that hasn't resolved the core problem."

**The divergence persists** (consistent wrong-direction shift across
every tested lambda, and the classic own-score-up/real-score-down
signature reappearing at λ=0.03 with adequate N=26). Per the explicit
gating rule, **do not proceed to D.4's multi-pocket confirmation** —
proceed directly to **D.5** (the DIAG1-equivalent mechanistic diagnostic
for synth-guidance).

This means: even after fixing BOTH previously-identified issues on the
synth-guidance side (bond-aware features in Phase 1, and now the
leakage-safe retrain in D.2a), the fundamental problem — guidance moving
its own metric in a direction uncorrelated with (here, opposed to) the
real target — has not been resolved. This mirrors the affinity side's own
history closely: fixing the predictor's data/training quality (Stage 0,
and now this leakage-safe synth retrain) does not by itself fix a broken
guidance mechanism.

## Next step

D.5: diagnose whether the synth-guidance gradient's direction correlates
with something plausibly related to real synthesizability, or with
something else entirely (matching DIAG1's method, adapted to
synthesizability-relevant features).
