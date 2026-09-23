# D.5: Synth-Guidance Gradient-Direction Diagnostic

**Standalone diagnostic, not a guidance-variant checkpoint** — mirrors
`guidance/DIAG1_SIZE_CONFOUND_FINDING.md`'s method for the affinity side,
adapted since `SynthGuidance` is ligand-only (no protein/pocket input at
all, so the original "distance-to-pocket" axis doesn't apply here).

## Question

D.3 confirmed the synth-guidance divergence (own score up, real RA-score
down) persists even after both prior fixes (bond-aware features, leakage-
safe retrain). D.5 asks *why*: does the gradient's direction correlate
with something plausibly related to real synthesizability, or with
something else?

## Method

Captured genuine mid-trajectory states from the **unguided** sampling
path (read-only hook, zero effect on the trajectory) at 3 timesteps
(i=150/500/850), 3 pockets, n=16 samples/pocket (n=144 total, matching
DIAG1's original scale). Computed the raw (unnormalized) synth-guidance
gradient (bond-aware features) at each captured state post-hoc, correlated
per-atom gradient magnitude against two proxies:
- **Size/heaviness**: expected atomic number under the current soft type
  distribution (same construct as DIAG1).
- **Aromaticity fraction**: expected P(aromatic) under the current soft
  type distribution — a cheap structural-complexity proxy, since RA-score
  is a fingerprint-based classifier sensitive to ring/aromatic
  substructure complexity.

## Result (n=144)

| Correlation | Mean | 95% CI | Interpretation |
|---|---|---|---|
| Gradient magnitude vs. size/heaviness | 0.032 | [-0.019, 0.083] | **Not significant** (CI includes 0) |
| Gradient magnitude vs. aromaticity | -0.014 | [-0.061, 0.036] | **Not significant** (CI includes 0) |

## Interpretation: unlike DIAG1, this diagnostic did NOT find an explanation

**This is a genuinely different result from the affinity side's DIAG1**,
which found a real, statistically clear positive correlation with
distance-to-pocket (+0.240, CI excluding 0) — a concrete, actionable
explanation for that gradient's failure. Here, **neither tested proxy
explains the synth-guidance gradient's misdirection**: it is not
systematically pushing toward heavier atoms, and not systematically
pushing toward (or away from) aromaticity.

This does not mean the gradient direction is fine — D.3 already
established it moves real RA-score in the wrong direction. It means the
two cheapest, most obvious candidate explanations (size-shortcut,
aromaticity-shortcut) are both ruled out, and the actual mechanism behind
the divergence remains an open question. Plausible remaining candidates,
none tested here (matching `SYNTH_GUIDANCE_FINDING.md`'s own "open
question" list, still unresolved): the gradient may be exploiting a
higher-order/adversarial-like direction in the model's input space (the
classifier-guidance literature's known failure mode) that isn't captured
by either simple scalar proxy tried here, or the fingerprint-based
RA-score target itself may simply have a training-distribution structure
that a smooth, continuous 3D gradient cannot usefully approximate
regardless of what feature it's chasing.

## Track D overall conclusion (D.2 through D.5)

| Step | Finding |
|---|---|
| D.2 | Prior results were exploratory-scale (1 pocket, n=8, no stats); model not converged; **no leakage-safe test set existed at all** |
| D.2a | Leakage-safe retrain: real test R²=0.414 (not the previously-reported 0.769) |
| D.3 | Divergence **persists** at proper power (n=30) even with both fixes applied — own score up, real RA-score down, same as the original finding |
| D.4 | Skipped per the explicit gating rule (D.3 did not show resolution) |
| D.5 | Neither size nor aromaticity explains the gradient's misdirection — mechanism remains unidentified |

**Track D's synthesizability-guidance investigation reaches the same
overall conclusion as Track A/B's affinity-guidance investigation**:
gradient-based guidance built on a real, validated, moderately-predictive
learned model (R²=0.41, a genuine, useful point-estimate signal) does not
transfer to a usable in-loop generative guidance signal — and, as on the
affinity side, this is not for lack of trying to fix it (two independent,
principled fix attempts: bond-aware features, leakage-safe retrain; two
independent diagnostics: this one and the original SMILES-divergence
check). Unlike the affinity side, the specific mechanistic "why" remains
open here rather than pinned down to a concrete confound — a real,
disclosed difference in how much this phase was able to explain on each
side of the "coupled" framing, not a gap to be minimized.

This completes Track D. Combined with the affinity side's `guidance/
DUAL_FALSIFICATION_CONCLUSION.md`, both halves of the thesis's originally-
proposed "coupled affinity-synthesizability guidance" mechanism have now
been independently, rigorously investigated and found not to transfer
gradient-based guidance to their respective real targets.
