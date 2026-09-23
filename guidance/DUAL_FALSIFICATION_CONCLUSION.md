# Dual-Falsification Conclusion: Gradient-Based Affinity Guidance Does Not Transfer to Real Docking Outcomes

**Status: both independent falsification routes required by the Stage 2+
addendum have now failed. This document is the phase-closing synthesis,
written to be usable directly in the thesis results chapter.**

## The question

Can gradient-based guidance of a target-conditional diffusion model
(TargetDiff), steering the reverse-diffusion trajectory's per-step
clean-data estimate using the gradient of a learned binding-affinity
predictor, improve the real, independently-measured binding quality
(AutoDock Vina Dock score, physical validity via PoseBusters) of
generated molecules?

The addendum's standing rule: **two independent falsification routes must
both fail before concluding this approach is not viable** — a failure of
the guidance *mechanism* alone (Track B) does not rule out that a better
*predictor architecture* (Track A) would work with the existing
mechanism, and vice versa. Both have now been tested to a properly-
powered conclusion.

## Route 1: Track A — predictor architecture quality

| Stage | Model | Predictive quality (test) | Heavy-atom-count confound | Guidance result |
|---|---|---|---|---|
| Stage 0 | EGNN (unchanged architecture, corrected leakage-safe split) | Pearson=0.609, R²=0.342 | 0.719 | Null (multi-pocket confirmation, `LPSPLIT_LAMBDA_RESWEEP_FINDING.md`) |
| A2-exploratory | GIGN+PIGNet2 (from scratch), partial training (5000 iters, 3 pockets, n=20) | Pearson=0.583, R²=-0.110 | not measured | Null but promising-looking (large raw effects, confounded by PB degradation, underpowered) |
| **A2-full** | GIGN+PIGNet2, full training (46,964 entries, converged, 8 pockets, n=30) | Pearson=0.581, R²=-0.189 | 0.640 | **Clean null (0/24 significant after correction; direction consistency weakened, not strengthened, vs. the exploratory tier)** |

Doubling predictive quality (Stage 0's own contribution) did not produce
a guidance effect. Replacing the architecture entirely with a physics-
anchored GIGN+PIGNet2 model — reasoned about from first principles,
implemented faithfully to the published methodology, trained to real
convergence on the complete available dataset, and evaluated at adequate
statistical power across 8 diverse pockets — **also** did not produce a
guidance effect, despite comparable-to-Stage-0 predictive quality and a
partially-reduced (0.719 → 0.640, though not below the originally hoped
<0.4) heavy-atom-count confound.

## Route 2: Track B — guidance mechanism

| Variant | Mechanism | Result |
|---|---|---|
| B3 | Timestep-windowed guidance (gate to early/middle/late third of trajectory) | Clean null (all p in [0.57, 0.98] raw) |
| B1 | Gradient-norm normalization (unit-per-atom rescale before lambda) | Hypothesis not supported (normalized indistinguishable from, often worse than, raw) |
| B2 | Classifier-guidance reformulation (new head, same frozen backbone, `grad log p(y|x)`) | No signal, wrong-direction-leaning (worse by median in 5/6 comparisons; most severe PB collapse of the phase) |

All three ways of changing *how* guidance is computed or applied — its
timing, its magnitude, and its training objective — failed to produce an
effect, using the same underlying (Stage 0) learned representation.

## The connecting diagnostic: DIAG1

A standalone diagnostic (`DIAG1_SIZE_CONFOUND_FINDING.md`, not a guidance
variant) measured the Stage 0 model's raw gradient *direction* against two
plausible confounds: atom-type heaviness (weak, slightly reassuring,
-0.127) and distance to the nearest pocket contact (**+0.240, a real,
moderate, statistically clear effect in the wrong direction** — the
gradient pushes hardest on atoms *farthest* from any binding contact, not
closest). This is a plausible mechanistic account for *why* every
mechanism-level fix in Track B failed: if the gradient's strongest
component is concentrated on atoms unlikely to matter for binding, no
amount of rescaling (B1), retiming (B3), or reformulating the training
objective on the same backbone (B2) would be expected to fix it, because
none of those changes touch what the backbone's representation actually
encodes.

## Why this is a genuine dual falsification, not a single negative result generalized

The two routes are mechanistically independent:
- Track B's null result could, in principle, have been explained by
  "the Stage 0 EGNN's representation is fine, but the guidance mechanism
  built on top of it is flawed" — ruled out by testing three structurally
  different mechanisms (timing, scale, objective) on that same
  representation, all null.
- Track A's null result could, in principle, have been explained by "the
  representation is the problem, but a better, physics-grounded one would
  fix it" — ruled out by building, training to convergence, and properly
  evaluating a materially different architecture (message-passing +
  explicit physical energy decomposition, not a black-box regressor).

Both explanations have now been tested and failed. A remaining logical
possibility — that some *combination* not yet tried (e.g. Track A's
physics model + Track B1's normalization) would work — was explicitly
considered per the addendum's cross-informing guidance (§4.2) and not
pursued further, because Track B1's own result was a clean non-effect on
the existing representation, giving no positive reason to expect it would
behave differently on Track A's.

## What this conclusion does NOT claim

- It does not claim affinity-aware guidance of diffusion-generated
  molecules is impossible in principle — only that gradient-based
  steering of the x0-hat estimate, using a trained point-predictor (or
  classifier) of experimental affinity as the guidance signal, did not
  transfer to real docking outcomes across every architecture, gradient
  scale, application timing, and training objective tested in this
  project.
- It does not claim the underlying affinity predictors are useless —
  Stage 0's EGNN and Track A's GIGN+PIGNet2 both achieve real, moderate
  point-prediction correlation (Pearson ~0.58-0.61) on held-out,
  leakage-safe test data. The predictors work as predictors; they do not
  work as differentiable guidance signals for this generative task.
- It does not rule out non-gradient-based mechanisms entirely untested
  here (e.g. rejection sampling / best-of-N filtering using the same
  predictors as a post-hoc ranker, rather than an in-loop gradient) —
  that is a different mechanism class from everything tested in this
  phase and was out of scope for the Stage 2+ addendum's gradient-guidance
  investigation specifically.
- It does not rule out that a fundamentally different representation
  (e.g. one explicitly trained to have gradients aligned with pocket
  contacts, rather than only trained on point-prediction accuracy) might
  behave differently — DIAG1's finding suggests *where* the problem likely
  lies (gradient direction, uncorrelated with or anti-correlated with
  interaction-relevance), which is a concrete, falsifiable lead for future
  work, not something this phase had the remaining scope to pursue.

## Recommendation for the thesis

This is a substantial, well-evidenced negative result spanning two
independent investigative routes, each pursued to a properly-powered
conclusion with pre-registered-style checkpoints (exploratory tier before
committing to full-tier cost), transparent reporting of confounds and
degraded-quality caveats throughout, and a converging mechanistic
diagnosis (DIAG1) rather than a bare "it didn't work." Per this project's
standing principle that null results are legitimate, reportable
scientific outcomes: **present this as a genuine finding** — gradient-
based affinity guidance, as commonly formulated in the diffusion-guidance
literature, does not straightforwardly transfer to structure-based drug
design's real evaluation metrics, and the likely reason (gradient
direction misallocated relative to binding-relevant geometry) is itself a
testable, actionable hypothesis for whoever continues this line of work.

**Update (post-phase):** the "synthesizability" half of the original
"Coupled Affinity-Synthesizability Guidance" title has since been
re-investigated to the same standard (Track D, `guidance/
TRACK_D_SYNTH_GUIDANCE_REPORT.md`) and a third, non-gradient mechanism
was also tested (Track C, `guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md`).
Both reached the same overall conclusion documented here. The complete,
four-track closing synthesis — including how these results fit together
and the recommended thesis framing — is `guidance/THESIS_FINDINGS_SYNTHESIS.md`.
