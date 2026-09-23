# Stage 2+ Cross-Track Summary: All Track B Variants + Track A Exploratory Tier Complete

Per the Stage 2+ addendum's explicit requirement, **two independent
falsification routes (Track A architecture-quality AND Track B mechanism)
must both fail before concluding affinity guidance is not viable** —
neither alone is sufficient. This summary consolidates where each route
stands now that all three Track B variants and Track A's exploratory tier
have been run.

## Track B (mechanism route): all three variants tested, none supported

| Variant | Mechanism | Verdict | Notable finding |
|---|---|---|---|
| B3 | Timestep-windowed guidance (gate guidance to early/middle/late third of trajectory) | no_signal, clean null | All effects negligible (p in [0.57, 0.98] raw); no window recovers a signal uniform guidance lacked |
| B1 | Gradient-norm normalization (unit-per-atom rescale before lambda) | Hypothesis not supported | Normalized indistinguishable from (often worse than) raw gradient at its own freshly-derived stable lambda |
| B2 | Classifier-guidance reformulation (new head, same frozen backbone, `grad log p(y|x)` target) | no_signal, wrong-direction-leaning | Worse (by median) in 5/6 comparisons; most severe PoseBusters collapse seen this phase (HDAC8: 0.60->0.05) |

**All three mechanism-level fixes tested on the Stage 0 model's
representation have failed to produce a detectable, correctly-directed,
undegraded guidance effect.** Combined with `DIAG1-size-direction`'s
diagnostic finding (gradient magnitude correlates *positively* with
distance-to-pocket, +0.240 — the model pushes hardest on atoms farthest
from any binding contact), a consistent picture emerges: **the Stage 0
EGNN backbone's learned gradient direction, not its scale (B1) or its
training objective (B2) or when it's applied (B3), is the limiting
factor.** No sampling-time-only or head-only fix was able to route around
this within the Track B budget.

## Track A (architecture route): exploratory tier only, not yet conclusive

The GIGN+PIGNet2 physics-anchored model's exploratory checkpoint (`A2-exp`)
also showed no signal surviving BH correction — but with a qualitatively
different, more promising pattern than any Track B result: large,
consistently-directed raw effect sizes (up to -1.48 mean shift, vs. Track
B's largest real effects being ~-0.7 at best), two nominally-significant
raw p-values before correction, confounded specifically by PoseBusters
degradation at the high end of its (much narrower) stable lambda range.
**This is only the exploratory tier** (partial training, 5000 iterations,
~1.67 epochs, test Pearson=0.583 vs. Stage 0's fully-converged 0.609) —
Track A's full Stage 2 training run (the expensive, multi-hour-to-multi-
day commitment) has not been attempted.

## Where this leaves the addendum's dual-falsification requirement

**Not yet met.** Track B (mechanism route) is now exhausted with no
supported hypothesis across all three variants. Track A (architecture
route) has only had its cheap exploratory check done — which, notably,
was the LEAST clean null of anything tried this phase, not a clear
failure. Per the addendum's own logic, this is exactly the situation
where Track A's full training becomes the more informative next step:
Track B's mechanism-level fixes are used up, and Track A's own
exploratory signal (however unconfirmed) is the only remaining thread
that looked meaningfully different from pure noise.

**This is also the point at which the addendum's "full Stage 2 training
is the most expensive item in the queue" framing becomes directly
relevant** — proceeding requires the kind of multi-hour-to-multi-day GPU
commitment this project's standing practice is to flag and let the user
decide on, rather than committing to autonomously. See the accompanying
report to the user for that decision point.

## What would constitute the phase's conclusion

Per the addendum's explicit routing rule: if Track A's full Stage 2
training ALSO fails to produce a signal (clearing the same dual-criterion
bar), that would satisfy the two-independent-falsification-routes
requirement, and the phase's conclusion — "affinity guidance, as
implemented via gradient-based x0-hat steering in this project, does not
transfer detectably to real docking outcomes across architecture,
gradient scale, training objective, and application timing" — would be
ready to write up. Until Track A's full tier is run (or explicitly
deprioritized by the user), that conclusion is not yet reachable.
