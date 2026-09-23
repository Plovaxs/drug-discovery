# DIAG1-size-direction: Size-Confound Gradient-Direction Diagnostic

**This is a standalone diagnostic, not a guidance-variant checkpoint.** No
pass/fail gate applies; it exists to help interpret *why* Track A/B
checkpoints do or don't show an effect, per the council-consultation
addendum's Part 2.

## Question

Stage 0's EGNN affinity model has a real, still-partially-present
heavy-atom-count correlation with its point predictions (0.719, per
`guidance/FLAGSHIP_ARCHITECTURE_RESULTS.md`). Since ligand atom count is
fixed by the prior *before* guidance begins each sample's trajectory, a
pure "guide toward more atoms" confound cannot be acted on by the
guidance gradient within one trajectory — but the model's gradient
*direction* at a fixed atom count could still be dominated by
weight/heaviness-correlated shortcuts (e.g. consistently preferring
heavier elements among the atom types still available) rather than by
genuinely interaction-relevant signal (e.g. orienting toward specific
pocket contacts).

## Method

Captured genuine mid-trajectory (pos0-hat, v0-hat) states from the
**unguided** sampling path (`guidance/guided_sampling.py`'s new
`capture_timesteps` hook — read-only, zero effect on the sampled
trajectory) at 3 representative timesteps (i=150/500/850, spanning
late/middle/early thirds), 3 pockets, n=16 samples/pocket. Computed the
raw (unnormalized) Stage 0 EGNN gradient at each captured state
post-hoc, then correlated per-atom gradient magnitude against:
- **Size/heaviness**: expected atomic number under that atom's *current,
  soft* type distribution (softmax of v0, not a premature argmax) — a
  continuous proxy for "how heavy is the model currently leaning toward
  making this atom."
- **Distance-to-pocket**: each atom's minimum distance to any protein
  pocket atom.

Per-sample Pearson correlations aggregated with a bootstrap 95% CI
(n_boot=2000).

## Result (n=144: 3 pockets x 3 timesteps x 16 samples)

| Correlation | Mean | 95% CI | Interpretation |
|---|---|---|---|
| Gradient magnitude vs. size/heaviness | **-0.127** | [-0.178, -0.075] | Small, but clearly nonzero (CI excludes 0) |
| Gradient magnitude vs. distance-to-pocket | **+0.240** | [0.194, 0.285] | Moderate, clearly nonzero (CI excludes 0) |

## Interpretation

**Size/heaviness (reassuring):** the correlation is small and, if
anything, in the *opposite* direction from a "push toward heavier atoms"
shortcut hypothesis — gradient magnitude is slightly LARGER for atoms the
model currently leans toward making lighter, not heavier. This does not
support the hypothesis that the model's gradient *direction* is dominated
by the same size-related shortcut that dominates its point predictions.
Combined with the mechanical argument that atom count itself is not
actionable mid-trajectory anyway, this specific confound looks unlikely to
be the main reason guidance hasn't moved real docking outcomes.

**Distance-to-pocket (a real, moderate concern):** the positive
correlation is the more interesting and less reassuring result. A
genuinely interaction-relevant gradient would plausibly show gradient
magnitude concentrated on atoms *near* specific pocket contacts (H-bond
partners, hydrophobic contacts in range) — i.e. a *negative* correlation
with distance. Instead, the largest gradient magnitudes are, on average,
on atoms *farther* from the pocket. This is a plausible partial
explanation for the persistent null/weak results across Stage 0, Track A,
and Track B3: if the model's strongest pull is on peripheral,
solvent-exposed-ish atoms rather than atoms actually forming binding
contacts, then even a well-scaled, well-timed, well-normalized gradient
(Track B1's own concern) would still be pushing on the "wrong" atoms most
of the time, independent of scale or timing.

## Implication for the rest of this phase

- This does not gate or block any other track's progress (diagnostic
  only), but should inform interpretation of Track B1's upcoming result
  and any future architecture work: **if a properly-scaled, properly-timed
  gradient still fails to move real Vina Dock scores, this diagnostic
  offers a concrete, testable reason why** — not "the model has no
  signal" in the abstract, but "the model's strongest gradient signal is
  concentrated on atoms that are mechanistically unlikely to matter for
  binding."
- Worth revisiting if Track A's full Stage 2 training proceeds: a
  physics-anchored model with an explicit distance-decaying interaction
  term (as GIGN+PIGNet2 has, via the physics head's ligand-protein pairwise
  potentials) might be expected to show a *negative* distance correlation
  by construction, unlike Stage 0's EGNN (message-passing based, no
  explicit distance-decay prior on the final gradient). This diagnostic
  was run on Stage 0's EGNN only; re-running it on Track A's checkpoint
  would be a natural, cheap follow-up if Track A's full training proceeds.

## Logging

Logged as `DIAG1-size-direction` in `guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`,
explicitly marked as a diagnostic, not a track pass/fail row.

Raw data: `guidance/diag_size_confound_results.json`.
