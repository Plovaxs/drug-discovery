# Direct Trace of the Guidance Mechanism: The Nudge Is Swamped by the Diffusion Model's Own Step Noise, Not Erased by a Corrective Force

Direct test of the best-supported-by-elimination hypothesis from
`FOLLOWUP_PHASE_POOLED_CORRECTION.md` and the per-checkpoint finding
docs: that the guidance mechanism itself (a deterministic gradient nudge
added to the per-step clean-data estimate, on a diffusion model that was
never trained to expect it) is the bottleneck, not any remaining defect
in the predictor. This asks directly: what actually happens, step by
step, when guidance is applied?

## Method

One molecule, sampled TWICE from bit-identical starting noise and
identical per-step stochastic draws (same seed reset immediately before
each call) -- once with lambda_affinity=0 (unguided), once with
lambda_affinity=1.0 on the gradient-alignment checkpoint (chosen because
its gradient is the most directly and successfully trained to point
toward the pocket of any checkpoint in this investigation -- if any
checkpoint should show a clean, persistent guidance effect, it is this
one). Since the diffusion core's forward pass is deterministic given its
input and guidance introduces no new randomness, the two trajectories
are identical up to the first guided step; every atom of divergence
after that is attributable only to the guidance nudge, not independent
sampling noise. New instrumentation
(`guided_sampling.py`'s `grad_capture_list`) records the nudge actually
applied at every step; `pos_traj`'s existing per-step output is compared
between the two runs for every atom in the molecule.

## Result

**The guided-vs-unguided position gap starts at exactly zero, grows
smoothly (not flat, not oscillating), accelerates sharply in the final
third of sampling, and ends tiny:**

| Step (raw timestep) | Mean all-atom gap |
|---|---|
| 0 (999, noisiest) | 0.0000 A |
| 100 (899) | 0.0006 A |
| 400 (599) | 0.0014 A |
| 700 (299) | 0.0035 A |
| 800 (199) | 0.0085 A |
| 900 (99) | 0.0167 A |
| 1000 (0, final) | **0.0512 A (mean), 0.1314 A (max, any single atom)** |

The single atom receiving the largest cumulative raw nudge (summed
|nudge| across all ~1000 guided steps = 9.49, in the guidance model's
own gradient units) ends the trajectory only **0.0362 A** away from
where it would have been with no guidance at all -- roughly 1% of what
the raw cumulative nudge magnitude would suggest if nudges simply added
up linearly.

## Interpretation: swamped by the model's own step noise, not actively cancelled

This is a specific, falsifiable mechanistic story, not just "the effect
is small": at each reverse-diffusion step, the posterior sampling step
adds `0.5*exp(log_variance) * randn()` -- the model's OWN intrinsic
stochastic noise for that step. Early in sampling (high raw timestep,
high posterior variance by the DDPM schedule), this intrinsic noise is
large relative to the small deterministic guidance nudge, so the nudge
is statistically buried in noise the model would have injected anyway --
consistent with the near-zero gap through roughly the first 80% of
sampling. Only as the schedule's posterior variance shrinks toward the
end of sampling does the same-sized deterministic nudge's relative
contribution grow enough to produce a measurable, compounding effect --
consistent with the sharp late-trajectory acceleration observed. The
nudge is not being "erased" by an active corrective force each step; it
is mostly drowned out by noise the model would be adding regardless of
guidance, with only a small residual surviving to compound.

## A puzzle this raises, checked rather than glossed over

Despite the tiny geometric gap (≤0.13 A for any atom), the two runs'
*final reconstructed molecules have completely different SMILES* (both
single-fragment, both valid). This was checked, not assumed away: the
most likely explanation is that atom-TYPE sampling
(`log_sample_categorical`, a Gumbel-max argmax -- a discrete,
decision-boundary-sensitive operation) is sensitive to even this small a
position-driven perturbation at some intermediate step, after which the
coupled position+type forward pass diverges in the discrete channel far
more than in the continuous position channel -- a "butterfly effect"
specific to discrete sampling, not evidence that the position effect
itself was actually large. This was not exhaustively diagnosed (which
specific step the type-sampling decision flips at was not isolated) and
is flagged as an open detail, not claimed as settled.

## What would have rejected the hypothesis (stated before running, not after)

If the gap had grown to be a large fraction of the raw cumulative nudge
(e.g. multiple Angstroms, comparable to the 9.49 raw-nudge-unit sum)
while the molecule remained chemically valid, that would point to the
guidance DIRECTION being the problem (a persistent, uncancelled push in
a direction that simply doesn't help Vina Dock), not the injection
mechanism. That is not what was observed -- the gap stays two orders of
magnitude below the raw nudge sum throughout, consistent with the
injection-mechanism hypothesis specifically, not a generic "nothing
happened" result.

## Scope and honest limits

- n=1 molecule, 1 checkpoint, 1 lambda. This is a mechanistic case study
  demonstrating a concrete, falsifiable phenomenon, not a population-
  level statistical claim (that evidence is `FOLLOWUP_PHASE_POOLED_
  CORRECTION.md`'s 38-test pool). Repeating this trace across more
  molecules/checkpoints/lambdas would strengthen it but was not done.
- This explains *why* the position-level nudge has limited reach; it
  does not by itself fully explain the molecule-identity divergence via
  discrete-type sampling, which is flagged as unresolved above.

## Artifacts

- `guidance/trace_guidance_trajectory.py` -- script.
- `guidance/guided_sampling.py` -- new `grad_capture_list` instrumentation hook (inert unless passed).
- `guidance/trace_guidance_trajectory_results.json` -- full per-step trace data.
