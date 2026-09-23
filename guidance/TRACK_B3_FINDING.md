# Track B3 Finding: Timestep-Windowed Guidance — Null Result

Per the Stage 2+ addendum's explicit priority order, Track B3 (timestep-windowed
guidance) was run first as the cheapest possible test: no new training, pure
sampling-time change on the existing Stage 0 model
(`guidance_models/affinity_egnn_lpsplit.pt`).

## Setup

- 2 pockets (exploratory tier, per Sec 1.2): `BSD_ASPTE_1_130_0`, `HDAC8_HUMAN_1_377_0`
  (indices 0 and 5 of `guidance/lpsplit_confirmation_pockets.json`).
- 5 conditions per pocket, same seed (2021) across all conditions per pocket
  (isolates the guidance-window intervention against otherwise-identical
  initial noise/trajectory): `unguided` (lambda=0), `uniform` (guidance
  active across all 1000 steps — the condition used throughout every prior
  stage), `early_third`/`middle_third`/`late_third` (guidance active only
  within that third of the trajectory).
- Fixed lambda_affinity = 10.0 (reused from the Stage 0 multi-pocket
  confirmation's known-safe operating point, not re-derived).
- 20 molecules/condition/pocket, real Vina Dock (exhaustiveness=8, lighter
  than Task-F's full-tier 16 since this is exploratory), PoseBusters run on
  all.
- Gradient-informativeness test (`guidance/gradient_informativeness_test.py`):
  KS test on real Vina Dock distributions (windowed condition vs. that
  pocket's unguided baseline), BH-corrected across all 8 tests run in this
  checkpoint (2 pockets × 4 windowed conditions).

## Result

| Comparison | KS p (raw) | KS p (BH-corrected) | Direction | LE degraded? | PB degraded? | Verdict |
|---|---|---|---|---|---|---|
| BSD_ASPTE:uniform | 1.000 | 1.000 | correct (more negative) | no | no | no_signal |
| BSD_ASPTE:late_third | 0.832 | 1.000 | wrong (more positive) | yes | no | no_signal |
| BSD_ASPTE:middle_third | 0.983 | 1.000 | correct (more negative) | yes | no | no_signal |
| BSD_ASPTE:early_third | 0.983 | 1.000 | correct (more negative) | no | no | no_signal |
| HDAC8:uniform | 0.571 | 1.000 | correct (more negative) | no | no | no_signal |
| HDAC8:late_third | 0.983 | 1.000 | wrong (more positive) | yes | no | no_signal |
| HDAC8:middle_third | 0.832 | 1.000 | correct (more negative) | no | no | no_signal |
| HDAC8:early_third | 0.832 | 1.000 | correct (more negative) | no | no | no_signal |

(Full table with raw stats: `guidance/track_b3_results/gradient_informativeness_results.csv`.)

**All 8 comparisons (2 pockets × 4 windowed conditions) returned `no_signal`.**
Raw KS p-values ranged from 0.57 to 0.98 (nowhere close to a nominal 0.05
even before correction); after BH correction all 8 corrected p-values were
≈0.9999997. Directional effect: 6/8 comparisons showed the "correct" (more
negative/better) mean-shift direction, but the shifts were tiny relative to
each pocket's own vina_dock spread (e.g. BSD_ASPTE:early_third mean_shift =
-0.002, against a baseline std of ~1.6) and none were remotely significant.
No ligand-efficiency or PoseBusters degradation was detected in any
condition, so this is not a case of guidance producing an effect that's
being masked by a Vina-hacking artifact — there is simply no detectable
distributional shift in either direction.

## Interpretation

Restricting affinity guidance to any single third of the diffusion
trajectory (early/coarse-structure, middle, or late/fine-detail) does not
recover a gradient-informativeness signal that uniform guidance (already
confirmed null in the Stage 0 multi-pocket confirmation,
`guidance/LPSPLIT_LAMBDA_RESWEEP_FINDING.md`) failed to produce. This is
consistent with — not independent evidence against — the Stage 0 finding:
if the underlying gradient carried no usable signal about real docking
outcomes at any lambda, gating *where in the trajectory* it's applied
would not manufacture a signal from nothing. The result is a clean null,
not an ambiguous one (no borderline p-values, no direction reversals worth
chasing).

## Implication for remaining Track B priority order

Per the addendum's own sequencing logic ("run this first... use B3's result
to inform how much effort to invest in B1 and B2"): B3's null result does
not by itself rule out B1 (relative/reference-normalized guidance) or B2
(classifier-guidance reformulation), since both change *what quantity* the
gradient is computed from, not just *when* it's applied — a different
failure mode than B3 tests. Per the addendum's explicit requirement that
two independent falsification routes (Track A architecture-quality AND
Track B mechanism) must both fail before concluding affinity guidance is
not viable, B3 alone is not grounds to skip B1/B2. Proceeding to Track A's
exploratory-tier checkpoint next (next in the priority order), with B1/B2
still queued after it.
