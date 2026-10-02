# ESM2-Augmented EGNN Guidance: Best Diagnostics and Predictive Quality Yet, Still No Confirmed Real Effect

This closes the loop on the ESM2 branch (`guidance/lp_split/
train_egnn_stage0_esm2.py`, `guidance/affinity_guidance_esm2.py`,
`guidance/diag_size_confound_esm2.py`): adding a global ESM2 protein
language-model feature (esm2_t12_35M_UR50D, mean-pooled, 480-dim →
32-dim projection, concatenated to the pooled graph representation before
the output head) to the Stage 0 EGNN, trained on the Vina-target recipe.

## Predictive quality: clear best of every variant tried

| Checkpoint | Test R² | Pearson | Spearman |
|---|---|---|---|
| EGNN noise-matched | 0.333 | 0.599 | - |
| EGNN vina-target | 0.400 | 0.652 | - |
| EGNN gradient-alignment | 0.345 | 0.626 | 0.615 |
| **EGNN + ESM2 (vina-target)** | **0.460** | **0.731** | **0.748** |

A real, substantial jump — adding genuine sequence-derived information
helped more than any training-curriculum, target-relabeling, or
auxiliary-loss intervention tried on this backbone.

## DIAG1 diagnostic: best and only checkpoint with a correctly-SIGNED, significant distance correlation

3 pockets, n=16/pocket, 144 captured atom-level gradient measurements:

| Checkpoint | Distance-to-pocket correlation (mean, 95% CI) |
|---|---|
| Original Stage 0 EGNN | +0.240 (wrong direction) |
| Noise-matched | +0.062 |
| Vina-target | +0.047 |
| Gradient-alignment (direct optimization target) | -0.66 to -0.73 (forced) |
| **ESM2 (vina-target)** | **-0.072, 95% CI [-0.106, -0.040]** |

ESM2's correlation is small in magnitude but is the only checkpoint
*besides* gradient-alignment (which directly optimizes this exact
quantity as a training loss) to reach the correct sign with a 95% CI that
excludes zero — and it gets there as a side effect of adding sequence
information, not by being told to.

## Real-docking test: promising at n=8, substantially weaker at n=24, with emerging validity cost

Prelambda check (`guidance/prelambda_esm2_results.json`, n=8/lambda, fixed
pocket SQHC_ALIAD_1_631_0/PDB 1h36, matching every other checkpoint's
sweep protocol) showed a monotonic dose-response unlike anything seen
elsewhere in this investigation: Vina Dock improving cleanly from -10.46
(λ=0) to -12.34 (λ=3.0) with single_fragment_rate=1.0 and validity=1.0
throughout, only collapsing at λ=10 (single_fragment_rate 0.38) and
λ=30 (0.0). Per this project's standing protocol (every encouraging
small-n signal gets a confirmatory run before any claim — see the soft-v
precedent, `guidance/lambda_sweep_vinatarget_softv_highlam_results.json`
vs `..._confirm_results.json`), this was **not** reported as a finding
until confirmed at n=24.

**Confirmation (`guidance/lambda_sweep_esm2_confirm_results.json`, n=24,
same fixed pocket, paired by seed):**

| Lambda | n=8 mean diff | n=8 Wilcoxon p | n=24 mean diff | n=24 Wilcoxon p | n=24 single_fragment_rate |
|---|---|---|---|---|---|
| 1.0 | -1.255 | 0.109 | **-0.483** | **0.345** | 0.96 |
| 3.0 | -1.879 | 0.109 | **-0.990** | **0.040** | **0.83** (down from 1.00 at n=8) |

Two things happen going from n=8 to n=24, both in the direction this
project has now seen repeatedly: the effect size shrinks substantially
(λ=1.0: -1.255→-0.483; λ=3.0: -1.879→-0.990), and a validity cost appears
at n=24 that was invisible at n=8 (single_fragment_rate 1.00→0.83 at
λ=3.0 — roughly 1 in 6 molecules now fragments under guidance at this
lambda, a cost not visible in the small sample that looked cleanest).

λ=3.0's raw p=0.040 is below the uncorrected 0.05 threshold, but
**Benjamini-Hochberg correction across these 2 tests gives an adjusted
p≈0.080** — it does not survive even this minimal within-checkpoint
correction, let alone pooling with every other test in this phase (see
below). Per this project's non-negotiable standard (`THESIS_FINDINGS_
SYNTHESIS.md` §1: no claim of improvement without a test surviving
multiple-comparison correction AND a quality-degradation check), this is
**not** a confirmed effect — it is reported as a near-miss with a real
validity cost, not papered over as a success because one raw p-value
happened to clear 0.05.

## Updated pooled correction

Adding these 2 tests to `FOLLOWUP_PHASE_POOLED_CORRECTION.md`'s 25-test
pool (27 total): minimum raw p becomes 0.0306 (unchanged, Track A
follow-up) — ESM2's λ=3.0 raw p=0.0400 is the second-smallest in the
entire pool, but still does not change the pooled conclusion (0/27
significant after BH correction across the full set).

## What this adds to the investigation's central claim

This is, if anything, the *strongest* version of the pattern this
investigation keeps finding: ESM2 is a genuinely different kind of
intervention (new information source, not a curriculum/target/loss
change) and it produced the best predictive quality AND the best-signed
diagnostic of anything tried — the two numbers most likely, a priori, to
predict a real guidance effect. It still did not clear this project's own
bar for a confirmed real-docking improvement. This is the tenth
independently-trained checkpoint (4 architectures counting ESM2's
addition as a distinct variant, 2 guidance mechanisms) to show this
pattern, and the one where diagnostic/predictive improvement came closest
to mattering — which makes its failure to clear the bar more informative
for the thesis's central claim, not less.

## Artifacts

- `guidance/lp_split/train_egnn_stage0_esm2.py`, `esm2_embed.py` — training, embedding pipeline.
- `guidance/affinity_guidance_esm2.py` — guidance wrapper (`set_pocket` wiring).
- `guidance/diag_size_confound_esm2.py`, `diag_size_confound_esm2_results.json` — diagnostic.
- `guidance/prelambda_esm2_results.json` — exploratory tier (n=8).
- `guidance/lambda_sweep_esm2_confirm_results.json` — confirmatory tier (n=24).
