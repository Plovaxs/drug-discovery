# ESM2 Pocket-Only: Worse Predictive Quality, Weak Dose-Response Confounded by Validity Collapse

Phase 3's second item: sharpens train_egnn_stage0_esm2.py's whole-protein
ESM2 feature to pocket residues only (guidance/lp_split/
esm2_pocket_embed.py, reusing alphafold_pocket_robustness.py's pocket-
extraction/alignment code -- 847/971 LP-split targets resolved, 87%
coverage). Tests whether a sharper, binding-site-specific signal does
better than the coarse whole-protein version on either predictive quality
or real-docking outcome.

## Predictive quality: worse than the whole-protein version

| | Test R² | Pearson | Spearman |
|---|---|---|---|
| ESM2 whole-protein | 0.460 | 0.731 | 0.748 |
| **ESM2 pocket-only** | **0.280** | **0.686** | **0.723** |

Best checkpoint at epoch 6 (early-stopped at epoch 11). Restricting the
embedding to ~3-60 pocket residues per target (far fewer than a typical
full chain) produced a noisier, less accurate predictor than pooling
over the whole protein -- the sharper signal did not help, and by this
metric clearly hurt. A plausible explanation (not verified further): a
small set of pocket residues gives ESM2's mean-pooling a much higher-
variance, less stable estimate per target than averaging over a full
chain of hundreds of residues.

## Diagnostic: healthy but unremarkable

3 pockets confirmed present in the pocket-embedding cache (HDAC8, CD38,
PAK4; others in the standard diagnostic set lack a resolvable pocket
embedding and were excluded rather than silently zero-filled), n=16/pocket:

| | size/heaviness corr | distance-to-pocket corr |
|---|---|---|
| ESM2 whole-protein | -0.016 | -0.072 (significant) |
| **ESM2 pocket-only** | **-0.061 (significant)** | -0.026 [-0.069, 0.017] (correct sign, not significant) |

## Real-docking test: weak monotonic trend, but confounded by validity collapse exactly where it gets closest to significance

Prelambda check (`guidance/prelambda_esm2pocket_results.json`, n=8,
standard 7-point grid, fixed pocket SQHC_ALIAD_1_631_0):

| Lambda | Mean diff vs. λ=0 | Wilcoxon p | single_fragment_rate |
|---|---|---|---|
| 0.1 | -0.110 | 0.195 | 1.00 |
| 0.3 | -0.130 | 0.148 | 1.00 |
| 1.0 | -0.248 | 0.313 | 1.00 |
| 3.0 | -0.328 | 0.945 | 1.00 |
| 10.0 | -0.957 | 0.063 | **0.62** |
| 30.0 | -2.799 (n=2) | 1.000 | **0.25** |

Unlike gradient-alignment's clean flat null, this checkpoint shows a
mean effect that grows monotonically with lambda up to 3.0 (though
non-significant at every point, p=0.15-0.95) and reaches its closest
approach to significance (p=0.063) at lambda=10 -- but exactly at that
lambda, single-fragment rate has already dropped to 0.62 (3 of 8
molecules fragmented, dropping the paired test to n=5). This is the same
"approaching significance only where validity is already compromised"
signature seen in the ESM2-whole-protein checkpoint's own near-miss
(`ESM2_GUIDANCE_FINDING.md`), not a new or cleaner pattern. Per this
project's protocol, a confirmatory run is reserved for signals that look
encouraging WITHOUT a validity confound; this one does not clear that
bar, so no confirmatory tier was run.

## Updated pooled correction: 48 tests, 0 significant

Adding these 5 tests: **48 tests pooled across the entire follow-up
phase, 0/48 significant after BH correction** (min raw p unchanged at
0.0306; min BH-corrected p=0.525).

## Conclusion

The 18th independently-trained checkpoint in this investigation, and the
first to actively make predictive quality WORSE by trying to sharpen a
signal (pocket-only vs. whole-protein ESM2) rather than add a new one.
Its real-docking result follows the same pattern as every other
checkpoint: any approach toward significance coincides with structural
validity breaking down, never a clean effect at a stable lambda.

## Artifacts

- `guidance/lp_split/esm2_pocket_embed.py` -- pocket-only embedding computation (reuses alphafold_pocket_robustness.py).
- `guidance/lp_split/train_egnn_stage0_esm2pocket.py` -- training.
- `guidance/affinity_guidance_esm2.py` -- extended with an optional `vec_for_target` override to support this checkpoint's per-target (not per-accession) embedding cache.
- `guidance/diag_size_confound_esm2pocket.py` -- DIAG1 diagnostic.
- `guidance/prelambda_esm2pocket_results.json` -- real-docking exploratory tier.
