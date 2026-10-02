# Kitchen-Sink Checkpoint: All Four Interventions Combined, No Emergent Synergy

Closes the one remaining untried cell in this phase's ablation space
(`guidance/lp_split/train_egnn_stage0_kitchensink.py`): noise-matched
curriculum + Vina-target label + gradient-alignment auxiliary loss +
ESM2 global feature, trained together in one checkpoint. Run because it
was the one combination not yet tested after 11 independently-trained
checkpoints showed the same null real-docking pattern, not because of a
specific mechanistic reason to expect synergy (see the module's own
docstring, which says this plainly before training).

## Predictive quality: good, but below ESM2 alone

| Checkpoint | Test R² | Pearson | Spearman |
|---|---|---|---|
| Vina-target | 0.400 | 0.652 | - |
| Gradient-alignment | 0.345 | 0.626 | 0.615 |
| **ESM2 alone** | **0.460** | **0.731** | **0.748** |
| **Kitchen-sink (all four)** | 0.413 | 0.664 | 0.686 |

Early-stopped at epoch 7; best checkpoint at epoch 2 (val loss 1.017).
Training loss kept falling sharply after epoch 2 (0.95 -> 0.54 -> 0.33 ->
0.24 -> 0.16 -> 0.14) while validation R² briefly went *negative*
(-0.929 at epoch 3) before partially recovering -- a visible overfitting
signature once the four combined objectives are optimized jointly for
too long, not present in any single-intervention checkpoint at this
training budget. Stacking everything did not beat ESM2 alone; if
anything, combining made the optimization landscape harder to hold at
its best point.

## Diagnostic: distance-correlation preserved near its best, size-confound at its worst

| Checkpoint | size/heaviness corr | distance-to-pocket corr |
|---|---|---|
| ESM2 alone | -0.016 | -0.072 |
| Gradient-alignment alone | **+0.331 (previously worst)** | **-0.784 (previously best)** |
| **Kitchen-sink** | **+0.354 (now worst)** | -0.757 (2nd best, ~preserved) |

Adding ESM2's global feature on top of gradient-alignment's loss did
**not** counteract gradient-alignment's size-confound side effect (the
one open question this combination could have answered) -- it made that
side effect slightly worse while leaving the distance-correlation fix
nearly intact. The two objectives did not interfere with each other
much, but they did not help each other either.

## Real-docking test: another clean, flat null

Prelambda check (`guidance/prelambda_kitchensink_results.json`, n=8,
standard 7-point grid, fixed pocket SQHC_ALIAD_1_631_0):

| Lambda | Mean diff vs. lambda=0 | Wilcoxon p | single_fragment_rate |
|---|---|---|---|
| 0.1 | +0.021 (worse) | 0.109 | 1.00 |
| 0.3 | +0.102 (worse) | 0.461 | 1.00 |
| 1.0 | -0.032 (better, tiny) | 0.742 | 1.00 |
| 3.0 | -0.014 (better, tiny) | 0.313 | 1.00 |
| 10.0 | +0.259 (worse) | 0.313 | 1.00 |
| 30.0 | +0.725 (worse) | 0.078 | **1.00** |

No encouraging direction at any lambda -- diffs are small and
inconsistently signed, worst at the highest lambda rather than showing a
dose-response. Notably, structural validity never degrades even out to
lambda=30 (every other checkpoint tested in this investigation collapses
well before that point) -- the ESM2 conditioning appears to make
generation more robust to guidance strength even though it does not make
the guidance itself effective.

## Updated pooled correction: 38 tests, 0 significant

Adding these 6 tests: **38 tests pooled across the entire follow-up
phase, 0/38 significant after BH correction** (min raw p unchanged at
0.0306; min BH-corrected p=0.462).

## Conclusion for this phase

The kitchen-sink result closes the ablation space cleanly: no pairing or
full combination of the four interventions tried in this investigation
(noise-matching, Vina-target, gradient-alignment, ESM2) -- alone, in
pairs (Track A follow-up), or all four together -- produces a real-
docking effect that survives multiple-comparison correction. The total
count of independently-trained checkpoints showing this pattern is now
**12**, spanning 3 architectures and every tested combination of 4
distinct mechanism-level interventions. Per the plan agreed for this
phase, this is the last "deepen the pattern" item; the next phase moves
to closely-connected but mechanistically different branches (PLIP
interaction-type validation) rather than further combinations of the
same four interventions.

## Artifacts

- `guidance/lp_split/train_egnn_stage0_kitchensink.py` -- training.
- `guidance/diag_size_confound_kitchensink_results.json` -- DIAG1 diagnostic.
- `guidance/prelambda_kitchensink_results.json` -- real-docking exploratory tier.
