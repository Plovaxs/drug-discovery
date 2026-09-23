# Flagship 5-Tier Affinity Guidance Architecture: Stage-by-Stage Results

Companion to `guidance/FLAGSHIP_ARCHITECTURE_RESEARCH.md` (technical
specification) and a direct response to `AFFINITY_MODEL_SWEEP_RESULTS.md`'s
generalization-gap finding. Each stage's checkpoint is reported here as
soon as it completes, per the addendum's mandatory gating rule: no stage
proceeds before the prior stage's checkpoint is reported and interpreted.

## Stage 0: Leakage-safe split + baseline checkpoint

### What was built

An LP-PDBBind-style split adapted to CrossDocked2020's labeled affinity
subset (pk != 0, rmsd < 2A: 1,041 targets / 76,803 entries), operating at
target granularity so all of a target's entries move together (guaranteeing
train/val/test are target-disjoint, unlike the old `build_labeled_splits`,
whose val shared 74% of its targets with train -- see
`AFFINITY_MODEL_STANDALONE_QUALITY.md`). Full implementation and two
real calibration bugs found and fixed during construction (a runaway
test-set-seeding rule, and a val-vs-test ligand-identity leak that slipped
through the original algorithm) are documented in
`guidance/lp_split/build_split.py`'s module docstring and inline comments.

**Final split**: Train 779 targets / 46,964 entries, Val 65 / 6,069, Test
127 / 11,855 (31 targets discarded for ligand contamination). Verified
zero target overlap between every pair of {train, val, test}.

### Checkpoint: deployed EGNN (unchanged) vs. ligand-only baseline, both on the new split

| Model | Val Pearson | Val R^2 | **Test Pearson (95% CI)** | **Test R^2 (95% CI)** |
|---|---|---|---|---|
| Ligand-only Ridge | 0.274 | -0.689 | 0.375 | -0.602 |
| **Ligand-only Random Forest** | 0.405 | 0.123 | **0.462** | **0.195** |
| **Deployed EGNN (lr=1e-4, 6L, unchanged)** | 0.637 | 0.394 | **0.609 (0.597, 0.621)** | **0.342 (0.322, 0.361)** |

(Test n=11,855 throughout -- 439x larger than the old split's n=27, hence
the very tight bootstrap CIs here compared to every previous checkpoint
in this project.)

Heavy-atom-count shortcut check (Diagnostic 5a's exact test, re-run on
this checkpoint): Pearson(predicted affinity, ligand heavy-atom count) =
**0.719** (down modestly from 0.799 on the old split, still a strong
effect).

### Honest interpretation -- this is a genuinely surprising, important result

**The addendum's own stated expectation for this checkpoint was: "expect
it to still show a gap (this split alone doesn't fix the model)."** That
expectation was wrong, and reporting that plainly matters more than
protecting the plan's premise. With NO architecture or hyperparameter
changes at all -- the literal currently-deployed EGNN config -- fixing
only the split produces a real, statistically robust positive result
(R^2=0.342, CI entirely above zero, on a test set 439x larger than
before) that clearly and meaningfully **beats the ligand-only baseline**
(R^2 0.342 vs 0.195, non-overlapping in the direction that matters) --
the opposite of the old split's finding, where ligand-only matched the
full model almost exactly.

This strongly suggests that a substantial part of the original
"generalization gap" documented in `AFFINITY_MODEL_STANDALONE_QUALITY.md`
and investigated at length in `AFFINITY_MODEL_SWEEP_RESULTS.md` was a
**split-construction artifact** -- the old val set's 74% target overlap
with train, and the old test set's tiny, high-variance n=27 sample --
rather than a fundamental EGNN architecture limitation or an
unfixable data-scarcity ceiling. The earlier diagnostics (ligand-only
matching the full model, Diagnostics 3-5) were correct as measurements
*on that specific split*, but the causal story built on top of them
("the gap is architecture-independent and needs a fundamentally
different model / physics decomposition / more data") looks, in light of
this checkpoint, like it was answering the wrong question -- the visible
gap was substantially a split artifact, not (only) a property of the
model or the available data's intrinsic information content.

**This does not mean the flagship architecture's remaining stages have no
value** -- the heavy-atom-count shortcut signature is still present and
strong (0.719), meaning there is real remaining room for a more
physically-grounded model (Stage 2's PIGNet2 decomposition specifically
targets this exact failure mode) to do better than a plain EGNN even on
this corrected split. But the size and direction of this result is
different enough from what the addendum anticipated that it changes the
relative priority of continuing to Stage 1 versus first re-testing
guidance (Task F) with this already-much-better, zero-additional-
architecture-work model -- a decision surfaced to the user rather than
assumed, given how directly it bears on how the remaining multi-week
effort should be sequenced.

### Reproducibility

- Split construction: `guidance/lp_split/build_split.py`, `guidance/lp_split/uniprot_client.py`, `guidance/lp_split/protein_similarity.py`, `guidance/lp_split/ligand_similarity.py`, `guidance/lp_split/pocket_similarity.py`
- Final split: `guidance/lp_split/leakage_safe_split.json`
- EGNN training (unchanged deployed config on the new split): `guidance/lp_split/train_egnn_stage0.py`, checkpoint at `logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt`
- Ligand-only baseline: `guidance/lp_split/ligand_only_baseline_stage0.py`, results in `guidance/lp_split/ligand_only_baseline_results.json`
- Bootstrap CI + heavy-atom-count check: `guidance/lp_split/stage0_checkpoint_analysis.py`, results in `guidance/lp_split/stage0_checkpoint_analysis_results.json`

## Guidance re-test (Stage 0 model): CONFIRMED null result

Following Stage 0's checkpoint, the user chose to re-test guidance
(Task F) with the corrected model before continuing to Stage 1+. Full
detail in `guidance/LPSPLIT_LAMBDA_RESWEEP_FINDING.md`; summary:

- Single-pocket lambda re-sweep (8 lambda values, 0.1-100): no
  statistically significant guidance effect on real Vina Dock score at
  any point (all p>0.69), though structural robustness improved
  markedly (fragmentation delayed from lambda=30 to lambda=100 vs. the
  old model).
- Multi-pocket confirmation (8 diverse pockets, 3 representative
  lambdas, real per-pocket Vina docking): **confirms the null result
  holds generally** -- all 24 per-pocket comparisons non-significant
  (p=0.105-1.000); one nominally-significant paired test at lambda=0.3
  (p=0.023) reflects a negligible, wrong-directioned effect (+0.03
  kcal/mol, i.e. very slightly worse binding under guidance), not a real
  positive signal, and does not survive scrutiny as a genuine finding.

**Conclusion**: fixing the affinity model's generalization gap (Stage 0)
was necessary but not sufficient for affinity guidance to work in this
project's current guidance design. This is now a confirmed, not merely
suggestive, negative result -- the natural next decision is whether to
continue to Stage 2+ (GIGN backbone, PIGNet2 physics anchoring) in
pursuit of a guidance-useful gradient, or to treat this as the thesis's
final honest account of the coupled-guidance approach's current limits.

**UPDATE (2026-09-12): this question has been resolved.** Stage 2+ was
pursued to completion -- a full GIGN+PIGNet2 physics-anchored model was
built, trained to convergence, and evaluated at full statistical power
alongside three independent guidance-mechanism variants (Track B). Both
routes returned clean null results. See
`guidance/DUAL_FALSIFICATION_CONCLUSION.md` for the phase-closing
synthesis and `guidance/TRACK_A_FULL_TIER_FINDING.md` /
`guidance/STAGE2_PLUS_EXPERIMENT_LOG.md` for the full evidence trail.
