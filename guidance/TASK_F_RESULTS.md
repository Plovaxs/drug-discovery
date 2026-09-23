# Task F: Rigorous Ablation Results (Baseline vs. Affinity-Guidance)

## Scope note (read first)

Synthesizability guidance was evaluated separately and found not to
provide a genuine ground-truth-verified benefit in its current design
(full detail in `guidance/SYNTH_GUIDANCE_FINDING.md`: at every tested
nonzero `lambda_synth`, the guidance model's own predicted score improved
while the real, independently-computed RA-score got worse, and a
follow-up fix attempt using geometry-derived bond-aware features made
fragmentation worse rather than resolving the divergence). The ablation
below therefore evaluates **baseline vs. affinity-guidance only**.
Synthesizability is still handled — via the existing post-hoc RA-score
filtering already present in `eval/honest_eval.py` (Task B) and the
blueprint report (Task G), which remain fully valid and unaffected by the
synth-guidance finding, since they score real, finished molecules against
the real RA-score model, with no guidance model or placeholder features in
that path at all.

The "dual-guidance" config (`configs/sampling_dual.yml`) has
`lambda_synth=0.0` for this reason and is not reported here as a separate
variant — it is identical to the affinity-only variant.

## Experimental setup

- **Pockets:** 20 of the 100 official CrossDocked2020 test pockets,
  stratified-sampled (every 5th pocket in the official test split order,
  with one substitution — `LMBL1_HUMAN` swapped for `MENE_BACSU` — because
  the mirror hosting the receptor PDBs was missing that specific
  cross-docked receptor file). **20/20 selected pockets have distinct
  protein families** (no concentration), spanning human, bacterial,
  fungal, and viral targets — see `guidance/task_f_pockets.json` for the
  full list with target names.
- **Seeds:** 3 per (variant, pocket) pair (seeds 1, 2, 3).
- **Samples:** 10 molecules per (variant, pocket, seed) triple, full 1000
  diffusion steps (no step-count reduction).
- **Variants:** `baseline` (unguided, `lambda_affinity=0`) and
  `affinity_only` (`lambda_affinity=1.0`, `lambda_synth=0`) — the value
  validated as keeping 100% molecular validity throughout the earlier
  single-pocket lambda sweep (`guidance/SYNTH_GUIDANCE_FINDING.md`'s
  sibling sweep results for affinity; see also
  `guidance/sweep_affinity.json`).
- **Total:** 120 (variant, pocket, seed) triples, 1,200 attempted samples,
  **1,193 successfully scored molecules** (596 affinity_only, 597
  baseline) — **0 triple failures** across the entire run.
- **Docking:** AutoDock Vina 1.2.6, `vina_dock` mode (score/minimize/dock),
  exhaustiveness=16, against each pocket's real, coordinate-matched
  receptor PDB (`data/test_set/`, sourced from the same mirror
  — `GlowBond/CrossDocked2020` — as the processed LMDB, to guarantee
  coordinate-frame consistency; an earlier attempt using a *different*
  mirror's receptor files silently produced meaningless zero-interaction
  docking scores due to a coordinate-frame mismatch between mirrors —
  caught and fixed before this run, see the run's dry-run verification
  notes).
- **Checkpointing:** run via `guidance/run_task_f.py`, one result directory
  per (variant, pocket, seed) triple, resumable (verified with an actual
  interrupt-and-resume test, and used for real when the run was
  interrupted partway through by a laptop shutdown and resumed cleanly
  from triple 96/120).

## Results table

| Metric | baseline (mean ± std) | affinity_only (mean ± std) | Mann-Whitney p | Wilcoxon (paired by pocket, n=20) p |
|---|---|---|---|---|
| Validity rate | 0.995 (597/600 scored) | 0.993 (596/600 scored) | — | — |
| Uniqueness rate | 1.000 | 1.000 | — | — |
| Diversity (mean 1−Tanimoto) | 0.9098 | 0.9099 | — | — |
| QED | 0.449 ± 0.185 | 0.452 ± 0.185 | 0.889 | 0.648 |
| SA score | 0.601 ± 0.122 | 0.600 ± 0.121 | 0.798 | 0.277 |
| RA score (real model) | 0.435 ± 0.386 | 0.429 ± 0.387 | 0.816 | 0.312 |
| Vina Dock (kcal/mol) | −6.813 ± 10.079 | −7.251 ± 2.409 | 0.961 | 0.546 |
| Ligand efficiency (kcal/mol/heavy atom) | −0.340 ± 0.749 | −0.372 ± 0.109 | 0.956 | 0.330 |
| PoseBusters pass rate | 0.551 | 0.545 | — | — |
| Standard success rate | 0.0704 | 0.0705 | — | — |
| Honest success rate | 0.0302 | 0.0268 | — | — |

**No metric shows a statistically significant difference between baseline
and affinity-guided sampling** (all p-values > 0.27; most > 0.6). This
holds under both the pooled, unpaired Mann-Whitney U test (treating every
molecule independently) and the more conservative Wilcoxon signed-rank
test paired by pocket (n=20 pockets, averaging each pocket's 3 seeds x 10
samples into one value per pocket per variant, which controls for
pocket-to-pocket variation).

Notable: baseline's Vina Dock and ligand-efficiency standard deviations
are far larger than affinity_only's (10.08 vs 2.41 for Vina Dock; 0.749
vs 0.109 for ligand efficiency) — baseline has a heavier-tailed
distribution, almost certainly from a small number of extreme outlier
poses/scores. This is itself worth a closer look in follow-up work (single
extreme values in a Vina Dock distribution are a known artifact when a
generated ligand happens to clash badly or dock into a spurious pocket)
but does not change the significance conclusion above -- the paired
Wilcoxon test, which is more robust to this kind of skew, agrees with the
unpaired test.

## Vina-hacking check

Affinity-guided sampling's mean Vina Dock score is numerically more
negative than baseline's (−7.25 vs −6.81) -- but per the significance
tests above, **this difference is not statistically distinguishable from
noise** (p=0.96 pooled, p=0.55 paired). Proceeding with the check anyway,
for completeness:

- Ligand efficiency did **not** worsen alongside the (statistically
  insignificant) Vina Dock change (−0.372 vs −0.340 -- actually slightly
  *better*/more negative for affinity_only) -- **not** consistent with the
  classic Vina-hacking signature of "better score through size inflation."
- PoseBusters pass rate dropped slightly (0.545 vs 0.551) -- a small,
  likely-noise-level difference given the sample sizes involved, not
  flagged as a meaningful physical-plausibility regression on its own.

**Conclusion: no Vina-hacking signature detected between these two
variants** -- but this is a secondary finding, since the primary one
(no significant Vina Dock difference at all) means there is no real
affinity gain here to interrogate for authenticity in the first place.

## Honest summary: what did affinity guidance achieve here?

Applying the same no-overstatement standard used throughout this project:
**at `lambda_affinity=1.0` -- the value validated as keeping 100% molecular
validity in the earlier single-pocket sweep -- affinity guidance produced
no statistically significant change in binding affinity, ligand efficiency,
drug-likeness, synthesizability, physical validity, or success rate,
relative to unguided sampling, across a diverse 20-pocket, 3-seed,
1,193-molecule evaluation.**

This is consistent with (not contradicted by) the earlier single-pocket
lambda sweep, which already showed affinity guidance's own predicted score
barely moving across the 0.1-3.0 range and only showing a small uptick at
lambda=10 -- a value known from that same sweep to come with substantial
QED/RA-score cost and, at lambda=30, outright fragmentation. `1.0` was
chosen specifically because it was the safest point with zero observed
validity cost in that sweep; this ablation now shows that safety came
at the cost of the guidance signal being too weak, at this strength, to
move any real downstream metric on a broader, more diverse pocket set.

This is a second honest null/mixed result for the coupled-guidance
approach, alongside the synth-guidance finding -- **not** a case for
declaring the underlying architecture broken, but concrete evidence that
the current position-only, constant-strength guidance design (see
`guidance/affinity_guidance.py`'s and `guidance/interfaces.py`'s
documented simplifications) needs either a stronger operating point
(traded off against the QED/RA-score cost the sweep already measured) or
one of the architectural directions raised in `SYNTH_GUIDANCE_FINDING.md`'s
open-questions section (e.g. time-dependent guidance scheduling, so
guidance strength isn't constant across a range where it's shown to be
both too weak to matter (low lambda) and too disruptive (high lambda) with
seemingly little useful middle ground at a *fixed* strength) before a
real effect can be claimed.

## Reproducibility

- Raw per-molecule results: `guidance/task_f_results/{baseline,affinity_only}/pocket<id>_seed<seed>/honest_eval.csv`
- Aggregated statistics: `guidance/task_f_analysis.json`
- Pocket selection + diversity check: `guidance/task_f_pockets.json`
- Analysis code: `guidance/analyze_task_f.py`
- Run orchestration (checkpointed/resumable): `guidance/run_task_f.py`
