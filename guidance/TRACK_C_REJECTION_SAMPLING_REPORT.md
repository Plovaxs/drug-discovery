# Track C: Rejection-Sampling Guidance — Statistical Findings

**Status: FALSIFIED.** Rejection sampling with the frozen EGNN affinity
ranker as a post-hoc top-k filter does not identify genuinely better
binders. The apparent improvement in raw Vina Dock score is attributable
almost entirely to the ranker preferring larger, more often
structurally-invalid molecules — a "Vina-hacking" artifact, not a real
affinity signal. This is the third independent falsification of this
project's affinity-guidance mechanisms, joining Track A (physics-anchored
gradient guidance) and Track B (3/3 gradient-guidance mechanism variants),
via a distinct route (non-gradient, post-hoc selection) that shares the
same underlying root cause identified by DIAG1: the affinity model's
learned signal is dominated by molecular size, not target-specific
chemistry.

## 1. Design recap

20-pocket scope was reduced mid-run (user-approved, see
`STAGE2_PLUS_EXPERIMENT_LOG.md`'s `C1-scope-v2` entry) to **15 pockets ×
2 seeds = 30 pools**, reusing Task F's exact pocket set for direct
comparability with this project's other ablations. Each pool: 300
molecules sampled unguided (`lambda_affinity=0.0`), reconstructed,
scored once with the frozen affinity ranker
(`guidance/affinity_point_estimate.py`, higher = predicted stronger
binder), and docked once with real AutoDock Vina (`vina_dock`,
exhaustiveness=8) plus PoseBusters. All pools completed with zero
failures (`run_log.jsonl`); molecule counts per pool ranged 285–300 of
300 attempted (normal RDKit reconstruction attrition).

Top-k / random-subset / scrambled-top-k are all post-hoc re-selections
of this **same already-docked pool** — no extra generation or docking
was needed for the analysis (`guidance/analyze_track_c.py`).

## 2. Method

Per pocket (both seeds' pools combined, ~576–600 molecules per pocket):

1. **Real top-k**: top 10% of molecules by predicted affinity score
   (primary threshold; sensitivity also run at 5%, 20%, 30%).
2. **Random-subset null**: 2000 bootstrap draws of a same-size random
   k-subset, giving an empirical one-sided p-value for whether real
   top-k's mean `vina_dock` beats chance.
3. **Scrambled-filter null (mandatory negative control)**: 2000
   permutations of the predicted-score labels, top-k re-selected under
   each permutation. Mathematically equivalent in expectation to
   random-subset; a **null-consistency check** (KS test between the two
   null distributions themselves) validates this holds in practice — no
   pocket failed this sanity check, confirming no scrambling
   implementation bug.
4. **KS test** (real top-k `vina_dock` distribution vs. full-pool
   distribution), BH-corrected across the 15 pockets — the primary
   significance test, matching this project's established convention
   (`guidance/gradient_informativeness_test.py`).
5. **Bootstrap 95% CI** on the effect size (case-resampling, 2000
   iterations).
6. **Diagnostics**: heavy-atom-count shift (DIAG1-style size-confound
   check), PoseBusters validity rate, and — decisively — **ligand
   efficiency** (`vina_dock` / heavy atoms, a size-normalized metric),
   independently KS-tested and BH-corrected the same way.

## 3. Results: raw Vina Dock looks like a win — until normalized

| Verdict | Count |
|---|---|
| `signal_but_confounded_by_size` | 12 / 15 |
| `signal_wrong_direction` | 2 / 15 |
| `no_signal` | 1 / 15 |
| `clear_signal` | **0 / 15** |

Raw effect size (real top-k mean `vina_dock` − full-pool mean): **mean
−1.31, median −1.33**, range [−3.73, +1.95] kcal/mol-equivalent units.
12 of 15 pockets show a large, BH-significant (p < 0.001 in 11/12 cases)
apparent improvement in raw Vina Dock score. Taken alone, this would look
like a real, usable rejection-sampling signal.

## 4. The decisive test: does it survive size normalization?

Re-running the same KS-test-plus-BH-correction procedure with **ligand
efficiency** as the outcome variable instead of raw `vina_dock`:

**14 of 15 pockets show significantly WORSE ligand efficiency for
top-k-selected molecules** (all BH-corrected p < 0.0001). Mean effect:
**+0.099** (positive = worse; only PNTM_STRAE showed a small −0.03
improvement, not enough to change the overall picture).

This directly falsifies the raw-score result: the ranker is not finding
better binders, it is finding **bigger** molecules, whose raw Vina Dock
score improves mechanically (more atoms → more contacts → more negative
raw score) while their *per-atom* binding efficiency gets worse.

Supporting diagnostics, meta-analyzed across all 15 pockets:

| Diagnostic | Mean | Median | Range |
|---|---|---|---|
| Heavy-atom count, top-k minus full pool | **+12.0** | +12.7 | [+7.4, +15.1] |
| PoseBusters valid-rate drop (full − top-k) | **28.8 pp** | 31.6 pp | [5.7 pp, 52.7 pp] |

Top-k-selected molecules are **~12 heavy atoms larger on average** than
the pool they were drawn from, and their PoseBusters pass rate drops by
an average of **29 percentage points** — a severe, consistent validity
collapse. This is the classic signature of Vina-hacking: larger,
frequently-invalid geometries (steric clashes, disconnected fragments)
can receive artificially favorable raw docking scores that do not
reflect real, physically meaningful binding.

## 5. Two pockets show the opposite failure mode

`RIBB_VIBCH_2_218_0` and `P2Y12_HUMAN_1_342_0` show the ranker's top-k
selection performing **significantly worse than random** on raw
`vina_dock` (p_random = 1.0000, i.e., zero of 2000 random draws did as
poorly as the real top-k selection), robust across all four k-fraction
thresholds tested. The ranker is not merely uninformative for these
targets — it is actively anti-correlated with real docking outcome.

## 6. Sensitivity across k-fraction thresholds

The qualitative pattern (most pockets show a raw-score effect that
weakens or reverses at very small or large k) is broadly stable across
5%/10%/20%/30% thresholds, with the ligand-efficiency degradation being
the more robust and decisive signal throughout. Full per-k results in
`guidance/track_c_analysis/track_c_analysis_results.json`
(`sensitivity_by_k_fraction`).

## 6b. Exploratory follow-up: does pocket size predict susceptibility?

The per-pocket PoseBusters drop in §4 ranges widely (5.7–52.7 pp) — a
natural follow-up question is whether this variation is itself
explainable, rather than unstructured noise. One cheap, testable
hypothesis: pockets with more physical room might tolerate the ranker's
size-shortcut better (or worse). Pocket size was operationalized as the
protein atom count in each target's CrossDocked2020 `pocket10` crop (the
same 10 Å pocket definition used throughout this project), correlated
against the PoseBusters valid-rate drop via Spearman's rank correlation
with a case-resampling bootstrap 95% CI (n=15 pockets).

**Result: r = −0.59, 95% CI [−0.83, −0.13], excludes zero** — robust
under leave-one-out (r stays in [−0.53, −0.71], raw p < 0.055 with any
single pocket removed) and confirmed by Pearson (r = −0.54, p = 0.039).
As a sanity check, pocket size also correlates with the *baseline*
molecule size the model generates there (r = +0.72, 95% CI [+0.20,
+0.96]) — larger pockets do get larger unguided molecules, as expected.

**The direction is the opposite of the naive hypothesis.** Larger
pockets are *more* tolerant of the ranker's size-push (smaller PB drop),
not more susceptible. A plausible interpretation: a small, tight pocket's
baseline-generated molecules are already closely fitted to the available
space; pushing toward the ranker's preferred larger size has nowhere to
go without producing steric clashes or fragmentation, so validity
collapses sharply. A large, roomy pocket can accommodate a somewhat
larger molecule without necessarily becoming physically implausible, so
the same size-shortcut does less structural damage there — even though
(per §4) it is *still* not a genuine affinity signal in either case.

**A second predictor was tested and adds independent explanatory power.**
Per-pocket pool diversity (mean 1−Tanimoto across the full combined-seed
pool, the same metric Task F already used) shows no significant
relationship with the PoseBusters drop on its own (r=+0.24, 95% CI
[−0.26,+0.59]). But the pool's heavy-atom-count *spread* (standard
deviation across the ~600-molecule pool) does predict the PB drop
**independently of pocket size**: partial Spearman correlation, pocket
size held fixed via linear residualization, r=+0.70, 95% bootstrap CI
[+0.17, +0.91] (n=15, case-resampling bootstrap, 5000 draws). Pockets
whose unguided pool already spans a wider range of molecule sizes see a
larger PoseBusters collapse after top-k selection — plausibly because a
high-variance pool hands the ranker easier access to unusually large,
tail-end molecules to select into the top 10%, compounding the plain
pocket-size effect above. Pocket size itself correlates with pool
diversity (r=−0.59, a side finding) but diversity does not independently
predict the PB-drop outcome the way size-spread does.

*(An earlier draft of this analysis mistakenly truncated the combined
pool to its first 300 molecules when computing these two pool-level
statistics — effectively using only one of the two seeds — which
produced a non-significant partial correlation. Re-running the saved,
reusable script below on the correct, full ~600-molecule pool per pocket
reversed that conclusion to the significant result reported here; this
correction was caught by re-deriving the numbers from a clean script run
rather than trusting the first exploratory pass, consistent with this
project's verification standard throughout.)*

**Caveats (explicitly not overclaimed):** this is a small set of
exploratory correlations at n=15, not pre-registered, and not
BH-corrected against the rest of this report's hypothesis tests (run
post-hoc, after seeing the PB-drop variation, specifically to explain
it). It should be read as a plausible, testable mechanistic hypothesis
for future work — e.g. an explicit pocket-volume vs. steric-clash-rate
model — not as a confirmed causal finding. Figure and full correlation
table in `guidance/all_tracks_figures/trackC_pocket_size_vs_pb_drop.{pdf,png}`.

## 7. Conclusion

Track C's rejection-sampling mechanism — using the same frozen EGNN
affinity model that failed as a gradient-guidance signal in Tracks A and
B — **also fails as a post-hoc ranker**, via a mechanistically distinct
but root-cause-identical failure: the model's affinity signal is
dominated by molecular size rather than target-specific chemistry (same
conclusion DIAG1 reached for the gradient-guidance route). Rejection
sampling additionally surfaces a *new* failure mode not visible in the
gradient tracks — a severe PoseBusters validity collapse among
top-ranked molecules — because unlike gradient guidance, top-k selection
can freely reach into the tail of geometrically implausible structures
that happen to score well.

**No k-fraction, and no pocket save one borderline null result, produced
a `clear_signal` verdict.** Combined with Track A's null (full-tier
training, dual-checkpoint verified) and Track B's 3/3 null (gradient
normalization, classifier-head, timestep-windowing), this is the third
independent, convergent falsification of affinity-guided generation
using this project's trained affinity models — three different
mechanisms (physics-anchored gradient, learned gradient variants,
non-gradient rejection sampling) all fail for a traceable, common reason.

## Artifacts

- `guidance/analyze_track_c.py` — analysis script (reusable, seeded RNG
  for reproducibility).
- `guidance/track_c_analysis/track_c_analysis_results.json` — full
  per-pocket, per-k-fraction results, seed-consistency check, and raw
  bootstrap CIs.
- `guidance/example_render/*.png` — qualitative pose renders (separate,
  illustrative only; not part of this statistical analysis).
- `guidance/analyze_pocket_size_confound.py`,
  `guidance/all_tracks_figures/trackC_pocket_size_vs_pb_drop.{pdf,png}`
  — §6b's exploratory pocket-size-vs-susceptibility follow-up analysis
  and figure.
