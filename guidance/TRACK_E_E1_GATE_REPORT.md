# Track E — E1 Gate Report (delta-learning target, Option F)

**Verdict: E1 FAILS the pre-registered gate (C1a–C1e). E2 was not run.** This is a well-powered negative result on the
decision-relevant side (the upper end of every confidence interval that matters is below the pass threshold), reported as a valid
finding. Pre-registration: `TRACK_E_DESIGN.md` (frozen at `8d913c6`; amendments A1–A10 changed no threshold). Raw results:
`guidance/track_e/e1_gate_results.json`; per-run predictions in `runs_track_e/*/test_preds.npz` (git-ignored).

## 1. Setup actually run
LP split (train 46,964 / val 6,069 / test 11,855; 127 test targets). Track A GIGN+PIGNet2 backbone (hidden 256, 0.494 M parameters), batch 1, val-loss early stopping (patience 12, cap 30 epochs),
test evaluated once from `best.pt` (positive control: `tests/test_no_test_selection.py`). Anchor pK_Vina = max(−vina, 0)/1.364; Δ standardised on train (mean 0.518, SD 1.753).
Arms × 3 seeds: **A0′** (absolute target, matched control), **E1** (Δ target, physics-sum head with free-sign scale + bias), **E1-MLP** (Δ, scalar MLP readout); plus 2 extra Stage-0 EGNN seeds (2022, 2023).
Statistics: paired target-cluster bootstrap (2,000 resamples, seed 20260925), an arm's value = mean over its seeds inside each resample; CIs at 90/95/99%; BH across tests.

## 2. Results (test set, pooled R², target-clustered 95% CI)

| Model | Per-seed R² | Mean R² [95% CI] | Within-target Spearman | Corr(ŷ, heavy atoms) |
|---|---|---|---|---|
| **E1** (Δ, physics head) | 0.102 / 0.176 / 0.074 | **0.117** [−0.197, 0.316] | 0.258 | 0.677 |
| E1-MLP (Δ, MLP head) | −0.391 / −0.368 / 0.123 | −0.212 [−0.806, 0.130] | 0.269 | 0.576 |
| A0′ (absolute target) | 0.275 / 0.259 / 0.242 | 0.259 [0.099, 0.360] | 0.225 | 0.611 |
| Stage 0 EGNN (pre-registered reference) | 0.342 | 0.342 [0.160, 0.465] | 0.195 | 0.719 |
| EGNN, 3 seeds (Stage 0 + 2022 + 2023) | 0.342 / 0.347 / 0.406 | 0.365 [0.196, 0.477] | 0.180 | – |
| Vina + heavy-atom linear | – | 0.362 [0.200, 0.470] | 0.266 | 0.940 |
| Vina only, calibrated / raw | – | 0.273 / 0.076 | 0.248 | 0.622 |
| Track A GIGN+PIGNet2 (no bias, existing) | – | −0.189 [−0.768, 0.162] | 0.233 | – |

## 3. Pre-registered criteria (E1, seed-averaged; differences are E1 minus reference)

| ID | Test | Difference [95% CI] (99% CI) | Verdict |
|---|---|---|---|
| C1a | R² vs Stage 0 EGNN (need ≥ +0.05, CI lower > 0) | **−0.224** [−0.422, −0.091] (−0.499, −0.063) | **fail** — significantly *worse* |
| C1b | R² vs Vina + heavy-atom linear (CI lower > 0) | **−0.244** [−0.444, −0.121] (−0.513, −0.092) | fail |
| C1c | within-target Spearman vs raw Vina (CI lower > 0) | +0.010 [−0.018, +0.038] (−0.028, +0.048) | fail (not different) |
| C1d | Corr(ŷ, heavy atoms) ≤ 0.605 | 0.677 | fail |
| C1e | R² vs A0′ (isolates the delta target) | **−0.141** [−0.318, −0.026] (−0.379, +0.007) | fail — delta target did not help; worse at 95% |
| sens. | R² vs 3-seed EGNN mean | −0.247 [−0.426, −0.134] | fail (same conclusion) |

Effect sizes: ΔR²/bootstrap-SD z = −2.63 (C1a), −2.97 (C1b), −1.83 (C1e); paired Cohen's d_z vs Vina + heavy-atom = −0.33. BH-adjusted one-sided p (H1: E1 better) = 1.0 for all improvement tests.
E1-MLP fails as well (ΔR² vs EGNN −0.553 [−1.085, −0.249]); its guard C1d passes only because it is a poorer predictor overall.
The PCGrad, λ-sensitivity and E2 arms were **not run** (E2 is scoped as an addition to a working E1).

## 4. Power, recomputed for the seed count actually used
Seed-averaged paired bootstrap SD of ΔR² is **0.075–0.085** (the pre-registration assumed 0.044 from single-model comparisons), so the probability of detecting a true +0.05 gain is only **≈ 8–10%**.
This concern **did not decide the outcome**: the result is not inconclusive, because the *upper* CI bounds (−0.091, −0.121, −0.026) lie below zero; the design could not have confirmed a small positive effect, but it clearly excludes the required one.
Baseline seed variance is now known: EGNN R² 0.342 / 0.347 / 0.406 (range 0.064), smaller than E1's shortfall (0.22–0.25).

## 5. Why it failed (evidence, not speculation)
- **Δ is close to unlearnable from structure by this model.** Best validation loss (standardised units) of E1 is 0.87–0.95, i.e. about the loss of a predictor that always outputs the mean Δ (≈ 0.92 on this validation set: val SD 1.657, mean shift 0.72 vs train 0.51), while the absolute-target control genuinely learns (0.68–0.73). E1-MLP (≈ 1.02) is worse than the constant.
- **The constant residual already scores R² 0.193** on test (ŷ = pK_Vina + mean Δ); E1 (0.117) does not beat it, so the learned Δ̂ adds nothing measurable beyond the Vina anchor.
- Both delta arms peak at epoch 1–2 (E1-MLP once at epoch 13) and then overfit (train loss falls to ≈ 0.25 while validation loss rises), as the absolute arm and Track A do.
- The reconstruction inherits the anchor's size behaviour (Corr(ŷ, heavy atoms) 0.68 vs 0.56 for ground truth).
- The GIGN+PIGNet2 backbone is itself weaker than the EGNN under this budget (A0′ 0.259 vs EGNN 0.34–0.41), so E1 was tested on the weaker of the two backbones; C1e is the backbone-independent comparison and it also fails.
- Untested hypotheses (do **not** state as findings): pK label heterogeneity, cross-docking provenance (A9) and the ≈ 0.8 kcal/mol Vina-preparation noise (D6) may cap how much of Δ any structure model can recover.

## 6. Limits of this conclusion
Scoped to: this architecture and training budget, 3 seeds, a 127-target test set, the CrossDocked (cross-docked-pose) label regime, and a fixed unweighted MSE loss. It does **not** show that delta-learning fails in general (tree-ensemble Δ-models on hand-built features succeed per the review, Sec. 1.1), nor that E2 cannot help a different E1.
E1 was tested only for *scoring*; C4 (gradient diagnostics) was not run because no arm passed C1.
Scalar-only: within-target ranking is the guidance-relevant metric and every arm sits at 0.18–0.27 with overlapping CIs (raw Vina 0.248); no model demonstrated within-pocket ranking clearly above raw Vina.

## 7. Budget, timing, interruptions
Per-run wall-clock: A0′ 2.2–2.6 h, E1 2.2–2.4 h (estimate 2.4 h), E1-MLP 1.5–2.8 h, EGNN seeds 3.5 h and 3.3 h; total E1 stage ≈ 27 GPU-h (started 25 Sep 21:34, finished 27 Sep 00:23 WIB), inside the 24–27 h estimate.
**Interruptions: none** (0 pause/resume/failure events in `progress.jsonl`), so equivalence of resumed vs uninterrupted runs was verified only in the dedicated tests (bitwise identical on CPU; within CUDA non-determinism on GPU), **not** on the production runs.

## 8. Consequence for the thesis and next decisions
Track E is a **completed negative result**: neither a Vina-anchored delta target nor (untested) PLIP supervision on top of it rescues the affinity model for guidance — consistent with DIAG1 and Tracks A–D. The E2 budget (≈ 28 GPU-h) is unspent.
Decisions for the user (none started): (i) write up as the Track E result in the Results chapter, novelty wording per A8; (ii) optional cheap analyses that need no retraining — anchor-perturbation robustness (A3) and BEDROC/EF re-scoring of previously generated molecules; (iii) any *new* hypothesis (e.g. E2 alone on the absolute target, a different backbone) needs its own scoping decision; it is not implied by this gate.
