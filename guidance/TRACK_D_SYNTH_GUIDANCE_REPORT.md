# Track D: Synthesizability-Guidance Re-Investigation — Consolidated Report

**Status: FALSIFIED, mechanism unresolved.** Gradient-based guidance
toward higher predicted synthesizability (RA-score) does not transfer to
the real, independently-computed RA-score of generated molecules — the
guidance model's own score reliably rises while the real target reliably
falls, a synthesizability-guidance analogue of "Vina-hacking." This
divergence survives two independent, principled fix attempts (bond-aware
input features; a leakage-safe retrain to real convergence) and remains
mechanistically unexplained after two diagnostics. This consolidates
D.2 through D.5 (previously scattered across four separate finding docs)
into one report, mirroring `TRACK_C_REJECTION_SAMPLING_REPORT.md`'s
structure.

## 1. Where this started: the original divergence finding

Before this phase, `guidance/SYNTH_GUIDANCE_FINDING.md` had already found
a striking pattern with the placeholder-feature guidance design (single
bundled pocket, n=8/point, no statistical testing): as `lambda_synth`
increased, the guidance model's own predicted score rose monotonically
(0.458 → 0.864 at λ=1.0) while the real, independently-computed RA-score
collapsed (0.361 → 0.126 at λ=0.3, before molecules fragmented entirely
at higher λ). The diagnosed root cause was a train/inference feature
mismatch: the model was trained on real RDKit-derived bond features
(Degree, NumHs, Hybridization), but the in-loop guidance gradient was
computed against placeholder "unknown" values for those same fields,
since the diffusion trajectory has no discrete bonds mid-generation. A
bond-aware feature fix was implemented and partially improved but did not
resolve the divergence at the time.

Per the Stage 2+ addendum's Track C/D directive, this phase re-opened the
question at publication-grade rigor rather than assuming the prior
exploratory-scale conclusion generalized.

## 2. D.2: Status re-establishment found two more gaps, not anticipated

Direct log inspection (not assumption) revealed the prior work needed
more than "just test at higher power":

1. **Exploratory-scale only.** Both prior checkpoints (placeholder and
   bond-aware) used 1 pocket, n=8/point, single seed, and zero statistical
   testing (no KS test, no BH correction, no bootstrap CIs) — directionally
   plausible given how large the raw effects were, but not rigorously
   established.
2. **The synth model itself was not trained to convergence.**
   `guidance_models/synth_ra_score.pt` hit a fixed `max_epochs=40` ceiling
   with its early-stop patience counter only at 1/6 — an artificial cutoff,
   not a genuine plateau, the same category of gap Track A's exploratory
   tier had.
3. **A genuine, previously-unflagged leakage-safety gap.** The synth
   dataset's train/val split (`datasets/synth_dataset.py`) was a plain
   random 90/10 split with **no held-out test set at all** and **no
   leakage check** — the exact class of problem Stage 0 was built to fix
   on the affinity side. The originally-reported quality (R²=0.769,
   Pearson=0.878) rested entirely on this un-audited split.

## 3. D.2a: Leakage-safe retrain — a major correction to reported quality

Fix reused 100% existing infrastructure: the same target-level leakage-safe
assignment already built for the affinity side
(`guidance/lp_split/leakage_safe_split.json`) was applied to the synth
dataset's 15,000 RA-score-labeled entries by target identity (no new
RA-score computation needed), giving **8,835 train / 1,015 val / 1,940
test** (3,210 discarded — target not covered by the leakage-safe
assignment). The identical `SynthPredNet` architecture was retrained with
no artificial epoch cap (early-stopped at epoch 20, patience=15; best
checkpoint epoch 5).

| | Original (leakage-unchecked, val only) | New (leakage-safe, held-out test) |
|---|---|---|
| R² | 0.769 | **0.414** |
| Pearson | 0.878 | **0.647** |
| Spearman | 0.871 | **0.649** |
| RMSE | 0.192 | **0.287** |

The genuine, held-out predictive quality is substantially lower than
previously reported (R² nearly halves) — not a training regression (this
run converged properly; the original did not), but the removal of
leakage-driven inflation. The corrected picture: the model still predicts
RA-score meaningfully better than chance, and still better than either
affinity model ever achieved (Pearson ~0.58–0.61) — RA-score is an easier
target to predict from 3D structure than experimental binding affinity.

## 4. D.3: Full lambda re-sweep — divergence persists at proper power

Single-pocket sweep (BSD_ASPTE_1_130_0), bond-aware features, using the
new leakage-safe checkpoint, n=30/lambda (up from n=8), full 1000
diffusion steps, grid 0.0–30.0.

| λ_synth | N reconstructed | Single-fragment rate | N scoreable | Own score (mean) | Real RA-score (mean) |
|---|---|---|---|---|---|
| 0.0 (baseline) | 30/30 | 0.90 | 27 | 0.5605 | 0.2581 |
| 0.01 | 30/30 | 0.90 | 27 | 0.5668 | 0.2515 |
| 0.03 | 29/30 | 0.90 | 26 | 0.5922 | 0.1916 |
| 0.1 | 30/30 | 0.77 | 23 | 0.5784 | 0.1863 |
| 0.3 | 29/30 | 0.21 | 6 | 0.5577 | 0.0659 |
| 1.0 | 28/30 | 0.07 | 2 | 0.7925 | 0.2391 |
| 3.0+ | — | 0.00 | 0 | — | — |

KS test (real RA-score, guided vs. unguided), BH-corrected across the 5
testable lambdas: **no signal survives correction** (min corrected
p=0.997) — matching the "no significant effect after correction" pattern
seen across every guidance mechanism tested in this project (Tracks A,
B1, B2, B3).

**But the direction is not ambiguous.** Every single tested lambda shows
the same-direction shift — real RA-score mean is lower than baseline at
every non-zero lambda, never once higher. At λ=0.03 (N=26, the largest
well-powered non-trivial point), the classic divergence signature
reappears: own score rises (+5.7%) while real RA-score falls (−25.8%) —
the same pattern as the original finding, now reproduced against a
properly leakage-safe, converged model. The λ=0.3 point shows the largest
raw effect (−0.192, own CI excludes zero) but rests on only N=6 scoreable
molecules (heavy fragmentation), exactly the kind of large-point-estimate
on a tiny, underpowered sample this project's rigor standard exists to
catch — not reported as a standalone finding.

**Per the pre-registered gating rule** ("if divergence persists even
partially, proceed directly to the mechanistic diagnostic rather than
spending multi-pocket confirmation effort on an unresolved single-pocket
result"): **D.4 (multi-pocket confirmation) was skipped**, proceeding
directly to D.5.

## 5. D.5: Gradient-direction diagnostic — mechanism remains unexplained

Mirrors `DIAG1_SIZE_CONFOUND_FINDING.md`'s method for the affinity side,
adapted since `SynthGuidance` is ligand-only (no protein/pocket input, so
DIAG1's "distance-to-pocket" axis does not apply). Captured genuine
mid-trajectory states from unguided sampling (read-only hook, zero effect
on the trajectory) at 3 timesteps × 3 pockets × n=16/pocket (n=144 total,
matching DIAG1's scale), correlating raw gradient magnitude against two
proxies plausibly related to real synthesizability:

| Correlation | Mean | 95% CI | Interpretation |
|---|---|---|---|
| Gradient magnitude vs. size/heaviness | 0.032 | [-0.019, 0.083] | Not significant |
| Gradient magnitude vs. aromaticity fraction | -0.014 | [-0.061, 0.036] | Not significant |

**Unlike DIAG1** (which found a real, statistically clear +0.240
correlation with distance-to-pocket — a concrete, actionable explanation
for the affinity-guidance failure), **neither proxy explains this
gradient's misdirection.** It is not systematically pushing toward
heavier atoms, and not systematically pushing toward or away from
aromaticity. This rules out the two cheapest, most obvious candidate
explanations, but the actual mechanism behind the divergence remains an
open question. Plausible untested candidates: a higher-order/
adversarial-like direction in the model's input space (a known failure
mode in the classifier-guidance literature), or the fingerprint-based
RA-score target may have a training-distribution structure that a smooth,
continuous 3D gradient cannot usefully approximate regardless of which
feature it chases.

## 6. Summary table (D.2 → D.5)

| Step | Finding |
|---|---|
| D.2 | Prior results were exploratory-scale (1 pocket, n=8, no stats); model not converged; no leakage-safe test set existed at all |
| D.2a | Leakage-safe retrain: real test R²=0.414 (not the previously-reported 0.769) |
| D.3 | Divergence **persists** at proper power (n=30) even with both fixes applied — own score up, real RA-score down |
| D.4 | Skipped per the explicit gating rule (D.3 did not show resolution) |
| D.5 | Neither size nor aromaticity explains the gradient's misdirection — mechanism remains unidentified |

## 7. Conclusion and how this compares to the affinity side

Track D's synthesizability-guidance investigation reaches the **same
overall conclusion** as Track A/B's affinity-guidance investigation:
gradient-based guidance built on a real, validated, moderately-predictive
learned model (R²=0.41, a genuine, useful point-estimate signal) does not
transfer to a usable in-loop generative guidance signal — and, as on the
affinity side, this is not for lack of trying to fix it (two independent,
principled fix attempts: bond-aware features, leakage-safe retrain; two
independent diagnostics: this one, and the original SMILES-divergence
check).

**One honest, disclosed difference from the affinity side**: DIAG1 found
a concrete, statistically clear mechanistic explanation for the
affinity-guidance failure (gradient direction anti-correlated with
pocket-contact proximity). D.5 did not find an equivalent explanation for
the synthesizability side — the two cheapest candidate proxies were both
ruled out, but the actual mechanism remains open. This is reported as a
genuine gap in this phase's explanatory power, not minimized or implied
to be equivalent to DIAG1's result.

Combined with the affinity side's `guidance/DUAL_FALSIFICATION_CONCLUSION.md`,
both halves of the thesis's originally-proposed "Coupled
Affinity-Synthesizability Guidance" mechanism have now been independently,
rigorously investigated and found not to transfer gradient-based guidance
to their respective real targets — see
`guidance/THESIS_FINDINGS_SYNTHESIS.md` for the full four-track
(A/B/C/D) closing synthesis.

## Artifacts

- `guidance/TRACK_D_STATUS_D2.md`, `guidance/TRACK_D_LEAKAGE_SAFE_RETRAIN_FINDING.md`,
  `guidance/TRACK_D_D3_LAMBDA_RESWEEP_FINDING.md`, `guidance/DIAG_SYNTH_DIRECTION_FINDING.md`
  — the four individual step reports this document consolidates (kept
  intact as detailed reference; not superseded, only summarized here).
- `guidance/SYNTH_GUIDANCE_REINVESTIGATION_LOG.md` — the step-by-step
  experiment log for this track.
- `guidance/lp_split/build_synth_lp_splits.py`, `guidance/lp_split/train_synth_lp.py`
  — the leakage-safe re-split and retrain scripts.
- `guidance/diag_synth_direction.py`, `guidance/diag_synth_direction_results.json`
  — D.5's diagnostic script and raw results.
