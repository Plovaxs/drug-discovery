# Thesis Findings Synthesis: Four Independent Falsifications of Guided Molecular Diffusion

**Status: all four investigative tracks (A, B, C, D) are complete. This
document is the full phase-closing synthesis for the thesis's Results/
Discussion chapters — it extends `guidance/DUAL_FALSIFICATION_CONCLUSION.md`
(Tracks A/B, gradient-based affinity guidance) to also cover Track C
(non-gradient rejection sampling) and Track D (synthesizability
guidance), and states the single narrative that ties all four together.**

The two track-pair documents (`DUAL_FALSIFICATION_CONCLUSION.md` for A/B,
`TRACK_D_SYNTH_GUIDANCE_REPORT.md` for D, `TRACK_C_REJECTION_SAMPLING_REPORT.md`
for C) remain the detailed technical reference for each track; this
document is the one-level-up synthesis that a reader should start from.

## 1. The thesis's original question

> **Coupled Affinity-Synthesizability Guidance for Target-Conditional
> Molecular Diffusion in Structure-Based Drug Design**

Can a target-conditional 3D diffusion model (TargetDiff) be steered,
using learned predictors of binding affinity and synthesizability, toward
molecules that are genuinely better on those two axes — not just
better-looking by the predictor's own metric, but better on real,
independently-computed outcomes (AutoDock Vina Dock score, structural
validity via PoseBusters, and real RA-score)?

Four mechanistically distinct approaches to this question were tested,
each governed by the same non-negotiable standard: **no claim of
improvement without a statistical test surviving multiple-comparison
correction, a negative/sanity control where applicable, and an explicit
check for the two failure modes that would make a "positive" result
meaningless anyway** — molecule-size confounds and Vina-hacking (raw
score gains that trade away structural validity).

## 2. The four tracks, side by side

| Track | Mechanism | Model(s) | Result | Full report |
|---|---|---|---|---|
| **A** | In-loop gradient guidance, physics-anchored predictor (GIGN + PIGNet2 energy decomposition, trained from scratch to full convergence) | GIGN+PIGNet2 | Clean null — 0/24 significant after BH correction; direction consistency *weakened* vs. the exploratory tier | `DUAL_FALSIFICATION_CONCLUSION.md` §Route 1 |
| **B** | In-loop gradient guidance, 3 mechanism variants on the same frozen backbone (norm normalization, classifier-head reformulation, timestep-windowing) | Stage 0 EGNN | Clean null, 3/3 — one variant wrong-direction-leaning with the phase's most severe validity collapse | `DUAL_FALSIFICATION_CONCLUSION.md` §Route 2 |
| **C** | Non-gradient rejection sampling (post-hoc top-k filtering of a fully-generated, fully-docked pool by a frozen affinity ranker) | Stage 0 EGNN (as ranker) | Falsified — raw-score "improvement" in 12/15 pockets is a molecule-size artifact: ligand efficiency gets significantly *worse* in 14/15 pockets, with a 29-point average PoseBusters validity collapse | `TRACK_C_REJECTION_SAMPLING_REPORT.md` |
| **D** | In-loop gradient guidance toward higher predicted synthesizability | SynthPredNet (RA-score) | Falsified — own score up, real RA-score down, persists after 2 independent fixes; mechanism unresolved after 2 diagnostics | `TRACK_D_SYNTH_GUIDANCE_REPORT.md` |

**Every mechanism tried, across both halves of the "coupled" title and
across both the gradient and non-gradient mechanism classes, failed to
produce a defensible improvement in real, independently-measured
outcomes.**

## 3. The connecting thread: two distinct failure signatures, one shared root

Two separate diagnostic efforts (DIAG1 for the affinity/gradient side,
D.5 for the synthesizability side) asked *why*, rather than stopping at
"no significant effect":

- **DIAG1** (`DIAG1_SIZE_CONFOUND_FINDING.md`) found the Stage 0 EGNN's
  raw gradient is significantly *anti-correlated* with pocket-contact
  proximity (+0.240 correlation between gradient magnitude and distance
  to the nearest binding contact — strongest on atoms *farthest* from
  where binding actually happens). This is a concrete, falsifiable
  explanation for why every gradient-guidance variant in Tracks A and B
  failed regardless of how the gradient was rescaled, retimed, or
  reformulated: none of those changes touch what the representation
  itself encodes.
- **Track C**, via a completely different, non-gradient mechanism
  (post-hoc top-k selection rather than trajectory steering), independently
  surfaces the *same underlying pathology* through a different signature:
  the ranker systematically prefers larger molecules (+12.0 heavy atoms on
  average) whose raw docking score improves mechanically, while their
  true per-atom binding efficiency does not. Two unrelated mechanisms
  (gradient steering, post-hoc filtering) built on the same class of
  learned affinity model both fail because that model's usable signal is
  dominated by molecular size, not target-specific chemistry.
- **D.5**, run with the same diagnostic method adapted to the
  synthesizability side, did **not** find an equivalent concrete
  explanation (neither size nor aromaticity proxies correlate with the
  synth-guidance gradient's misdirection). This is reported as a genuine,
  disclosed asymmetry — the affinity side's failure has a diagnosed
  mechanism; the synthesizability side's does not, yet — not smoothed
  over to make the two halves of the thesis look more symmetric than the
  evidence supports.

**The unifying finding, stated at the level the thesis can defend:**
learned point-predictors of affinity, trained to real convergence with
leakage-safe evaluation and achieving genuine, moderate predictive
quality (Pearson 0.58–0.65 across every model variant tried), do not
straightforwardly convert into usable *generative guidance signals* —
neither as an in-loop gradient (Tracks A, B, D) nor as a post-hoc ranker
(Track C) — and where the mechanism has been traced, molecular size is
the specific confound standing in the way.

## 4. Why this qualifies as four *independent* falsifications, not one result restated

Each track closes off a different, mechanistically distinct explanation
for why the previous track(s) might have failed:

- Track B's null could have been "the representation is fine, the
  mechanism on top of it is broken" → ruled out by Track A's from-scratch,
  physics-grounded alternative representation, independently null.
- Track A's null could have been "the guidance *mechanism* (gradient
  steering itself) is the problem, not the predictor" → ruled out by Track
  B's three mechanism variants on a fixed, adequate representation, all
  null.
- Both A and B's null could have been "gradient steering specifically is
  the wrong tool, but the predictor is fine for other uses" → tested
  directly by Track C's non-gradient, post-hoc mechanism, using the exact
  same predictor as a pure ranker — also null, and for a diagnosable
  reason (size confound).
- All three could have been dismissed as "affinity is just a hard target"
  → ruled out as a general claim by Track D testing the *other* half of
  the thesis's title (synthesizability, a categorically different,
  fingerprint-based target with its own, independently-trained,
  higher-quality predictor) — also null.

A remaining logical possibility — some untried combination (e.g., Track
C's rejection-sampling mechanism applied to Track A's physics-anchored
predictor instead of the Stage 0 EGNN) — was considered and not pursued,
because Track C's failure mode (size confound in the *ranker*, not the
*selection procedure*) gives no positive reason to expect a different
predictor architecture would behave differently as a ranker, absent
evidence that architecture reduces the size confound specifically (Track
A's own heavy-atom-count correlation, 0.640, was barely improved from
Stage 0's 0.719 despite a completely different architecture).

## 5. What this synthesis does NOT claim

- It does not claim affinity- or synthesizability-aware generation is
  impossible in principle for structure-based drug design generally —
  only that every mechanism tested in this project (in-loop gradient
  steering across 2 predictor architectures and 3 mechanism variants;
  post-hoc rejection sampling) failed to transfer learned point-prediction
  quality into a usable generative signal.
- It does not claim the underlying predictors are useless — all of them
  (Stage 0 EGNN, GIGN+PIGNet2, SynthPredNet) achieve real, moderate,
  leakage-safe-validated point-prediction correlation. They work as
  predictors; none of them worked as guidance signals, in either the
  gradient or non-gradient form tested.
- It does not claim the size-confound explanation is complete — it
  accounts for the affinity-guidance failures (DIAG1, Track C) but was
  explicitly tested for and *not found* to account for the
  synthesizability-guidance failure (D.5); that mechanism remains an open
  question, honestly reported as such.
- It does not rule out representations explicitly trained for gradient
  alignment with binding-relevant geometry (rather than trained only for
  point-prediction accuracy) — DIAG1's finding is a concrete, falsifiable
  lead for such future work, not something this project had remaining
  scope to build and test.

## 6. Recommendation for the thesis narrative

Frame the contribution as **a rigorous, four-track ablation study of
learned-guidance mechanisms for structure-based molecular diffusion**,
with the negative results themselves as the primary scientific content —
not a failed attempt to report positive results, but a systematic,
statistically disciplined mapping of where a widely-assumed-workable
class of methods (gradient guidance and rejection sampling using trained
affinity/synthesizability predictors) does not work, plus a partial,
honestly-scoped mechanistic account of *why* (the size confound,
diagnosed twice via two independent mechanisms; the synthesizability
side's mechanism left as an open question for future work).

This survives scrutiny specifically because of the methodology, not in
spite of a lack of positive results: every track used pre-registered-style
dual/multi-criterion checkpoints (predictive quality *and*
gradient-informativeness or selection-informativeness), Benjamini-Hochberg
correction across every test run within a checkpoint, bootstrap 95% CIs
on effect sizes rather than bare means, and — critically — mandatory
negative/sanity controls (Track C's scrambled-filter permutation test,
explicitly checked for its own implementation correctness before being
trusted) and quality-degradation checks (PoseBusters validity, ligand
efficiency) on every apparent "positive" signal before it was allowed to
stand.

## Artifacts index

- `guidance/STAGE2_PLUS_EXPERIMENT_LOG.md` — single source of truth,
  every experiment across Tracks A, B, C logged row-by-row.
- `guidance/SYNTH_GUIDANCE_REINVESTIGATION_LOG.md` — Track D's step log.
- `guidance/DUAL_FALSIFICATION_CONCLUSION.md` — Tracks A/B detailed synthesis.
- `guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md` — Track C detailed report.
- `guidance/TRACK_D_SYNTH_GUIDANCE_REPORT.md` — Track D detailed report.
- `guidance/DIAG1_SIZE_CONFOUND_FINDING.md`, `guidance/DIAG_SYNTH_DIRECTION_FINDING.md`
  — the two mechanistic diagnostics.
- `guidance/track_c_figures/`, `guidance/example_render/` — publication
  figures (statistical + structural) for the Results chapter.
