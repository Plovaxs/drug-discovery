# Synthesizability-Guidance Finding: Surrogate/Ground-Truth Divergence

## Summary

Across every tested nonzero `lambda_synth` in the placeholder-feature
guidance design (`guidance/atom_features.py`'s default,
`use_bond_aware=False`), the synth guidance model's own predicted score on
the final generated molecule moves **up**, while the real,
independently-computed RA-score (`eval/honest_eval.py`'s `RAScorer`, the
actual reymond-group model — not the guidance model's internal copy of a
similar signal) moves **down**. This is the synthesizability-guidance
analogue of "Vina-hacking": a guidance signal that looks like it's working
by its own metric while making the real target worse. It is exactly the
kind of failure mode this project's honest-evaluation harness (Task B,
novelty #2) exists to catch — and here it caught the project's *own*
in-loop guidance model doing it, not just a baseline sampling artifact.

This finding motivated a root-cause investigation and a documented,
architecture-level fix attempt (bond-aware features, see below) rather
than a quiet workaround.

## Root cause hypothesis

`guidance/train_synth_model.py` trains `SynthPredNet` on real CrossDocked2020
ligands, featurized via `utils/transforms_prop.FeaturizeLigandAtom` — which
includes real per-atom **Degree**, **NumHs**, and **Hybridization**,
computed by RDKit from actual bonds (`datasets/protein_ligand.get_ligand_atom_features`).

At *inference*, inside the guided sampling loop, the diffusion process has
no discrete bonds at all — only atom positions and a per-step atom-type
estimate (`v0`). `guidance/atom_features.py`'s original (and still default)
behavior fills Degree/NumHs/Hybridization with a fixed "unknown" one-hot
placeholder for every atom, regardless of its actual local geometry. That
is a real train/inference input-distribution mismatch: the model has never
seen "every atom has degree-class 0, numhs-class 0, hybridization-class 0"
during training (real molecules have varied values across these fields),
so its behavior on that input region is unconstrained by training and can
do essentially anything — including assign a higher score to geometries
that happen to exploit whatever the network learned to do with
out-of-distribution placeholder inputs, with no guarantee that direction
correlates with genuine synthesizability.

## Results: placeholder-feature design (the original, default configuration)

Single-signal sweep, `guidance/lambda_sweep.py --stage synth`, bundled
example pocket (`examples/1h36_..._pocket10.pdb`), n=8 samples/point, full
1000 diffusion steps, `lambda_affinity=0` throughout.

| λ_synth | single-fragment rate | reconstructable | guidance's own score | **real RA-score** |
|---|---|---|---|---|
| 0.0 (baseline) | 100% | 100% | 0.458 | **0.361** (0.01-probe run: 0.361; 30-point run: 0.299 — two independent baseline draws, same seed, see note below) |
| 0.01 | 100% | 100% | 0.487 (+6%) | 0.365 (+1%, but see per-molecule note) |
| 0.03 | 100% | 100% | 0.591 (+29%) | 0.168 (**−53%**) |
| 0.1 | 100% | 100% | 0.645 (+41%) | 0.154 (−48%) |
| 0.3 | 100% | 100% | 0.735 (+61%) | 0.126 (−58%) |
| 1.0 | 12.5% | 100%* | 0.864 | 0.727 (n=1 valid, not meaningful) |
| 3.0 | 0% | 87.5% | 0.864 | — (0 scoreable) |
| 10.0 | 0% | 75% | 0.601 | — (0 scoreable) |
| 30.0 | 0% | 100%* | 0.119 (collapses too) | — (0 scoreable) |

\*"reconstructable" ≠ single connected molecule; at λ=30 all 8 reconstruct
but every one is fragmented.

Note on the two baseline readings (0.361 vs 0.299): both come from
`misc.seed_all(42)` at the start of `run_one_point`, but the 0.01/0.03
probe run and the original 0.0–30.0 sweep run were separate process
invocations with different total RNG consumption before reaching the
`lambda=0.0` point (different grids → different number of prior draws in
some runs), so the exact baseline sample set differs between the two runs.
This is expected and does not affect the within-run comparisons the
conclusions below are based on.

### Per-molecule detail at λ=0.01 and λ=0.03 (Part 1 confirmation probe)

Aggregate means can hide a real effect behind cancellation. Per-sample
real RA-scores, same 8 initial noise draws across all three points
(0.0, 0.01, 0.03):

| sample | baseline (λ=0) | λ=0.01 | λ=0.03 |
|---|---|---|---|
| 0 | 0.1665 | 0.1665 (unchanged) | 0.0107 (↓) |
| 1 | 0.8867 | 0.8867 (unchanged) | 0.9091 (≈unchanged, near ceiling) |
| 2 | 0.0452 | 0.3007 (↑) | 0.0189 (↓ vs baseline) |
| 3 | 0.8660 | 0.8660 (unchanged) | 0.1876 (↓) |
| 4 | 0.3267 | 0.3267 (unchanged) | 0.1342 (↓) |
| 5 | 0.0659 | 0.0659 (unchanged) | 0.0397 (↓) |
| 6 | 0.5234 | 0.2986 (↓) | 0.0392 (↓) |
| 7 | 0.0063 | 0.0128 (≈unchanged) | 0.0038 (≈unchanged) |

At λ=0.01: 6/8 samples unchanged, 1 up, 1 down — genuinely ambiguous, not
a clean safe/beneficial window (an aggregate mean that looks "flat" here
is flat because of cancellation, not because guidance had no effect).

At λ=0.03: 6/8 samples degrade, 0 improve meaningfully, 2 ≈unchanged —
a clear, systematic divergence, consistent with the 0.1/0.3 points.

## Decision (per the project's explicit ambiguity rule)

Per working rule: *"If Part 1 produces an ambiguous result... do not
resolve the ambiguity by guessing — report the ambiguity plainly and
default to the null-result path."* λ=0.01's result is ambiguous (mixed
per-molecule direction, n too small to call it a real effect either way);
λ=0.03 and above show a clear, worsening divergence. Combined, there is no
evidence of a genuine safe-and-beneficial operating window for
`lambda_synth` in the placeholder-feature design across the full tested
range (0.0–30.0).

**Declared: synth-only guidance, in the placeholder-feature design, is a
null/negative result.** It does not provide a ground-truth-verified
synthesizability benefit; at every nonzero value tested it either has no
detectable real effect, actively worsens real RA-score while the guidance
model's own score improves (0.03–3.0), or destroys molecular validity
(≥1.0, worsening further through 30.0).

## Root-cause fix attempt: bond-aware features (architectural-upgrade addendum, Phase 1)

`guidance/bond_estimator.py` replaces the fixed placeholder with a
parameter-free, differentiable estimate of Degree/NumHs/Hybridization from
the diffusion state's own geometry — reusing the covalent bond-length
tables already in `utils/evaluation/analyze.py` (soft/sigmoid version of
the same single/double/triple bond-order logic used there for the hard
stability check) and `models/common.GaussianSmearing` (already used
elsewhere in this codebase for continuous-to-vector featurization) to bin
the continuous estimates into the fixed-width slots
`FeaturizeLigandAtom`'s real feature layout expects.

Two real bugs were caught and fixed during implementation, both verified
via a synthetic-geometry unit test before trusting the estimator on real
sampling runs:
1. Element pairs with no such bond order at all (e.g. C–H has no
   double/triple bond) were incorrectly scored via `sigmoid(inf) = 1`
   instead of being masked to 0 — every such pair silently inherited the
   single-bond indicator's value.
2. The sigmoid steepness constant (originally 4.0 per picometer) saturated
   within a fraction of a picometer of the reference bond length, making
   the "soft" indicator behave as a hard step function with ~zero gradient
   almost everywhere on realistic (non-idealized) geometry — defeating the
   purpose of a differentiable estimator. Reduced to 0.2/pm (a ~20pm-wide
   soft transition band, comparable to real bond-length variability),
   verified to produce nonzero `pos.grad` on a non-idealized synthetic
   test geometry.

### Phase 1 checkpoint result: does NOT resolve the divergence

`guidance/lambda_sweep.py --stage synth --use_bond_aware`, same protocol
(bundled example pocket, n=8, 1000 steps), grid 0.0-1.0:

| λ_synth | single-fragment (bond-aware) | single-fragment (placeholder, for comparison) | guidance's own score | real RA-score |
|---|---|---|---|---|
| 0.0 | 100% | 100% | 0.479 | 0.361 |
| 0.1 | **62.5%** | 100% | 0.536 (+12%) | 0.304 (−16%) |
| 0.3 | **12.5%** | 100% | 0.597 (+25%) | 0.010 (n=1 complete, unreliable) |
| 1.0 | **0%** | 12.5% | 0.689 (+44%) | — (0 complete) |

Two findings, both against the explicit pass/fail criterion ("does the
real RA-score move in the same direction as the guidance model's own
score at any nonzero lambda, for at least one value that doesn't collapse
validity?"):

1. **The divergence is not fixed.** At λ=0.1, the one point with enough
   intact molecules (5/8) to compare meaningfully, guidance's own score
   still rises while the real RA-score still falls — the same direction
   of divergence as the placeholder-feature design, just at a different
   overall scale.
2. **Fragmentation gets *worse*, not better, at matched lambda values.**
   Bond-aware λ=0.1 already drops to 62.5% single-fragment, where
   placeholder-feature λ=0.1 was still 100%; bond-aware λ=0.3 is down to
   12.5%, where placeholder λ=0.3 was still 100%; by bond-aware λ=1.0,
   validity has fully collapsed (0%), a state placeholder-features didn't
   reach until λ=10-30. The likely reason: bond-aware features make
   `ligand_atom_feature` itself a differentiable function of `pos`, adding
   a second gradient pathway (position → estimated bonds → features →
   score) on top of the pathway that already existed through the model's
   geometric encoder (position → score directly). The same nominal lambda
   now drives a larger effective gradient, so lambda values aren't
   directly comparable across the two feature modes -- but there was no
   value found, at any scale tested, where fragmentation was under
   control AND the real RA-score moved with (rather than against) the
   guidance model's own score.

**Conclusion: Phase 1 (real bond-aware features) does not resolve the
synth-guidance divergence.** Per the addendum's explicit instruction, this
is documented plainly rather than treated as a stepping stone to
automatically try Phase 2 (time-dependent guidance scheduling) without
checking in first -- Phase 2 is a substantially larger, separately-scoped
change (new schedule-function interface, config schema changes, threading
timestep through every guidance call site, then re-running this same
checkpoint protocol again), and the evidence so far does not point at
"timing of when guidance is applied" as an obvious next suspect the way it
pointed at "feature mismatch" before Phase 1 -- so it is being surfaced as
a decision point rather than assumed to be the next right move.

The placeholder-feature path remains the default
(`use_bond_aware=False` in both `AffinityGuidance` and `SynthGuidance`);
the bond-aware path is available via `use_bond_aware=True` /
`--use_bond_aware` for anyone who wants to reproduce or extend this
checkpoint, but is not used in any `sampling_*.yml` config as a result of
this finding.

## Phases 2-3: considered, not attempted

Time-dependent guidance scheduling (Phase 2) and a richer guidance-model
architecture (Phase 3) were both scoped in the architectural-upgrade
addendum as candidate follow-ups to Phase 1. After Phase 1's result, both
were explicitly **not attempted**, as a reasoned decision rather than an
oversight or a time-saving shortcut:

Phase 1 tested the leading, concrete root-cause hypothesis (the guidance
model sees inference-time inputs -- constant placeholder Degree/NumHs/
Hybridization -- unlike anything in its training distribution) and the
result was not merely "no improvement" but a *worse* outcome (earlier,
sharper fragmentation at the same nominal guidance strength) while the
core divergence pattern (own score up, real RA-score down) persisted
regardless. That is evidence the problem is not simply "the model doesn't
know how to interpret certain fixed placeholder values" -- if it were,
giving the model more realistic, in-distribution-shaped inputs should have
made behavior *more* sensible, not less. Instead, giving the guidance
gradient a second, richer pathway into geometry (through estimated bonds)
made the same nominal step size push harder in whatever direction the
model's gradient happens to point -- which is a symptom more consistent
with the model's gradient direction itself being poorly correlated with
real synthesizability, independent of what features it's fed.

Phase 2 (scheduling) targets a different mechanism entirely -- *when*
during the trajectory guidance is applied, not what it's computed from.
Nothing in Phase 1's result specifically implicates timing (there is no
observed pattern here, e.g., of guidance behaving reasonably early and
only breaking down late, or vice versa, that would motivate trying a
schedule). Phase 3 (a richer guidance architecture) requires retraining and
would only be justified if there were a specific architectural
insufficiency identified as the cause -- Phase 1's result doesn't identify
one; it points more toward the guidance model's learned gradient direction
itself being unreliable, which a larger architecture is not guaranteed to
fix and could equally make worse (a more expressive model has more
capacity to find a different, still-uncorrelated-with-ground-truth
direction to exploit).

Continuing to Phase 2 or 3 without a specific hypothesis they would test
would be searching for a fix rather than testing one, which is a weaker
basis for further GPU-time investment than the targeted test Phase 1 was.

## Open question for future work

This finding surfaces a genuine, unresolved question rather than a solved
problem: **why does gradient-based guidance on this learned synthesizability
surrogate consistently point in a direction uncorrelated with (here,
opposed to) the real RA-score, across two different feature
representations?** Candidate directions for actually diagnosing this,
none attempted here:

- **Interpretability analysis of the guidance gradient itself** -- e.g.
  visualize/inspect what geometric changes `grad_pos` actually proposes at
  a few individual sampling steps, rather than only observing the
  downstream effect on the finished molecule. This would show whether the
  gradient is doing something locally sensible that fails to compose well
  over 1000 steps, or is locally nonsensical from the start.
- **A different synthesizability surrogate**, trained the same way but on
  a different underlying signal (e.g. a rule-based synthetic-accessibility
  score instead of the learned RA-score classifier), to test whether this
  divergence is specific to guiding against *this* model, or a more
  general problem with using any learned molecular classifier as a
  gradient-based diffusion guidance signal.
- **Direct comparison against classifier guidance literature** in image
  diffusion, where a known failure mode is guidance exploiting
  high-frequency/adversarial-like directions in the classifier's input
  space that improve its score without corresponding to the semantic
  property it was trained to recognize -- checking whether that same
  mechanism (adversarial-direction exploitation rather than genuine
  semantic guidance) explains what's observed here would be a concrete,
  literature-grounded next hypothesis, distinct from both the feature-
  mismatch hypothesis (tested, ruled out as the primary cause) and a
  timing-based one (untested).

This is presented as an open research question the thesis surfaces, not a
gap to be quietly minimized -- correctly diagnosing that a guidance signal
doesn't work, and showing two independent, principled attempts to fix it
both failing in an informative way, is itself a real contribution to
understanding where classifier-style diffusion guidance breaks down for
molecular generation.

## What this means for Task F

Per the adjusted scope: the main ablation proceeds with baseline and
affinity-only guidance (whose own single-signal sweep showed no such
divergence — see the affinity-only rows in the sweep logs — though also a
very weak effect on its own target metric; reported separately). Synth-only
is **not** run as part of the main ablation table, since its results are
already fully and more informatively captured here. If the dual-guidance
variant is reported, `lambda_synth` is set to 0 (or the Phase 1/2/3
checkpoint value, if a fix is found and validated) and the variant is
labeled accurately rather than as "dual-guidance" implying both signals
were meaningfully active.

Post-hoc synthesizability filtering (Task B's honest-eval RA-score column,
Task G's blueprint reports) is unaffected by this finding — both score
*real, finished* molecules against the *real* RA-score model, with no
guidance model or placeholder features in that path at all.
