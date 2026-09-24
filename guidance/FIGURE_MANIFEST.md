# Thesis Figure Manifest

**Purpose:** a single, numbered, sequential index of every publication
figure produced in this project, in the order they should appear in the
thesis Results chapter. `guidance/thesis_figures/` holds a numbered copy
of each (`fig01_...` through `fig12_...`, both `.pdf` for LaTeX and `.png`
for quick preview/slides) — pull directly from that folder when inserting
figures into the thesis document. Original files remain in their
track-specific folders (`guidance/all_tracks_figures/`,
`guidance/track_c_figures/`, `guidance/example_render/`) exactly as
referenced by each track's own finding report; this manifest does not
replace those cross-references, it adds one reading order on top of them.

Caption text below is written ready to paste under each figure (adjust
only the figure number to match your thesis's actual numbering).

---

## Fig. 1 — Track A: effect-size forest plot, exploratory vs. full tier

**File:** `fig01_trackA_effect_forest.{pdf,png}`
**Source:** `guidance/plot_all_tracks_figures.py::fig_trackA_forest()`, reading
`guidance/track_a_{exploratory,full}_results/gradient_informativeness_results.csv`

> **Figure 1.** Per-(pocket, λ) mean Vina Dock shift (guided − unguided)
> for Track A's physics-anchored GIGN+PIGNet2 gradient guidance, comparing
> the exploratory tier (3 pockets, n=20; panel A) against the full,
> converged tier (8 pockets, n=30; panel B). Blue = correct-direction
> shift, red = wrong-direction shift, green = significant after
> Benjamini-Hochberg correction (none). Expanding to more pockets and
> training to full convergence did not concentrate points toward a
> significant, correct-direction effect.

## Fig. 2 — Track A: direction-consistency summary

**File:** `fig02_trackA_direction_summary.{pdf,png}`

> **Figure 2.** Percentage of (pocket, λ) conditions with a correct-
> direction Vina Dock shift, exploratory vs. full tier. Consistency fell
> from 89% to 67% as the model and evaluation scaled up — the opposite of
> what would be expected if the exploratory tier's apparent promise
> reflected a real, just-underpowered effect.

## Fig. 3 — Track A: PoseBusters dose-response

**File:** `fig03_trackA_posebusters.{pdf,png}`

> **Figure 3.** PoseBusters structural-validity pass rate of guided
> samples as a function of guidance strength (λ_affinity), full tier, all
> 8 pockets (thin lines) with the cross-pocket mean overlaid (thick red).
> The dashed line marks the unguided baseline. Validity collapses
> monotonically with guidance strength, reaching a mean of ~17% (vs. ~58%
> unguided) at λ=1.0.

## Fig. 4 — Track B: combined forest plot across three mechanism variants

**File:** `fig04_trackB_forest.{pdf,png}`

> **Figure 4.** Mean Vina Dock shift for every condition tested across
> Track B's three gradient-guidance mechanism variants: B1 (gradient-norm
> normalization), B2 (classifier-head reformulation), B3 (timestep-
> windowed guidance). Each panel uses its own x-axis scale, since B1/B2
> include guided-vs-guided comparisons on a different scale from B3's
> guided-vs-unguided comparisons. Colors as in Fig. 1. All three
> mechanisms, applied to the same frozen Stage 0 representation, returned
> a null result.

## Fig. 5 — Predictive quality across every model trained in this project

**File:** `fig05_predictive_quality.{pdf,png}`

> **Figure 5.** Held-out test-set Pearson correlation for every affinity/
> synthesizability predictor trained during this project. All achieve
> genuine, moderate-to-strong predictive quality (Pearson 0.58–0.65 once
> leakage-corrected) — the guidance failures documented in Figs. 1–4 and
> 9–11 are not attributable to poor predictors. The red bar (Track D's
> original synthesizability model) is leakage-inflated and superseded by
> the leakage-safe retrain (green); see `TRACK_D_SYNTH_GUIDANCE_REPORT.md` §3.

## Fig. 6 — Mechanistic diagnostics: DIAG1 vs. D.5

**File:** `fig06_diagnostics_comparison.{pdf,png}`

> **Figure 6.** Correlation between guidance-gradient magnitude and two
> candidate confounds, with 95% bootstrap CIs, for the affinity-guidance
> diagnostic (DIAG1, left) and the synthesizability-guidance diagnostic
> (D.5, right). DIAG1 found a significant, wrong-direction correlation
> with distance to the nearest pocket contact (red) — a concrete
> explanation for Tracks A/B's failure. D.5's two candidate proxies (size,
> aromaticity) both cross zero — the synthesizability-guidance failure's
> mechanism remains an open question.

## Fig. 7 — Track C: statistical falsification summary (3 panels)

**File:** `fig07_trackC_combined.{pdf,png}`

> **Figure 7.** Track C's rejection-sampling result across 15 held-out
> pockets. **(A)** Raw Vina Dock effect size of top-10%-by-predicted-
> affinity selection vs. the full pool, with 95% CIs, colored by verdict.
> **(B)** The same selection re-evaluated on size-normalized ligand
> efficiency — 14/15 pockets get significantly *worse*, revealing (A)'s
> apparent improvement as a molecule-size artifact. **(C)** PoseBusters
> valid-rate drop between the full pool and the top-10% selection, the
> structural-validity signature of that artifact.

## Fig. 8 — Track C: representative 3D binding poses

**File:** `fig08_trackC_structural_examples.{pdf,png}`

> **Figure 8.** Representative docked poses (highest-scoring, PoseBusters-
> valid molecule per pocket) for three of Track C's targets — SQHC_ALIAD,
> XANLY_BACGL, CD38_HUMAN. Ligand carbons in orange, pocket residues
> (within 4.5 Å of the ligand) in blue, hydrogen-bond contacts as yellow
> dashes with the interacting residue labeled. Illustrative only — not
> part of the statistical evidence in Fig. 7.

## Fig. 9 — Track D: own-score-vs-real-score divergence

**File:** `fig09_trackD_divergence.{pdf,png}`

> **Figure 9.** The synthesizability-guidance model's own predicted score
> (orange) rises with guidance strength (λ_synth) while the real,
> independently-computed RA-score (blue) falls — the synthesizability-
> guidance analogue of Vina-hacking. Hollow markers denote underpowered
> points (N<20 scoreable molecules out of 30 attempted, per fragmentation
> loss); the apparent reversal at λ=1.0 rests on only N=2 molecules and
> should not be read as the divergence resolving (see Fig. 10).

## Fig. 10 — Track D: sample-size collapse

**File:** `fig10_trackD_sample_size.{pdf,png}`

> **Figure 10.** Number of scoreable (non-fragmented, successfully
> reconstructed) molecules out of 30 attempted, by guidance strength.
> Fragmentation removes most of the pool by λ=0.3, which is why large
> point estimates beyond that λ in Fig. 9 must be read with reduced
> confidence.

## Fig. 11 — Grand summary: verdicts across every track and mechanism

**File:** `fig11_grand_summary.{pdf,png}`

> **Figure 11.** Verdict distribution (% of tested conditions) for every
> track and mechanism in this project: Track A (both tiers), Track B1–B3,
> and Track C. No track produced a "clear signal" verdict in any
> condition. Track C uniquely decomposes into "confounded by size" and
> "wrong direction" bands because it is the only track with a diagnosed,
> quantified confound (Fig. 7B); Tracks A/B/D's failures are reported as
> plain nulls (gray) at this summary level, with their own dose-response/
> direction-consistency detail in Figs. 1–4 and 9–10.

## Fig. 12 — Track C: pocket size predicts susceptibility (exploratory)

**File:** `fig12_trackC_pocket_size_correlation.{pdf,png}`
**Source:** `guidance/analyze_pocket_size_confound.py`; see
`TRACK_C_REJECTION_SAMPLING_REPORT.md` §6b.

> **Figure 12.** Exploratory post-hoc analysis of what predicts the
> per-pocket PoseBusters damage from Fig. 7C (n=15 pockets). **(A)**
> Pocket size (protein atom count, CrossDocked2020 pocket10 crop) vs. the
> PB valid-rate drop: Spearman r = −0.59 (95% bootstrap CI [−0.83, −0.13],
> excludes zero; robust under leave-one-out and confirmed by Pearson).
> Counter-intuitively, *smaller* pockets are more susceptible, not
> larger ones — plausibly because a tightly-fitted small pocket has less
> room to accommodate the ranker's preferred larger molecules without
> steric clash, while a roomier pocket absorbs the same size push with
> less structural damage. **(B)** The unguided pool's own heavy-atom-count
> spread predicts additional damage *independent of pocket size* (partial
> Spearman r = +0.70, 95% bootstrap CI [+0.17, +0.91]) — a pool that
> already spans a wider size range hands the ranker easier access to
> unusually large tail-end molecules. Both exploratory (not
> pre-registered, not BH-corrected against the rest of Track C's tests)
> — testable mechanistic leads for follow-up work, not confirmed causal
> claims. See `TRACK_C_REJECTION_SAMPLING_REPORT.md` §6b for the full
> analysis, including a documented correction (an earlier draft
> under-sampled the pool to 300/600 molecules, which understated panel
> B's effect).

---

## Not included in this numbered sequence (separate deliverable classes)

- **Interactive 3D viewer** (`https://claude.ai/artifact/5NLVR8iQKKVEXPsgWTsuaf`,
  "Candidate Explorer") — for live demonstration during the defense, not
  a static thesis figure.
- **PowerPoint 3D models** (`guidance/pptx_3d_models/*.glb`) — for
  offline, rotatable display during the defense presentation itself.
- **Individual Track C panel figures** (`guidance/track_c_figures/panel_{a,b,c}_*`)
  — the same content as Fig. 7's three sub-panels, kept as standalone
  files in case any panel is needed at a larger size on its own page.
- **Individual structural pose renders** (`guidance/example_render/*_render.png`)
  — the same content as Fig. 8's three sub-panels, standalone.
- **`guidance/generated_candidates/*/candidates.csv`** — raw candidate-
  molecule tables (CD38, HDAC8, P2Y12); reference data, not a figure.
