# Track E — Root-Cause-Targeted Modification of the Affinity Guidance Model: Design & Pre-Registration

**Status: PLANNING ONLY. No Track E model has been trained.** This document, committed before any training, *is* the
pre-registration: criteria in Sec. 2.4 are frozen at that commit; any later change must be logged as a dated amendment
at the bottom, never edited in place. Literature backing is in `guidance/TRACK_E_LITERATURE_REVIEW.md`
(citations below as [n] refer to it). Everything below marked **MEASURED** was computed in this planning phase with
inference-only or CPU-only feasibility checks; scripts and raw outputs are in `guidance/track_e/`.

Track E is not a generic backbone swap. DIAG1 and Tracks A–D point to a training-signal/objective problem, not backbone capacity.
Two targeted changes are tested on the **existing** Track A GIGN + PIGNet2 model:

- **E1 (delta-learning):** target Δ = pK_exp − pK_Vina; reconstruction ŷ = pK_Vina + Δ̂.
- **E2 (multi-task PLIP supervision):** auxiliary heads predict PLIP interaction labels from the shared encoder.

---

## 0. Premise checks that change the plan (read first)

1. **"The negative-R² EGNN baseline" is not one number, and the bar it implies is too low.**
   Three existing baselines, all on the leakage-safe LP split:

   | Reference model | Test R² | Note |
   |---|---|---|
   | Old EGNN, old split, n = 27 pockets | −0.537 | old split with 74% val/train target overlap; n=27 |
   | Track A GIGN+PIGNet2 (full tier) | −0.189 [−0.274, −0.114]* | Pearson 0.581; negative R² is a calibration gap (no bias term in the physics-sum readout, per the Track A log), not absent signal |
   | **Stage 0 EGNN on the LP test set (n = 11,855)** | **+0.342** | Pearson 0.609 |

   \*complex-level bootstrap CI. Beating a negative R² would be trivially easy; the pre-registered bar (Sec. 2.4) is the Stage 0 EGNN **and** simple baselines below.

2. **MEASURED: simple baselines are as good as the EGNN, and the honest CI is ~7× wider than previously reported.**
   Stage 0's CI (0.322–0.361) resampled *complexes*; the independent unit is the **target** (127 test targets, median 14
   complexes/target, max 1,392). Target-clustered bootstrap, LP test set:

   | Predictor (anchor clipped, see Sec. 2.3) | Pooled R² [95% CI, target-clustered] | Pearson | Corr(ŷ, heavy atoms) | Within-target Spearman (mean, 73 targets with ≥10 cx) |
   |---|---|---|---|---|
   | Train-mean constant | 0.000 | – | – | – |
   | Vina only, raw (−vina/1.364, clipped ≥0) | 0.076 [−0.208, 0.263] | 0.530 | 0.622 | 0.248 [0.157, 0.333] |
   | Vina only, linearly calibrated | 0.273 [0.163, 0.338] | 0.530 | 0.622 | 0.248 |
   | Heavy-atom count only (linear) | 0.307 [0.136, 0.429] | 0.555 | 1.000 | 0.191 [0.045, 0.329] |
   | **Vina + heavy-atom (2-feature linear)** | **0.362 [0.194, 0.472]** | 0.602 | 0.940 | **0.266 [0.155, 0.369]** |
   | **Stage 0 EGNN** | **0.342 [0.164, 0.464]** | 0.609 | 0.719 | **0.195 [0.072, 0.312]** |

   (Ground truth: Pearson(pK, heavy atoms) = 0.555 on test.) Consequences: (a) the EGNN is statistically indistinguishable
   from a two-scalar linear model; (b) **within a pocket — which is where guidance acts — raw Vina ranks better than the EGNN**
   (0.248 vs 0.195), which is direct, previously unavailable support for the delta premise; (c) any E-arm must beat
   *Vina + heavy atoms*, not merely an EGNN.

3. **Δ is size-neutral but not lower-variance. MEASURED (train, n = 46,964):** with the clipped anchor, mean Δ = 0.518,
   SD(Δ) = 1.753 vs SD(pK) = 1.801, Pearson(Δ, heavy atoms) = 0.052; with the *unclipped* anchor the comparison values are
   Pearson(pK, heavy atoms) = 0.497, Pearson(pK_Vina, heavy atoms) = 0.534 and Pearson(pK, pK_Vina) = 0.437
   (Vina explains ≈19% of train pK variance). Delta-learning therefore removes the size component of
   the target but does not shrink its spread: E1 can only win if the residual is *learnable from structure*. That is a testable
   assumption, not a given.

4. **Guidance caveat for E1 (the thesis question is guidance, not scoring).** ŷ = pK_Vina + Δ̂, but Vina is an external binary
   with no differentiable path in our pipeline, so at generation time only **∇Δ̂** is available. ∇Δ̂ is the gradient of a
   *correction*, not of affinity. Scoring improvement (E1's Sec. 2.4 tests) therefore does **not** imply guidance improvement;
   Criterion C4 (a DIAG1-style gradient diagnostic) is the gate for that, and generative re-testing is out of scope for this plan.

5. **PLIP labels are deterministic functions of the input geometry.** E2 cannot add information the encoder does not already
   see; it can only act as an inductive-bias regulariser. This lowers the prior for E2 (cf. [38], [41] in the review, which
   find no gain from adding interaction-oriented information) and is why the scrambled-label control is mandatory.

---

## 1. Decisions I need from you at the gate (Part 3)

| # | Decision | My recommendation |
|---|---|---|
| D1 | Approve the staged compute plan (Sec. 2.6): option R (3 seeds/arm, E1 then E2) | R, ≈ 38 GPU-h core |
| D2 | π-stacking/π-cation label: keep in the primary head (as the addendum lists) or drop | **Drop from primary** (MEASURED unreliable: atom-level Jaccard 0.16 under 0.2 Å jitter); labels still stored |
| D3 | Baseline bar for E1: Stage 0 EGNN only (addendum) or also Vina+heavy-atom linear | **Both** (Sec. 2.4) — otherwise E1 could "pass" while adding nothing beyond two scalars |
| D4 | Add 2 extra EGNN seeds (optional, ≈ 4.4 GPU-h) to bound baseline seed variance | Optional; skip unless you want the strictest baseline |
| D5 | Guidance re-test after a passing E-arm (Task F–style / Track C-style) | Not in this plan; separate scope and separate go-ahead |
| D6 | Verify the semantics of the `vina` column in `affinity_info.pkl` (score-only vs minimised) before freezing Δ | Yes, first action after go-ahead (≈30 min, CPU only); Δ must be defined against the same quantity we later compute for generated molecules |

---

## 2.1 Architecture

**Backbone = the existing Track A model, reused, not reimplemented.** `guidance/track_a/model.py` (`GIGNPignetAffinity`):
separate protein/ligand atom embeddings → 3 GIGN heterogeneous interaction layers (`hil.py`, ported near-verbatim from GIGN: intra/covalent-proxy and
inter/non-covalent edges share one sum-aggregating message pass and are differentiated by separate MLPs — not sum-vs-mean as in
the paper description in `FLAGSHIP_ARCHITECTURE_RESEARCH.md`) → PIGNet2-style pairwise physics terms (Morse vdW, H-bond, metal, hydrophobic;
learned per-pair ε and width from atom-pair embeddings) summed per complex. Hidden 64, 3 layers, batch size 1.
Edge construction (`features.py:build_intra_inter_edges`) already handles the missing protein bond list (distance proxy).

**Shared encoder, separate heads.**

```
pocket+ligand → embeddings → HIL x3 → x (per-atom, ligand+protein)
                                        ├─ H_Δ  (E1):  Δ̂ = b + s · (−ΣE_phys)        [primary]
                                        │              or MLP(pool(x_lig), pool(x_pocket-contact))   [readout ablation]
                                        └─ H_PLIP (E2): per-LIGAND-atom MLP(x_i) → logits ∈ R^C  (C = 3 primary classes)
```

- **E1 delta head.** Primary = Track A's physics-sum readout **plus a learned bias b and a learned scale s of free sign**
  (the Track A log identifies the missing bias as the cause of its negative R²; a residual needs an offset). This is the
  minimal change to a validated model and keeps the PIGNet2-informed structure. s is *not* sign-constrained: a positive Δ
  need not mean "more attraction" (it may mean Vina over- or under-rewards contacts). Because a physics-sum readout may
  under-represent an arbitrary residual, a **readout ablation** (scalar MLP on pooled embeddings) is a pre-registered arm.
  Runs are cheap (Sec. 2.6), so this does not distort the budget.
  **Reconstruction:** Δ̂ is trained on standardised Δ; at inference
  `ŷ = pK_Vina + (Δ̂·σ_Δ + μ_Δ)` with `pK_Vina = max(−vina, 0)/1.364` and (μ_Δ, σ_Δ) = (0.518, 1.753) from train.
- **E2 auxiliary head.** Per-ligand-atom binary multi-label logits (atom participates in interaction class c). Chosen because
  (i) PLIP reports interactions at single-atom level [26]; (ii) the closest precedent (DeepICL [29]) uses atom-level interaction
  labels from PLIP with exactly these four types; (iii) a complex-level count/fingerprint would discard *where* the interaction
  is, which is what a position gradient needs. **Primary classes: H-bond, hydrophobic, salt bridge** (MEASURED stable).
  π-stacking merged with π-cation is computed and stored but excluded from the primary loss (decision D2). Halogen, water-bridge
  and metal classes are dropped (sparse / not reconstructible from pocket-only LMDB entries).
- **E2 independently disable-able.** Config flag `aux.enabled`; when false the head is not built, PLIP labels are not loaded,
  and the run is byte-for-byte the E1-alone code path. `aux.labels ∈ {plip, scrambled}` selects the negative control.
- **Guidance interface** (for the later, separate scope): Δ̂ is differentiable w.r.t. ligand positions through the physics readout
  (as in `track_a/gign_pignet_guidance.py`, position-only, `grad_v = 0`). The gradient character depends on the head: with the
  physics-sum head ∇Δ̂ is a re-weighted contact-energy gradient of the same functional family as Track A's (which showed
  PoseBusters collapse at λ = 1.0), so a *free-sign* s matters; the MLP head has an unconstrained gradient. C4 tests both.

## 2.2 Loss and weighting

`L = L_Δ + λ_aux · L_aux`

- **L_Δ = MSE on standardised Δ.** MSE is matched to the Stage 0/Track A training loss and to the R² metric, so the *only*
  difference between arms A0′ and E1 is the target. Huber is not used in the primary arms (label noise is real — pK mixes
  Ki/Kd/IC50 — but a different loss would confound the delta effect); a Huber sensitivity arm is not budgeted.
- **L_aux = masked class-balanced BCE** over ligand atoms and primary classes, `pos_weight_c = min((1−p_c)/p_c, 10)`
  with p_c the train atom-level positive rate (MEASURED on 200 train complexes: H-bond 0.111, hydrophobic 0.170,
  π-stack 0.132, salt bridge 0.032 → weights 8.0, 4.9, (excluded), 10 after clipping). Averaged over atoms per complex, then over the
  batch; masked out for the 1% of complexes with zero detected interactions (MEASURED).
- **Weighting scheme: fixed λ_aux = 0.3, set a priori, with loss-scale normalisation and a conflict monitor.** Justification from the
  literature: adaptive schemes — learned uncertainty ([11], caveats in [12]), GradNorm [13], dual-balancing [16], gradient
  surgery [14], [15] — each add hyper-parameters and per-step cost, none has been validated for a *residual* main task with
  *deterministic geometry-derived* auxiliary labels (not found in literature), and negative transfer is documented ([17], [18]).
  A single fixed weight keeps the E1 vs E1+E2 contrast attributable to the labels rather than to a balancing algorithm.
  Safeguards, all pre-declared: (i) log cos(∇_shared L_Δ, ∇_shared L_aux) every 200 steps (one extra backward on the last shared
  layer, negligible cost); (ii) **pre-declared fallback:** if the median cosine is < −0.2 over the second half of training, one
  additional 3-seed arm with PCGrad [14] is run and reported alongside — never substituted silently; (iii) λ sensitivity at
  {0.1, 1.0}, single seed, reported as sensitivity only and never as evidence for the claim.
- **Scrambled-label negative control.** Permute ligand-atom label matrices **across complexes with the same number of ligand
  atoms** (fixed seed). Class prevalence, per-size label statistics, loss scale and code path are preserved; the mapping
  geometry → label is destroyed.

## 2.3 Data

**Vina scores can be reused.** `data/affinity_info.pkl` supplies `vina` for all 64,888 LP-split complexes (train 46,964 /
val 6,069 / test 11,855; **0 missing**, MEASURED). Nothing needs re-docking.
Caveats: (i) semantic of the column (score-only vs locally minimised) is not documented in the pickle — decision D6 resolves it;
(ii) 79 poses have *positive* Vina scores (53 train, 5 val, 21 test; max +24.9 kcal/mol, i.e. clashes) that would make Δ an
extreme outlier → **anchor is clipped: pK_Vina = max(−vina, 0)/1.364**, applied identically to training targets, reconstruction and
every Vina-based baseline; test poses are *not* removed.

**PLIP at training scale (MEASURED, 200 random train complexes, 2 CPU workers).**

| Quantity | Value |
|---|---|
| Success | 200/200 complexes, 0 errors |
| Throughput | 13.3 complexes/s (mean 0.072 s CPU per complex) |
| Zero-interaction complexes | 1.0% |
| Complexes with ≥1 label | H-bond 93%, hydrophobic 91%, π-stack 44%, salt bridge 28.5% |
| Ligand-atom positive rate | H-bond 11.1%, hydrophobic 17.0%, π-stack 13.2%, salt bridge 3.2% |
| Ligand-index mapping check | 0 element mismatches over all ring/charge-group atoms |
| Reconstruction vs real receptor (earlier check, 8 complexes) | 7/8 identical interaction counts, 1 differs by one H-bond |

- **Cost:** all 64,888 complexes ≈ **81 min** on 2 workers (≈ 2.7 h on 1 worker). Labels are computed **once, offline**, and cached as
  one ragged `uint8` array (n_lig × 4) per LMDB index (≈ 8 MB total); no PLIP call ever runs inside the training loop.
- **Pocket reconstruction.** Raw PDB/SDF files are not on disk for the training data; pockets are rebuilt from the LMDB
  (residue starts at each backbone N, residue name from the amino-acid type, ligand as a `LIG` HETATM in LMDB atom order).
  `guidance/track_e/plip_label_extraction.py`. Limitation: only 8 complexes were checked against real receptors.
- **Data-quality filters (pre-registered):** drop PLIP failures (none observed); mask aux loss for zero-interaction complexes;
  train/val exclude Vina > 0 poses (58 complexes) — test keeps them; pK = 0 / rmsd ≥ 2 Å entries are already excluded upstream by the LP split.
- **Label noise — MEASURED (100 complexes, isotropic Gaussian σ = 0.2 Å on ligand coordinates, same PLIP pipeline):**

  | Class | Atom-level Jaccard (orig vs jittered) | Complex-level presence agreement |
  |---|---|---|
  | H-bond | 0.81 | 0.95 |
  | Hydrophobic | 0.69 | 0.98 |
  | Salt bridge | 0.81 | 0.95 |
  | **π-stack (incl. π-cation)** | **0.16** | **0.69** |

  This is a stress test of the whole pipeline (jitter also perturbs bond/aromaticity perception), not an error rate against a
  ground truth — no atom-level ground truth exists — but it shows the π-stack label is not usable as a primary supervision signal
  and that even the stable classes have ~20–30% atom-level churn at 0.2 Å. Errington et al. [32] document that interaction
  profiles of *predicted* poses diverge from experimental ones, so labels on **diffusion-generated** poses (noisier than 0.2 Å)
  should be assumed less reliable than on the CrossDocked training poses E2 uses; E2 never trains on generated poses.
- **Splits:** LP split `guidance/lp_split/leakage_safe_split.json` unchanged (target-disjoint; 779/65/127 targets). Train on the
  **full** 46,964 (Track A already did, in 2.4 h); early stopping on val loss with patience 12 as in Track A, *and* checkpoint by
  val within-target Spearman logged alongside (Track A found val-loss selection picks epoch 1 while val-Pearson selection picks a
  different epoch — selection rule is a pre-registered choice: **val loss**, to stay matched to the baselines).

## 2.4 Pre-registered checkpoint and falsification criteria

### Arms (all: Track A backbone, full LP train, batch 1, hidden 64, 3 layers, 3 seeds each unless noted)

| Arm | Target | Head | Aux | Purpose |
|---|---|---|---|---|
| **A0′** | absolute pK | physics-sum + bias | – | matched control: isolates the *target* effect from backbone/bias effects |
| **E1** | Δ (standardised) | physics-sum + bias + free-sign scale | – | E1 (primary) |
| E1-MLP | Δ | scalar MLP | – | readout ablation |
| **E1+E2** | Δ | as best E1 head | PLIP (true) | E2 (primary) |
| **E1+E2-scr** | Δ | as best E1 head | PLIP (scrambled) | negative control |
| References (no new training) | – | – | – | Stage 0 EGNN, Track A GIGN+PIGNet2 (both existing), Vina-only, heavy-atom-only, Vina + heavy-atom (Sec. 0) |

### Metrics and statistics

- **M1** pooled R² on the LP test set (n = 11,855). **M2** within-target mean Spearman over test targets with ≥10 complexes (n = 73)
  — the guidance-relevant metric. **M3** Pearson(ŷ, heavy atoms).
- **Unit of resampling = target.** Paired cluster bootstrap over the 127 test targets, 2,000 resamples, seed 20260925; per-arm value =
  mean over its 3 seeds *within the same resample*. Multiple comparisons: Benjamini–Hochberg across the pre-registered tests in each stage.
  KS test used only as declared in C2.
- Baseline seed variance is unknown for EGNN (one existing run); reported as a limitation unless D4 is approved.

### Power, disclosed in advance — MEASURED

The paired SD of ΔR² between two predictors on the 127 target clusters is 0.044 (EGNN vs Vina-calibrated), 0.053 (EGNN vs heavy-atom-only)
and 0.018 (two linear baselines). With SD ≈ 0.044 the 95% CI half-width is ≈ 0.09; the probability that the CI lower bound exceeds 0 is ≈ 21% for a
true ΔR² of +0.05, ≈ 62% for +0.10 and ≈ 93% for +0.15 (normal approximation, illustrative). GIGN-arm-to-arm paired SDs may be smaller and
will be measured. **The +0.05 margins below are a floor for practical relevance, not a promise of detectability; an outcome whose point
estimate meets the margin but whose CI includes 0 is reported as INCONCLUSIVE, never as a positive or a null.**

### Criteria

**Stage E1 (arms A0′, E1, E1-MLP).**

| ID | Test | PASS requires |
|---|---|---|
| C1a | M1: E1 − Stage 0 EGNN | ΔR² ≥ +0.05 **and** paired-CI lower bound > 0 (BH-adjusted) |
| C1b | M1: E1 − (Vina + heavy-atom linear) | paired-CI lower bound > 0 |
| C1c | M2: E1 − raw Vina (within-target Spearman) | paired-CI lower bound > 0 |
| C1d | M3 guard | Pearson(ŷ_E1, heavy atoms) ≤ 0.605 (ground truth 0.555 + 0.05; EGNN 0.719, Track A 0.640) |
| C1e | M1: E1 − A0′ | paired-CI lower bound > 0 (attributes any gain to *delta*, not to backbone or bias) |

Interpretation table (all outcomes are reportable thesis results):

| Outcome | Reading |
|---|---|
| C1a–e all pass | Delta-learning improves scoring on unseen targets beyond simple baselines |
| C1a passes, C1b fails | Gain is over the EGNN only; the network adds nothing beyond Vina + size |
| C1a and C1b pass, C1e fails | Gain comes from the bias/backbone/anchor, not delta as a target formulation |
| Point estimates meet margins, CI includes 0 | INCONCLUSIVE (power-limited) |
| C1a fails | **E1 falsified at this scale** |

**Stage E2 (only after E1 is reviewed; arms E1+E2, E1+E2-scr, uses best E1 head).**

| ID | Test | PASS requires |
|---|---|---|
| C2a | M1: (E1+E2) − E1 | ΔR² ≥ +0.03 and paired-CI lower bound > 0 |
| C2b | M2: (E1+E2) − E1 within-target Spearman | paired-CI lower bound > 0 |
| C2c | KS test on the per-target |residual| distributions of E1+E2 vs E1 (127 targets) | reported with BH-adjusted p; supporting, not decisive (non-independent samples) |
| C2d | aux head validity: macro-AUROC on the 3 primary classes, test targets | ≥ 0.70, else E2 is uninterpretable ("aux head did not learn") |
| C3 | Negative control | (E1+E2) − (E1+E2-scr) CI lower bound > 0 **and** (E1+E2-scr) − E1 CI includes 0 or is negative; if scrambled ≈ true the gain is a regularisation/capacity artefact and **E2's interaction claim is rejected** |

**Gate to generative use (C4, cheap, before any sampling study).** DIAG1-style diagnostic (n = 144 captured mid-trajectory states, unguided
TargetDiff) on ∇Δ̂ of every arm that passed its stage: gradient-magnitude vs distance-to-pocket correlation must have a **95% CI upper bound < +0.10**
(DIAG1's affinity gradient: +0.240) and |gradient vs size| no worse than DIAG1's. Failing C4 while passing C1 is a *scoring-only* success and is reported as such.

**Stage order (hard rule).** E1 arms → report → *stop for your review* → E2 arms. E2 is not launched automatically even on an E1 PASS,
and its rationale is weak after an E1 that adds nothing beyond Vina + size; that decision is yours.

**Not pre-registered on purpose:** tuning of hidden size, layers, learning rate (fixed to Track A's), and any additional PLIP classes.

## 2.5 Novelty framing

**Reapplication (cite, do not claim):**
Δ-ML [1]; delta-learning on Vina/linear baselines with tree ensembles [2], [4]; Vina-anchored ML rescoring [3]; GIGN [25] and PIGNet [22] /
PIGNet2 [23] architectures (Track A spliced them); PLIP [26]–[28] as a label source and its four-type atom-level taxonomy as used by DeepICL [29]; multi-task
loss balancing [11]–[16]; leakage-controlled splits [39], [40].

**Thesis's own combination (claim only what the searches support):**
1. *Delta target + PLIP auxiliary supervision on a guidance model whose gradient steers a diffusion generator* — **no prior work found** (Review Sec. 1.5; **see amendment A8**, which supersedes the earlier wording and records that InterDiff applies an interaction-type classification loss inside the generator's denoiser), with the stated search limits;
   the final wording must be "we did not find" until a manual Scholar/Semantic-Scholar check is done and PLANET [21] / DPDiff [31] are read in full.
2. *Applied to guiding a 3D diffusion generator*, evaluated with a pre-registered, target-clustered, size- and validity-aware protocol, and gated by a gradient-direction diagnostic (C4)
   rather than by scoring metrics alone. The nearest work injects interactions into the *generator* ([29], [30], [31]); Track E puts them in the *guidance model*.
3. *A negative or inconclusive result is a first-class outcome*: together with DIAG1 and Tracks A–D it would show that neither target reformulation
   (E1) nor interaction supervision (E2) rescues gradient guidance — a diagnosis-complete negative result.

Do not describe Track E as introducing a new architecture (the backbone is Track A's) or as "the first" anything without the manual check above.

## 2.6 Compute cost, wall-clock, and the gate (Part 3)

**Grounding (measured, this project):** Track A full-tier training: 46,964 train entries, batch size 1, hidden 64, 3 layers, 13 epochs
(patience-12 early stop) in **8,620 s = 2.4 h** on the RTX 3050 (≈ 71 it/s) — far cheaper than the EGNN (≈ 12 it/s, 4 GB card, 2.5 M params, ≈ 11.6 min per 6,000-entry epoch incl. validation).
Per-run estimates for Track E are therefore anchored to Track A, with +8% for the aux head/label loading (**estimate, to be measured in a 100-step benchmark first**) and a
±25% band for early-stopping variability. Track A's own pre-launch estimate (1.5–3 days) was over by ≈ 15–30×, because it included validation overhead in the per-iteration rate —
this estimate deliberately uses the measured epoch time instead.

| Item | Runs | GPU-h (central [range]) | Wall-clock |
|---|---|---|---|
| Implementation (delta data path, aux head, label cache loader, analysis/bootstrap script, smoke tests) | – | 0.3 (smoke only) | ≈ 2 working days |
| PLIP labelling, all 64,888 complexes (CPU, 2 workers) | – | 0 GPU | 81 min (run before E1, or 1 worker concurrently ≈ 2.7 h) |
| D6 Vina-column semantics check (CPU) | – | 0 | ≈ 0.5 h |
| **Stage E1:** A0′, E1, E1-MLP × 3 seeds | 9 | 21.6 [16–31] | ≈ 22 h continuous |
| Stage E1 analysis + report | – | 0.2 | ≈ 0.5 day |
| *(review gate — your go-ahead)* | | | |
| **Stage E2:** E1+E2, E1+E2-scr × 3 seeds | 6 | 15.6 [12–20] | ≈ 16 h continuous |
| C4 gradient diagnostics (per passing arm; ~20 min each, from the earlier DIAG1/D.5 timing) | ≤ 3 | ≈ 1 | ≈ 1 h |
| Stage E2 analysis + report | – | 0.2 | ≈ 0.5 day |
| **Core total (option R)** | 15 training runs | **≈ 38 [≈ 29–52]** | **≈ 1.6 days continuous; ≈ 3–4 calendar days at ~12 h/day unattended, plus ≈ 2 days implementation** |

Scope options, mirroring the earlier Track A/Track C scope decisions:

| Option | Content | GPU-h | Wall-clock | Assessment |
|---|---|---|---|---|
| **S — minimal** | 1 seed/arm, no readout ablation (A0′, E1, E1+E2, E1+E2-scr = 4 runs) | ≈ 10 | ≈ 0.5 day | Cannot separate seed noise from effect; power already limited by 127 targets. Not recommended. |
| **R — recommended** | 3 seeds/arm, staged E1 → E2, readout ablation | ≈ 38 | ≈ 1.6 days | Matches the project's standard (3 seeds, KS + BH + bootstrap + negative control) |
| **F — extended** | R + 2 EGNN seeds (4.4 h) + λ sensitivity ×2 (5.2 h) + conditional PCGrad arm 3 seeds (≈ 10 h) | ≈ 58 | ≈ 2.4 days | Only if R yields an ambiguous, high-stakes result; each add-on is separately justified above |
| *Rejected* | Full re-test of guided generation (Task F/Track C-style docking sweeps) | multi-day–week | – | Separate scope and go-ahead (D5); the same single-GPU constraint that blocked Track D concurrency applies |

Risks, stated up front: (1) 127 test targets bound the achievable statistical power (Sec. 2.4) — the honest outcome may be INCONCLUSIVE;
(2) the aux labels are a deterministic function of the input, so E2 may reduce to a regulariser; (3) E1's guidance value is unproven even if scoring improves (C4);
(4) Track A's best epoch was epoch 1 — the delta target may train equally briefly, making early-stopping the dominant source of seed variance;
(5) PLIP labels on generated poses are less reliable than on the training poses used here.

**Nothing in this plan starts without your explicit go-ahead. Only the feasibility checks listed below have been run.**

## Appendix — feasibility checks run in this planning phase (no training)

| Check | Script | Output |
|---|---|---|
| PLIP on 8 reconstructed vs real complexes | `guidance/track_e/plip_reconstruction_check.py` (run with `python ... <outdir>` from the repo root, `PYTHONPATH=.`) | `plip_reconstruction_check_results.json` |
| PLIP scale + jitter stability, 200 train complexes | `guidance/track_e/plip_scale_check.py` | `plip_scale_check_results.json` |
| Reference baselines incl. target-clustered CIs; Stage 0 EGNN inference only | `guidance/track_e/reference_baselines_check.py` | `reference_baselines_results.json` |
| Paired-bootstrap power, within-target ranking, clipped-anchor reference table | `guidance/track_e/paired_power_check.py` | `paired_power_results.json` |
| PLIP label extraction library | `guidance/track_e/plip_label_extraction.py` | – |

Regeneration: `PYTHONPATH=. python guidance/track_e/reference_baselines_check.py` (writes the two `_*.npz` caches, which are git-ignored) then `paired_power_check.py`.

## Amendments

*Pre-registration frozen at commit `8d913c6`. All amendments below were made on 2026-09-25 after the go-ahead for Option F and
**before any Track E model was trained**; none changes a pass/fail threshold.*

**A1 — decisions D1–D6 resolved by the user.** D1: Option F (~58 GPU-h budget), scope below. D2: π-stack/π-cation dropped from the
primary auxiliary head. D3: E1 counts as a positive result only if it beats **both** the Stage 0 EGNN (R² 0.342) **and** the
Vina + heavy-atom linear baseline (R² 0.362); this is exactly C1a AND C1b in Sec. 2.4 (both must pass; C1a alone is explicitly *not* a pass — a
"PARTIAL" outcome). D4: the 2 extra EGNN seeds are part of Option F (not additional). D5: guided-generation re-test deferred. D6: see A3.

**A2 — limitation stated explicitly (D2).** π-stacking (merged with π-cation) PLIP labels are **not** used as training targets: atom-level
Jaccard between labels of the original and 0.2 Å-jittered pose is 0.16 (complex-level presence agreement 0.69), against 0.69–0.81 for H-bond,
hydrophobic and salt bridge. The labels are computed and cached (column `pi_stack`) but excluded from the loss; E2 therefore tests
H-bond, hydrophobic and salt-bridge supervision only, and cannot speak to π-stacking.

**A3 — D6 result (Vina column semantics), `guidance/track_e/d6_vina_semantics_check.py`, n = 100 CrossDocked test-set reference poses with real receptor
+ SDF on disk, Vina 1.2.6 through this project's own preparation.** The stored `vina` value is a Vina-type score of (approximately) the stored pose but is
**not reproducible exactly** with our pipeline: mean(stored − recomputed) = −0.79 kcal/mol (score-only) / −0.44 (minimised), MAE 0.82 / 0.79 kcal/mol,
Pearson 0.86 / 0.89. **Score-only and minimised cannot be told apart by this check**: for 90 of 100 poses they differ by < 0.2 kcal/mol under our preparation
(poses already sit at a local minimum), and for the 5 poses where they differ neither matches the stored value (MAE ≈ 4–5 kcal/mol; stored value
tracks minimised in 3/5); a Wilcoxon test of |stored − minimised| < |stored − score-only| is not significant (p = 0.97). The SDF files carry no `minimizedAffinity`
field. Conclusion: the stored anchor is treated as an **opaque, fixed, pose-attached Vina-type score with ≈ 0.8 kcal/mol (≈ 0.6 pK) preparation noise
relative to our recomputation**. This does **not** affect any scoring result on CrossDocked poses (train and test both use the stored anchor), but it means an anchor
recomputed for *generated* molecules (D5, deferred) would carry a systematic offset of about +0.3–0.6 pK that would have to be calibrated. The stretch item "rerun
the headline result under the other Vina interpretation" is **not feasible at scale** (raw receptors exist only for the 100 test targets, not the 64,888 LP-split complexes); its planned
substitute is a **robustness test** that perturbs the anchor at test time (offset −0.5 kcal/mol and Gaussian noise SD 0.8 kcal/mol) and reports the change in ΔR².

**A4 — corrections and execution details (no threshold changes).**
(i) Track A's full-tier model used **hidden 256 / 0.494 M parameters** (not 64 as written in Sec. 2.1); Track E arms use hidden 256, matching the reference model.
(ii) Epoch cap **30** (Track A used 100): worst case 30 × 10.9 min ≈ 5.5 h per run; early stopping keyed on validation loss of the *main* (Δ or absolute) loss, patience 12, for every arm.
(iii) Test predictions are produced **once**, from `best.pt`, at the end of each run (Track A evaluated the test set at every improvement; selection is unchanged because it uses validation loss only).
(iv) Train/val exclude Vina > 0 poses for all GIGN arms (A0′ included); the test set is unfiltered. The 2 extra EGNN seeds (2022, 2023) replicate `train_egnn_stage0.py`
exactly (same 6,000-entry training subsample, seed-2021 subsample kept fixed; only the training seed varies) and are combined with the existing Stage 0 checkpoint into a 3-seed EGNN baseline used as a **sensitivity** reference;
the decision reference stays the pre-registered Stage 0 checkpoint.
(v) The "best E1 head" for E2 is chosen on **validation loss only** (seed-mean best val loss, `phys` vs `mlp`), never on test.
(vi) The λ_aux sensitivity sweep and the conditional PCGrad arm belong to the E2 stage (λ_aux exists only there); they are not run in the E1 stage. The PCGrad arm is launched only if the
logged cosine between the Δ and auxiliary gradients on the shared layers has median < −0.2 over the second half of training, as declared in Sec. 2.2.
(vii) Uncertainty is reported at 90/95/99% (percentile bootstrap); decisions use 95% and one-sided p-values with Benjamini–Hochberg across each stage's tests. Effect sizes: ΔR²/bootstrap-SD and paired Cohen's d_z on per-target MSE differences.
(viii) Aux loss is masked for complexes with no detected interaction in the three primary classes: 2.6% of train, 1.9% of val, **7.3% of test** (full-cache MEASURED; the 200-complex sample gave 1%).

**A5 — confirmed compute (100-step benchmark, RTX 3050, idle CPU; `train_e.py --benchmark_steps 100`).**

| Arm | Train it/s | Eval it/s | Epoch (46,911 train + 6,069 val) | Run (13 epochs = patience-12 minimum) |
|---|---|---|---|---|
| A0′ / E1 (physics head) | 76.7 | 140 | 10.9 min | **≈ 2.4 h** (matches estimate) |
| E1-MLP | 123 | 166 | 7.1 min | ≈ 1.5 h |
| E1+E2 (aux head) | 74.8 | 132 | 11.2 min | **≈ 2.4 h** (estimate was 2.6 h) |
| E1+E2 + PCGrad | 59.9 | 129 | 13.5 min | ≈ 2.9 h |
| Stage 0 EGNN (extra seeds) | 11.9 | 32 | 11.6 min (6,000-entry epoch) | 2.3–3.9 h (patience 5 / cap 20 epochs) |

(An earlier benchmark taken while PLIP labelling ran on 4 CPU cores gave ≈ 60 it/s; the difference was CPU contention, so **labelling and training must not overlap** — labelling is finished.)
Revised Option F budget: E1 stage 3 × (2.4 + 2.4 + 1.5) + 2 EGNN ≈ **24–27 GPU-h**; E2 stage 6 × 2.4 + λ-sweep 2 × 2.4 + PCGrad 3 × 2.9 ≈ **28 GPU-h**; total ≈ **52–55 GPU-h**, inside the ~58 GPU-h budget.

**A6 — infrastructure verification (before any real run).** Unit tests for the delta target/reconstruction and the scramble control (`guidance/track_e/tests/test_core.py`, 10 tests).
Interrupt/resume: SIGTERM mid-epoch → atomic checkpoint, exit 75, exact resume; SIGKILL → resume from the last periodic checkpoint. **A trainer bug was found and fixed during this test**: the model was built before
`seed_all`, so initial weights were not a function of the seed. After the fix, an interrupted-and-resumed run is **bitwise identical** to an uninterrupted run on CPU (single thread: prediction and parameter
differences exactly 0.0, interrupted mid-epoch 2), and on GPU epoch 1 is bit-identical while later epochs differ at the level of CUDA atomic non-determinism (max |Δŷ| 0.048 for resumed vs 0.059 between two uninterrupted runs), i.e. **resuming is indistinguishable from re-running on this hardware**.
Analysis pipeline check: the Track A model reproduces its recorded R² = −0.189 / Pearson 0.581 through the new inference path, and the fast target-clustered bootstrap reproduces the frozen reference CIs.

**A7 — novelty framing corrected after the literature follow-up (no threshold change).**
(i) *E2 is not the first interaction-like auxiliary supervision of an affinity network.* PLANET [21] (peer-reviewed, *J. Chem. Inf. Model.*; abstract read, full text not accessible with the available tools) trains affinity jointly with a protein–ligand
**contact map** and a ligand distance matrix. What remains untested in the literature we found is the **PLIP-typed, ligand-atom-level** form (H-bond / hydrophobic / salt bridge) **combined with a delta target relative to a docking score**, used to build a *guidance* model for a 3D diffusion generator.
Sec. 2.5 should be read with "first" removed from every claim about auxiliary interaction supervision; the defensible claims are the specific combination (E1+E2), the pre-registered target-clustered evaluation, and the C4 gradient gate.
(ii) DeepICL [29] remains a near miss (PLIP labels as generation-time conditioning, autoregressive VAE). DPDiff [31] (full text read) is a near miss on the generator side: its "interaction priors" are latent features of pretrained **affinity-prediction networks**, extracted from intermediate diffusion predictions; it has no delta target and no PLIP labels.
(iii) **Independent corroboration for C4.** DPDiff's own ablation reports that a geometry-based prior pretrained on accurate poses *degrades* generation when queried on noisy intermediate structures (Vina Score −3.58 vs −5.04 without it). This is the same train/inference mismatch that Tracks A–D documented for this project's guidance models, and it is why a scoring gain from E1/E2 (trained on clean docked poses) cannot be assumed to transfer to guidance.
(iv) Literature items resolved: 33/45 sources are now PubMed-verified (PLANET, LUNA and LP-PDBBind now have journal versions); 12 remain Consensus-only (see the review's status line). IPDiff, cited by DPDiff, has not been read. A manual Scholar/Semantic-Scholar search for delta-learning + auxiliary-interaction combinations remains a **user task** (not possible with the available tools).

**A8 — novelty section revised after an independent 12-paper literature check supplied by the user (2026-09-25). Supersedes A7(i)–(ii) and Sec. 2.5 item 1; no threshold change.**
*Provenance and my verification level.* The check was supplied by the user (12 papers reported read in full). I re-verified only the claims that carry the novelty wording, by fetching the sources (secondary, model-summarised page text, so treat as spot-checks, not full-text reads):
(a) **BInD** (Lee, Zhung, Seo, Kim, *Adv. Sci.* 2025; PMC12463045): PLIP-derived interactions are **co-generated** as a categorical diffusion channel with their own loss and used for guidance terms after the network prediction — **confirmed**; no delta target.
(b) **"Delta Score" / SBE-Diff** (Ren, Gao, Qiang, Lan, arXiv:2311.12035; Gao et al., arXiv:2403.12987): "delta" is an evaluation/specificity metric with contrastive, energy-guided training — **consistent with the user's claim; a name collision, not a concept collision**. Correction to the supplied text: these papers are by Ren/Gao/Qiang/Lan, **not Ragoza et al.**; the abstract pages I could reach did not show the exact definition, so the contrastive off-target definition is *not independently confirmed*.
(c) **InterDiff** (Wu et al., *Brief. Bioinform.* 2024, bbae174): **contradicts the supplied wording.** Its denoiser is trained with a composite loss that **includes a training-time cross-entropy classification of protein-atom features into BINANA interaction types (λc·Cls(h^(p)))**, in addition to cross-attention "interaction prompts" at inference. It conditions by cross-attention, **not** by gradients of a separate model, and uses no delta target.

*Revised claim (replaces A7 and the corresponding wording in Sec. 2.5).* Across the delta-learning, interaction-guided-diffusion and multi-task interaction-prediction literatures checked, we found **no method that combines** (a) a delta-learning target against a docking score, (b) typed per-atom interaction classification (PLIP) as an **auxiliary supervised loss on a separate guidance predictor**, and (c) use of that predictor's **gradient** to steer a frozen 3D diffusion generator. Typed interactions do enter diffusion pipelines in three distinct ways, none identical to E2: **inference-time conditioning** (DeepICL, DiffPharma), **co-generation as a diffusion output** (BInD), and **an auxiliary classification loss inside the generator's own denoiser** (InterDiff). What E2 changes is *where* the supervision sits — on the guidance model whose gradient is used, not on the generator. The earlier stronger phrasing "never used as an auxiliary supervised training loss" is **withdrawn**: it is false for InterDiff. "We did not find" remains the correct wording pending the user's manual Scholar check.
*Terminology.* "Delta" in Ren/Gao et al. ("Delta Score") is unrelated to this project's residual-against-Vina target and must be disambiguated wherever cited.
*Nearest-neighbour map (supplied; entries not marked verified above were **not** independently re-checked by me):* PLANET (contact-map auxiliary loss, no typed labels, no diffusion; abstract read); Interformer (interaction-aware MDN, PubMed-verified); DPDiff (full text read: affinity-network latent priors); DeepICL (PLIP conditioning; full text read); pocket-aware evolutionary-conserved interaction-guided diffusion (arXiv:2505.05874); BADGER / general binding-affinity guidance (*JCIM* 2025, EGNN approximating absolute Vina energy); DiffPharma (*npj Drug Discov.* 2026, interaction particles as conditioning); DiffInt (*JCIM* 2024, H-bond pseudo-particles; **training-loss details unconfirmed — the one remaining paper needing a manual full-text read**, and after the InterDiff correction any "conditioning-only" statement about it should not be relied on); DeepRLI (arXiv:2401.10806, PLIP only for post-hoc interpretation).

**A9 — checkpoint-selection audit, methodology statement, and CrossDocked provenance limitation (2026-09-25; no threshold change).**

*Methodology statement (for the thesis).* In every Track E run, checkpoint selection (`best.pt`), the learning-rate scheduler, early stopping, and the choice of E1 readout for E2 are determined **only from validation data**. Test predictions are computed **once**, from `best.pt`, after training has completed. This is enforced by an automated static test (`guidance/track_e/tests/test_no_test_selection.py`, 5 tests: the test split is referenced only at dataset construction and inside `final_test`; `Runner.run` never touches it; `best.pt` is written only in the validation-improvement branch) and is a positive control against test-set checkpoint selection. Hyper-parameters are fixed to Track A's and were not tuned on test.

*Audit of earlier tracks (code-verified).* All within-run checkpoint selection is by validation criteria: Stage 0 EGNN (`train_egnn_stage0.py`, best validation loss), Track A full tier (`best.pt` by validation loss; `best_by_pearson.pt` by validation Pearson), leakage-safe synthesizability model (`train_synth_lp.py`, best validation loss), and the old learning-rate/depth sweep (documented as validation-loss selection). Disclosed caveats: (1) Stage 0, Track A and the synth trainer **compute and log test metrics at every validation improvement**, so test numbers were visible during development even though they were not used for selection; Track E removes this exposure by evaluating test once at the end. (2) In the old sweep (`AFFINITY_MODEL_SWEEP_RESULTS.md`, n = 27 test pockets, old split) one configuration is labelled the "best config" (lr = 3e-5), apparently on the basis of its test Pearson (I found no stated selection rule), and that configuration was reused for the DIAG1 data-fraction learning curve. It was **not** used for any deployed or guidance model (Stage 0 reused the deployed configuration unchanged; Track A has its own), but DIAG1's learning-curve reading should carry this caveat. (3) The Track A observation that the validation-Pearson checkpoint "did not hold up on test" was informational; validation loss remained the pre-declared criterion.

*CrossDocked provenance (limitation).* CrossDocked2020 is built by **cross-docking** protein–ligand pairs from PDBbind, so its poses are docking outputs, not crystal structures (stated in Gao et al., arXiv:2406.08980, Sec. 2 and 3.2 — read in the arXiv text). Consequences for Track E: (i) PLIP labels and the Vina anchor are computed on **docked** poses (already the case, and the stored `vina` value carries the D6 preparation noise); (ii) file names such as `3upr_C_rec_3vri_1kx_lig` show the receptor and the ligand can come from **different PDB entries**, so the experimental pK attached to a pose refers to the ligand's measured binding, not necessarily to that receptor conformation — a label-provenance noise source **that I have not verified in the affinity file's documentation** and which affects all tracks equally (it does not bias the arm comparisons but limits absolute R²).

*Verification status of the supplied "research frontier" claims about Gao et al.* Confirmed in the arXiv text: Vina scores can be inflated simply by increasing atom count ("susceptibility to overfitting"); TargetDiff "may be overfitting to Vina scores without generating truly useful molecular structures"; CrossDocked is derived from cross-docking rather than crystal structures; the paper's three-part framework (similarity to actives, virtual-screening metric, delta-score/DrugCLIP binding estimate). **Not found in the arXiv version, so not confirmed:** the quoted sentence about "trained Vina predictors guid[ing] sampling… hacking of Vina scores", and the statement about checkpoint selection on the test set — these may be in the ICLR 2025 camera-ready, which I could not read; do not quote them until checked. Note also that Gao et al.'s evidence concerns **evaluation-metric overfitting of generative models**, not gradient-guidance failure, so it corroborates the size/Vina-gaming side of this thesis's findings rather than the guidance-gradient diagnosis. Their "delta score" is the specificity metric (defined in Gao et al. 2024), a name collision with this project's residual target.

**A10 — closes the open item in A9 (2026-09-25; no threshold change).** Both quotes were **independently confirmed by me** from the text of the ICLR 2025 camera-ready PDF (proceedings.iclr.cc / OpenReview), extracted locally; they are **not** in the arXiv version (arXiv:2406.08980, "From Theory to Therapy"), which is a different, earlier text. The two versions must be cited separately.
- *Citation to use for both quotes:* Gao, B.; Tan, H.; Huang, Y.; Ren, M.; Huang, X.; Ma, W.-Y.; Zhang, Y.-Q.; Lan, Y. **Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs.** *ICLR 2025* (camera-ready; OpenReview forum RyWypcIMiE).
- *Verbatim, Sec. 4.5.1 (Analysis):* "some models use trained Vina predictors to guide sampling, prioritizing the hacking of Vina scores rather than generating reliable and actual effective molecules." (the source reads "actual effective", not "actually effective").
- *Verbatim, dataset-limitations paragraph:* "we observed that many existing models use the test set as a validation set to select checkpoints during training, which risks data leakage." The same paragraph also states that CrossDocked structures are "generated by docking software instead of real complexes" and that data selection "relies on docking software, which leads to bias in the dataset, as the ligands selected for the training data tend to be favored by the docking software" — a further, documented CrossDocked bias relevant to every track.
- *What each version supports.* arXiv version: Vina scores inflate with atom count; cross-docked data; the three-part evaluation framework. ICLR camera-ready: additionally the "hacking" framing and the checkpoint-leakage observation.
- *Scope caveat for use in the thesis.* The "hacking" sentence is an observation in the analysis discussion; the paper does not run an experiment on gradient guidance from a Vina predictor and names no model, so it **corroborates the direction** of Tracks A–D and the size shortcut (A.6), it is not independent evidence for the guidance-gradient diagnosis (DIAG1). The checkpoint-leakage sentence is a field-level observation and is the citation for the positive control in A9.

**A11 — two further details of Gao et al. (ICLR 2025 camera-ready) checked against the PDF text (2026-09-27; no threshold change).**
(i) *Confirmed:* Vina scores correlate with chemistry-level "overfitting" factors — more hydroxyl groups, a lower N+O percentage and more halogen atoms (main text and Appendix E, Figs. 15 and 17), in addition to atom count.
(ii) *Definition of "delta score" (Eq. 2):* Delta Score(y_i) = mean_j [ −S(x_ij, y_i) + S(x_ij, y_k) ], with S a Glide (SP/XP) docking score and y_k a **randomly sampled other target** (k ≠ i) — i.e. **target vs random off-target pocket**, not "relative to a reference ligand" as a supplied summary stated. This confirms the A8 name-collision note (it is unrelated to this project's residual-against-Vina target Δ = pK_exp − pK_Vina); reference ligands only appear as a comparison row in their tables.

