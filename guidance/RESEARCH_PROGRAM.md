# Unified research programme: why learned affinity guidance fails, as an information problem

Status: 2026-10-10. This document exists to replace a pile of separately-motivated tracks with one
question, and to say exactly which evidence we already hold, which is in flight, and which is missing.
Everything cited as "ours" has a script and a results file in this repository; everything cited as a
field result has a reference.

---

## 1. The one question

> When a learned affinity predictor is used to guide a pocket-conditioned diffusion model, and the
> guidance fails, **what exactly failed?**

The thesis currently answers "all five routes failed their pre-declared criteria" and documents it
rigorously. That is a result about *outcomes*. It does not say what the failure *is*. Three explanations
were live and none had been separated:

* **(A) a tuning problem** — the guidance scale, mechanism, or timestep window was wrong;
* **(B) an accuracy problem** — the surrogate is not good enough to guide with;
* **(C) an information problem** — the surrogate's accuracy does not come from the structural
  information it is given, so its gradient has no structural information to give.

The thesis already rules out (A) across three orders of magnitude of guidance strength plus three
mechanism variants, and explicitly rejects (B) by noting the predictors reach Pearson 0.58–0.65 on
leakage-safe test sets. **The work of 2026-10-10 establishes (C), and (C) is what makes the rejection of
(B) coherent rather than puzzling.**

---

## 2. The claim

The surrogate's held-out accuracy is carried by information channels that are *mutually redundant* and
*not pose-specific*: gross ligand size, gross pocket properties, and contact counts that re-encode size.
A predictor whose accuracy does not depend on pose cannot supply a pose-specific gradient. Guidance
therefore fails for a reason that no amount of tuning or extra training data addresses, and — because
point-wise uncertainty estimation also fails on this task — the failure cannot be detected
per-prediction either. Only distribution-free conformal intervals remain valid.

That is one sentence that each of the five thesis tracks, the A1–A1g uncertainty suite, and the new
cheminformatics layer all serve. It is also the link the published literature has not made: the field has
established the premise (accuracy without structure) and separately established the detection failure
(uncertainty methods do not rank error), but has not drawn the consequence for *generative guidance*.

---

## 3. Evidence we already hold

### 3.1 The gradient does not point at the pocket (established, in the thesis)
`guidance/diag_gradient_magnitude.py`, `diag_size_confound*.py`. Gradient magnitude correlates
**positively** with an atom's distance from the pocket, r = +0.24. The guidance signal is strongest
exactly where the protein is not.

### 3.2 The accuracy does not need the pocket (new, 2026-10-10)
`guidance/cheminformatics/ligand_only_baseline.py`, `compare_vs_structure.py`. Paired target-clustered
bootstrap over the 127 test targets, B = 2000, BH over 40 pairs:

| model | test R² |
|---|---|
| ridge on 14 RDKit descriptors, **no protein** | **0.3534** |
| ligand heavy-atom count, **one feature** | 0.3069 |
| EGNN Stage 0 (the thesis's structure model) | 0.3415 |
| EGNN, 3 seeds | 0.3437 ± 0.0377 |
| EGNN deep ensemble (A1c) | 0.4072 |

After BH: **0 of 40 pairings where a ligand-only model loses**, and structure vs the 14-descriptor ridge
is a tie at every seed and as an ensemble. Three marginal "wins" (p = 0.039–0.049) were withdrawn by BH
at q = 0.12–0.13.

### 3.3 Neither does it need the ligand (new, 2026-10-10)
`guidance/cheminformatics/pocket_features.py`, `representation_ladder.py`. Both marginals work alone and
are redundant with each other:

| information source | dim | test R² |
|---|---|---|
| ligand + pocket | 52 | 0.3832 |
| ECIF raw contact counts | 72 | 0.3701 |
| ligand descriptors | 14 | 0.3534 |
| **pocket descriptors, no ligand at all** | 38 | **0.2889** |
| ECIF size-normalised (interaction *density*) | 72 | 0.2077 |

`ligand+pocket` vs `ligand`: tie. vs `pocket`: tie after BH. **Neither source carries information the
other lacks at descriptor level** — a full replication of Volkov et al. 2022 on a leakage-controlled,
target-disjoint split, which that paper did not have.

### 3.4 Contact counts are ligand size in disguise (new, and a control the field omits)
ECIF raw (0.3701) − ECIF size-normalised (0.2077) = **+0.1625, q = 0.013, significant.** Raw interaction
counts work mainly by re-encoding heavy-atom count. Any paper reporting raw ECIF-style counts as evidence
of learned interactions, without this control, is reporting ligand size.

### 3.5 We cannot detect the failure per-prediction (established, A1–A1g)
Five point-wise σ families (MC dropout, deep ensemble, heteroscedastic Gaussian NLL, deep evidential,
and the corrected evidential variant) all fail the three pre-registered tests. Split-conformal is the
only method with valid coverage. The evidential result that looked positive at one seed (p = 0.042) did
not replicate across three — between-seed sd is 0.0377 R², larger than the effects being claimed.

### 3.6 Model size is not the constraint (established, A1g)
Ternary (1.58-bit) weights via TWN + STE with QAT give test R² 0.3437 ± 0.0377 versus FP32 0.3420 —
statistically indistinguishable. A model whose weights can be reduced to three values without measurable
loss is not accuracy-limited by capacity; it is limited by what its inputs tell it.

### 3.7 The split is sound, which is what makes the above interpretable
* Tanimoto 1-NN lookup **fails badly** (R² −0.92) and every structure model beats it at q = 0.002, so
  this split does not reward memorising the nearest training ligand — unlike the PDBbind setting Volkov
  et al. analysed.
* 64% of test ligands are below Tanimoto 0.35 to any training ligand (median 0.316).
* 40.1% of test molecules sit on a generic Bemis-Murcko scaffold seen in training, from only 1,779
  distinct generic scaffolds across 46,964 training ligands. Held-out target ≠ held-out scaffold; this is
  a property of CrossDocked's narrow framework vocabulary and must be disclosed.
* Scaffold familiarity does **not** buy accuracy: on RMSE, 0 of 13 models are worse on unseen scaffolds.
* Four leakage filters applied: receptor PDB ID, ChEMBL→UniProt accession, sequence identity ≥90%, and
  ligand InChIKey connectivity skeleton.

### 3.8 A methodological result worth stating in its own right
R² must never be compared across data subsets with different label variance. Our scaffold audit first
concluded the opposite of the truth because pK sd is 1.72 on seen-scaffold versus 1.29 on unseen — a
1.77× variance ratio — and R² = 1 − SSE/SST. What exposed it was keeping **Vina**, an untrained physics
scoring function, in the table: it showed the identical collapse. *A pattern present in an untrained
predictor is a property of the data partition, not of learning.* Keeping an untrained baseline in every
subset-comparison table is a cheap, general safeguard.

---

## 4. Field gaps, and where each attaches to our branches

| field gap (reference) | what it says | where it attaches |
|---|---|---|
| Volkov et al. 2022, *J. Med. Chem.* | interaction descriptions give no advantage over ligand *or* protein descriptors; memorisation dominates | §3.2, §3.3 — replicated and extended with a leakage-controlled split and a size confound control |
| Mattsson et al. 2026, bioRxiv | sequence-identity splits insufficient down to 0.2 identity ("target mirroring"); ligand-only reaches r = 0.66; proposes novelty tiers | §3.7 — we use 90% and *keep* a 50–90% gray zone. Now a referenced limitation, not our judgement call. Novelty tiers implemented. |
| Graber et al. 2025, *Nat. Mach. Intell.* | PDBbind↔CASF leakage inflates all reported numbers; cleaning it collapses them | §3.7 — our four filters, and the Tanimoto-NN failure as positive evidence the split is clean |
| Schuh et al. 2026, *Chem. Sci.* | widely used biomolecular benchmarks carry cross-split contamination and label conflicts | §3.7 — our audits are exactly this, applied to our own data |
| Brown 2025, *PNAS* (CORDIAL) | the fix is an interaction-only inductive bias, denied direct structure parameterisation; survives leave-superfamily-out | §3.3 — **tested and not supported here**: ECIF-norm 0.2077 is a tie with both marginals on our target-disjoint split |
| Rayka et al. 2025, *Sci. Rep.* | five UQ methods compared on Leak-Proof PDBBind | §3.5 — **this scoops our UQ comparison.** Our remaining contribution is the 3D-GNN model class, the target-clustered inference, and the replication failure |
| Seitzer et al. 2022 | Gaussian NLL is a known pathology; β-NLL is the published fix | §3.5 — **in flight**, see §5 |
| Gibbs et al. 2023, *JRSS-B* | exact conditional coverage is impossible in finite samples; a marginal↔conditional spectrum exists | §5 — the framework for the missing conformal work |
| Jeliazkova et al. 2026 | conformal formalises what applicability-domain heuristics approximate; coverage drops when exchangeability breaks | §5 — predicts our conformal result should degrade on novel chemistry. Untested. |

---

## 5. What is missing, ranked by what it would settle

### KEYSTONE — pose-sensitivity of the predictor (not done, not in the literature in this form)
Everything above is circumstantial for the claim in §2. The direct test is: **perturb the ligand pose
inside the pocket and measure whether the prediction moves.** Rigid-body rotations and translations of
increasing magnitude, plus decoy poses (another ligand's pose for the same pocket), holding the molecule
and the pocket identical so only the pose changes.

If the prediction is insensitive to pose, then the predictor cannot provide a pose gradient, and §3.1's
finding (gradient points away from the pocket) stops being a curiosity and becomes a corollary. If it
*is* sensitive, the claim in §2 is wrong and the failure lies elsewhere — which is equally worth knowing.
Either way this is the experiment the programme turns on. Volkov et al. compared free versus bound
states, which is adjacent but not this; DIAG1 measured the gradient, not the prediction.

Cost: inference only, no training. Needs GPU, currently occupied; queue behind the A1e-β chain.

### HIGH — PLIP interaction types as an eighth representation block
ECIF element-pair counts are a coarse notion of interaction. PLIP gives *typed, pose-specific*
interactions: hydrogen bonds, hydrophobic contacts, π-stacking, salt bridges, halogen bonds. The
infrastructure already exists in `guidance/track_e/` (`PlipLabelStore`). If typed interactions also add
nothing over the marginals, §3.3's redundancy claim becomes very strong. If they add something, that is
the positive result and it names the fix. CPU only.

### HIGH — conditional conformal coverage
Our one positive UQ result (A1d) is *marginal* coverage. Jeliazkova et al. 2026 predict it should
degrade where exchangeability breaks, i.e. on novel chemistry; Gibbs et al. 2023 give the framework.
Stratify coverage by novelty tier (we have the tiers) and by protein family. A valid marginal interval
that silently under-covers exactly the novel compounds a practitioner cares about is a result worth
reporting, and it is the most promising remaining novelty in the UQ leg. CPU only.

### MEDIUM — protein family / target class stratification
Where does it fail? Kinases versus proteases versus nuclear receptors. Cheap, and it turns a pooled
number into a usable one. Requires a family mapping (UniProt entry names are already in the anchor
table). CPU only.

### MEDIUM — A1e-β (β-NLL), running
`scripts/a1e_beta_chain.sh`, queued behind Arm B. Balanced accrual: β2021 → plain2022 → β2022 →
plain2023 → β2023, so the comparison is 1v1, 2v2, 3v3 at every interruption point (A1e as it stands is
one seed). If β-NLL also fails, §3.5 becomes much stronger — the negative result survives the field's
own correction.

### LOW / blocked
Sequence conservation and MSA features for pocket residues; pocket druggability descriptors (volume,
enclosure, buriedness — needs fpocket or CASTp); LP-PDBBind structures (licence); Binding MOAD (manual
download).

---

## 6. What this changes about the thesis

Nothing is discarded. The reframing is a strengthening, and it is small in terms of text:

* **Chapter IV** gains a section: the information-source audit (§3.2–§3.4, §3.8).
* **Chapter X, conclusion (1)** currently argues *"the failures were not attributable to low predictor
  accuracy: the predictors reached Pearson correlations of 0.58 to 0.65."* That argument now leaks,
  because models with no protein input reach the same correlations. It must become: the failures were
  not caused by low accuracy but by *where the accuracy comes from* — which is also why no gradient was
  available.
* **Abstract** gains one sentence.
* **Arm B / Arm C** (surrogate data scale-up) keep running, but the question improves: not "does R² go
  up" but "does more data close the gap to a ligand-only model". The ligand-only baseline becomes a
  mandatory comparator for every arm.

The contribution statement moves from *"a documented negative result that locates where learned guidance
signals fail"* to *"a documented negative result that **explains why** learned guidance signals fail,
with the mechanism isolated by an information-source audit and the detection failure characterised."*

---

## 7. Reproduction

```bash
PYTHONPATH=. python guidance/cheminformatics/chem_data.py                  # SMILES, ECFP4, descriptors, scaffolds
PYTHONPATH=. python guidance/cheminformatics/pocket_features.py            # pocket descriptors + ECIF
PYTHONPATH=. python guidance/cheminformatics/ligand_only_baseline.py       # the ladder's ligand rung
PYTHONPATH=. python guidance/cheminformatics/compare_vs_structure.py       # paired, BH-corrected
PYTHONPATH=. python guidance/cheminformatics/representation_ladder.py      # all information sources
PYTHONPATH=. python guidance/cheminformatics/novelty_tiers.py              # Mattsson-style tiers
PYTHONPATH=. python guidance/cheminformatics/scaffold_audit.py             # with the untrained-baseline control
PYTHONPATH=. python guidance/aggregate_results.py <logdirs> --compare A B   # multi-seed, with a power check
PYTHONPATH=. python guidance/run_tests.py                                  # 7 suites, 91 tests
```

All statistics use the project's standing rules: target-clustered bootstrap (B = 2000, seed 20260925)
resampling targets rather than complexes, Benjamini-Hochberg over every family of tests actually run,
model selection on validation only, and pre-registered analysis plans reused verbatim rather than
re-derived per experiment.

---

# Part II — Deep gap analysis: three failure mechanisms, and which one is binding

Added 2026-10-10 after a systematic search of the guidance-theory, reward-hacking and co-folding
literatures, which Part I had not touched. Part I established that the *guide* is uninformative. The
search shows that is one of **three** independently documented ways this kind of system fails, that the
literature has named and partially solved the other two, and that the three make different, testable
predictions about which published remedy would help us. That turns the programme from "run more
experiments" into "predict, then test".

## 8. The three mechanisms

### M1 — the guide is uninformative (an information problem)
**Ours:** ligand descriptors alone match the 3D GNN; pocket descriptors alone reach R² 0.289 on unseen
targets; the two are redundant; ECIF raw counts are ligand size in disguise; and the factorial strip
experiment shows the model responds ~5× more to how many protein atoms exist than to where the ligand
sits, with deleting the ligand's actual contacts costing *less* than deleting atoms that touch nothing.

**Literature:** Volkov et al. 2022. And, critically, at the frontier: **Bret et al. 2026** (J. Chem. Inf.
Model.) challenged **Boltz-2** — the state-of-the-art co-folding model — with *target mutation and target
shuffling*, and found its affinity classification "remains insensitive to key binding site mutations and
even in some cases to target exchange", with predictions "relatively independent of pose quality".

That matters more than any result of ours. It means this is **not an artefact of our small EGNN or our
4 GB budget**: the best available model has the same pathology, probed the same way. Our factorial strip
design (equal atom-count deletion, differing interface content) goes one step further than their target
shuffling, because it holds the amount of protein constant while varying only which protein is in
contact — but the phenomenon is the same, independently confirmed, and now citable.

### M2 — the gradient is misaligned (a mechanism problem)
**Ours:** DIAG1 — gradient magnitude correlates *positively* with distance from the pocket (r = +0.24).

**Literature, and this names our mechanism exactly:** Wallace et al. 2023 (DOODL, ICCV) state that
classifier guidance "requires either training new **noise-aware models** to obtain accurate gradients or
using a **one-step denoising approximation** of the final generation, which leads to **misaligned
gradients** and sub-optimal control". Our guidance is position-only on the clean-data estimate — i.e.
precisely the one-step approximation they identify as the problem. Burns et al. 2026 give the finite-
sample theory: likelihood approximations at intermediate timesteps propagate into erroneous posteriors,
mis-estimate posterior spread, and hallucinate modes, and this arises from a multimodal prior alone.

**Published remedies we never tried:** DOODL (optimise diffusion latents against gradients taken on the
true generated output, via an invertible diffusion process); a genuinely noise-aware predictor trained
on intermediate noisy states; or SVDD (Li et al. 2024) — derivative-free guidance using soft value
functions that look ahead from noisy states, which sidesteps the need for a differentiable proxy
altogether and is demonstrated on molecule generation.

**A tension worth recording:** Kynkäänniemi et al. 2024 (266 cites) find guidance is *harmful* at high
noise, unnecessary at low noise, and beneficial only in a middle interval — improving ImageNet-512 FID
from 1.81 to 1.40. Our thesis tested timestep windowing as a mechanism variant and it failed. Either the
molecular setting differs, or the window was wrong, or M1 dominates so completely that no scheduling
helps. M1 predicts the third.

### M3 — the objective is gameable (reward over-optimisation)
**Ours:** rejection sampling improved the raw docking score in 12 of 15 pockets but worsened ligand
efficiency in 14 of 15 (+12 heavy atoms on average, −29 pp PoseBusters validity). Synth guidance raised
the predictor's own score while lowering the independently computed RA-score.

**Literature:** Gao et al. 2022, *Scaling Laws for Reward Model Overoptimization* (1,335 cites) — this is
Goodhart's law for proxy rewards, measured, with **different functional forms for RL versus best-of-n
sampling**. Our rejection sampling *is* best-of-n, so their best-of-n curve is the right reference and we
never used it. Yoshizawa et al. 2025 (Nature Communications) address exactly this in molecular design and
propose applicability-domain gating as the mitigation.

**Independent corroboration of the evaluation side:** Harris et al., *PoseCheck* — generative SBDD models
produce "significantly more physical violations and fewer key interactions compared to baselines, calling
into question the implicit assumption that providing rich 3D structure information improves molecule
complementarity."

## 9. The dissociation that makes this a contribution rather than a complaint

Thomas et al. 2021 (J. Cheminformatics) guided REINVENT with **Glide docking** — real physics, not a
learned surrogate — and report that it beats a ligand-based scorer, reaches affinities past known actives,
and "learns to generate molecules that satisfy crucial residue interactions, which is information only
available when taking protein structure into account."

So structure-based guidance is **not** broken in general. What fails is guidance by a **learned affinity
surrogate**. Put next to M1, the two statements isolate the cause: the surrogate never encodes the
interface, so it cannot pass on interface information no matter how the gradient is computed. That is a
sharper claim than "guidance failed", and it is one this project is unusually well placed to defend,
because it holds the generator, pockets, evaluation harness and statistics fixed across both.

## 10. Research questions, in falsifiable form

**RQ1.** Is guidance failure attributable to the guide (M1), the gradient (M2), or the objective (M3),
and which is binding? *Status:* M1 established; M2 and M3 documented in the literature but not isolated
here. The three are separable by experiment, which is what Part II proposes.

**RQ2.** Does a guide that *does* read the interface produce usable gradients on the same generator and
pockets? *Operationalisation:* re-run the guidance ablation with Vina or a PLIP-derived interaction score
in place of the learned surrogate. Thomas et al. predict yes. If yes, M1 is confirmed as binding and the
thesis has a positive control for its own negative result — which it currently lacks.

**RQ3.** Does fixing the gradient without fixing the guide help? *Operationalisation:* DOODL-style
gradients through the denoiser, or SVDD derivative-free guidance, with the existing surrogate.
**M1 predicts NO.** This is the sharpest prediction the programme makes, because the literature expects
these methods to help and our mechanism says they cannot.

**RQ4.** Can conformal intervals gate reward over-optimisation? A1d established that split-conformal is
the *only* valid uncertainty on this task — every point-wise σ method failed. Yoshizawa et al. use
applicability domain for exactly this purpose. **M3 predicts yes for the rejection-sampling route; M1
predicts no for the gradient route.** A single experiment that confirms one and refutes the other would
be strong evidence for the whole decomposition.

**RQ5.** How much is there to win at all? The noise ceiling is R² 0.578, the ligand-only floor 0.353, and
70% of even that floor is a class-level prior rather than SAR. Does any amount of extra data move a model
beyond the class prior? *Arm B and Arm C now answer this question*, which is a better question than the
one they were launched with.

## 11. Novelty candidates, ranked honestly

1. **Conformal-gated guidance** — apply the guidance gradient (or accept a rejection-sampling candidate)
   only where the surrogate's conformal interval is narrow enough to trust. The components all exist
   separately: conformal for molecular property (Parks 2020; Rayka 2024; Jeliazkova 2026), AD-gating
   against reward hacking (Yoshizawa 2025), confidence-dependent guidance strength (Azangulov et al.
   2025, theoretical and in images). **The combination, for structure-based generative design, appears
   unoccupied.** Honest caveat: our own M1 evidence predicts it will help the rejection route and *not*
   the gradient route — which is precisely why it is worth running as a pre-registered prediction rather
   than proposed as a method that works.

2. **The factorial interface-vs-bulk ablation** as a general diagnostic. Bret et al. 2026 did target
   shuffling; holding atom count constant while varying interface content is a step finer and applies to
   any affinity model. Small, clean, reusable.

3. **The noise-ceiling power analysis of the task.** The window between the ligand-only floor and the
   label-noise ceiling is 0.2245 R², narrower than one of our confidence intervals. Published
   comparisons reporting 0.02–0.05 R² gains on data of this kind are reporting inside that window. This
   is a methodological result with implications well beyond our thesis.

4. **The class-prior decomposition** (within-target y-randomisation). 70% of ligand-only performance
   survives destroying all within-target SAR. The right zero point for this task is 0.247, not 0.

**Not novel, do not claim:** the UQ method comparison (Rayka et al. 2025 did five methods on Leak-Proof
PDBBind); conformal for affinity (Parks 2020); descriptor models matching deep learning (van Tilborg
2022, 314 cites; Deng 2023, 213 cites); affinity models ignoring the protein (Volkov 2022; Bret 2026).

## 12. What changes in the plan

RQ2 is now the highest-value experiment in the programme and it was not previously on the list: **the
thesis has no positive control.** Every route it tested failed, which leaves open the objection that the
harness cannot detect success at all. Guiding with Vina or a PLIP score — a guide that demonstrably does
read the interface — would either produce the improvement Thomas et al. report, validating the harness
and confirming M1, or fail too, which would move the explanation to M2/M3 and be equally decisive.

Revised order: RQ2 (positive control) → RQ4 (conformal gating, both routes) → RQ3 (DOODL/SVDD) → RQ5
(Arms B/C, already running).

---

# Part III — What the finished thesis looks like, and what is actually novel

Added 2026-10-10. Written so the target is fixed before the work, not reconstructed after it.

## 13. The claim the 11 items are assembling

> Learned-surrogate guidance in structure-based generative design does not fail for want of tuning,
> accuracy, or data. It fails because the surrogate's accuracy is carried by channels that are not
> pose-specific — so it has no interface signal to pass into a gradient. We isolate that mechanism
> directly, show the detection of such failures is itself unreliable (only conformal intervals are
> valid), and bound how much the task could ever have rewarded: the window between what a
> protein-blind model achieves and what the label noise permits is narrower than one of our confidence
> intervals.

Three sentences. Everything else is evidence for one of them.

## 14. What each item contributes

| # | contributes | if it comes out the other way |
|---|---|---|
| 1 positive control (Vina/PLIP guide) | **validates the harness** and confirms M1 as binding | if Vina guidance also fails, the cause moves to M2/M3 and our M1 work explains less than claimed — the single biggest risk in the programme |
| 2 PLIP typed interactions | makes the redundancy claim near-airtight | if typed interactions DO add signal, that is a positive result and it names the fix: use interaction types, not element-pair counts |
| 3 conditional conformal coverage | the clearest standalone novelty in the UQ leg | if coverage holds conditionally, conformal is stronger than we thought — also publishable |
| 4 EGNN y-randomisation | closes the one caveat on the class-prior decomposition | if the EGNN's class prior differs much from the descriptor one, the decomposition needs restating per architecture |
| 5 activity-cliff metric | turns an observed pattern into a measured one | — |
| 6 matched molecular pairs | tests local SAR, the complement to the class prior | if MMP direction IS captured, the model knows more than the pooled numbers show |
| 7 family stratification | turns one pooled number into a usable one | — |
| 8 Arm A + Arm C | answers RQ5: does more data move anything past the class prior | if it does, the data-scaling route is alive after all |
| 9 writing | makes it defensible: 14 missing citations, the leaked Ch. X argument, the limitations | — |
| 10 conformal-gated guidance | pre-registered split prediction (helps rejection, not gradient) | either half being wrong is informative about the decomposition |
| 11 DOODL / SVDD | the sharpest prediction: M1 says these will NOT help | if they DO help, M2 was binding all along and the thesis's centre of gravity shifts |

Items 1, 2 and 11 can each overturn part of the story. That is a feature: the programme makes
falsifiable predictions rather than accumulating confirmations.

## 15. Novelty, tiered honestly

**Tier 1 — defensible as novel.**
* The factorial interface-vs-bulk ablation: three deletions of exactly 50% of protein atoms, differing
  only in interface content. Destroying the ligand's real contacts costs *less* than deleting atoms that
  touch nothing (1.058 vs 1.245 pK, overlapping CIs), while destroying the pose costs 0.278 pK. Bret et
  al. 2026 did target shuffling on Boltz-2; holding atom count constant is a step finer and reusable on
  any affinity model.
* The power analysis of the task: floor 0.3531 (protein-blind), ceiling 0.5776 (label noise), window
  0.2245 R², mean CI width 0.3871. Published comparisons reporting 0.02–0.05 R² gains are reporting
  inside a window their design cannot resolve. This applies well beyond our thesis.
* The class-prior decomposition: within-target y-randomisation reaches R² 0.2468 (sd 0.0019), so ~70%
  of the best protein-blind model's performance survives destroying all within-target SAR. The right
  zero point for this task is 0.247, not 0.
* Connecting the information audit to guidance-gradient failure. The premise (accuracy without
  structure) and the detection failure (point-wise σ does not rank error) are each established in the
  literature; nobody has drawn the consequence for generative guidance.
* Ternary (1.58-bit) QAT on a 3D affinity GNN, test R² indistinguishable from FP32.

**Tier 2 — replication that adds rigour. Valuable; do not call it new.**
Volkov et al.'s result on a leakage-controlled, target-disjoint split, extended with both marginals and
their mutual redundancy; descriptor models matching deep learning; affinity models being insensitive to
which protein they are given.

**Tier 3 — not novel, cite and move on.**
The UQ method comparison (Rayka et al. 2025, five methods on Leak-Proof PDBBind). Conformal prediction
for affinity (Parks et al. 2020).

## 16. What this is and is not

It is a falsification study with a mechanism and a measurement limit. Its value is telling the field
where not to spend effort, with evidence, on data whose leakage is controlled four ways and whose
statistics resample targets rather than complexes.

It is not a better model, and it will not become one. Sized honestly: a strong S2 thesis, and a credible
methods or negative-results paper (JCIM, J. Cheminformatics, or a workshop). Not a Nature paper. Anyone
promising otherwise from these numbers is misreading the window in section 15.

## 17. The strategic conclusion that falls out, and where to dig next

The ceiling is set by **label noise**, not by architecture. Raising repeat-measurement agreement from
Pearson 0.76 to 0.85 — which Landrum et al. 2024's published "maximal curation" is designed to do —
moves the ceiling from R² 0.578 to 0.72. That is more headroom than any architecture change in this
literature has demonstrated. **The highest-leverage move in this field is better labels, not better
models.** It is also the cheapest thing we have not done.

Directions that follow naturally, by field:

*Cheminformatics.* Maximal curation of the Arm B/C labels (raises the ceiling, cheap, immediate).
Activity-cliff-aware training objectives. Synthesizability as an active design objective rather than a
filter.

*Bioinformatics.* Leave-superfamily-out splits — ours is per-target, which is easier, and CORDIAL's
validation standard. Evolutionary conservation and coupling as pocket features. Cross-species transfer.

*Biomolecular engineering / informatics.* Everything here uses a single static pocket: protein dynamics
and ensemble pockets are untouched, as are water networks and desolvation. And the open question Bret et
al. 2026 leaves behind: a co-folding model has the same insensitivity to target identity, so whether it
can nonetheless *guide* better than our surrogate is unknown and directly testable with the harness this
project already has.

---

# Part IV — Four fields: gaps, questions, and the algorithms each needs

Added 2026-10-10 after searching the generative-architecture, interaction-conditioning, synthesizability,
retrobiosynthesis, equivariance and active-learning literatures. Part III fixed the target; this fixes
where the work goes afterwards, and names the methods rather than gesturing at directions.

## 18. The finding that gives our negative result somewhere to land

Our M1 result says a learned surrogate cannot pass interface information through a gradient. The obvious
follow-up question — "so what should one do instead?" — now has a published answer, and it is not a
better surrogate. It is to stop guiding and start **conditioning**.

* **ShEPhERD-2** (Abeywardane et al. 2026) argues that an **interaction profile** — shape,
  electrostatics, directional pharmacophores — is "a sufficient and transferable design specification
  for molecular design", and generates 3D structures **conditioned on explicit interaction profiles**.
  One model covers bioisosteric fragment merging, dual-target design, selectivity engineering and
  modality hopping **without task-specific retraining**. They call interaction profiles "a
  chemotype-agnostic interface between structure hypotheses and molecular design".
* **FLOWR / FLOWR.MULTI** (Cremer et al. 2025, Nature Comp. Sci.) — flow matching with equivariant
  optimal transport and pocket conditioning; beats diffusion baselines on PoseBusters validity, pose
  accuracy and **interaction recovery**, 70× faster, and samples ligands matching **predefined
  interaction profiles** without retraining or resampling. Ships SPINDR, a cocrystal dataset built to
  fix known data-quality problems.
* **PGMG** (Zhu et al. 2023, Nature Comms, 90 cites) — the earlier pharmacophore-conditioned instance.
* **Zhang et al. 2026** (J. Comput.-Aided Mol. Design) survey 100+ SBDD methods and name
  **"shift from implicit to explicit conditioning"** as one of four field-wide trends — alongside
  SE(3)-equivariance, sequential→parallel generation, and synthesizability in the generative prior.
  They also name **"geometric relaxation artefacts in docking-based metrics"** as a known evaluation
  problem, which is our Vina-hacking finding under the field's own term.

**What this does to our conclusion.** It stops being "guidance failed, future work unclear" and becomes:
the failure is *informational*, and the remedy is *architectural* — replace a surrogate's implicit
gradient with an explicit interaction specification the generator is conditioned on. Our contribution is
the diagnosis and the prediction; theirs is the instantiation. Chapter X should say exactly that and
cite them, because a negative result that names its own successor is far more useful than one that does
not.

**What it does NOT do:** it does not scoop the diagnosis. Nobody has shown *why* gradient guidance by a
learned affinity surrogate cannot work. That is still ours.

## 19. The four fields

### Cheminformatics
*Gap.* Label noise sets the ceiling (R² 0.578) and nobody treats it as the binding constraint. Our own
Arm B/C labels mix Kd/Ki/IC50 with a composition that shifts between splits.
*Question.* Does curating labels buy more than any architecture change? Landrum et al. 2024's maximal
curation is designed to raise repeat-measurement agreement; Pearson 0.76 → 0.85 moves the ceiling
0.578 → 0.72.
*What we can solve.* Re-curate Arm B/C with their published code and re-run one arm. Cheap, and it is
the highest-leverage move available to us.
*Methods needed.* `rinikerlab/overlapping_assays` (maximal curation); MoleculeACE for activity-cliff
metrics; matched molecular pairs for local SAR; Mervin et al.'s Probabilistic Random Forest, which
propagates the per-measurement σ into training instead of pretending labels are exact.

### Bioinformatics
*Gap.* Our split is per-target. CORDIAL's standard is **leave-superfamily-out**, which is strictly
harder, and Mattsson et al. 2026 show sequence-identity splits leak down to 0.2 identity via target
mirroring.
*Question.* Does any of our conclusions survive a superfamily-level split? Our class-prior decomposition
predicts the ligand-only floor should drop sharply, since chemotype→protein-class priors stop
transferring.
*What we can solve.* Build a superfamily-holdout split from the UniProt entry names already in the
anchor table and re-run the ladder. No new data, no new models.
*Methods needed.* Pfam/InterPro family assignment; MMseqs2 or the existing Biopython aligner for
clustering; conservation from an MSA (ConSurf-style) as pocket features; ESM embeddings — note
`logs_lp_split_stage0_esm2*` already exists in this repo, so that was tried and its result should be
re-read in light of the information audit.

### Biomolecular engineering / informatics
*Gap.* Everything here uses one static pocket. No dynamics, no ensemble, no water, no induced fit.
*Question.* Bret et al. 2026 showed a co-folding model (Boltz-2) is also insensitive to target exchange
— but can it nonetheless *guide* better than our surrogate? Unknown, and directly testable with the
harness this project already has.
*What we can solve.* The positive control (item 1) with Vina, then the same harness with a Boltz-2
affinity head, then with a PLIP interaction score. Three guides of increasing physical content on one
generator — that is a clean dose-response on "how much interface content does a guide need".
*Methods needed.* PLIP (eight non-covalent interaction types; 1,792 cites) — already wired in
`track_e/PlipLabelStore`; Boltz-2 affinity module and its fine-tuning framework
(`molecularinformatics/Boltz2_affinity`); ensemble pockets from MD snapshots or AlphaFold variants
(`alphafold_pocket_robustness.py` in this repo is the start of that).

### Bio-synthetic (synthetic biology / biosynthesis)
*Gap.* This is the field our Track D touched and then dropped. Synth guidance used **RA-score, a
heuristic**, and failed — and Guo et al. 2024 show precisely that "over-reliance on synthesizability
heuristics can overlook promising molecules", while heuristic/retrosynthesis correlation holds for
bioactives but breaks for other chemotypes. Meanwhile an entire adjacent objective is untouched:
**bio**synthesizability — can an enzymatic route make this molecule.
*Question.* Our strongest untested idea in this area: **is biosynthesizability a better-behaved design
objective than chemical synthesizability?** A retrobiosynthesis route is grounded in enzyme reaction
rules rather than in a learned heuristic, so it is far harder to reward-hack — which is exactly the
failure mode Track D hit.
*What we can solve.* Re-run the synth-guidance ablation with a retrosynthesis model in the loop instead
of RA-score (Guo et al.'s saturn shows this is now computationally feasible), and separately with a
retrobiosynthesis route score. The second has, as far as this search found, never been used as a
guidance objective in 3D structure-based generation.
*Methods needed.* AiZynthFinder or Syntheseus for retrosynthesis in the loop; `schwallergroup/saturn`
for sample-efficient RL; TANGO (Guo et al. 2026, Nature Comp. Sci.) for *constrained* synthesizability
— enforcing specific building blocks; RetroPath/novoStoic-style rule-based retrobiosynthesis
(Gricourt et al. 2024 review the AI methods); enzyme discovery and de novo enzyme generation for orphan
reactions (Chen et al. 2025) if a route needs a catalyst that does not exist yet.

## 20. Algorithms and models to have on the shelf

| need | method | why this one |
|---|---|---|
| generator, if replacing TargetDiff | SE(3)-equivariant **flow matching** (FLOWR) | Zhang et al. 2026 name it the promising direction; 70× faster, better interaction recovery |
| guidance without differentiable proxies | **SVDD** (Li et al. 2024) soft value functions | lookahead from noisy states; no gradient needed, which is the point if M1 holds |
| guidance with correct gradients | **DOODL** (Wallace et al. 2023) | removes the one-step denoising approximation that misaligns our gradients |
| conditioning instead of guidance | **ShEPhERD-2**, FLOWR.MULTI, PGMG | explicit interaction profile; sidesteps M1 entirely |
| reward over-optimisation control | applicability-domain / **conformal** gating (Yoshizawa 2025; our A1d) | our only valid uncertainty is conformal, which makes it the natural gate |
| oracle budget | **RL-AL** (Dodds et al. 2024) | 5–66× more hits per oracle call; makes expensive guides (FEP, Boltz-2) affordable |
| label noise | maximal curation (Landrum 2024); **Probabilistic RF** (Mervin 2021) | raises the ceiling and propagates σ instead of ignoring it |
| data efficiency | equivariant architectures (**NequIP**, 2,464 cites) | up to 3 orders of magnitude fewer training points; equivariance is not the thing that failed for us |
| harder split | leave-superfamily-out (CORDIAL) | our per-target split is the easier standard |
| synthesizability in the loop | **saturn** (Guo 2024), **TANGO** (Guo 2026) | retrosynthesis directly in optimisation, and constrained variants |
| biosynthesis | RetroPath-class rule systems; retrobiosynthesis reviews (Yu 2023; Gricourt 2024) | the untouched objective, and the one hardest to reward-hack |

## 21. Novelty, re-scored after this search

**Still open and ours.** The diagnosis — why a learned affinity surrogate cannot supply a usable
guidance gradient — with the factorial interface-vs-bulk ablation, the class-prior decomposition, and
the task-level power analysis. None of the papers above does this; they propose fixes without
establishing the mechanism the fixes are needed for.

**Newly scooped, do not claim.** "Condition on interaction profiles instead of guiding" — ShEPhERD-2 and
FLOWR.MULTI. Cite them as the direction our result points at, not as our idea.

**New candidate, appears unoccupied.** Biosynthesizability as a guidance/conditioning objective in 3D
structure-based generation. The components exist separately (retrobiosynthesis route planning; RL with
retrosynthesis in the loop; SBDD generators), and our own Track D failure gives a concrete motivation:
a rule-grounded enzymatic route is much harder to reward-hack than a learned synthesizability heuristic.
Scope before committing — and check the claim again, since this was a single search.

---

# Part V — Our result has a name, and the field has a taxonomy for it

Added 2026-10-10 after searching the shortcut-learning, scaling-law, multi-objective, benchmark,
prospective-validation, protein-design and biocatalysis literatures. Three of these reframe findings we
already hold; one answers a question we were about to spend 16 GPU-hours on.

## 22. The reframing: this is shortcut learning, and that is a literature

Our factorial result — the model responds ~5× more to how many protein atoms exist than to where the
ligand sits, and deleting the ligand's real contacts costs *less* than deleting atoms that touch nothing
— is not merely "the model ignores the interface". It is a textbook instance of **shortcut learning**:
the model latched onto a cheap proxy (bulk protein size, ligand size) that correlates with the label in
training, instead of the causal feature (interface complementarity).

* **Steinmann et al. 2024** give a unifying taxonomy of shortcut learning, a formal definition, and an
  organised survey of detection and mitigation methods, explicitly bridging the scattered terms —
  shortcuts, Clever Hans behaviour, spurious correlation, confounders. Our ablation is a
  shortcut-*detection* method; naming it that places it in a methodology rather than leaving it as a
  one-off diagnostic.
* **Lee et al. 2023 (CMRL, KDD)** build a structural causal model for *molecular relational learning*
  and use conditional intervention to isolate the causal substructure while removing the confounding
  effect of shortcut substructures. This is a published **mitigation** for our exact problem class, on
  molecular pair prediction. We should cite it as the principled answer and, if anything is to be built,
  build that rather than another surrogate.
* **Joe et al. 2026** found LLMs predicting molecular properties near-perfectly *via shortcut cues* —
  specifically molecular weight — and collapsing under nonlinear transformations, while simpler ML
  baselines stayed robust. Molecular weight is our heavy-atom-count result in a different model class.

**Consequence for the thesis.** Chapter IV's framing should be "the surrogate learned a shortcut, and
here is the ablation that proves which one", with the shortcut-learning taxonomy as the methodological
frame. That is stronger than a bespoke diagnostic and it connects our work to a much larger literature.

## 23. Scaling laws answer RQ5 before Arm C runs

We were going to spend ~16 GPU-hours per arm asking whether more data moves anything past the class
prior. The literature already bounds the answer:

* **Frey et al. 2023** (Nature Mach. Intell., 200 cites) measure neural scaling for chemical models
  across orders of magnitude up to a billion parameters: scaling **exponent 0.17** for chemistry
  language models and **0.26** for equivariant GNN interatomic potentials. At an exponent near 0.2,
  halving the error needs roughly a thousandfold more data.
* **Lee et al. 2025** ("Are neural scaling laws leading quantum chemistry astray?") find the largest
  foundation models on the largest datasets still fail on H₂ dissociation and cannot reproduce the
  repulsive curve of two bare protons. Scaling alone does not install the missing physics.
* **Chen et al. 2023** confirm a power law in molecular representation learning and show data *pruning*
  can beat data *addition*.

Put beside our own numbers — a label-noise ceiling at R² 0.578 and a 0.2245 window between the
protein-blind floor and that ceiling — the prediction is specific: **BindingNet's extra 14k rows should
not move the test score measurably past the class prior.** Arm B is already running, so we will have the
measurement. The point is that it is now a *pre-registered prediction with a quantitative basis*, not an
open-ended hope, and if the arms do show movement that is the surprising and interesting outcome.

## 24. Vina-hacking is a scalarisation failure, and Pareto is the named fix

Our Track C result — rejection sampling improved the docking score in 12 of 15 pockets while worsening
ligand efficiency in 14 of 15 (+12 heavy atoms, −29 pp PoseBusters validity) — is a textbook
demonstration of why collapsing objectives into one number fails.

**Fromer & Coley 2022** (Patterns, 145 cites) state it directly: scalarisation "imposes assumptions
about relative importance and uncovers little about the trade-offs between objectives", whereas Pareto
optimisation requires no such assumption and exposes the trade-off. Current instantiations:
**PMMG** (Liu et al. 2025, Advanced Science) — Pareto Monte-Carlo tree search, 51.65% success at
*seven* simultaneous objectives, 2.5× the previous state of the art; and **AI-MedCraft** (Barakat et al.
2026, JCIM) — adaptive Pareto-guided RL, broader Pareto coverage than REINVENT 4 at matched cost.

So Track C's failure is not just "rejection sampling is bad". It is evidence for a known methodological
point, and the fix has a name. Report it that way.

## 25. Prospective validation exists — and only where constraints are explicit

This matters because it bounds what the field has actually achieved, and because the pattern is the same
one our M1 result predicts.

* **PocketFlow** (Jiang et al. 2024, Nature Mach. Intell., 101 cites) — structure-based generation with
  **chemical knowledge explicitly encoded**; 100% chemically valid; wet-lab-confirmed actives for HAT1
  and YTHDC1 with binding modes **verified by X-ray crystallography**. Their ablation shows the explicit
  knowledge is what makes it work.
* **ClickGen** (Wang et al. 2024, Nature Comms) — assembly from modular click-chemistry reactions plus
  RL; PARP1 inhibitors synthesised and assayed in **20 days**, two leads at nanomolar activity.
* **Papidocha et al. 2026** ("The elephant in the lab") — most generative work never reaches the wet
  lab, and limited synthesizability is the main reason.
* Contrast from protein design: **RFdiffusion** (Watson et al. 2023, Nature, 1,613 cites) is celebrated,
  yet **Jiang et al. 2025** independently designed and tested five binders for each of six targets and
  report that only two Strep-Tag II binders worked at all, none matching antibodies, the rest failing on
  expression, non-specific binding or undetectable affinity. A negative replication in the neighbouring
  field, and a useful citation: computational success does not transfer, and the field needs more work
  like ours, not less.

Every prospective success above constrains generation **explicitly** — reaction templates, chemical
rules, interaction profiles. None of them guides with a learned scalar's gradient. That is our M1
prediction holding across the literature.

## 26. Benchmark fragmentation, and what we should map onto

Both major surveys name this as a field-level problem: **Zhang et al. 2025** (ACM Computing Surveys)
list "insufficient evaluation metrics and large-scale benchmarks" and "the need for experimental
validation" among the field's open challenges; **CBGBench** (Lin et al. 2024) exists because "diverse
settings, complex implementation, difficult reproducibility and task singularity" prevent systematic
understanding.

Our harness is bespoke. It is also unusually rigorous — target-clustered bootstrap, BH, pre-registration,
four leakage filters. The cheap, high-value move is to **report our metrics in CBGBench's and MolScore's
terms as well as our own**, so the numbers are comparable without giving up the statistics. Relevant
tooling: `Edapinenut/CBGBench` (de novo, linker, fragment, scaffold, sidechain tasks; interaction,
chemical, geometric and substructure validity), **MolScore** (Thomas et al. 2024, J. Cheminformatics —
re-implements GuacaMol, MOSES, MolOpt, three lines to integrate), and `zaixizhang/Awesome-SBDD`.

And one directly usable asset: **EPoCS** (Oruç et al. 2024, Bioinformatics) builds binding-site
representations from protein language models plus 3D tessellation, explicitly to "define challenging
train-test splits for believable benchmarking of pocket-centric machine-learning models". That is a
*pocket-similarity* split — harder than our per-target split, off-the-shelf, and a natural companion to
the superfamily holdout in §19.

## 27. Bio-synthetic: the tool for our novelty candidate already exists

§21 floated biosynthesizability as a guidance objective. The enabling method is published and open:

* **Probst et al. 2022** (Nature Comms, 105 cites) — "Biocatalysed synthesis planning using data-driven
  learning". A Molecular Transformer extended to biocatalysis with **EC-number class tokens** that
  capture catalysis patterns across enzymes in the same hierarchy. Forward prediction top-1 49.6%;
  retrosynthetic single-step round-trip top-1 39.6%. Dataset and models public.
* Supporting: **Yu et al. 2023** (Nature Catalysis, 108 cites) and **Gricourt et al. 2024** (ACS Synth.
  Biol.) review retrobiosynthesis; **Chen et al. 2025** cover de novo enzyme generation for orphan
  reactions; **Tripathi et al. 2025** (Biotech. Advances) and **Qiu et al. 2026** review enzyme-activity
  prediction and flag dataset imbalance as the limiting factor.
* And a cross-field confirmation of our own result: **van Lent et al. 2023** (ACS Synth. Biol.) test ML
  methods over simulated design-build-test-learn cycles in metabolic engineering and find **gradient
  boosting and random forests outperform the alternatives in the low-data regime**, robust to training
  bias and experimental noise. Descriptor-and-tree models beating deep learning when data are scarce and
  labels noisy is not a quirk of our task; it recurs in an entirely different field.

**The claim, stated narrowly enough to defend:** a retrobiosynthesis route score, grounded in enzyme
reaction rules rather than a learned heuristic, used as a design objective in 3D structure-based
generation. Motivation from our own Track D failure: RA-score is a heuristic and the guidance hacked it
(predictor's score up, independent RA-score down), whereas a rule-grounded route is far harder to game.
Still a single-search claim — re-check before committing.

## 28. The pattern across all four fields

One observation recurs independently in structure-based drug design (ours; van Tilborg et al. 2022;
Deng et al. 2023), metabolic engineering (van Lent et al. 2023), LLM chemistry (Joe et al. 2026) and
quantum chemistry (Lee et al. 2025):

> **Where data are scarce and labels noisy, simple models match or beat deep ones, scaling does not
> rescue a missing inductive bias, and explicitly encoded constraints beat learned guidance.**

Our thesis does not discover this. What it contributes is a **mechanism** for one instance of it: the
surrogate's accuracy comes from a shortcut, so there is nothing for a gradient to carry, and no amount
of tuning, capacity or data changes that. The methodological pieces — the factorial shortcut ablation,
the class-prior decomposition, the noise-ceiling power analysis — are the transferable part.

---

# Part VI — The physics agrees with our ceiling, and the objective itself is suspect

Added 2026-10-10 after searching the free-energy, binding-kinetics, thermodynamics, interpretability,
self-driving-lab and enzyme-kinetics literatures. Two of these are the strongest external support the
thesis has found; one is a limitation we have not yet stated anywhere.

## 29. The gold-standard physics method lands on the same ceiling we computed

We estimated a label-noise ceiling of R² = 0.578 from repeat-measurement agreement (Part II §3). An
entirely independent line of work reaches the same place from physics.

* **Chen et al. 2023** (J. Chem. Inf. Model., 99 cites) run **absolute** binding free-energy perturbation
  (ABFEP, in FEP+) over eight congeneric series against eight receptors: weighted **R² = 0.55** across
  the dataset, RMSE **1.1 kcal/mol**.
* **Ross et al. 2023** (Communications Chemistry, 117 cites) assemble what they describe as the largest
  public FEP dataset, ask how accurate FEP "is and can ever be", and *survey the reproducibility of the
  experimental measurements themselves*. Their conclusion: with careful preparation, **FEP achieves
  accuracy comparable to experimental reproducibility.** The gold standard is label-limited.
* **OpenFE** (Baumann et al. 2026, JCIM) — 15 pharmaceutical companies, >1,700 ligands: weighted RMSE
  **1.73 kcal/mol** on 58 public systems and **2.44 kcal/mol** on 37 blinded private systems, with only
  2 of 37 reaching sub-kcal/mol. No single dominant error source.

At 298 K, 1 pK unit = 1.364 kcal/mol, so those errors are **0.81 pK** (ABFEP), **1.27 pK** (OpenFE
public) and **1.79 pK** (OpenFE private). Our EGNN's test RMSE is **1.27–1.38 pK**.

And **ABFEP's R² = 0.55 sits within 0.03 of our independently derived ceiling of 0.578.** Two
methods with nothing in common — alchemical molecular dynamics and a ChEMBL repeat-measurement survey —
agree on where the ceiling is.

**The caveat, stated plainly because the comparison is easy to overstate.** FEP in these benchmarks is
mostly *relative* free energy within congeneric series: discriminating small differences inside one
chemical series, which is a harder discrimination per unit of error than our across-series task, and the
RMSEs are per-system. So "our laptop EGNN matches industrial FEP" is **not** a claim we may make. What
we may claim is narrower and still strong: the error floor of the field's most rigorous method is
0.8–1.8 pK, our models sit at 1.27–1.38 pK, and the most comparable number — ABFEP's absolute-affinity
R² of 0.55 — matches our ceiling estimate. The ceiling is real, and it is not an artefact of our data
handling.

## 30. There is also a PHYSICAL reason, not only a statistical one

Our Part II argument was statistical: noisy labels cap R². The thermodynamics literature supplies a
second, independent reason the quantity itself is hard to predict.

* **Jiménez et al. 2024** (Biophysica) analyse 3,025 protein-ligand ΔG values and find they occupy a
  **narrow range of Gibbs free energy**, with widespread apparent **enthalpy-entropy compensation
  (EEC)**: pushing ΔH more negative to gain affinity is met by compensating entropy, giving "a barely
  noticeable increase in affinity". They hypothesise the narrow ΔG range is a *product of protein
  evolution* — a homeostatic mechanism keeping affinities tuned to physiological ligand concentrations.
* **Olsson et al. 2011** (Protein Science, 166 cites) show the compensation is real rather than an
  artefact of ITC constraints, using ΔΔ-plots over 32 proteins: strong compensation in 22% of ligand
  modifications (twice chance), reinforcement in 15%. "Strong but imperfect."
* **Verteramo et al. 2019** (JACS, 116 cites) dissect a diastereomer pair on galectin-3 with ITC, X-ray,
  NMR relaxation and MD: nearly equal ΔG, substantially different ΔH and ΔS, with **conformational
  entropy dominating over solvation entropy**.

**Why this matters to us specifically.** R² is variance-explained. If ΔG occupies a narrow range by
thermodynamic and evolutionary construction, then the variance available to explain is small *before*
any measurement noise is added — and structural modifications that ought to improve binding often do
not, because entropy compensates. That is a physical explanation for three things we observed
empirically: the low label variance, the weak within-series SAR signal that our class-prior
decomposition exposed, and the activity-cliff behaviour in the novelty tiers. It is a far better
explanation than "the model is inadequate", and it is citable physical chemistry rather than ML
speculation.

## 31. The objective itself may be the wrong one — and we have never said so

This is a limitation the thesis does not currently state anywhere, and it is a large one.

* **Wang et al. 2022** (J. Chem. Theory Comput., 64 cites): "the drug molecule residence time or
  dissociation rate has been shown to **correlate with their efficacies better than binding
  affinities**."
* **Bernetti et al. 2019** (Annual Review of Physical Chemistry, 159 cites): residence time "could
  predict drug efficacy in vivo, perhaps even more effectively than conventional thermodynamic
  parameters (free energy, enthalpy, entropy)", and may become "a go/no go step".
* **Liu et al. 2026** (Expert Opinion on Drug Discovery): binding affinity "reflects only the
  equilibrium state of drug-target interactions and **often correlates poorly with in vivo
  pharmacological responses**."

Our entire thesis — generator, guidance, surrogate, evaluation — optimises **equilibrium affinity**. If
the field's own view is that koff / residence time tracks efficacy better, then even a *perfect* affinity
guide would be optimising a proxy with known weak transfer to the outcome anyone cares about. That
belongs in the limitations, and it opens a direction: **guidance toward residence time or koff appears
essentially unexplored in 3D generative structure-based design.** It is harder — kinetics needs
transition states and unbinding paths, not an equilibrium snapshot — which is exactly why it is open.

## 32. Interpretability tools that complement our ablation

Our factorial ablation detects *that* a shortcut exists. These quantify *where* it lives, and they are
off-the-shelf.

* **SME** (Wu et al. 2023, Nature Comms, 196 cites) — substructure mask explanation, built on chemically
  meaningful segmentations rather than arbitrary nodes or edges; explicitly "alerts them to unreliable
  performance". Chemist-interpretable by construction.
* **Komissarov et al. 2025** (JCIM) — systematically combine post-hoc attribution with uncertainty
  quantification and find "a strong synergy": attributing *uncertainty* to specific atoms or
  substructures exposes data gaps and model limits. **Feature Ablation** and **Shapley Value Sampling**
  worked best. Our A1d conformal result plus attribution is exactly this combination.
* **Rao et al. 2021** (Patterns, 89 cites) — five XAI benchmarks for molecular property with ground
  truth, six XAI methods on four GNNs, benchmarked against **seven medicinal chemists** of varying
  experience. Gives us a way to validate an explanation rather than merely produce one.

## 33. Bio-synthetic, deeper: enzyme kinetics has our exact problem

* **DLKcat** (Li et al. 2022, Nature Catalysis, **415 cites**) — kcat from substrate SMILES plus protein
  sequence; captures mutation effects; genome-scale predictions for 300+ yeast species, feeding
  enzyme-constrained metabolic models.
* **TurNuP** (Kroll et al. 2022, Nature Comms, 169 cites) — organism-independent, generalises to enzymes
  *dissimilar* to the training set, using differential reaction fingerprints plus a retrained protein
  Transformer. Earlier models "provide inaccurate predictions except for enzymes highly similar to
  proteins in the training set" — the same generalisation failure mode as affinity models.
* **MMKcat** (Sun et al. 2025) — multimodal with a missing-modality training mechanism; beats DLKcat,
  TurNuP, UniKP and others by 6.4% RMSE / 22.2% R² / 8.2% SRCC on BRENDA and SABIO-RK.

Note the framing in DLKcat's own abstract: "experimentally measured kcat data are **sparse and noisy**".
The enzyme-kinetics field states our Part II problem in its first sentence. If the biosynthesizability
direction (§27) is pursued, its labels carry the same ceiling, and the same analysis applies — which is
an argument for doing the ceiling analysis *first* there rather than after.

## 34. Where the loop closes: self-driving labs

* **Tom et al. 2024** (Chemical Reviews, **654 cites**) — the comprehensive review of self-driving
  laboratories: hardware, software, integration, and real-world examples with their levels of automation.
* **Bai et al. 2024** (Nature Comms, 91 cites) — distributed SDLs on a dynamic knowledge graph; they
  linked robots in Cambridge and Singapore for closed-loop optimisation of a pharmaceutically relevant
  aldol condensation and **generated a cost-yield Pareto front in three days**.
* **Häse et al. 2019** (Trends in Chemistry, 370 cites) — the original framing.

Relevance to us: every result in this thesis is in silico, and §25's lesson is that only
explicitly-constrained generation has reached the wet lab. The SDL literature is where the
design-make-test loop actually closes, and it is also where RL-AL's oracle-efficiency argument (§20)
pays off. This is context for the discussion, not work we can do — but a thesis that ends by naming the
loop it did not close is more honest than one that does not mention it.

## 35. Updated limitation list for Chapter X

Collecting what Parts IV–VI have added, since the limitations section is where most of this must land:

1. **Equilibrium affinity may be the wrong objective** (§31) — koff/residence time tracks efficacy better.
2. **Enthalpy-entropy compensation** narrows the predictable range for physical, possibly evolutionary
   reasons (§30) — the low label variance is not purely a data-quality artefact.
3. The ceiling is corroborated by physics (§29): ABFEP reaches R² 0.55; FEP is limited by experimental
   reproducibility.
4. Per-target split is easier than leave-superfamily-out (§19) and than pocket-similarity splits (§26).
5. Scaling exponents of 0.17–0.26 (§23) mean data growth cannot close the gap.
6. Scalarised objectives hide trade-offs; Pareto is the named alternative (§24).
7. The surrogate's failure is shortcut learning, a named phenomenon with published mitigations (§22).
8. Our harness is bespoke; CBGBench/MolScore are the field's comparability standards (§26).

---

# Part VII — A confound in our input data, and what the literature already proved about it

Added 2026-10-10. One finding here requires rephrasing our central claim; another partially scoops the
diagnosis; a third is a cheap rigor win we should simply take.

## 36. OUR POSES ARE DOCKED, NOT CRYSTALLOGRAPHIC — and the literature has measured what that does

This is the most consequential thing found in this round, and it is a confound nobody on this project
raised. CrossDocked, by construction, contains **ligands docked into non-cognate pockets**, not
crystallographic complexes. The "structure" our EGNN reads is itself a prediction.

**Boyles et al. 2021** (J. Chem. Inf. Model.) measured exactly this and the result maps onto our headline
finding almost line for line:

> "the performance of a structure-based machine learning scoring function trained and tested on **docked**
> poses is **lower** than that of the same scoring function trained and tested on **crystallographic**
> poses. We construct a **hybrid** scoring function by combining both structure-based and **ligand-based**
> features, and show that its ability to predict binding affinity using docked poses is **comparable to
> that of purely structure-based scoring functions trained and tested on crystal poses**."

In other words: when poses are docked, the structural channel degrades and **ligand-based features take
up the slack**. That is our result — ligand descriptors matching the 3D GNN — predicted in advance, for
a reason that is about the *data* rather than about what the architecture can learn.

**What this does to our claim.** It does not refute the information audit: the factorial strip result
stands on its own, because deleting the ligand's real contacts costing *less* than deleting atoms that
touch nothing is a statement about the trained model regardless of how the poses were produced. But it
supplies a second, independent, published explanation for *why* the model ended up that way, and it
means our claim must be scoped:

> On a split built from **docked** poses, a structure-based GNN is not distinguishable from a ligand-only
> descriptor model, and the factorial ablation shows the model keys on bulk protein content rather than
> interface geometry. Whether this persists on crystallographic poses is **untested here**, and Boyles
> et al. predict that the structure/ligand gap should widen if it were tested.

**The experiment this implies**, and it is now the second-highest-value item after the positive control:
re-run the representation ladder on crystallographic complexes. PDBbind structures are licence-blocked
for us, but the LP-PDBBind metadata we already use carries the PDB IDs, so a subset could be assembled
from the PDB directly. If the gap widens, the thesis has both the docked-pose caveat *and* the
demonstration that it matters — a much stronger chapter than either alone.

## 37. We match the reference model on our own dataset

**Francoeur et al. 2020** (JCIM, 195 cites), the CrossDocked2020 paper itself, report their best model —
an ensemble of five densely connected CNNs — at **RMSE 1.42** and **Pearson 0.612** on the affinity task.
Our EGNN sits at **RMSE 1.27–1.38, Pearson 0.60–0.65**. So our numbers are not weak for this dataset;
they are at the dataset authors' own reference level.

Two further points from that paper are directly relevant:
* They state plainly that "current methods of model evaluation are **overly optimistic** in measuring
  generalization to new targets" — which is the premise our whole leakage programme operationalises.
* They report that "training with docked poses **imparts pose sensitivity** to the predicted affinity".
  That sits in tension with our pose-sensitivity result (8 Å displacement moves the prediction only
  0.278 pK). The tension is probably not a contradiction: their pose metric is a *classification* AUC
  (0.956 at distinguishing good from bad poses) from a separate head, whereas ours measures how the
  *affinity regression* responds to displacement. Worth stating explicitly rather than ignoring, and
  worth testing: a pose-classification head on our data might be sensitive where the affinity head is
  not, which would localise the problem to the regression objective.

## 38. The diagnosis is partially scooped — and that is good news

**Scantlebury et al. 2023** (JCIM, 40 cites; PointVS) state the general observation we have been
building toward:

> "many scoring functions make predictions based on **data set biases rather than an understanding of
> the physics of binding**. These scoring functions perform well when tested on similar targets to those
> in the training set but fail to generalize to dissimilar targets. To test what a machine
> learning-based scoring function has learned, **input attribution** ... can be applied."

They then do the constructive half: with thorough train/test filtering, PointVS's attributions have
**high correlation with a distance-based interaction profiler**, "unlike those extracted from other
scoring functions", and they use the extracted pharmacophores for fragment elaboration with improved
docking scores.

**Consequences, both ways.** We must cite them and stop presenting "ML scoring functions learn dataset
bias, not physics" as our observation — it is theirs, and attribution is their method for showing it.
What remains ours: the **factorial** ablation that holds protein quantity constant while varying
interface content (attribution tells you which atoms matter, not whether the response is to *amount*
versus *identity*), the class-prior decomposition, the noise-ceiling power analysis, and the link to
guidance gradients. And the good news is substantive: PointVS demonstrates a carefully filtered model
*can* learn real interactions, so our result is a statement about this model and this data regime, not
an impossibility proof. That is a more defensible and more useful claim.

## 39. Reporting standards: the cheapest rigor win available

**Kapoor & Narayanan 2023** (Patterns, **1,201 cites**) survey leakage across ML-based science: **17
fields, 294 affected papers**, an eight-type leakage taxonomy, and a reproducibility study in which
"when the errors are corrected, complex ML models do not perform substantively better than decades-old
logistic regression" — our descriptor-beats-GNN result in another domain. They propose **model info
sheets** as the remedy.

Alongside:
* **Artrith et al. 2021** (Nature Chemistry, 419 cites) — "Best practices in machine learning for
  chemistry": the elements needed for reliable, repeatable, reproducible models.
* **TRIPOD+AI** (Collins et al. 2024, BMJ, **3,884 cites**) — a 27-item reporting checklist. Clinical in
  origin, but the discipline transfers directly.
* **IMPACT** (Gangwal et al. 2026) — model cards, containerised environments, blind validation,
  uncertainty quantification, FAIR data.
* **Lu et al. 2022** (JAMA Network Open) — 15 guidelines collectively request **220 unique items**;
  deployed models document a median of **39%**. The items most often missing are exactly the ones about
  *reliability*: external validation, uncertainty measures, missing-data strategy.

**Action.** Fill out Kapoor & Narayanan's model info sheet and an Artrith-style checklist as a thesis
appendix. We already satisfy an unusual number of these — four leakage filters, pre-registration,
target-clustered inference, conformal uncertainty, seed replication, 91 regression tests, a reproducible
environment — but none of it is presented as a checklist, which is how a reader verifies it quickly.
This is hours of work for a disproportionate credibility return, and it converts our rigor from
something a reader must infer into something they can check.

## 40. Receptor flexibility: the static-pocket limitation has off-the-shelf answers

* **Mohammadi et al. 2022** (Scientific Reports, 35 cites) — ensemble learning *over* ensemble docking:
  rank receptor conformers by their importance to final accuracy, and a **few of the most important
  conformers suffice to reach 1 kcal/mol**, with better early enrichment than unweighted ensemble
  docking. Graph-based redundancy removal beats clustering for selecting representatives.
* **SubPEx** (Hellemann et al. 2023, JCTC) — weighted-ensemble path sampling to accelerate
  binding-pocket conformational sampling; open source.
* **FlowDock** (Li et al. 2026, Acta Pharm. Sin. B) — Bayesian Flow Networks doing pose generation and
  affinity prediction jointly *while modelling protein flexibility*; state of the art on physical
  plausibility.

Our `alphafold_pocket_robustness.py` is the beginning of this line in this repo and should be read in
light of §36: if pose quality is a confound, receptor conformational choice is part of the same problem.

## 41. A calibration point: the literature's good numbers come from an easier setting

Worth stating because it is easy to feel our R² is poor by comparison:

* **Li et al. 2019** (J. Med. Chem., 80 cites) — multitask DNN over **391 kinases**, auROC 0.90
  internal, and an experimentally validated 0.75 auROC over 1,410 kinase-compound pairs.
* **Schifferstein et al. 2024** (JCIM) — docking-informed ML across the kinome, **R² 0.63–0.74** on
  unseen inhibitors.

Those are substantially better than our 0.34–0.41 — but they are **within a single, data-rich protein
family, with the targets seen in training**. Our split is target-disjoint across 127 unseen targets from
many families. The settings are not comparable, and the difference is a property of the task definition,
not of the models. Any thesis table that places our numbers beside kinome-wide numbers must say so.

## 42. Running tally of what is still ours

After seven rounds of searching, the defensible novelty has narrowed but held:

1. The **factorial interface-vs-bulk ablation** — equal protein deletion, differing interface content.
   Attribution methods (PointVS, SME, Shapley) say *which atoms* matter; this says whether the response
   is to *amount* or *identity*. Not found elsewhere.
2. The **class-prior decomposition** via within-target y-randomisation — the correct zero point for this
   task is 0.247, not 0.
3. The **task-level power analysis** — floor 0.3531, ceiling 0.5776, window 0.2245 R², narrower than one
   of our confidence intervals; now corroborated from physics by ABFEP's R² 0.55 (§29).
4. The **link from the information audit to guidance-gradient failure**.
5. **Ternary QAT** on a 3D affinity GNN.

And the honest scoping that Part VII forces: all of it is demonstrated on **docked** poses, which is a
caveat to state in the abstract, not only in the limitations.

---

# Part VIII — The rabbit hole: three load-bearing findings, two of which go against us

Added 2026-10-10, final deep-dive round. This part is uncomfortable and should stay that way. One finding
substantially scoops our ligand-only programme, one undermines a metric the thesis leans on, and one
reveals a disagreement in the literature that our ceiling estimate quietly took a side in.

## 43. Our ligand-only baseline programme has been done, systematically, with a released tool

**Durant et al. 2023**, *"Robustly interrogating machine learning-based scoring functions: what are they
learning?"* (Bioinformatics):

> "We compared the performance of a diverse set of popular MLBSFs (RFScore, SIGN, OnionNet-2, Pafnucy,
> and PointVS) to **our proposed baseline models that can only learn dataset biases** on a range of
> benchmarks. We found that **these baseline models were competitive in accuracy to these MLBSFs in
> almost all proposed benchmarks, indicating these models only learn dataset biases.**"

They released the platform: `github.com/guydurant/toolboxsf` (ToolBoxSF).

This is not adjacent to our work — **it is our work**, done across five published scoring functions
rather than one, with a tool for others to repeat it. Constructing "baselines that can only learn dataset
bias" and showing the real models do not beat them *is* the ligand-only ladder.

Combined with Volkov et al. 2022, Boyles et al. 2021, Scantlebury et al. 2023 and Sieg et al. 2019, the
position is now unambiguous: **"ligand-only / bias-only baselines match structure-based ML scoring
functions" is established prior art with five independent demonstrations.** We must present our version
as a *replication on a leakage-controlled, target-disjoint split with target-clustered inference*, and
nothing more. Writing it as a discovery would be indefensible, and a reviewer who knows this literature
would find it immediately.

What remains ours after this is narrower, and I state it plainly in §47.

## 44. Ligand efficiency is mathematically contested — and our Track C conclusion leans on it

The thesis abstract says rejection sampling "improved the raw docking score in 12 of 15 pockets but
**worsened ligand efficiency in 14 of 15** (on average 12 more heavy atoms, and 29 percentage points
lower PoseBusters validity)."

**Kenny 2018** (J. Cheminformatics, 95 cites), *"The nature of ligand efficiency"*:

> LE "has a nontrivial dependency on the concentration unit used to express affinity that stems from the
> **inability of the logarithm function to take dimensioned arguments**. Consequently, perception of
> efficiency varies with the choice of concentration unit and **it is argued that the ligand efficiency
> metric is not physically meaningful nor should it be considered to be a metric.**"

**Polanski et al. 2017** (J. Cheminformatics) explain the characteristic LE trend as a simple 1/MW
dependency and conclude LE "should not be interpreted as a molecular descriptor connected with a single
molecule but as a property (binding per gram)" — the hyperbolic trend is "not a real increase in binding
potency but a physical limitation". **Zhao 2025** (ACS Med. Chem. Lett.) adds that LE's "mathematical
construction embeds a **strong size bias** that distorts cross-size comparisons", and that
size-independent variants inherit sensitivity to an arbitrary standard-state choice.

**Why this matters to us specifically, and it is a real problem.** LE = affinity / heavy-atom count.
Rejection sampling made the molecules **12 heavy atoms bigger**. Affinity scales sublinearly with size.
So LE *must* fall, largely as an arithmetic consequence of the size increase. Presenting "LE worsened in
14 of 15" as independent evidence of failure **double-counts the size effect**.

**The fix, and the good news.** The conclusion survives intact once reframed on the quantities that do
not have this problem: the molecules became **12 heavy atoms larger** and **29 percentage points less
PoseBusters-valid**. Both are direct, unnormalised measurements. That is the honest and still-damning
version. Chapter VI/VII should lead with heavy-atom count and validity, mention LE only as corroboration,
and cite Kenny, Polanski and Zhao when it does. Dropping LE entirely would be over-correction —
medicinal chemists use it — but it cannot carry the argument.

## 45. The label-noise literature disagrees with itself, and our ceiling took a side

Our ceiling of R² = 0.578 rests on Hernandez-Garrido et al. 2023 (RMSE 1.04, Pearson 0.76 for mixed
Kd/Ki/IC50) and on Landrum et al. 2024, who call mixing "a source of significant noise". But:

**Kalliokoski et al. 2013** (PLoS ONE, **311 cites**) reach a *milder* conclusion from the same kind of
analysis:

> "The standard deviation of IC50 data is **only 25% larger** than the standard deviation of Ki data,
> suggesting that mixing IC50 data from different assays, even not knowing assay conditions details,
> **only adds a moderate amount of noise** to the overall data." And: "Augmenting mixed public IC50 data
> by public Ki data does not deteriorate the quality of the mixed IC50 data, **if the Ki is corrected by
> an offset** … a Ki–IC50 conversion factor of **2** was found to be the most reasonable."

Landrum et al. 2024 (158 cites) measure the same phenomenon and are far more pessimistic — 65% of
repeated IC50 pairs differing by >0.3 log units, Kendall τ 0.51. **The two disagree, and I presented only
the pessimistic one because it supported the argument I was building.** That was selective.

**What the honest version looks like.** The ceiling is an *estimate with real uncertainty*, bounded below
by Landrum's pessimism and above by Kalliokoski's moderation. Report it as a range, state which source
gives which end, and note the corroboration from a completely different method — ABFEP's R² = 0.55
(§29) — which sits inside that range and is the strongest single reason to believe a ceiling near 0.55–0.6
exists at all.

**And an actionable item falls out:** apply Kalliokoski's **Ki→IC50 offset (factor of 2)** to our labels
before pooling, which neither we nor LP-PDBBind does. That is a one-line correction with a 311-citation
justification, and it is a cheaper first step than Landrum's full maximal curation.

## 46. AVE bias is computable on our splits, and it would let us make a positive claim

**Wallach & Heifets 2017** (JCIM, 212 cites), *"Most Ligand-Based Benchmarks Measure Overfitting Rather
than Accuracy"*:

> "the amount of **AVE bias strongly correlates with the performance** of ligand-based predictive methods
> irrespective of the predicted property, chemical fingerprint, similarity measure, or previously applied
> unbiasing techniques. Therefore, it may be the case that the previously reported performance of most
> ligand-based methods can be **explained by overfitting to benchmarks** rather than good prospective
> accuracy."

AVE (asymmetric validation embedding) is a *number you compute on a split*, from nearest-neighbour
similarity distributions between train and test. **LIT-PCBA** (Tran-Nguyen et al. 2020, 278 cites) was
constructed by AVE-unbiasing; **Davis et al. 2020** propose weighted refinements; **TocoDecoy** (Zhang
et al. 2022) attacks the same problem from decoy generation.

**This is the one item in Part VIII that works in our favour.** Everything else we have done about split
quality is a *negative* check — we removed leakage. AVE would let us state something *positive and
quantitative*: our LP split's AVE bias is X, compared to published values for DUD-E and LIT-PCBA. Given
that our Tanimoto 1-NN baseline fails badly (R² −0.92) and 64% of test ligands sit below Tanimoto 0.35 to
training, we have good reason to expect a low AVE — and if so, it is a defensible, citable statement that
our split is not the kind Wallach & Heifets found most benchmarks to be. Cheap, CPU-only, uses the
fingerprints already cached in `guidance/cheminformatics/cache/`.

Related and worth citing when discussing benchmark quality: **Chen et al. 2019** (PLoS ONE, 250 cites)
showed DUD-E's analogue and decoy bias explains CNN enrichment, and that deep models trained on PDBbind
were "**not superior to the performance of the docking program AutoDock Vina**" — the same comparison we
ran, with the same outcome at single-seed level. **Sieg et al. 2019** (JCIM, 279 cites) demonstrated that
"bias is learned implicitly and unnoticed from standard benchmarks" and give guidelines for bias-controlled
validation that our four filters largely satisfy.

One more, relevant to the pose-sensitivity tension in §37: **Vinardo** (Quiroga & Villa 2016, PLoS ONE,
342 cites) found that "the traditional approach to train empirical scoring functions, using **linear
regression to optimize the correlation of predicted and experimental binding affinities, does not result
in a function with optimal docking capabilities**". Scoring power and docking power are different
objectives — which is independent support for the idea that our affinity head can be pose-insensitive
while a pose-classification head on the same data would not be.

## 47. Final honest novelty position, after eight rounds

**Established prior art. Do not claim; cite and replicate.**
Ligand-only / bias-only baselines matching structure-based ML scoring functions (Volkov 2022; **Durant
2023 with ToolBoxSF**; Boyles 2021; Scantlebury 2023; Sieg 2019). Affinity models insensitive to target
identity (Bret 2026 on Boltz-2). Descriptors matching deep learning (van Tilborg 2022; Deng 2023).
Conditioning on interaction profiles instead of guiding (ShEPhERD-2; FLOWR.MULTI). UQ method comparison
(Rayka 2025). Conformal for affinity (Parks 2020). Benchmark bias and overfitting (Wallach 2017; Chen
2019; Sieg 2019; Kapoor 2023).

**What survives as ours, stated as narrowly as the evidence allows.**
1. **The factorial interface-vs-bulk ablation.** Three deletions of exactly 50% of protein atoms, equal in
   quantity and differing only in interface content, showing the response is to *amount* not *identity*
   (interface destroyed 1.058 pK vs intact 1.245 pK, overlapping CIs; pose destroyed only 0.278 pK).
   Attribution methods say which atoms matter; bias-only baselines say the model is not better than
   bias; neither separates amount from identity. Not found in eight rounds of searching.
2. **The class-prior decomposition.** Within-target y-randomisation puts the correct zero point for this
   task at R² 0.247, not 0 — 70% of the best protein-blind model's score needs no within-target SAR.
3. **The task-level power analysis.** Floor 0.3531, ceiling ~0.55–0.6, window ~0.2 R², narrower than one
   of our own confidence intervals — with independent corroboration from ABFEP (§29) and a physical
   mechanism (enthalpy–entropy compensation, §30).
4. **The link from the information audit to guidance-gradient failure**, which is the thesis's actual
   subject and which none of the scooping papers addresses.
5. **Ternary (1.58-bit) QAT on a 3D affinity GNN.**

**And the scoping that Parts VII–VIII force into the abstract, not the appendix:** all of it is on
**docked** poses, with mixed Kd/Ki/IC50 labels whose pooled noise is itself disputed in the literature.

## 48. The six cheap, high-value items this deep dive produced

In rough order of value per hour:

1. **Reframe Track C on heavy-atom count and PoseBusters validity**, demote LE to corroboration, cite
   Kenny/Polanski/Zhao. Text change only; removes a real vulnerability.
2. **Compute AVE bias on the LP split.** CPU-only, fingerprints already cached, yields a *positive*
   claim about split quality.
3. **Apply Kalliokoski's Ki→IC50 offset** before pooling labels. One line, 311-citation justification.
4. **Report the ceiling as a range** with both sources named, rather than a single number.
5. **Fill out Kapoor & Narayanan's model info sheet** plus an Artrith-style checklist as an appendix
   (§39).
6. **Run ToolBoxSF's bias-only baselines** on our data, so our replication uses the field's own tool and
   is directly comparable to Durant et al.'s five scoring functions.
