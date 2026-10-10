# Reporting checklist and model info sheet

Intended as a thesis appendix. Written for an examiner or reviewer who wants to verify this project's
methodology in a few minutes rather than infer it from scattered chapters.

Why it exists. Kapoor & Narayanan 2023 (*Patterns*, 1,201 cites) surveyed leakage across ML-based
science — 17 fields, 294 affected papers, an eight-type taxonomy — and found that in one case study
"when the errors are corrected, complex ML models do not perform substantively better than decades-old
logistic regression". They propose *model info sheets* as the remedy. Artrith et al. 2021 (*Nature
Chemistry*, 419 cites) give best-practice guidelines for ML in chemistry; Lu et al. 2022 (*JAMA Network
Open*) found that across 15 reporting guidelines, 220 unique items are collectively requested and
deployed models document a median of 39% of them — with the items most often missing being exactly those
about *reliability*: external validation, uncertainty measures, missing-data strategy.

This project satisfies an unusual number of those items. None of it was presented as a checklist, which
is how a reader checks rather than trusts. That is what this document fixes.

Every "yes" below names the script or file that implements it, so each line is falsifiable.

---

## 1. The task, stated without inflation

| item | answer |
|---|---|
| Prediction target | Protein–ligand binding affinity, pK, as a scalar regression |
| Label provenance | PDBbind affinities via CrossDocked2020; **mixed Kd 37% / Ki 26% / IC50 37%** |
| Input | 3D coordinates of a 10 Å pocket and a ligand pose |
| **Pose provenance** | **DOCKED, not crystallographic.** CrossDocked ligand files are `_lig_tt_min_*.sdf` / `_lig_tt_docked_*.sdf`, i.e. Vina-docked and minimised into non-cognate pockets. The structural input is itself a prediction. |
| Downstream use claimed | Gradient guidance of a frozen pocket-conditioned diffusion generator |
| Downstream use NOT claimed | Prospective hit identification; no molecule was synthesised or assayed |

## 2. Train/test separation — the eight things that could leak, and what was done

| # | potential leak | addressed? | how |
|---|---|---|---|
| 2.1 | No clean test set | yes | Test set untouched during all development; `--skip_test_logging` enforced during training, and A1b/A1e/A1g train with test logging disabled by default |
| 2.2 | Pre-processing fitted on train+test | yes | Scalers and ridge hyperparameters fitted on train, selected on val, applied to test once (`representation_ladder.fit_block`) |
| 2.3 | Hyperparameter / model selection on test | yes | Early stopping and alpha selection on validation loss only; `guidance/track_e/tests/test_no_test_selection.py` is a regression test for this |
| 2.4 | Duplicate records across splits | yes | **Four** independent filters: receptor PDB ID → ChEMBL-target/UniProt accession → sequence identity ≥90% (`leakage_audit.py`) → ligand InChIKey connectivity skeleton (`compound_overlap_audit.py`) |
| 2.5 | Illegitimate features (target proxies) | yes | Features are coordinates and atom types only; no assay, publication or date metadata reaches the model |
| 2.6 | Non-independence between train and test | yes | Split is **target-disjoint**: 779 train / 65 val / 127 test protein targets, no target shared. All inference resamples **target clusters**, not complexes (`stats_common.cluster_bootstrap_*`) |
| 2.7 | Temporal leakage | **not applicable / not controlled** | No date-based split. PDBbind deposition dates exist in LP-PDBBind and were not used. Stated as a limitation. |
| 2.8 | Test distribution not the one of interest | **partially** | Target-disjoint but **not** superfamily-disjoint (CORDIAL's standard) and not pocket-similarity-disjoint (EPoCS). 40.1% of test molecules sit on a generic Bemis-Murcko scaffold seen in training. Disclosed, not fixed. |

**Quantified residual redundancy.** `ave_bias.py` computes the Wallach & Heifets 2017 AVE statistic:
our split **+0.046** (mean over four pK cuts, CI excludes zero) versus **+1.044** for a random ligand
split of the same molecules — a 23-fold reduction, with mean nearest-neighbour similarity to own class
0.29–0.31 versus 0.934. Mild residual redundancy, not zero, and reported as such.

## 3. Uncertainty — reported, not assumed

| item | answer |
|---|---|
| Uncertainty method | Split-conformal prediction (`conformal_prediction.py`) |
| Methods tried and **rejected** | MC dropout, deep ensemble, heteroscedastic Gaussian NLL, deep evidential regression, recalibrated evidential — all five fail the three pre-registered tests |
| Pre-registration | Analysis plan fixed in A1's docstring before results; reused verbatim for A1b–A1g rather than re-derived per method |
| Multiplicity control | Benjamini–Hochberg over every family of tests actually run (40 paired model comparisons; 4 novelty tiers; 20 representation pairs) |
| Known failure disclosed | A1f (evidential) reached p = 0.042 at one seed and **did not replicate** across three; between-seed sd is 0.0377 R² |
| Loss variant caveat | A1e used vanilla Gaussian NLL, which Seitzer et al. 2022 document as pathological; β-NLL is queued as A1e-β |

## 4. Replication and seeds

| item | answer |
|---|---|
| Seeds per configuration | 3 (2021/2022/2023) for A1c, A1g and the surrogate arms; **1** for Stage 0 and original A1e — both disclosed |
| Between-seed variability reported | Yes, as sd: 0.0377 R² for the same architecture |
| Known small-n error | At n = 1 this project reported "QAT matches FP32"; at n = 2 "a systematic ~0.02 cost"; both were refuted at n = 3. Recorded rather than quietly fixed. |
| Aggregation | `guidance/aggregate_results.py` refuses an sd at n = 1 (reports `undefined`, never 0.0000) and prints a minimum-detectable-difference for every comparison |

## 5. Baselines — including the ones that beat us

| baseline | test R² | why included |
|---|---|---|
| Global mean | −0.0000 | floor |
| Ligand heavy-atom count, one feature | 0.3069 | size shortcut |
| Vina (physics, untrained) | 0.2601 | non-ML reference, and a confound detector |
| 14 RDKit descriptors, ridge, **no protein** | **0.3531** | the control that matters |
| 38 pocket descriptors, **no ligand** | 0.2889 | the other marginal |
| ECIF contact counts (raw) | 0.3701 | revealed as ligand size in disguise (q = 0.013 vs size-normalised) |
| Tanimoto 1-NN lookup | −0.9243 | memorisation probe |
| EGNN Stage 0 / ensemble | 0.3415 / 0.4072 | the model under study |

Paired, target-clustered, BH-corrected over 40 pairs: **0 pairings where a ligand-only model loses**, and
structure vs the 14-descriptor ridge is a tie at every seed and as an ensemble — unchanged after label
harmonisation (0/5 significant, closest p = 0.075).

## 6. Chance-correlation and shortcut controls

| control | result |
|---|---|
| y-randomisation, global (Rücker et al. 2007) | R² +0.0037 (max +0.0141) — the real 0.3531 **passes** by +0.3390 |
| y-randomisation, **within target** | R² **+0.2468** (sd 0.0019) — 70% of ligand-only performance needs no within-target SAR |
| Untrained-baseline confound detector | Vina included in every subset-comparison table; it caught a range-restriction artefact that had produced the **opposite** conclusion in the scaffold audit |
| Pose-sensitivity (factorial) | Deleting the ligand's real contacts costs *less* (1.058 pK) than deleting atoms that touch nothing (1.245 pK); 8 Å displacement costs 0.278 pK |

## 7. What the numbers are measured against

| item | answer |
|---|---|
| Label-noise ceiling | **R² ≈ 0.55–0.65**, point estimate 0.578 (Pearson 0.76 repeat-measurement agreement). Reported as a **range**: Landrum et al. 2024 pessimistic, Kalliokoski et al. 2013 milder |
| Physics-side anchor | Absolute BFEP reaches R² 0.55 (Chen et al. 2023); FEP is itself limited by experimental reproducibility (Ross et al. 2023) |
| Performance as a fraction of achievable | Best model 0.4072 = **70.5%** of the point-estimate ceiling, RMSE 1.22× the noise floor |
| Dataset reference model | CrossDocked2020's own best model: RMSE 1.42 / Pearson 0.612 (Francoeur et al. 2020). Ours: 1.27–1.38 / 0.60–0.65 |
| **Resolvable window** | Floor 0.3531 → ceiling 0.5776 = **0.2245 R²**, against a mean 95% CI width of **0.3871**. The design cannot resolve position within the window; reported as an interval, never a ranking |

## 8. Reproducibility

| item | answer |
|---|---|
| Code | This repository; every number has a named script and a results JSON |
| Environment | `guidance/thesis/environment_actual.yml`, `requirements-lock.txt` |
| Random seeds | Fixed and recorded in checkpoints (`'seed'` key), read from the checkpoint rather than the directory name |
| Bootstrap seed | 20260925 for every cluster bootstrap in the project |
| Automated tests | 7 suites, 91 tests (`python guidance/run_tests.py`); pre-commit hook installed; FAIL and SKIP paths both verified |
| Data regeneratable | Yes; the five multi-megabyte BindingNet index CSVs are gitignored with the regeneration chain documented |
| Resumability | Every training run writes `last.pt` each validation epoch; the work queue records completed steps by name |

## 9. Limitations, consolidated

1. **Docked, not crystallographic, poses.** Boyles et al. 2021 measured that structure-based scoring degrades on docked poses while ligand features compensate — which predicts our headline result for a reason about the data rather than the architecture. Untested here on crystal poses.
2. **Equilibrium affinity may be the wrong objective.** Residence time / k_off tracks in vivo efficacy better (Wang et al. 2022; Bernetti et al. 2019; Liu et al. 2026).
3. **Enthalpy–entropy compensation** narrows the predictable range for physical, possibly evolutionary reasons (Jiménez et al. 2024; Olsson et al. 2011) — the low label variance is not purely a data artefact.
4. **Split is target-disjoint but not superfamily- or pocket-disjoint**; 40.1% scaffold overlap; 50–90% sequence-identity homologs deliberately retained and disclosed.
5. **Label mixture** of Kd/Ki/IC50 with composition shifting between splits; harmonised post hoc (`label_harmonisation.py`) but models were trained uncorrected.
6. **Ligand efficiency** is mathematically contested (Kenny 2018); conclusions reframed on heavy-atom count and PoseBusters validity.
7. **Scaling exponents of 0.17–0.26** for chemical models (Frey et al. 2023) mean data growth cannot close the gap.
8. **Bespoke evaluation harness**; CBGBench and MolScore are the field's comparability standards and were not used.
9. **No prospective validation.** Everything is in silico. The prospective successes in this literature (PocketFlow, ClickGen) all constrain generation explicitly rather than guiding with a learned scalar.

## 10. Claims this project does and does not make

**Does:** on a target-disjoint, four-way-leakage-filtered split of docked poses, a structure-based EGNN is
not statistically distinguishable from ridge regression on 14 ligand descriptors; the model responds to
bulk protein content rather than interface geometry; point-wise uncertainty fails while split-conformal
holds; ternary weights cost nothing measurable; and the window in which any of this could have been
resolved is narrower than our own confidence intervals.

**Does not:** that learned guidance fails in general; that affinity prediction is impossible; that our
model is state of the art or that it is uniquely bad; that ligand-only baselines matching structure-based
models is a new observation (it is established — Volkov 2022; Durant 2023; Boyles 2021; Scantlebury 2023;
Sieg 2019); or that any generated molecule is a drug candidate.
