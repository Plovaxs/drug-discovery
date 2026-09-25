# Track E — Literature Review (planning phase; reusable as thesis Chapter II material)

**Scope.** Prior work relevant to two targeted modifications of the affinity-scoring model that
supplies guidance signals to a frozen 3D diffusion generator (TargetDiff):

- **E1 — delta-learning:** regress the residual Δ = pK_exp − pK_Vina (pK_Vina = −E_Vina/1.364) instead of pK_exp, and
  reconstruct ŷ = pK_Vina + Δ̂.
- **E2 — multi-task PLIP supervision:** auxiliary heads predict PLIP-derived interaction labels
  (H-bond, hydrophobic, π-stacking, salt bridge) from a shared encoder.

**Verification convention.** Every entry carries one tag:

- **[PubMed-verified]** — authors/year/journal/volume/pages/DOI checked against PubMed records retrieved in this session.
- **[Consensus-listed]** — title, first author, year and venue name come from a Consensus search record; volume/pages/DOI
  and the full author list were **not** independently verified and must be completed from the publisher page before submission.
  Author lists are shown as "First A. et al." for these entries.
- **[PREPRINT]** — not peer-reviewed as listed; flagged wherever used as evidence. Four items first found as preprints (PLANET, LUNA, LP-PDBBind, and Track E's other candidate PIGNet2) were re-checked: **PLANET, LUNA and LP-PDBBind now have PubMed-verified journal versions and are cited as such; only PIGNet2 [23] remains a preprint** (no published version found).

**Verification status after the 2026-09-25 follow-up:** 33 of 45 entries are PubMed-verified. The 12 that remain Consensus-listed only and still need volume/pages/DOI/author lists completed by hand are: [10] Rodríguez-López, [11] Kendall, [12] Kirchdorfer, [13] GradNorm, [14] PCGrad, [15] RECON, [17] ForkMerge, [18] Malhotra (TMLR), [19] Allenspach, [23] PIGNet2 (arXiv), [43] Pereira da Silva, [45] Harren — none of them is PubMed-indexed under the searched titles (ML-conference and non-biomedical venues, or too recent).

arXiv identifiers for [11], [13], [14] and the conference venues for [13]–[15] come from the author's
general knowledge and were **not tool-verified**; confirm them on the publisher/arXiv page.

Claims in the "Relevance" notes are limited to what the abstract (or, for DeepICL, the full text) supports. Anything
beyond that is marked **(not verified in literature)**. 45 distinct sources in total.

---

## 1.1 Delta-learning and residual-correction scoring

**[1] Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A.** Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach. *J. Chem. Theory Comput.* **2015**, *11* (5), 2087–2096. DOI: 10.1021/acs.jctc.5b00099. [PubMed-verified]
Origin of Δ-ML: an ML model is trained to add a correction to a cheap approximate method so that it reproduces a high-level target; trained on 1–10% of 134k molecules it reproduces DFT-level enthalpies for the rest. The recipe is the template for E1. The abstract does not itself report a direct-vs-delta learning-curve comparison — that comparison lives in the paper's figures and **has not been verified here**. Caveat: the baseline there (semi-empirical QM) approximates the *same physical quantity* as the target; Vina vs experimental affinity is not the same kind of relation (our inference, not a literature claim).

**[2] Wang, C.; Zhang, Y.** Improving Scoring-Docking-Screening Powers of Protein-Ligand Scoring Functions Using Random Forest. *J. Comput. Chem.* **2017**, *38* (3), 169–177 (online 2016). DOI: 10.1002/jcc.24667. [PubMed-verified]
Introduces ΔVinaRF20: a random forest with 20 descriptors added to the AutoDock Vina score. The abstract states that direct random-forest scoring functions improve scoring power but under-perform traditional functions in docking and screening power, whereas the ΔRF parameterization is superior in all power tests of CASF-2013 and CASF-2007. This is the closest thing to delta-vs-direct evidence for protein–ligand scoring, but the "direct" arm is a different feature set on a random-forest, and CASF benchmarks overlap PDBbind training data (see [39], [40]).

**[3] Li, H.; Leung, K.-S.; Wong, M.-H.; Ballester, P. J.** Improving AutoDock Vina Using Random Forest: The Growing Accuracy of Binding Affinity Prediction by the Effective Exploitation of Larger Data Sets. *Mol. Inform.* **2015**, *34* (2–3), 115–126. DOI: 10.1002/minf.201400132. [PubMed-verified]
RF-Score-v3: machine-learning re-scoring of Vina poses; the abstract reports that gains grow with more training data whereas additive-form regressions do not improve. Whether the model regresses a residual or takes Vina terms as input features must be read from the full paper **(not verified here)**; we cite it as evidence that Vina-anchored ML rescoring improves on Vina, not as a delta-target study.

**[4] Yang, C.; Zhang, Y.** Delta Machine Learning to Improve Scoring-Ranking-Screening Performances of Protein-Ligand Scoring Functions. *J. Chem. Inf. Model.* **2022**, *62* (11), 2696–2712. DOI: 10.1021/acs.jcim.2c00485. [PubMed-verified]
ΔLin_F9XGB: XGBoost Δ-ML on the linear empirical baseline Lin_F9, reporting CASF-2016 scoring-power Pearson R of 0.853 / 0.839 / 0.813 for optimized, redocked and ensemble-docked poses, plus LIT-PCBA screening. Shows the delta paradigm remains state of the art for classical-baseline anchoring and works on *docked* poses, which is our regime. Note the baseline is Lin_F9, not Vina, and the evaluation is on the CASF core set (leakage caveat, [39]).

**[5] Trott, O.; Olson, A. J.** AutoDock Vina: Improving the Speed and Accuracy of Docking with a New Scoring Function, Efficient Optimization, and Multithreading. *J. Comput. Chem.* **2010**, *31* (2), 455–461. DOI: 10.1002/jcc.21334. [PubMed-verified]
Defines the empirical scoring function that supplies the delta baseline. Validated for pose accuracy on its training set; the abstract makes no claim about affinity-ranking accuracy across chemotypes, which is why a learned correction is plausible.

**[6] Eberhardt, J.; Santos-Martins, D.; Tillack, A. F.; Forli, S.** AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. *J. Chem. Inf. Model.* **2021**, *61* (8), 3891–3898. DOI: 10.1021/acs.jcim.1c00203. [PubMed-verified]
Reference for the Vina 1.2 series used to compute this project's scores (we use 1.2.6; the paper describes 1.2.0). Adds Python bindings and batch mode; relevant because scores must be reproducible when the delta target is defined.

**[7] Xu, M.; Shen, C.; Yang, J.; Wang, Q.; Huang, N.** Systematic Investigation of Docking Failures in Large-Scale Structure-Based Virtual Screening. *ACS Omega* **2022**, *7* (43), 39417–39428. DOI: 10.1021/acsomega.2c05826. [PubMed-verified]
On DUD-E, the Vina scoring function "showed a bias toward compounds with higher molecular weights" (abstract). This is the primary citation for the critique that a Vina-anchored target inherits size bias; it is a screening-enrichment study, so its relevance to CrossDocked affinity regression is by analogy. Our own data (Sec. 2.3 of the design) measure this directly.

**[8] Quiroga, R.; Villarreal, M. A.** Vinardo: A Scoring Function Based on Autodock Vina Improves Scoring, Docking, and Virtual Screening. *PLoS ONE* **2016**, *11* (5), e0155183. DOI: 10.1371/journal.pone.0155183. [PubMed-verified]
Re-parameterises Vina's terms. The abstract states that the traditional approach of fitting empirical scoring functions by linear regression to maximise correlation with experimental affinity "does not result in a function with optimal docking capabilities", and that Vinardo outperforms Vina in all tests reported. Supports the premise that the Vina baseline is systematically imperfect (hence correctable) and that optimising one power can cost another; the baseline choice defines Δ.

**[9] Kautish, A.; Sivakumar, A.; Prasad, V. K.** Quantum Kernel-Based Delta (Δ)-Learning for Correcting a Semiempirical Method in the Prediction of Reaction Barrier Heights. *J. Chem. Phys.* **2026**, *165* (3). DOI: 10.1063/5.0336022. [PubMed-verified]
Δ-learning applied to correct the semi-empirical PM7 method towards ωB97X-D3/def2-TZVP; the top models improve on the uncorrected PM7 baseline by ≈ 54% in MAE. Cited only as evidence that the paradigm is still actively used outside protein–ligand scoring; different physical setting (correcting an approximation of the *same* quantity).

**[10] Rodríguez-López, O. et al.** Efficient Machine Learning Barrier Height and Reaction Enthalpy Corrections for Low- and Midlevel Quantum Chemistry Methods. *Small Struct.* **2026**. [Consensus-listed; volume/DOI not verified; abstract not read]
Same status as [9]: corroborates general use of ML corrections on low-level baselines. No quantitative claim used.

**Section 1.1 synthesis and gaps.**
(i) Delta-learning for protein–ligand scoring is established ([2], [4]) but always with **tree ensembles on hand-crafted features**; we found **no** study of a delta target on a **3D graph neural network** trained on a **leakage-controlled** split (Consensus ×3 and PubMed queries; absence not proof).
(ii) Head-to-head evidence that Δ beats an absolute target *at matched backbone and split* is **not found in literature**; [2]'s comparison is confounded by features and model class. Track E therefore must run its own matched-backbone control (Sec. 2.4 arm A0).
(iii) Vina's size bias ([7]) is the main critique; our measurement shows Δ is size-neutral by construction of the physical conversion, but the reconstruction ŷ = pK_Vina + Δ̂ re-inherits Vina's size behavior and must be checked.

---

## 1.2 Multi-task learning, loss balancing, and negative transfer

**[11] Kendall, A.; Gal, Y.; Cipolla, R.** Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)* **2018**. arXiv:1705.07115. [Consensus-listed (CVPR 2018 venue as listed); pages/DOI not verified]
Learns per-task weights from homoscedastic uncertainty so that no manual weight search is needed. Candidate scheme for E2, but it assumes the tasks' noise levels are learnable and comparable — see [12].

**[12] Kirchdorfer, L. et al.** Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative. *Int. J. Comput. Vis.* **2025**. [Consensus-listed; volume/DOI not verified; only title/venue read]
Devoted to the behavior of uncertainty weighting and an analytical alternative (per title). Read the full paper before finalizing the weighting choice; used here to justify *not* relying on learned uncertainty weights as the sole scheme.

**[13] Chen, Z.; Badrinarayanan, V.; Lee, C.-Y.; Rabinovich, A.** GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *Proc. 35th Int. Conf. Machine Learning (ICML)* **2018**. arXiv:1711.02257. [Consensus lists this as an arXiv record (2017); the ICML 2018 venue is from the authors' knowledge and **not tool-verified**]
Balances tasks by equalizing gradient magnitudes at a shared layer, with a hyper-parameter α controlling training-rate asymmetry. Relevant because E2's auxiliary gradients could dominate or vanish relative to the delta loss.

**[14] Yu, T.; Kumar, S.; Gupta, A.; Levine, S.; Hausman, K.; Finn, C.** Gradient Surgery for Multi-Task Learning. *Adv. Neural Inf. Process. Syst. (NeurIPS)* **2020**, *33*. arXiv:2001.06782. [Consensus lists an arXiv record; NeurIPS 2020 venue **not tool-verified**]
PCGrad projects a task gradient onto the normal plane of any conflicting task gradient. A principled defence against E2 gradients that oppose the delta objective; adds a per-step cost (pairwise gradient projections) we must budget on a 4 GB card.

**[15] Shi, G. et al.** Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning. *ICLR* **2023** (arXiv). [Consensus lists an arXiv record; ICLR 2023 venue **not tool-verified**]
Argues conflicting gradients originate in shared layers and mitigates them at the root, complementing surgery-style methods like PCGrad. Cited as an alternative if E2 shows conflict.

**[16] Lin, B.; Jiang, W.; Ye, F.; Zhang, Y.; Chen, P.; Chen, Y.-C.; Liu, S.; Tsang, I. W.; Kwok, J. T.** Dual-Balancing for Multi-Task Learning. *Neural Netw.* **2025**, *195*, 108317. DOI: 10.1016/j.neunet.2025.108317. [PubMed-verified; Consensus listed 2023 — an earlier version]
DB-MTL balances tasks from both the loss side (logarithmic transformation of each task loss) and the gradient side (normalising task gradients to comparable magnitudes using the maximum gradient norm). Completes the loss-balancing family; not planned for use.

**[17] Jiang, J. et al.** ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning. *Adv. Neural Inf. Process. Syst. (NeurIPS)* **2023**, *36*. [Consensus-listed; NeurIPS 36 as listed; pages not verified]
Directly targets negative transfer when a main task is trained with auxiliary tasks — exactly the E2 setting. Motivates our staged design (E1 alone before E2) and the pre-registered requirement that E1+E2 beat E1-alone; a fork-and-merge fallback is available if E2 hurts.

**[18] Malhotra, A. et al.** Dropped Scheduled Task: Mitigating Negative Transfer in Multi-Task Learning Using Dynamic Task Dropping. *Trans. Mach. Learn. Res.* **2023**. [Consensus-listed; not verified further]
Dynamic dropping of tasks during training to reduce negative transfer. Supports an E2 schedule that anneals the auxiliary weight to zero late in training if it is found to hurt the main task.

**[19] Allenspach, S.; Hiss, J. A.; Schneider, G.** Neural Multi-Task Learning in Drug Design. *Nat. Mach. Intell.* **2024**, *6*, 124–137 (volume from Consensus/author knowledge; PubMed lookup was ambiguous — **verify pages/DOI**). [Consensus-listed]
Reviews when multi-task learning helps or hurts in drug design (title/venue verified by Consensus; content beyond that not read here). Cited for the general expectation that gains depend on task relatedness and data regime.

**[20] Dey, V.; Ning, X.** Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation. *J. Cheminform.* **2024**, *16*, 85. DOI: 10.1186/s13321-024-00880-7. [PubMed-verified]
Jointly trains pretrained molecular GNNs with auxiliary tasks and studies how to measure auxiliary-task relevance and to combine task gradients (gradient surgery, including a proposed rotation of conflicting gradients) to counter negative transfer, reporting up to 7.7% gain over fine-tuning. Directly relevant prior art for auxiliary supervision of a molecular GNN and for the negative-transfer risk in E2; it is a property-prediction, not affinity-scoring, setting.

**[21] Zhang, X.; Gao, H.; Wang, H.; Chen, Z.; Zhang, Z.; Chen, X.; Li, Y.; Qi, Y.; Wang, R.** PLANET: A Multi-Objective Graph Neural Network Model for Protein–Ligand Binding Affinity Prediction. *J. Chem. Inf. Model.* **2023** (online 15 June), *64* (7), 2205–2220. DOI: 10.1021/acs.jcim.3c00253. [PubMed-verified; abstract read, full text not accessible with the available tools; a bioRxiv preprint also exists — cite the journal version]
**Nearest neighbour to E2 among affinity models.** Per the abstract, PLANET is trained "through a multi-objective process with three related tasks": binding affinity, **protein–ligand contact map** and ligand distance matrix, with non-binder decoys added. So auxiliary *structural* supervision of an affinity network is established, and E2 is **not** the first to supervise an affinity model with interaction-like auxiliary targets. What the abstract does *not* describe: PLIP interaction *types* (H-bond/hydrophobic/salt bridge) as labels, a delta target relative to a docking score, or use in guiding a generator. The contact map is a distance-based structural label; E2's labels are rule-based chemical interaction classes on ligand atoms.

**[22] Moon, S.; Zhung, W.; Yang, S.; Lim, J.; Kim, W. Y.** PIGNet: A Physics-Informed Deep Learning Model toward Generalized Drug–Target Interaction Predictions. *Chem. Sci.* **2022**, *13* (13), 3661–3673. DOI: 10.1039/d1sc06946b. [PubMed-verified; Consensus's "2020" is the arXiv posting]
Predicts atom–atom pairwise interaction terms from physics-informed equations parameterised by a network and sums them to the affinity; augments training with a broader range of poses and ligands. A structured decomposition (a built-in multi-component inductive bias) rather than a multi-task loss; this is the "PIGNet2-informed" element of the E backbone.

**[23] Moon, S. et al.** PIGNet2: A Versatile Deep Learning-Based Protein–Ligand Interaction Prediction Model for Binding Affinity Scoring and Virtual Screening. *arXiv* **2023**. [Consensus-listed; **[PREPRINT]** as listed; published version not checked]
Successor to [22] aimed at both scoring and screening. Cited as the source of the physics-decomposed readout and decoy-augmentation ideas in the flagship architecture note.

**[24] Lai, H.; Wang, L.; Qian, R.; Huang, J.; Zhou, P.; Ye, G.; Wu, F.; Wu, F.; Zeng, X.; Liu, W.** Interformer: An Interaction-Aware Model for Protein–Ligand Docking and Affinity Prediction. *Nat. Commun.* **2024**, *15*, 10223. DOI: 10.1038/s41467-024-54440-6. [PubMed-verified]
Graph-transformer with an interaction-aware mixture density network plus a negative-sampling correction for affinity prediction; per the abstract, modelling specific interactions improves docking and generalization. It couples interaction modelling with affinity (adjacent to E2) but through a distance-likelihood energy rather than explicit interaction-class labels, and does not use a delta target.

**[25] Yang, Z.; Zhong, W.; Lv, Q.; Dong, T.; Yu-Chian Chen, C.** Geometric Interaction Graph Neural Network for Predicting Protein–Ligand Binding Affinities from 3D Structures (GIGN). *J. Phys. Chem. Lett.* **2023**, *14* (8), 2020–2033. DOI: 10.1021/acs.jpclett.2c03906. [PubMed-verified]
Heterogeneous interaction layer unifying covalent and non-covalent message passing with E(3)-invariance; reported state of the art on three external test sets. Primary backbone reference for Track E (the CIGConv/NIGConv layers documented in `FLAGSHIP_ARCHITECTURE_RESEARCH.md`).

**Section 1.2 synthesis and gaps.** Loss-balancing methods fall into three families: learned-uncertainty ([11], [12]), gradient-magnitude ([13], [16]) and gradient-conflict ([14], [15]). None was validated for a *residual* main task with *deterministic, geometry-derived* auxiliary labels **(not found in literature)**. Negative transfer is documented and has dedicated remedies ([17], [18]); it is the principal E2 risk, and the design responds with staging and a fixed-weight ablation rather than committing to an adaptive scheme up front.

---

## 1.3 PLIP and interaction-fingerprint supervision

**[26] Salentin, S.; Schreiber, S.; Haupt, V. J.; Adasme, M. F.; Schroeder, M.** PLIP: Fully Automated Protein–Ligand Interaction Profiler. *Nucleic Acids Res.* **2015**, *43* (W1), W443–W447. DOI: 10.1093/nar/gkv315. [PubMed-verified]
Rule-based detection of seven non-covalent interaction types at single-atom level; accepts custom complexes (e.g. from docking) and does not require structure preparation; command-line mode is intended for high-throughput profiling. Primary label source for E2.

**[27] Adasme, M. F.; Linnemann, K. L.; Bolz, S. N.; Kaiser, F.; Salentin, S.; Haupt, V. J.; Schroeder, M.** PLIP 2021: Expanding the Scope of the Protein–Ligand Interaction Profiler to DNA and RNA. *Nucleic Acids Res.* **2021**, *49* (W1), W530–W534. DOI: 10.1093/nar/gkab294. [PubMed-verified]
Maintenance release extending PLIP to nucleic acids; states the engine had served over a million queries. Cited for provenance/version stability of the tool; irrelevant to protein-only pockets beyond that.

**[28] Schake, P.; Bolz, S. N.; Linnemann, K.; Schroeder, M.** PLIP 2025: Introducing Protein–Protein Interactions to the Protein–Ligand Interaction Profiler. *Nucleic Acids Res.* **2025**, *53* (W1), W463–W465. DOI: 10.1093/nar/gkaf361. [PubMed-verified]
Current release; states PLIP detects eight non-covalent interaction types. The version we run should be pinned and reported in the thesis.

**[29] Zhung, W.; Kim, H.; Kim, W. Y.** 3D Molecular Generative Framework for Interaction-Guided Drug Design (DeepICL). *Nat. Commun.* **2024**, *15*, 2688. DOI: 10.1038/s41467-024-47011-2. [PubMed-verified; full text read]
**The key precedent for E2's label choice.** DeepICL uses exactly four interaction types — hydrogen bonds, salt bridges, hydrophobic interactions and π–stackings — and, during training, extracts interaction conditions from the reference structures with PLIP (full-text, Results/Methods), on 11,284 training and 2,109 validation PDBbind-2020 complexes. Interactions are used as **generation-time conditioning**, not as auxiliary supervision targets of an affinity regressor, and the model is an autoregressive VAE, not a diffusion model. The paper reports that interaction-conditioned generation gave lower (better) SMINA scores than the baseline (mean −7.67 vs −6.52 kcal/mol), a docking-score result of the kind this thesis has learned to treat cautiously.

**[30] Zhang, J.; Chen, H.** Molecule Design Using Molecular Generative Models Constrained by Ligand–Protein Interactions. *J. Chem. Inf. Model.* **2022**, *62* (14), 3291–3306. DOI: 10.1021/acs.jcim.2c00177. [PubMed-verified]
An RNN generator coupled with a ligand–protein interaction fingerprint constructed on ligand *docking poses* as a constraint; constrained models tend to generate compounds with similar binding modes. Second precedent for interaction information entering a generative pipeline (conditioning, not auxiliary supervision of an affinity model), and for computing interaction fingerprints on docked poses.

**[31] Huang, Z.; Yang, L.; Qin, C.; Xing, Y.; Yu, H.; Zhou, X.; Zheng, B.; Wang, Y.; Gao, X.; Yang, W.** Disentangled Diffusion Model for 3D Molecular Generation with Protein–Ligand Interaction Priors (DPDiff). *Bioinformatics* **2026**, *42* (6), btag165. DOI: 10.1093/bioinformatics/btag165. [PubMed-verified; **full text read**]
A 3D diffusion SBDD model on CrossDocked2020 that injects "interaction priors" extracted **during sampling from intermediate diffusion predictions**. The priors are latent features of two pretrained **binding-affinity prediction networks** (a geometry-based G-net and a sequence-based S-net), not PLIP labels; there is no delta target and no multi-task interaction supervision. **Key empirical finding:** in its ablation, adding the geometry-based prior alone *degrades* generation (Vina Score −3.58 vs −5.04 baseline) because it was pretrained on accurate poses and receives noisy intermediate structures at sampling time — an independent report of the train/inference mismatch this project documents for its own guidance models, and the reason C4 (gradient diagnostic on noisy states) is in the design. It also cites IPDiff (an earlier affinity-network-prior diffusion model), which is **not** in this review and should be read before the novelty claim is written.

**[32] Errington, D.; Schneider, C.; Bouysset, C.; Dreyer, F. A.** Assessing Interaction Recovery of Predicted Protein–Ligand Poses. *J. Cheminform.* **2025**, *17*, 76. DOI: 10.1186/s13321-025-01011-6. [PubMed-verified]
Shows that ignoring interaction fingerprints can overestimate pose-model performance and that recent co-folding models often fail to recapitulate key interactions. Direct evidence that interaction profiles of *predicted* poses can differ from those of experimental poses, i.e. PLIP-style labels are **pose-quality dependent**; motivates our jitter-stability check.

**[33] Bouysset, C.; Fiorucci, S.** ProLIF: A Library to Encode Molecular Interactions as Fingerprints. *J. Cheminform.* **2021**, *13*, 72. DOI: 10.1186/s13321-021-00548-6. [PubMed-verified]
Python library producing interaction fingerprints for complexes from MD, experimental structures and docking, with re-parameterizable interaction definitions. The main alternative label source; PLIP is preferred here for continuity with [29].

**[34] Wójcikowski, M.; Kukiełka, M.; Stepniewska-Dziubinska, M. M.; Siedlecki, P.** Development of a Protein–Ligand Extended Connectivity (PLEC) Fingerprint and Its Application for Binding Affinity Predictions. *Bioinformatics* **2019**, *35* (8), 1334–1341. DOI: 10.1093/bioinformatics/bty757. [PubMed-verified]
PLEC fingerprints implicitly encode interactions by pairing ECFP environments of ligand and protein; even a linear model reaches R_p = 0.817 on the PDBbind v2016 core set. Evidence that interaction-encoding features carry affinity signal — as *input* features, on a benchmark subject to the leakage concerns in Sec. 1.4.

**[35] Desaphy, J.; Raimbaud, E.; Ducrot, P.; Rognan, D.** Encoding Protein–Ligand Interaction Patterns in Fingerprints and Graphs. *J. Chem. Inf. Model.* **2013**, *53* (3), 623–637. DOI: 10.1021/ci300566n. [PubMed-verified]
Triplet interaction fingerprint (TIFP) from hydrophobic, aromatic, H-bond, ionic and metal interactions detected on the fly (≈ 10,000 complexes). An early, PLIP-independent instance of the same interaction taxonomy.

**[36] Fassio, A. V.; Shub, L.; Ponzoni, L.; McKinley, J.; O'Meara, M. J.; Ferreira, R. S.; Keiser, M. J.; de Melo Minardi, R. C.** Prioritizing Virtual Screening with Interpretable Interaction Fingerprints (LUNA). *J. Chem. Inf. Model.* **2022**, *62* (18), 4300–4318. DOI: 10.1021/acs.jcim.2c00695. [PubMed-verified; an earlier bioRxiv preprint has since been published]
Trains ML models to reproduce DOCK3.7 scores from interaction fingerprints on 1 million docked complexes (EIFP-4096, R² = 0.61). Precedent for computing interaction descriptors **on docked poses at large scale**; note the labels there were docking scores, not interactions.

**[37] Wang, D. D.; Xie, H.; Yan, H.** Proteo-Chemometrics Interaction Fingerprints of Protein–Ligand Complexes Predict Binding Affinity. *Bioinformatics* **2021**, *37* (17), 2570–2579. DOI: 10.1093/bioinformatics/btab132. [PubMed-verified]
Interaction fingerprints combined with ML outperform several scoring models on PDBbind v2019 core and CSAR-HiQ. Again interaction features as *inputs*; core-set benchmark caveat.

**Section 1.3 synthesis and gaps.**
(i) PLIP has been used **as a label generator at scale for a generative model** ([29]) — this is the direct precedent for E2's label extraction and its four-type taxonomy.
(ii) We found **no** study using PLIP-derived interaction labels as **auxiliary supervised targets of an affinity regressor** on a leakage-controlled split, and none on **diffusion-generated** poses **(not found in literature)**. The closest affinity model we located is PLANET ([21], peer-reviewed; abstract read), which supervises a **protein–ligand contact map** and a ligand distance matrix alongside affinity — distance-based structural targets, not PLIP interaction types — so interaction-*like* auxiliary supervision of an affinity network is **not novel**; only the PLIP-typed, ligand-atom-level form is not found.
(iii) Reliability on non-crystal poses is **not** settled: [32] documents interaction-recovery gaps for predicted poses. No source characterises PLIP label stability under coordinate noise, so we measured it (design Sec. 2.3).

---

## 1.4 Critiques, limits, and the small-GPU leakage-controlled regime

**[38] Volkov, M.; Turk, J.-A.; Drizard, N.; Martin, N.; Hoffmann, B.; Gaston-Mathé, Y.; Rognan, D.** On the Frustration to Predict Binding Affinities from Protein–Ligand Structures with Deep Neural Networks. *J. Med. Chem.* **2022**, *65* (11), 7946–7958. DOI: 10.1021/acs.jmedchem.2c00487. [PubMed-verified]
GNNs given an explicit description of non-covalent interactions show **no advantage over ligand or protein descriptors alone**, and nearest-neighbour baselines already perform well — "memorization largely dominates true learning". The strongest critique of E2's premise: supplying interaction labels may not make the encoder learn transferable interactions. Also a warning for E1: any gain must be tested against ligand-only and Vina-only baselines.

**[39] Graber, D.; Stockinger, P.; Meyer, F.; Mishra, S.; Horn, C.; Buller, R.** Resolving Data Bias Improves Generalization in Binding Affinity Prediction. *Nat. Mach. Intell.* **2025**, *7* (10), 1713–1725. DOI: 10.1038/s42256-025-01124-5. [PubMed-verified]
PDBbind CleanSplit: benchmark performance of top models "drops substantially" when leakage is removed, so much reported performance is leakage-driven; a sparse-graph model with language-model transfer keeps its performance. Justifies our leakage-safe target-level split and the requirement that every claim be on unseen targets.

**[40] Li, J.; Guan, X.; Zhang, O.; Sun, K.; Wang, Y.; Bagni, D.; Head-Gordon, T.** Leak Proof PDBBind: A Reorganized Data Set of Protein–Ligand Complexes for More Generalizable Binding Affinity Prediction. *J. Phys. Chem. B* **2026**, *130* (2), 730–740. DOI: 10.1021/acs.jpcb.5c08598. [PubMed-verified; the arXiv version (arXiv:2308.09639, a **preprint**) preceded it — cite the journal version]
Reorganises PDBbind into leak-controlled train/val/test sets (high sequence and structural similarity removed) and reports that models retrained on it (Vina, RF-Score, IGN, DeepDTA) perform better on a new independent set (BDB2020+); IGN is recommended for scoring/ranking of new systems. Our `guidance/lp_split` is an LP-PDBBind-*style* target-level adaptation to CrossDocked2020, not this dataset.

**[41] Balcı, M. A.; Çetin, E.; Calibasi-Kocal, G.; Akgüller, Ö.** Pocket-Surface Discrete Differential Geometry as a Leakage-Robust Feature Class for Protein–Ligand Binding Affinity Prediction. *Molecules* **2026**, *31* (11). DOI: 10.3390/molecules31111899. [PubMed-verified]
Reports that, on stricter splits and LP-PDBBind DataSAIL-S2, four strategies for injecting the extra geometric descriptor into a SchNet-style GNN produced **no statistically significant lift** — a documented null for adding hand-crafted information to a GNN, relevant as a prior on E2.

**[42] Scantlebury, J.; Vost, L.; Carbery, A.; Hadfield, T. E.; Turnbull, O. M.; Brown, N.; Chenthamarakshan, V.; Das, P.; Grosjean, H.; von Delft, F.; Deane, C. M.** A Small Step Toward Generalizability: Training a Machine Learning Scoring Function for Structure-Based Virtual Screening. *J. Chem. Inf. Model.* **2023**, *63* (10), 2960–2974. DOI: 10.1021/acs.jcim.3c00322. [PubMed-verified]
States that many scoring functions "make predictions based on data set biases rather than an understanding of the physics of binding", filters train/test to reduce bias, and shows attribution on the resulting model correlates with a distance-based interaction profiler. A validation *method* for E2: check whether E2 changes attribution toward profiled interactions.

**[43] Pereira da Silva, M. M. et al.** Data-Centric Training Enables Meaningful Interaction Learning in Protein–Ligand Binding Affinity Prediction. *J. Cheminform.* **2026**. [Consensus-listed; volume/DOI not verified]
Argues that under stringent OOD splits (Pfam-CV) the generalization gap reflects insufficient coverage of diverse binding profiles in the data rather than model capacity. Consistent with this project's own finding that architecture changes were not the lever, and a caution that E1/E2 may not close a data-coverage gap.

**[44] Shen, C.; Zhang, X.; Hsieh, C.-Y.; Deng, Y.; Wang, D.; Xu, L.; Wu, J.; Li, D.; Kang, Y.; Hou, T.; Pan, P.** A Generalized Protein–Ligand Scoring Framework with Balanced Scoring, Docking, Ranking and Screening Powers. *Chem. Sci.* **2023**, *14* (30), 8129–8146. DOI: 10.1039/d3sc02044d. [PubMed-verified; Consensus attributed it to the senior author, T. Hou]
Notes that most ML scoring functions serve a single task and that balance across scoring/docking/screening is a grand challenge; adds an adjustable binding-affinity term to a mixture-density-network training. Relevant to baseline dependence: optimising one power (here, affinity regression on a Vina-anchored target) can cost another (here, ranking of *generated* molecules).

**[45] Harren, T. et al.** Modern Machine-Learning for Binding Affinity Estimation of Protein–Ligand Complexes: Progress, Opportunities, and Challenges. *WIREs Comput. Mol. Sci.* **2024**. [Consensus-listed; volume/DOI not verified]
Review of real-world applicability and pitfalls of ML scoring. Used as general support for the standard of reporting we adopt.

**Section 1.4 synthesis.**
- **Baseline dependence:** a delta model can be no better than its anchor allows where the anchor's error is not learnable from the input; Vina is size-biased ([7]). E1 must beat (a) Vina alone (raw and linearly calibrated) and (b) a heavy-atom-only baseline, not just an absolute-target GNN.
- **Negative transfer:** documented ([17], [18]); addressed by staging.
- **Shortcut/memorization risk for E2:** [38], [41] both find that adding interaction-oriented information to a GNN does not automatically improve generalization.
- **4 GB single-GPU, leakage-safe CrossDocked regime:** **not found in literature.** The cited leak-controlled sets ([39], [40]) are PDBbind-based, and none of the cited works reports training on a 4 GB card; the compute assumptions in the design come from this project's own timing logs (Sec. 2.6), not from a source.

---

## 1.5 Prior work combining delta-learning AND multi-task interaction supervision

**Result: none found.** Searches run (all in this session): two targeted Consensus queries combining delta learning, multi-task/auxiliary supervision, interaction labels and docking-score correction (returned only [4], [24], [44]); a PubMed boolean query for (delta learning OR Δ-ML) AND (multi-task OR auxiliary) AND protein–ligand AND (interaction OR PLIP) AND binding affinity (**0 records**); plus the topic-specific searches behind Sec. 1.1–1.4.

**Limits of this finding.** Consensus returns three records per query and PubMed indexing lags preprints; "none found" therefore means *not found by these searches*, not *does not exist*. **Status of the follow-up (2026-09-25):** DPDiff [31] was read in full (affinity-network latent priors on the generator side; no delta target, no PLIP labels). PLANET [21] was read at abstract level only (published in *J. Chem. Inf. Model.*; full text is not accessible with the available tools) and must be read in full by the user. IPDiff (cited by [31]) has not been read. A manual Google Scholar / Semantic Scholar search for combinations of "Δ-learning" or "residual to Vina" with "auxiliary interaction" or "PLIP" **still has to be done by the user** (it cannot be done with the available tools) before any novelty claim is written.

**Nearest neighbours (none is the combination):** delta-learning without auxiliary interaction supervision ([2], [4]); interaction-aware affinity modelling without a delta target ([24]); PLIP labels used for generation-time conditioning ([29], [30], [31]); multi-objective affinity models with structural auxiliary targets ([21], contact map + distance matrix) and auxiliary-learning frameworks with gradient surgery for molecular GNNs ([20]); and generator-side interaction priors from affinity networks ([31]).

---

## Consolidated reference list (paste-ready, alphabetical by first author)

Numbering matches the entries above. Entries marked † are the 12 that are still Consensus-listed only and need volume/pages/DOI and full author lists completed from the publisher before submission; entries marked ‡ are preprints (only PIGNet2 remains one).

- Adasme, M. F.; Linnemann, K. L.; Bolz, S. N.; Kaiser, F.; Salentin, S.; Haupt, V. J.; Schroeder, M. PLIP 2021: Expanding the Scope of the Protein–Ligand Interaction Profiler to DNA and RNA. *Nucleic Acids Res.* **2021**, *49*, W530–W534. https://doi.org/10.1093/nar/gkab294 [27]
- † Allenspach, S.; et al. Neural Multi-Task Learning in Drug Design. *Nat. Mach. Intell.* **2024**, *6*. [19]
- Balcı, M. A.; Çetin, E.; Calibasi-Kocal, G.; Akgüller, Ö. Pocket-Surface Discrete Differential Geometry as a Leakage-Robust Feature Class for Protein–Ligand Binding Affinity Prediction. *Molecules* **2026**, *31* (11). https://doi.org/10.3390/molecules31111899 [41]
- Bouysset, C.; Fiorucci, S. ProLIF: A Library to Encode Molecular Interactions as Fingerprints. *J. Cheminform.* **2021**, *13*, 72. https://doi.org/10.1186/s13321-021-00548-6 [33]
- Chen, Z.; Badrinarayanan, V.; Lee, C.-Y.; Rabinovich, A. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML* **2018**; arXiv:1711.02257. [13]
- Desaphy, J.; Raimbaud, E.; Ducrot, P.; Rognan, D. Encoding Protein–Ligand Interaction Patterns in Fingerprints and Graphs. *J. Chem. Inf. Model.* **2013**, *53*, 623–637. https://doi.org/10.1021/ci300566n [35]
- Dey, V.; Ning, X. Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation. *J. Cheminform.* **2024**, *16*, 85. https://doi.org/10.1186/s13321-024-00880-7 [20]
- Eberhardt, J.; Santos-Martins, D.; Tillack, A. F.; Forli, S. AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. *J. Chem. Inf. Model.* **2021**, *61*, 3891–3898. https://doi.org/10.1021/acs.jcim.1c00203 [6]
- Errington, D.; Schneider, C.; Bouysset, C.; Dreyer, F. A. Assessing Interaction Recovery of Predicted Protein–Ligand Poses. *J. Cheminform.* **2025**, *17*, 76. https://doi.org/10.1186/s13321-025-01011-6 [32]
- Fassio, A. V.; Shub, L.; Ponzoni, L.; McKinley, J.; O'Meara, M. J.; Ferreira, R. S.; Keiser, M. J.; de Melo Minardi, R. C. Prioritizing Virtual Screening with Interpretable Interaction Fingerprints. *J. Chem. Inf. Model.* **2022**, *62*, 4300–4318. https://doi.org/10.1021/acs.jcim.2c00695 [36]
- Graber, D.; Stockinger, P.; Meyer, F.; Mishra, S.; Horn, C.; Buller, R. Resolving Data Bias Improves Generalization in Binding Affinity Prediction. *Nat. Mach. Intell.* **2025**, *7*, 1713–1725. https://doi.org/10.1038/s42256-025-01124-5 [39]
- † Harren, T.; et al. Modern Machine-Learning for Binding Affinity Estimation of Protein–Ligand Complexes: Progress, Opportunities, and Challenges. *WIREs Comput. Mol. Sci.* **2024**. [45]
- Shen, C.; Zhang, X.; Hsieh, C.-Y.; Deng, Y.; Wang, D.; Xu, L.; Wu, J.; Li, D.; Kang, Y.; Hou, T.; Pan, P. A Generalized Protein–Ligand Scoring Framework with Balanced Scoring, Docking, Ranking and Screening Powers. *Chem. Sci.* **2023**, *14*, 8129–8146. https://doi.org/10.1039/d3sc02044d [44]
- Huang, Z.; Yang, L.; Qin, C.; Xing, Y.; Yu, H.; Zhou, X.; Zheng, B.; Wang, Y.; Gao, X.; Yang, W. Disentangled Diffusion Model for 3D Molecular Generation with Protein–Ligand Interaction Priors. *Bioinformatics* **2026**, *42* (6), btag165. https://doi.org/10.1093/bioinformatics/btag165 [31]
- † Jiang, J.; et al. ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning. *Adv. Neural Inf. Process. Syst.* **2023**, *36*. [17]
- Kautish, A.; Sivakumar, A.; Prasad, V. K. Quantum Kernel-Based Delta (Δ)-Learning for Correcting a Semiempirical Method in the Prediction of Reaction Barrier Heights. *J. Chem. Phys.* **2026**, *165* (3). https://doi.org/10.1063/5.0336022 [9]
- † Kendall, A.; Gal, Y.; Cipolla, R. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR* **2018**; arXiv:1705.07115. [11]
- † Kirchdorfer, L.; et al. Investigating Uncertainty Weighting for Multi-Task Learning: Insights and Analytical Alternative. *Int. J. Comput. Vis.* **2025**. [12]
- Lai, H.; Wang, L.; Qian, R.; Huang, J.; Zhou, P.; Ye, G.; Wu, F.; Wu, F.; Zeng, X.; Liu, W. Interformer: An Interaction-Aware Model for Protein–Ligand Docking and Affinity Prediction. *Nat. Commun.* **2024**, *15*, 10223. https://doi.org/10.1038/s41467-024-54440-6 [24]
- Li, J.; Guan, X.; Zhang, O.; Sun, K.; Wang, Y.; Bagni, D.; Head-Gordon, T. Leak Proof PDBBind: A Reorganized Data Set of Protein–Ligand Complexes for More Generalizable Binding Affinity Prediction. *J. Phys. Chem. B* **2026**, *130*, 730–740. https://doi.org/10.1021/acs.jpcb.5c08598 (preprint: arXiv:2308.09639) [40]
- Li, H.; Leung, K.-S.; Wong, M.-H.; Ballester, P. J. Improving AutoDock Vina Using Random Forest: The Growing Accuracy of Binding Affinity Prediction by the Effective Exploitation of Larger Data Sets. *Mol. Inform.* **2015**, *34*, 115–126. https://doi.org/10.1002/minf.201400132 [3]
- Lin, B.; Jiang, W.; Ye, F.; Zhang, Y.; Chen, P.; Chen, Y.-C.; Liu, S.; Tsang, I. W.; Kwok, J. T. Dual-Balancing for Multi-Task Learning. *Neural Netw.* **2025**, *195*, 108317. https://doi.org/10.1016/j.neunet.2025.108317 [16]
- † Malhotra, A.; et al. Dropped Scheduled Task: Mitigating Negative Transfer in Multi-Task Learning Using Dynamic Task Dropping. *Trans. Mach. Learn. Res.* **2023**. [18]
- Moon, S.; Zhung, W.; Yang, S.; Lim, J.; Kim, W. Y. PIGNet: A Physics-Informed Deep Learning Model toward Generalized Drug–Target Interaction Predictions. *Chem. Sci.* **2022**, *13*, 3661–3673. https://doi.org/10.1039/d1sc06946b [22]
- † ‡ Moon, S.; et al. PIGNet2: A Versatile Deep Learning-Based Protein–Ligand Interaction Prediction Model for Binding Affinity Scoring and Virtual Screening. arXiv **2023**. [23]
- † Pereira da Silva, M. M.; et al. Data-Centric Training Enables Meaningful Interaction Learning in Protein–Ligand Binding Affinity Prediction. *J. Cheminform.* **2026**. [43]
- Quiroga, R.; Villarreal, M. A. Vinardo: A Scoring Function Based on Autodock Vina Improves Scoring, Docking, and Virtual Screening. *PLoS ONE* **2016**, *11* (5), e0155183. https://doi.org/10.1371/journal.pone.0155183 [8]
- Ramakrishnan, R.; Dral, P. O.; Rupp, M.; von Lilienfeld, O. A. Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach. *J. Chem. Theory Comput.* **2015**, *11*, 2087–2096. https://doi.org/10.1021/acs.jctc.5b00099 [1]
- † Rodríguez-López, O.; et al. Efficient Machine Learning Barrier Height and Reaction Enthalpy Corrections for Low- and Midlevel Quantum Chemistry Methods. *Small Struct.* **2026**. [10]
- Salentin, S.; Schreiber, S.; Haupt, V. J.; Adasme, M. F.; Schroeder, M. PLIP: Fully Automated Protein–Ligand Interaction Profiler. *Nucleic Acids Res.* **2015**, *43*, W443–W447. https://doi.org/10.1093/nar/gkv315 [26]
- Scantlebury, J.; Vost, L.; Carbery, A.; Hadfield, T. E.; Turnbull, O. M.; Brown, N.; Chenthamarakshan, V.; Das, P.; Grosjean, H.; von Delft, F.; Deane, C. M. A Small Step Toward Generalizability: Training a Machine Learning Scoring Function for Structure-Based Virtual Screening. *J. Chem. Inf. Model.* **2023**, *63*, 2960–2974. https://doi.org/10.1021/acs.jcim.3c00322 [42]
- Schake, P.; Bolz, S. N.; Linnemann, K.; Schroeder, M. PLIP 2025: Introducing Protein–Protein Interactions to the Protein–Ligand Interaction Profiler. *Nucleic Acids Res.* **2025**, *53*, W463–W465. https://doi.org/10.1093/nar/gkaf361 [28]
- † Shi, G.; et al. Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning. *ICLR* **2023**. [15]
- Trott, O.; Olson, A. J. AutoDock Vina: Improving the Speed and Accuracy of Docking with a New Scoring Function, Efficient Optimization, and Multithreading. *J. Comput. Chem.* **2010**, *31*, 455–461. https://doi.org/10.1002/jcc.21334 [5]
- Volkov, M.; Turk, J.-A.; Drizard, N.; Martin, N.; Hoffmann, B.; Gaston-Mathé, Y.; Rognan, D. On the Frustration to Predict Binding Affinities from Protein–Ligand Structures with Deep Neural Networks. *J. Med. Chem.* **2022**, *65*, 7946–7958. https://doi.org/10.1021/acs.jmedchem.2c00487 [38]
- Wang, D. D.; Xie, H.; Yan, H. Proteo-Chemometrics Interaction Fingerprints of Protein–Ligand Complexes Predict Binding Affinity. *Bioinformatics* **2021**, *37*, 2570–2579. https://doi.org/10.1093/bioinformatics/btab132 [37]
- Wang, C.; Zhang, Y. Improving Scoring-Docking-Screening Powers of Protein-Ligand Scoring Functions Using Random Forest. *J. Comput. Chem.* **2017**, *38*, 169–177. https://doi.org/10.1002/jcc.24667 [2]
- Wójcikowski, M.; Kukiełka, M.; Stepniewska-Dziubinska, M. M.; Siedlecki, P. Development of a Protein–Ligand Extended Connectivity (PLEC) Fingerprint and Its Application for Binding Affinity Predictions. *Bioinformatics* **2019**, *35*, 1334–1341. https://doi.org/10.1093/bioinformatics/bty757 [34]
- Xu, M.; Shen, C.; Yang, J.; Wang, Q.; Huang, N. Systematic Investigation of Docking Failures in Large-Scale Structure-Based Virtual Screening. *ACS Omega* **2022**, *7*, 39417–39428. https://doi.org/10.1021/acsomega.2c05826 [7]
- Yang, C.; Zhang, Y. Delta Machine Learning to Improve Scoring-Ranking-Screening Performances of Protein-Ligand Scoring Functions. *J. Chem. Inf. Model.* **2022**, *62*, 2696–2712. https://doi.org/10.1021/acs.jcim.2c00485 [4]
- Yang, Z.; Zhong, W.; Lv, Q.; Dong, T.; Yu-Chian Chen, C. Geometric Interaction Graph Neural Network for Predicting Protein–Ligand Binding Affinities from 3D Structures (GIGN). *J. Phys. Chem. Lett.* **2023**, *14*, 2020–2033. https://doi.org/10.1021/acs.jpclett.2c03906 [25]
- Yu, T.; Kumar, S.; Gupta, A.; Levine, S.; Hausman, K.; Finn, C. Gradient Surgery for Multi-Task Learning. *Adv. Neural Inf. Process. Syst.* **2020**, *33*; arXiv:2001.06782. [14] †
- Zhang, X.; Gao, H.; Wang, H.; Chen, Z.; Zhang, Z.; Chen, X.; Li, Y.; Qi, Y.; Wang, R. PLANET: A Multi-Objective Graph Neural Network Model for Protein–Ligand Binding Affinity Prediction. *J. Chem. Inf. Model.* **2024**, *64*, 2205–2220 (online 2023). https://doi.org/10.1021/acs.jcim.3c00253 [21]
- Zhang, J.; Chen, H. Molecule Design Using Molecular Generative Models Constrained by Ligand–Protein Interactions. *J. Chem. Inf. Model.* **2022**, *62*, 3291–3306. https://doi.org/10.1021/acs.jcim.2c00177 [30]
- Zhung, W.; Kim, H.; Kim, W. Y. 3D Molecular Generative Framework for Interaction-Guided Drug Design. *Nat. Commun.* **2024**, *15*, 2688. https://doi.org/10.1038/s41467-024-47011-2 [29]

*Bibliographic details for the PubMed-verified entries were retrieved from PubMed records in this session.*

## Addendum (2026-09-25): independent cross-check of Sec. 1.5, supplied by the user

Additional near-neighbour sources (not in the numbered list of 45; tags show how far I verified them): **InterDiff** — Wu, P.; Du, H.; Yan, Y.; Lee, T.-Y.; Bai, C.; Wu, S. *Guided diffusion for molecular generation with interaction prompt.* *Brief. Bioinform.* **2024**, *25* (3), bbae174 [page fetched: training-time interaction-type cross-entropy loss inside the denoiser + cross-attention prompts; not PubMed-checked]. **BInD** — Lee, J.; Zhung, W.; Seo, J.; Kim, W. Y. *Adv. Sci.* **2025**, *12* (35), e02702; PMC12463045 [page fetched: PLIP interactions co-generated as a diffusion channel]. **Delta Score** — Ren, M.; Gao, B.; Qiang, B.; Lan, Y. arXiv:2311.12035 (2023) and **SBE-Diff** — Gao, B. et al. arXiv:2403.12987 (2024) [**preprints**; "delta" = a specificity/evaluation metric — name collision only]. Reported by the user's check but **not re-verified by me**: DiffPharma (*npj Drug Discov.* 2026, 10.1038/s44386-026-00040-x), DiffInt (*JCIM* 2024, 4c01385; **training-loss details unconfirmed**), BADGER (*JCIM* 2025, 5c01166), arXiv:2505.05874, DeepRLI (arXiv:2401.10806).

**Correction to the user-supplied summary:** InterDiff *does* use an auxiliary interaction-type classification loss at training time (in the generator's denoiser), so the statement that no paper uses typed interactions as an auxiliary training loss is not supported; the defensible gap is the placement of that supervision on the **guidance model** together with a delta target. See `TRACK_E_DESIGN.md` amendment A8.

