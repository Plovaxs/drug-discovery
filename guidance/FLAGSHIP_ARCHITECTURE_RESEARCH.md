# Flagship 5-Tier Affinity Guidance Architecture: Research Synthesis

This document records the technical specification behind the flagship
architecture addendum, provided directly by the user as an exhaustive
literature review and reimplementation-ready extraction (superseding an
earlier, less precise synthesis authored during this session). It anchors
`FLAGSHIP_ARCHITECTURE_RESULTS.md`'s stage-by-stage implementation and
checkpoints, and responds directly to `AFFINITY_MODEL_SWEEP_RESULTS.md`'s
finding that the deployed affinity model's generalization gap is not
EGNN-architecture-specific (a ligand-only baseline matches it) but is
consistent with data scarcity/superficial shortcut reliance (0.80 Pearson
between predicted affinity and ligand heavy-atom count).

**Status note (added after Stage 0's actual checkpoint result):** Stage
0's real result deviated from this document's own anticipated outcome
("the baseline's R^2 will likely drop further") -- fixing only the split
produced a real, statistically robust positive result (test R^2=0.342,
tight CI, beating the ligand-only baseline) with the unchanged EGNN. See
`FLAGSHIP_ARCHITECTURE_RESULTS.md`'s Stage 0 section for the full,
honest account. Per the user's explicit decision after that checkpoint,
the 5-stage build is paused in favor of first re-testing guidance (Task
F) with the corrected model; this document remains the reference for
resuming Stage 1+ afterward.

## Motivation

Three concrete failure modes from the prior investigation anchor every
stage's pass/fail checkpoint:
1. Negative test R^2 on unseen targets (deployed model, every epoch).
2. A ligand-only ECFP4+RandomForest baseline matches the full EGNN's test
   performance (R^2 0.047 vs 0.046, old split).
3. Predicted affinity correlates with ligand heavy-atom count at Pearson
   0.799 (old split) / 0.719 (Stage 0's corrected split) -- a superficial
   shortcut, not learned pocket chemistry.

## A.1 Physics-Informed and Physics-Anchored Scoring Models

PIGNet and its successor PIGNet2 use a neural network to parameterize
coefficients of established biophysical equations rather than predicting
affinity directly -- decomposing binding free energy into van der Waals
(modified Lennard-Jones), hydrogen bonding, metal-ligand, and hydrophobic
terms, penalized by a ligand rotor-entropy term. This structural
bottleneck is intended to prevent the model from exploiting a simple
heavy-atom-count shortcut. Delta-learning frameworks (ΔVinaRF20,
ΔVinaXGB, RF-Score-v3/v4) learn corrections atop classical SFs like
AutoDock Vina but remain limited by the base SF's rigid functional form.
Caveat from the literature: PIGNet-style models trained with docking-
engine-generated decoys (PDA/NDA) risk learning to identify that
specific docking engine's geometric artifacts rather than genuine
physics (a CASF-documented risk) -- worth checking for in Stage 2/3's
evaluation, not just assuming away.

## A.2 Interaction-Explicit Graph Neural Network Architectures

GIGN and its 2025 successor EIGN use heterogeneous message-passing:
covalent (intramolecular) edges via sum-aggregation Covalent Interaction
Graph Convolutions (CIGConv, preserves exact valency/topology),
non-covalent (intermolecular) edges via mean-aggregation Non-covalent
Interaction Graph Convolutions (NIGConv, normalizes the dense, variable
proximity-contact count). EIGN adds dynamic edge updates and structural
noise simulation (randomly dropping/adding intermolecular edges up to 4
A) for pose-inaccuracy robustness, reaching CASF-2016 Pearson R=0.861
(RMSE 1.126) vs GIGN's R=0.810 (RMSE 1.278) and IGN's R=0.805 (RMSE
1.303) -- all trained on PDBbind v2020. Controlled benchmarks on
CSAR-HiQ/LP-PDBBind consistently rank these heterogeneous architectures
best for cold-target generalization specifically because they force
evaluation of interface geometry rather than bulk ligand properties.
CheapNet (ICLR 2025) shows hierarchical cross-attention between
coarse/fine-grained graphs can match this predictive power more
efficiently -- a possible later efficiency upgrade, not required for the
core 5-tier build.

## A.3 Uncertainty Quantification for OOD Generalization

eMOSAIC (embedding Mahalanobis Outlier Scoring and Anomaly Identification
via Clustering) computes Mahalanobis distance between an unseen
instance's embedding and clusters of known training-case embeddings,
demonstrating better calibration than deep ensembles for multitarget
affinity prediction. TerraBind's Epinets (epistemic neural networks)
instead train a small auxiliary network to predict a frozen base model's
residuals -- far cheaper than deep ensembles or full Mahalanobis
clustering, and the mechanism actually specified for this project's Tier
3 given the 4GB VRAM constraint (deep ensembles and eMOSAIC-style
clustering are both explicitly ruled out as infeasible at this budget --
this supersedes the earlier session's 4-seed-ensemble Diagnostic-4-style
plan, which should be read as feasible-but-not-optimal by comparison).
Conformal prediction (TESSERA, ENS-Score) gives distribution-free
coverage guarantees but tends toward intervals too wide to be actionable
for generative rejection sampling.

## A.4 Leakage-Aware Benchmarking and Dataset Curation

LP-PDBBind enforces protein sequence similarity strictly <50% and ligand
Tanimoto similarity strictly <0.99 across splits. Retraining established
SFs on LP-PDBBind: 3D structure-based models (IGN, RF-Score, AutoDock
Vina) are resilient/improve; purely 1D sequence-to-SMILES models
(DeepDTA) collapse catastrophically -- direct evidence that 3D
interaction-geometric models generalize better to novel targets, and
that the split methodology itself (not just architecture) is a primary
determinant of measured generalization. BDB2020+ (BindingDB records
matched to post-2020 PDB depositions, Kd/Ki only, IC50 excluded for assay
variance) provides an independent, zero-overlap temporal benchmark
outside this project's scope but worth noting as a future external
validation option. For CrossDocked2020 specifically: apply Dice
similarity for ligand clustering and strict sequence-identity cutoffs
for protein clustering (this project's Stage 0 substituted Needleman-
Wunsch identity gated by UniProt-Pfam category, plus a documented pocket-
composition proxy where ProBiS itself was unavailable -- see
`FLAGSHIP_ARCHITECTURE_RESULTS.md`).

## A.5 Equivariant Architectures and Docking-Pose Augmentation

Full E(3)-equivariant architectures (MACE, NequIP; MACE+DiffDock per
Shirasuna et al. 2025) are ruled out under the 4GB VRAM constraint. GIGN
sidesteps this by using coordinate-free invariant inputs (pairwise
distances, angles) plus geometric augmentation instead of spherical
harmonics. PIGNet2's three-way loss (crystal MSE + PDA near-native MSE +
NDA decoy hinge) is the specified mechanism for multi-pose robustness
(this project's Tier 5 / Stage 3).

## A.6 Known Failure Mode: Heavy-Atom-Count Shortcut

This project's own observed failure (Pearson(pred, heavy_atom_count) =
0.799 on the old split / 0.719 on Stage 0's corrected split, alongside
negative-then-marginal test R^2) is a documented CASF pathology: larger
ligands have more atoms available for dispersive vdW contacts, so
unconstrained networks latch onto this as a fast, low-loss shortcut
during early training, at the expense of learning orientation-dependent
H-bonding/electrostatics that actually drive specificity. The "Delta
Score" SBDD evaluation paradigm shows generative diffusion models will
exploit this by outputting larger, unoptimized structures to game a
size-biased learned SF -- directly relevant to this project's guidance
use case, since Task F's guided-sampling loop is exactly the setting
where such gaming could occur undetected without the Vina-hacking checks
already built into `eval/honest_eval.py`.

## Part B: Five-Tier Flagship Architecture

- **Tier 1 (Physics-Anchored Core):** PIGNet2 energy decomposition
  (vdW/H-bond/metal/hydrophobic, rotor-entropy-normalized) as the final
  readout instead of a scalar MLP head -- structurally prevents unbounded
  affinity output for a large, non-complementary ligand.
- **Tier 2 (Interaction-Explicit Backbone):** GIGN/EIGN heterogeneous
  graph (CIGConv sum-aggregation for covalent edges, NIGConv
  mean-aggregation for non-covalent edges) replacing the homogeneous
  EGNN backbone.
- **Tier 3 (Uncertainty/OOD-Robustness Head):** TerraBind-style Epinet
  (lightweight MLP trained on frozen backbone representations to predict
  residual error) -- explicitly the specified mechanism for this
  project's 4GB VRAM budget, replacing the heavier eMOSAIC/deep-ensemble
  options ruled infeasible at this scale.
- **Tier 4 (Leakage-Safe Training Protocol):** LP-CrossDocked strategy
  (Stage 0, already implemented -- see `FLAGSHIP_ARCHITECTURE_RESULTS.md`).
- **Tier 5 (Pose-Diversity Augmentation):** 3-way energy-landscape
  training (native diffusion pose + PDA near-native + NDA decoy) via the
  project's existing Vina infrastructure.

### Novelty scope

PIGNet2 (Tiers 1 & 5), GIGN (Tier 2), TerraBind Epinets (Tier 3), and
LP-PDBBind (Tier 4) each exist independently; synthesizing GIGN's
heterogeneous CIGConv/NIGConv layers as PIGNet2's backbone (replacing its
native homogeneous GatedGAT) plus an Epinet head for generative diffusion
guidance has not been published as a single system.

### 4GB VRAM feasibility

- GIGN's invariant-distance-input design avoids spherical-harmonic cost.
- Batch size must drop to 4-8 with gradient accumulation over 8-16 steps
  to approximate PIGNet2's reported batch 64 / GIGN's reported batch 128
  (this project's actual EGNN runs so far have used batch_size=1 with no
  accumulation, given per-atom-graph memory pressure at even modest
  batch sizes on this specific model+data combination -- accumulation
  should be implemented and validated empirically at Stage 2/3, not
  assumed to transfer directly from the literature's batch sizes).
- Neither PIGNet2 nor GIGN ship CrossDocked2020-finetuned weights;
  architectural code is open-source or reimplementable (see Part C below
  and the vendored reference excerpts fetched into
  `guidance/reference_code/`).

### Staged go/no-go roadmap (as specified by the user)

1. **Stage 1 (Tier 4, data rigor):** LP-PDBBind-style split on
   CrossDocked2020; evaluate the unchanged EGNN. *This project's Stage 0
   already executed this -- with a result that deviated from the
   anticipated "R^2 will likely drop further": R^2 instead rose to 0.342
   with a tight CI, beating the ligand-only baseline. See
   `FLAGSHIP_ARCHITECTURE_RESULTS.md`.*
2. **Stage 2 (Tier 2, GIGN backbone):** swap EGNN for GIGN heterogeneous
   backbone, scalar output head retained (physics head deferred).
   Checkpoint: heavy-atom Pearson correlation below 0.80* and cold-target
   R^2 above zero. (*Stage 0 already achieved R^2 above zero and heavy-
   atom correlation of 0.719 with the unchanged EGNN -- so this stage's
   checkpoint bar should be read as "improve further on Stage 0's
   0.719/0.342," not the old split's 0.799/negative numbers, when/if this
   stage resumes.)
3. **Stage 3 (Tiers 1 & 5, physics + pose augmentation):** replace scalar
   head with PIGNet2's energy decomposition; add 3-way Vina-decoy loss.
   Checkpoint: correct decoy-vs-native hinge-loss ranking; heavy-atom
   shortcut should weaken substantially due to the Lennard-Jones-style
   functional constraint.
4. **Stage 4 (Tier 3, guidance integration):** freeze backbone, train
   Epinet on residuals; deploy into the diffusion guidance loop.
   Checkpoint: does rejecting/down-weighting high-Epinet-uncertainty
   intermediate steps improve real Vina scores and structural validity?

## Part C: Exact Reimplementation Specs (as provided)

### C.1 PIGNet2

`E_total = (E_vdW + E_Hbond + E_Metal + E_Hydrophobic) / T_rotor`

- **vdW (modified 12-6 Lennard-Jones):**
  `E_vdW = sum_{i,j} c_ij * [ (d'_ij/d_ij)^12 - 2*(d'_ij/d_ij)^6 ]`
  where `c_ij` is a learned interaction-energy coefficient, `d_ij` is true
  Euclidean distance, and the corrected minimum distance
  `d'_ij = r_i + r_j + c * b_ij` (r_i, r_j = literature vdW radii, b_ij =
  learned distance-correction parameter, c=0.2 fixed). This b_ij
  correction (vs. PIGNet1's static radii) is the key PIGNet1->PIGNet2
  vdW change, preventing over-penalizing minor steric clashes in
  near-native docked poses.
- **H-bond / Metal / Hydrophobic (shared piecewise form):**
  `e_ij = w` if `d_ij - d'_ij < c1`; `w * (d_ij - d'_ij - c2)/(c1 - c2)` if
  `c1 <= d_ij - d'_ij < c2`; `0` if `d_ij - d'_ij >= c2`. `(c1, c2) =
  (-0.7, 0.0)` for H-bond and Metal; `(0.5, 1.5)` for Hydrophobic. `w` is
  a learned scalar weight per pair.
- **Rotor penalty:** `T_rotor = 1 + C_rotor * N_rotor`, `N_rotor` = RDKit
  rotatable-bond count (deterministic), `C_rotor` learned scalar.
- **Backbone:** Gated Graph Attention Network (GatedGAT): unnormalized
  attention `e_ij = (W2 h_i)^T (W2 h_j)`, softmax-normalized, GRU-updated
  node states across depth to prevent oversmoothing:
  `h_i^{(n+1)} = GRU(h'_i^{(n)}, h_i^{(n)})`.
- **Three-way loss:** MSE on crystal + PDA (near-native, local-docking-
  perturbed) poses; hinge `max(y_decoy - y_crystal - 1.0, 0)` for NDA
  high-RMSD re-docked decoys; hinge `max(-6.8 - y_random, 0)` for
  random cross-docked "assumed non-binder" screening pairs (-6.8 kcal/mol
  ~ Kd~1e-5 M generic non-binder threshold).
- **Hyperparameters:** Adam, lr=0.0004, batch=64 (accumulated),
  dropout=0.1, target=exact pKd/pKi.
- **Repos:** `github.com/ACE-KAIST/PIGNet2`, `github.com/ACE-KAIST/PIGNet`.
- **CASF-2016:** Pearson R=0.747, Spearman rho=0.651, Top-1 docking
  success 66.7% (93.0% loose), Top-1%/0.5% screening EF 24.9/20.0.

### C.2 GIGN

- **CIGConv (covalent, sum aggregation):**
  `m_i^{(t+1)} = f_theta( h_i^{(t)} + sum_{j in N(v_i)} sigma(h_j^{(t)} + h_ji^{(t)}) )`,
  applied only to same-molecule (ligand-ligand or protein-protein) edges.
- **NIGConv (non-covalent, mean aggregation):**
  `m_i^{(t+1)} = sigma( W_alpha h_i^{(t)} + W_beta * (1/|N(v_i)|) * sum_{k in N(v_i)} h_k^{(t)} (.) h_ki^{(t)} )`,
  applied only to intermolecular ligand-protein edges. Final node repr:
  `[h_i^{(t+1)}]_l = [m_i^{(t+1)}]_{(l,l)} + [m_i^{(t+1)}]_{(l,p)}`.
- **Invariance:** scalar Euclidean distances via RBF encoding, concatenated
  with one-hot invariant atomic identity (element, degree, implicit
  valence, hybridization, aromaticity) -- no absolute-orientation signal
  reaches the network at all.
- **Architecture:** 2 CIGConv + 2 NIGConv layers, global add-pooling over
  ligand nodes post-message-passing, MLP readout to scalar affinity.
- **Hyperparameters:** Adam, batch=128, receptive field 2-4 hops
  (application-dependent).
- **Repo:** `github.com/guaguabujianle/GIGN`.
- **CASF-2016 (PDBbind v2020):** GIGN RMSE 1.278/R=0.810; EIGN (2025
  successor, adds dynamic edge updates + up-to-4A edge noise simulation)
  RMSE 1.126/R=0.861; IGN RMSE 1.303/R=0.805.

### C.3 LP-PDBBind

- Protein: sequence-identity clustering (CD-HIT/MMseqs2), strict <50%
  cross-split identity (no test target >=50% identical to any train
  target).
- Ligand: Tanimoto on Morgan/ECFP fingerprints, strict <0.99 cross-split
  similarity.
- BDB2020+: independent BindingDB Kd/Ki records (IC50 excluded for assay
  variance) matched to post-2020 PDB depositions -- guarantees zero
  overlap with any PDBBind-era training data. Not used in this project
  (CrossDocked2020-only scope) but noted as a possible future external
  check.
- Qualitative retraining trend (exact numeric arrays not recoverable from
  available sources, flagged as such by the source review): DeepDTA (1D
  sequence-to-SMILES) collapses catastrophically on the leak-proof split;
  IGN (3D interaction graph) and AutoDock Vina/RF-Score (classical/delta)
  remain resilient or improve -- directionally consistent with, though
  not a substitute for, this project's own Stage 0 finding that fixing
  the split alone materially changed the EGNN's measured generalization.
- **CrossDocked2020 adaptation steps (as specified):** extract FASTA
  sequences for pockets in the filtered subset; cluster via MMseqs2 at
  50% identity; compute all-by-all Tanimoto (Morgan, radius 2, 2048 bits)
  for bound ligands; segregate into train/val/test clusters with no edge
  crossing either threshold; use exclusively this held-out test cluster
  for every later checkpoint. (This project's actual Stage 0
  implementation used Needleman-Wunsch global-alignment percent identity
  rather than CD-HIT/MMseqs2 clustering directly, gated by UniProt-Pfam
  category to keep the O(n^2) comparison tractable at ~1000 targets, and
  1024-bit Morgan fingerprints with Dice rather than 2048-bit/Tanimoto --
  functionally equivalent similarity signals, documented substitutions
  rather than the literal tool names specified here, given environment
  constraints -- e.g. no MMseqs2/CD-HIT binary available -- see
  `guidance/lp_split/build_split.py`.)

## Reference implementations fetched for direct reimplementation

Vendored excerpts (model code, loss functions, data augmentation
scripts) from the repositories cited above are saved under
`guidance/reference_code/` for faithful reimplementation at Stage 2/3,
rather than working from paper text alone. See that directory's own
notes for exactly what was retrievable and what wasn't.
