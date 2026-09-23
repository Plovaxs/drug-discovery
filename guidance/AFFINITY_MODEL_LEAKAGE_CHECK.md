# Affinity Guidance Model: Data Leakage &amp; Coverage Check (Part 3.1-3.2)

Mandatory first step of Part 3 (deepening the affinity model), run before any
retraining/hyperparameter work, per this project's own working rule that a
leakage check is top priority if there is any reason to suspect it.

## Training pipeline (as it actually exists)

- Training script: `guidance/train_affinity_model.py`, trains `PropPredNet`
  (EGNN, `models/property_pred/prop_model.py`) with MSE regression to a
  single pK value; config `configs/prop/crossdocked_affinity_egnn.yml`;
  checkpoint saved on best val loss (max 20 epochs, early-stop patience 5).
- Base geometry: the same CrossDocked2020 LMDB TargetDiff itself trains on
  (`PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')`).
- Split membership comes from `./data/crossdocked_pocket10_pose_split.pt`
  -- **the official TargetDiff train/test split** (99,990 train / 100 test
  pocket-instances; 1,905 unique train targets / 93 unique test targets;
  the official split itself already has 0 target overlap between its own
  train and test).
- Labels: `./data/affinity_info.pkl`, 184,087 entries keyed by CrossDocked
  ligand filename stem (76,803 with nonzero pK) -- **real experimental
  pKd/pKi/pIC50 values, PDBBind-derived**, not docking/Vina scores and not
  synthetic. This matters: it rules out a subtler "masked leakage" concern
  (training on Vina scores computed with a docking config later reused at
  evaluation time) -- the labels here are independent of any evaluation-time
  docking configuration entirely.
- `val` is carved 90/10 out of the official *train* split only; the
  returned `test` split is exactly the officially-test pockets that happen
  to carry a real label -- never trained on.
- Actual run used `train_subsample: 6000`: final labeled sets were
  train=6,000 (507 unique targets), val=4,061 (456 unique targets),
  test=27 (26 unique targets).

## Pocket-level leakage vs. Task F: NO

Cross-referenced all 20 `guidance/task_f_pockets.json` target IDs directly
against the affinity model's actual `train_final`/`val_final` index sets
(rebuilt with the real training seed/config). **Intersection is empty for
both.** All 20 Task F targets are a subset of the official CrossDocked2020
test-split targets, and the official split has zero target overlap with
train by construction. The affinity guidance model never received a
gradient update from any pocket that appears in Task F.

**Caveat (not gradient leakage, flagged for transparency):** 7 of the 20
Task F targets (`IMA1_HUMAN_68_497_0`, `Y635_MYCTU_1_158_0`,
`AKT1_HUMAN_1_137_0`, `CD38_HUMAN_44_300_0`, `BSD_ASPTE_1_130_0`,
`DFPA_LOLVU_2_314_0`, `SQHC_ALIAD_1_631_0`) are also members of the
affinity model's own 27-pocket held-out `test_final` set, evaluated
(metrics only, not used for optimization or checkpoint selection beyond
val loss) during training. No weights were updated on this signal, so this
is not classical leakage, but is disclosed here in case it matters for a
stricter reading of "independent test."

## Target-family-level leakage: negligible, one coincidence

Compared gene-symbol prefixes (UniProt-mnemonic style) across Task F
targets and the affinity train set. One coincidental match:
`CHIA_SERMA_19_563_0` (Task F -- bacterial chitinase, *Serratia
marcescens*) vs `CHIA_HUMAN_20_404_0` (affinity train) -- same 4-letter
mnemonic, different gene/organism/fold family, not a real target match. No
other prefix collisions. **Caveat:** this was a coarse gene-symbol string
match only; no UniProt-ID or sequence-identity clustering was available to
check subtler family relationships (e.g. shared kinase fold, shared
binding-site pharmacophore). Flagged as uncertain beyond this coarse
check, not asserted as a clean bill of health at the fold-family level.

## Data coverage (Part 3.2)

- 184,087 total keyed affinity entries; 76,803 with nonzero pK (rest are
  presumably unlabeled/placeholder entries in the source annotation file).
- Actual training pool after subsampling: 6,000 train (507 unique targets,
  ~11.8 examples/target average -- thin per-protein redundancy), 4,061 val
  (456 unique targets), 27 test (26 unique targets).
- pK label range ~0.5-14; train mean 6.76 +/- 1.84, val 6.85 +/- 1.89, test
  5.71 +/- 1.81 -- broadly overlapping distributions, no severe class
  imbalance, but a noticeably lower mean/tighter test set (n=27 is also
  small enough that this could just be sampling noise, not a real shift).
- Diversity is wide (500-1900 unique targets touched across the full pool)
  but shallow (few examples per target) -- consistent with a model that
  may have learned coarse, cross-target trends rather than target-specific
  affinity structure, which is a plausible contributing explanation for
  the flat dose-response found in `AFFINITY_LAMBDA_RESWEEP_FINDING.md`:
  a guidance gradient trained mostly on cross-target signal has little
  reason to sharpen affinity for one particular pocket during sampling.

## Verdict

**No disqualifying data leakage found.** Task F's null result and the
lambda re-sweep's flat dose-response are not explained by the affinity
model having "seen the answer" for these test pockets -- pocket-level
overlap is empty, and the training labels are real experimental values
independent of any evaluation-time docking configuration. The most
plausible remaining explanation, given the thin per-target example count
noted above, is a data-coverage/generalization issue rather than a
leakage issue: proceeding to Part 3.3 (held-out validation + light
hyperparameter sweep) and Part 3.4 (standalone predictive-quality report)
to characterize this directly.

## Reproducibility

- Training script: `guidance/train_affinity_model.py`
- Split/label construction: `datasets/crossdocked_affinity.py` (`build_labeled_splits`)
- Config: `configs/prop/crossdocked_affinity_egnn.yml`
- Split file: `data/crossdocked_pocket10_pose_split.pt`
- Label file: `data/affinity_info.pkl`
