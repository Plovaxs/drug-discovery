# 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/guanjq/targetdiff/blob/main/LICIENCE)

## This fork: thesis extension (Coupled Affinity-Synthesizability Guidance)

This repository is a fork of the [official TargetDiff implementation](https://github.com/guanjq/targetdiff)
(all credit for the base model, training pipeline, and original codebase
below belongs to the original authors — see [Citation](#citation)). It
adds a Master's thesis research project built on top of TargetDiff's
frozen, pretrained diffusion model:

> **Coupled Affinity-Synthesizability Guidance for Target-Conditional
> Molecular Diffusion in Structure-Based Drug Design**

Full thesis document: `guidance/thesis/Thesis_Coupled_Guidance_SBDD.pdf`
(176 pages, 10 chapters + 5 appendices, 33 figures, built via
`guidance/thesis/build_thesis.py`). Appendix E documents the full
follow-up investigation below (8 further independently-trained
checkpoints, three phases, and a direct mechanistic trace) in the same
statistical rigor as the core thesis.

### Core thesis: four independent falsifications

The thesis investigates whether generation can be steered toward
molecules with better predicted binding affinity and synthesizability,
using four independent, statistically rigorous tracks — each governed by
pre-registered dual-criterion checkpoints (KS tests with
Benjamini-Hochberg correction across all comparisons, bootstrap 95% CIs,
and mandatory negative controls, never a bare mean-difference claim):

| Track | Mechanism tested | Result |
|---|---|---|
| **A** | Physics-anchored gradient guidance (GIGN + PIGNet2 energy decomposition) | Falsified — full-tier training, dual-checkpoint verified null |
| **B** | 3 gradient-guidance variants (norm normalization, classifier-head reformulation, timestep-windowing) | Falsified — 3/3 null |
| **C** | Non-gradient rejection sampling (post-hoc top-k filtering by a frozen affinity ranker) | Falsified — apparent raw-score gain is a molecule-size artifact; ligand efficiency gets significantly *worse* in 14/15 pockets |
| **D** | Synthesizability guidance re-investigation (leakage-safe retrain, λ re-sweep, direction diagnostics) | Mechanism unresolved after ruling out the obvious confounds |

All four tracks converge on the same root cause (diagnosed in
`guidance/DIAG1_SIZE_CONFOUND_FINDING.md`): the trained affinity model's
signal is dominated by molecular size, not target-specific binding
chemistry.

### Follow-up investigation: does fixing the diagnosed mechanism help? (8 checkpoints, three phases, still no)

A second phase, reported in full as Appendix E of the thesis, tested
whether directly repairing DIAG1's diagnosed mechanism — rather than
working around it — produces a real guidance effect, then pushed the
question harder with independent validation, the strictest generalization
split available, and a direct trace of the guidance mechanism itself.

**Phase 1 — repair the predictor (6 checkpoints).** Four interventions
tried alone, in combination (Track A follow-up), and all stacked together
(kitchen-sink), on top of the original Stage 0 EGNN and Track A's
GIGN+PIGNet2 backbones:

| Checkpoint | Best fix | Real-docking effect |
|---|---|---|
| Noise-matched EGNN | Diffusion-matched noise curriculum | Null |
| Vina-target EGNN | Vina-derived pK label instead of experimental | Null |
| Gradient-alignment EGNN | Directly minimizes DIAG1's gradient/distance correlation (-0.78, strongest fix of any checkpoint) | Null — flattest result of all, no encouraging direction at any λ |
| ESM2-augmented EGNN | Adds a protein language-model (ESM2) global feature; best predictive quality (R²=0.460) and best correctly-signed diagnostic (-0.072, significant) of any checkpoint | Closest near-miss — raw p=0.04 at λ=3, but BH-adjusted p≈0.08 and a new validity cost appears at n=24 |
| Track A follow-up (GIGN+PIGNet2 + 2 fixes) | Noise-matching + Vina-target combined | Null — 87.5% direction-consistency (best of this architecture) still 0/16 significant |
| **Kitchen-sink** (all 4 combined) | Every intervention above, stacked | Null — closes the ablation space; no emergent synergy, worst size-confound of any checkpoint |

**Phase 2 — independent validation and the hardest generalization test (1
new checkpoint + 2 analyses on existing checkpoints).** `guidance/PLIP_INTERACTION_VALIDATION_FINDING.md`
asks a sharper question than "did the Vina score improve": does guidance
increase the number of *real, named* interactions (H-bonds, hydrophobic
contacts, pi-stacking, salt bridges, via PLIP) a docked pose actually
forms? An n=8 exploratory signal (interactions down −3.71, Vina Dock up
−2.27 kcal/mol) reversed sign on confirmation at n=16 (interactions
+1.00) — not weakened, reversed. `guidance/FAMILY_HOLDOUT_GENERALIZATION_FINDING.md`
retrains with an entire protein family (PF00069, kinases) held out of
training and tests on CDK6, a major real-world kinase target the model
never saw any relative of: predictive quality holds (R²=0.507) and the
diagnostic is correctly signed and significant (−0.086) for the first
time on a non-forced checkpoint, but the real-docking effect is still
null. `guidance/ALPHAFOLD_POCKET_ROBUSTNESS_FINDING.md` (side analysis)
checks whether AlphaFold DB models could substitute for crystal pockets
across 835 LP-split targets: median pocket RMSD 0.47 Å, pLDDT-RMSD
correlation r=−0.63 (p≈1e-92), but 13/835 (1.6%) targets are "confidently
wrong" (pLDDT>80 yet RMSD>5 Å) — a failure mode pLDDT alone cannot flag.

**Phase 3 — sharpen the signal and check the labels (1 new checkpoint + 1
label-verification analysis).** `guidance/ESM2POCKET_GUIDANCE_FINDING.md` restricts the ESM2 feature to
only the binding-pocket residues instead of the whole protein: predictive
quality gets *worse* (R²=0.280, down from 0.460), the first checkpoint in
this investigation to actively regress by trying to sharpen a signal.
`guidance/CHEMBL_CROSSVALIDATION_FINDING.md` independently cross-checks
this project's affinity labels against ChEMBL for 4 recurring targets: 3
of 4 agree, but CD38's labels disagree completely (this project: pK
3.6–4.3; ChEMBL: pChEMBL 7.9–9.5) — a genuine discrepancy reported as
such, with a plausible but unverified explanation (PDBBind vs. ChEMBL
literature-selection bias).

**Direct mechanistic trace.** `guidance/GUIDANCE_MECHANISM_TRACE_FINDING.md`
answers the question directly: one molecule, sampled twice from
bit-identical noise (unguided vs. guided), with every reverse-diffusion
step's position gap measured atom by atom. The gap is exactly zero until
guidance starts, grows smoothly, and only accelerates in the final third
of sampling, ending at 0.05 Å mean / 0.13 Å max — about 1% of what the
raw cumulative nudge magnitude (9.49 units) would suggest if nudges
simply added up. The guidance signal is not being actively cancelled; it
is mostly drowned out by the sampler's own intrinsic step noise for most
of the trajectory, a specific and falsifiable account of why position-only
guidance is too weak to matter. (The two molecules' discrete atom types
still end up completely different — a separate, flagged-as-open "butterfly
effect" in the categorical sampling channel.)

**Pooled result across the entire follow-up phase: 48 independent
statistical tests, 0 significant after Benjamini-Hochberg correction**
(`guidance/FOLLOWUP_PHASE_POOLED_CORRECTION.md`). The central finding:
every mechanism-level fix improves its own diagnostic or predictive
target, independent chemistry-level validation and the hardest
generalization split both agree with the raw-score result, and the
mechanistic trace shows specifically why — the guidance mechanism itself,
not any single diagnosed confound, is the bottleneck.

**Start here:**
- `guidance/FOLLOWUP_PHASE_POOLED_CORRECTION.md` — the single pooled statistical verdict across the whole follow-up phase
- `guidance/GUIDANCE_MECHANISM_TRACE_FINDING.md` — the direct, step-by-step trace of what guidance does to one molecule
- `guidance/STAGE2_PLUS_EXPERIMENT_LOG.md` — single source of truth, every core-thesis experiment logged
- `guidance/DUAL_FALSIFICATION_CONCLUSION.md`, `guidance/TRACK_C_REJECTION_SAMPLING_REPORT.md` — core-thesis full write-ups
- `guidance/GRADALIGN_GUIDANCE_FINDING.md`, `guidance/ESM2_GUIDANCE_FINDING.md`, `guidance/ESM2POCKET_GUIDANCE_FINDING.md`, `guidance/KITCHENSINK_GUIDANCE_FINDING.md`, `guidance/TRACK_A_FOLLOWUP_FULL_TIER_FINDING.md` — Phase 1 per-checkpoint write-ups
- `guidance/PLIP_INTERACTION_VALIDATION_FINDING.md`, `guidance/FAMILY_HOLDOUT_GENERALIZATION_FINDING.md`, `guidance/CHEMBL_CROSSVALIDATION_FINDING.md` — Phase 2/3 write-ups
- `guidance/thesis/chapters/11_appendices.txt` (Appendix E) — all of the above assembled into the thesis document itself, with 14 figures and 7 tables
- `guidance/generate_molecules_for_target.py` — generate & score candidate molecules for any of the 100 test-set pockets
- `guidance/render_examples.py`, `guidance/build_pptx_glb.py` — publication-quality and PowerPoint-ready 3D visualizations of generated binding poses

This section documents the thesis contribution; everything below is the
original TargetDiff paper/codebase documentation.

-----

This repository is the official implementation of 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction (ICLR 2023). [[PDF]](https://openreview.net/pdf?id=kJqXEPXMsE0) 

<p align="center">
  <img src="assets/overview.png" /> 
</p>

## Installation

### Dependency

The code has been tested in the following environment:


| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8       |
| PyTorch           | 1.13.1    |
| CUDA              | 11.6      |
| PyTorch Geometric | 2.2.0     |
| RDKit             | 2022.03.2 |

### Install via Conda and Pip
```bash
conda create -n targetdiff python=3.8
conda activate targetdiff
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install rdkit openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge

# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
The code should work with PyTorch >= 1.9.0 and PyG >= 2.0. You can change the package version according to your need.

### (Alternatively) Install via Mamba
Install Mamba

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh  # accept all terms and install to the default location
rm Mambaforge-$(uname)-$(uname -m).sh  # (optionally) remove installer after using it
source ~/.bashrc  # alternatively, one can restart their shell session to achieve the same result
```

Create Mamba environment
```bash
mamba env create -f environment.yaml
conda activate targetdiff  # note: one still needs to use `conda` to (de)activate environments
```

-----
# Target-Aware Molecule Generation
## Data
The data used for training / evaluating the model are organized in the [data](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link) Google Drive folder.

To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
* `crossdocked_pocket10_pose_split.pt`

To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), save it into `data/CrossDocked2020`, and run the scripts in `scripts/data_preparation`:
* [clean_crossdocked.py](scripts/data_preparation/clean_crossdocked.py) will filter the original dataset and keep the ones with RMSD < 1A.
It will generate a `index.pkl` file and create a new directory containing the original filtered data (corresponds to `crossdocked_v1.1_rmsd1.0.tar.gz` in the drive). *You don't need these files if you have downloaded .lmdb file.*
    ```bash
    python scripts/data_preparation/clean_crossdocked.py --source data/CrossDocked2020 --dest data/crossdocked_v1.1_rmsd1.0 --rmsd_thr 1.0
    ```
* [extract_pockets.py](scripts/data_preparation/extract_pockets.py) will clip the original protein file to a 10A region around the binding molecule. E.g.
    ```bash
    python scripts/data_preparation/extract_pockets.py --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket10
    ```
* [split_pl_dataset.py](scripts/data_preparation/split_pl_dataset.py) will split the training and test set. We use the same split `split_by_name.pt` as 
[AR](https://arxiv.org/abs/2203.10446) and [Pocket2Mol](https://arxiv.org/abs/2205.07249), which can also be downloaded in the Google Drive - data folder.
    ```bash
    python scripts/data_preparation/split_pl_dataset.py --path data/crossdocked_v1.1_rmsd1.0_pocket10 --dest data/crossdocked_pocket10_pose_split.pt --fixed_split data/split_by_name.pt
    ```
## Training
### Training from scratch
```bash
python scripts/train_diffusion.py configs/training.yml
```
### Trained model checkpoint
https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz?usp=share_link

## Sampling
### Sampling for pockets in the testset
```bash
python scripts/sample_diffusion.py configs/sampling.yml --data_id {i} # Replace {i} with the index of the data. i should be between 0 and 99 for the testset.
```
You can also speed up sampling with multiple GPUs, e.g.:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 0 0
CUDA_VISIBLE_DEVICES=1 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 1 0
CUDA_VISIBLE_DEVICES=2 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 2 0
CUDA_VISIBLE_DEVICES=3 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs 4 3 0
```

### Sampling from pdb file
To sample from a protein pocket (a 10A region around the reference ligand):
```bash
python scripts/sample_for_pocket.py configs/sampling.yml --pdb_path examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb
```

## Evaluation
### Evaluation from sampling results
```bash
python scripts/evaluate_diffusion.py {OUTPUT_DIR} --docking_mode vina_score --protein_root data/test_set
```
The docking mode can be chosen from {qvina, vina_score, vina_dock, none}

Note: It will take some time to prepare pqdqt and pqr files when you run the evaluation code with vina_score/vina_dock docking mode for the first time.

### Evaluation from meta files
We provide the sampling results (also docked) of our model and CVAE, AR, Pocket2Mol baselines [here](https://drive.google.com/drive/folders/19imu-mlwrjnQhgbXpwsLgA17s1Rv70YS?usp=share_link).

| Metafile Name                   | Original Paper                                                                                      |
|---------------------------------|-----------------------------------------------------------------------------------------------------|
| crossdocked_test_vina_docked.pt | -                                                                                                   |
| cvae_vina_docked.pt             | [liGAN](https://arxiv.org/abs/2110.15200)                                                           |
| ar_vina_docked.pt               | [AR](https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html) |
| pocket2mol_vina_docked.pt       | [Pocket2Mol](https://proceedings.mlr.press/v162/peng22b.html)                                       |
| targetdiff_vina_docked.pt       | [TargetDiff](https://openreview.net/pdf?id=kJqXEPXMsE0)                                             |

You can directly evaluate from the meta file, e.g.:
```bash
python scripts/evaluate_from_meta.py sampling_results/targetdiff_vina_docked.pt --result_path eval_targetdiff
```

**One can reproduce the results reported in the paper quickly with [notebooks/summary.ipynb](notebooks/summary.ipynb)**

-----
# Binding Affinity Prediction

## Data
* In the unsupervised learning setting, we still use the CrossDocked2020 dataset and find the data with experimentally measured binding affinity (saved in `affinity_info.pkl`) for further analysis. 

* In the supervised learning setting, we use the PDBBind dataset, which can be downloaded from: http://www.pdbbind.org.cn. 
The downloaded refined / general set should be saved in data/pdbbind_v{YEAR} directory. 

Take the PDBBind v2016 for example, you need to first unzip the data:
```bash
mkdir -p data/pdbbind_v2016 && tar -xzvf data/pdbbind_v2016_refined.tar.gz -C data/pdbbind_v2016
```
Then, you can extract 10A pockets and split the dataset using the following commands:
```bash
# extract pockets
python scripts/property_prediction/extract_pockets.py --source data/pdbbind_v2016 --subset refined --refined_index_pkl data/pdbbind_v2016/pocket_10_refined/index.pkl

# split dataset
python scripts/property_prediction/pdbbind_split.py --index_path data/pdbbind_v2016/pocket_10_refined/index.pkl  --save_path data/pdbbind_v2016/pocket_10_refined/split.pt
```

## Training
One can train the binding affinity prediction model with:
```bash
python scripts/property_prediction/train_prop.py configs/prop/pdbbind_general_egnn.yml
```

It is also possible to enhance the model with extra features extracted from the unsupervised generative model. You need to first export the hidden states with:

```bash
python scripts/likelihood_est_diffusion_pdbbind.py
```

This command will dump various meta information and 
you need to specify the feature you want to use in the training config (like `configs/prop/pdbbind_general_egnn.yml`) of the following supervised prediction model.

### Trained model checkpoint
_NOTE: For the supervised learning setting, since the training results on PDBBind v2020 are lost by accident, 
we can only provide the model checkpoint trained on PDBBind v2016 in the preliminary experiments for now. 
However, it can already make accurate prediction for the practical use. 
We will retrain the models on PDBBind v2020 and provide the trained checkpoints as soon._

https://drive.google.com/drive/folders/1-ftaIrTXjWFhw3-0Twkrs5m0yX6CNarz?usp=share_link


## Evaluation
* For the unsupervised learning evaluation, please check [notebooks/analyze_affinity.ipynb](notebooks/analyze_affinity.ipynb)

* For the supervised learning evaluation, one can use the following command to evaluate on the tes set:
```bash
python scripts/property_prediction/eval_prop.py --ckpt_path pretrained_models/egnn_pdbbind_v2016.pt
```
Expected results:

| RMSE  | MAE   | R^2   | Pearson | Spearman |
|-------|-------|-------|---------|----------|
| 1.316 | 1.031 | 0.633 | 0.797   | 0.782    |


## Inference
To predict the binding affinity of a complex, one need to prepare the PDB file and SDF/MOL2 file first
(**Important: for the supervised learning model trained on PDBBind v2016, both protein and ligand need to have hydrogen atoms**). 
Then, the binding affinity can be predicted with [scripts/property_prediction/inference.py](scripts/property_prediction/inference.py). For example,
```bash
python scripts/property_prediction/inference.py \
  --ckpt_path pretrained_models/egnn_pdbbind_v2016.pt \
  --protein_path examples/3ug2_protein.pdb \
  --ligand_path examples/3ug2_ligand.sdf \
  --kind Kd
```

Expected prediction: Kd=5.23 nm.  Ground-truth: Kd=5.6 nm

## Citation
```
@inproceedings{guan3d,
  title={3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction},
  author={Guan, Jiaqi and Qian, Wesley Wei and Peng, Xingang and Su, Yufeng and Peng, Jian and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
