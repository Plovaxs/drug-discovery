# Family-Holdout Generalization: Null Guidance Pattern Holds Even for a Protein Family Never Seen in Training

Phase 2's second closely-connected branch: the LP-split's own leakage
safety (protein-identity threshold) still allows a test target to share
a fold/family with something seen in training. This asks a strictly
harder question: does the null guidance pattern (16 checkpoints,
`FOLLOWUP_PHASE_POOLED_CORRECTION.md`) also hold when the test target's
entire PROTEIN FAMILY was held out of training -- the closest proxy
this project can build to "a genuinely novel drug target."

## Split

`guidance/lp_split/build_family_holdout_split.py`: PF00069 (protein
kinase domain), the single largest family in the LP-split pool (per
cached Pfam annotations), held out entirely -- 53 targets / 8,880
entries moved to test, remaining 918 targets / 50,408+5,600 entries used
for train+val. No kinase-family target appears anywhere in training.

## Predictive quality: generalizes better than expected

Trained with the vina-target + noise-matching recipe (the best-performing
"simple" recipe in this investigation), tested on the entirely unseen
kinase family:

| | Test R² | Pearson | Spearman |
|---|---|---|---|
| Vina-target (same-distribution LP test) | 0.400 | 0.652 | - |
| **Family-holdout (unseen kinase family)** | **0.507** | **0.713** | **0.755** |

Point-prediction quality does not collapse on a never-seen protein
family -- if anything it is somewhat better here than the standard
same-distribution checkpoint's own numbers. This is itself informative:
whatever limits guidance is not simply "the predictor falls apart on
novel targets."

## Diagnostic: healthy, in fact better-signed than the familiar-distribution checkpoint

3 kinase pockets (CDK6_HUMAN via 2 docked poses, DYRK2_HUMAN), n=16/pocket:

| | size/heaviness corr | distance-to-pocket corr |
|---|---|---|
| Vina-target (familiar distribution) | -0.084 | +0.047 (wrong sign) |
| **Family-holdout (unseen kinase family)** | +0.056 [0.005, 0.104] | **-0.086 [-0.136, -0.037] (correct sign, significant)** |

## Real-docking test: same null pattern, on a target family the model has never seen

Prelambda check on CDK6_HUMAN_1_312_0 (`guidance/prelambda_family_holdout_results.json`,
n=8, standard 7-point grid; receptor/pocket wiring required adding
`--data_id`/`--receptor_pdb` to `lambda_sweep.py`, since every prior
checkpoint's test reused one fixed pocket not eligible for this
generalization question):

| Lambda | Mean diff vs. λ=0 | Wilcoxon p | single_fragment_rate |
|---|---|---|---|
| 0.1 | -0.009 | 0.195 | 1.00 |
| 0.3 | +0.007 | 0.688 | 1.00 |
| 1.0 | +0.007 | 0.844 | 1.00 |
| 3.0 | +0.161 | 0.641 | 1.00 |
| 10.0 | +0.241 | 0.313 | 0.75 |
| 30.0 | n=1 (most samples fragmented) | - | 0.12 |

No significant effect at any lambda, and the same stable-then-collapse
validity pattern seen on every other checkpoint. (Absolute Vina Dock
values here are much weaker, -3.4 to -4.8 kcal/mol vs. -10 to -11 on the
fixed pocket every other checkpoint was tested on -- an artifact of this
being a different, presumably shallower pocket, not a guidance effect;
only the within-pocket guided-vs-unguided comparison is meaningful here.)

## Conclusion

The null real-docking pattern is not an artifact of every tested pocket
being from a protein family the predictor had already seen somewhere in
training. It holds even under the strictest generalization test this
project can construct: a predictor trained with zero exposure, at any
training stage, to the target's entire protein family, tested on that
family's own most prominent member (CDK6, a major real-world kinase drug
target). This is the 17th independently-trained checkpoint in this
investigation to show the same pattern, and the one under the hardest
train/test separation.

## Artifacts

- `guidance/lp_split/build_family_holdout_split.py`, `train_egnn_stage0_family_holdout.py` -- split and training.
- `guidance/family_holdout_kinase_pockets.json` -- kinase pocket list for diagnostic/docking tests.
- `guidance/diag_size_confound_family_holdout_results.json` -- DIAG1 diagnostic.
- `guidance/prelambda_family_holdout_results.json` -- real-docking exploratory tier.
- `guidance/lambda_sweep.py`'s new `--data_id`/`--receptor_pdb` flags -- lets any diffusion-native test pocket be used for a real-docking test, not just the one fixed example pocket every prior checkpoint shared.
