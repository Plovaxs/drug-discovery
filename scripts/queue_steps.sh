#!/usr/bin/env bash
# The work queue, in execution order. Sourced by scripts/master_queue.sh -- it defines STEPS and nothing
# else, so the queue can be reordered or extended without touching the runner.
#
# Format, one step per line:   name|kind|command
#   name     a stable identifier. The state file records completed steps BY NAME, so renaming a step
#            makes it run again. Never rename a finished step.
#   kind     gpu | cpu. Only affects the thermal gate: a cpu step still gets its rest period, but is not
#            held back waiting for the GPU to go idle.
#   command  run from the repo root with PYTHONPATH=. already set.
#
# Appending work: add a line. The runner skips whatever is already recorded as done, so appending to a
# queue that has partially run is safe and will pick up only the new steps.

PY="$HOME/miniconda3/envs/drugdisc/bin/python"
CFG=configs/prop/crossdocked_affinity_egnn.yml
A1E=guidance/uncertainty_a1/train_egnn_heteroscedastic.py

STEPS=(
  # --- Arm B seeds 2022 and 2023.
  #
  # These were originally launched by scripts/a1e_beta_chain.sh's sibling, the armBmatch chain, which was
  # written BEFORE the thermal gate existed: it ran its three seeds back to back with no gate and no rest.
  # Measured, that means ~8 hours continuously at 92 C CPU. Moved here so each seed waits for the card and
  # the CPU to cool first. Seed 2021 was already running when this move was made and was left to finish.
  #
  # --no_compound_filter is REQUIRED, not optional. Seed 2021 ran before the ligand-side compound filter
  # existed, so it trained on the unfiltered pool. Letting 2022 and 2023 use the filter would mean the
  # three seeds of one arm were trained on different data, and their spread would no longer estimate
  # seed variance. The pre-registered rule in TASKS.md covers how Arm B is to be reported given this;
  # Arm C is the run that uses the clean pool.
  "armB_s2022|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armBmatch --cd_rows 6000 --bn_rows 14000 --bn_match_pk --batch_size 1 --seed 2022 --no_compound_filter"
  "armB_s2023|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armBmatch --cd_rows 6000 --bn_rows 14000 --bn_match_pk --batch_size 1 --seed 2023 --no_compound_filter"
  "armB_aggregate|cpu|$PY guidance/aggregate_results.py logs_surrogate_arms --metric R2 --min_epochs 2 --json guidance/surrogate_data/armB_table.json"

  # --- A1e-beta, alternating so plain and beta stay balanced at every interruption point.
  #     A1e on record is ONE seed (2021), so the chain tops plain up to three while building beta to
  #     three: after step 1 it is 1v1, after step 3 it is 2v2, after step 5 it is 3v3. Stopping early
  #     costs power, never validity.
  "a1e_beta_s2021|gpu|$PY $A1E $CFG --skip_test_logging --beta_nll 0.5 --logdir ./logs_a1e_beta_heteroscedastic --tag a1eb_s2021 --seed 2021"
  "a1e_plain_s2022|gpu|$PY $A1E $CFG --skip_test_logging --beta_nll 0.0 --logdir ./logs_a1e_heteroscedastic --tag a1e_s2022 --seed 2022"
  "a1e_beta_s2022|gpu|$PY $A1E $CFG --skip_test_logging --beta_nll 0.5 --logdir ./logs_a1e_beta_heteroscedastic --tag a1eb_s2022 --seed 2022"
  "a1e_plain_s2023|gpu|$PY $A1E $CFG --skip_test_logging --beta_nll 0.0 --logdir ./logs_a1e_heteroscedastic --tag a1e_s2023 --seed 2023"
  "a1e_beta_s2023|gpu|$PY $A1E $CFG --skip_test_logging --beta_nll 0.5 --logdir ./logs_a1e_beta_heteroscedastic --tag a1eb_s2023 --seed 2023"

  # --- analysis of the above, cheap, runs on the CPU right after
  "a1e_aggregate|cpu|$PY guidance/aggregate_results.py logs_a1e_heteroscedastic logs_a1e_beta_heteroscedastic --metric R2 --compare a1e_beta a1e_hetero --json guidance/uncertainty_a1/a1e_beta_table.json"

  # --- todo items whose scripts already exist and need no new code
  "ladder_refresh|cpu|$PY guidance/cheminformatics/representation_ladder.py --b 2000"
  # Full-scale AVE bias: all 46,964 training fingerprints, 2000 target-cluster resamples. Piloted at
  # max_train=8000 (ours +0.046 vs random ligand split +1.044); queued rather than run inline because
  # the CPU is the thermal bottleneck on this machine and a training job already owns it.
  "ave_bias|cpu|$PY guidance/cheminformatics/ave_bias.py --b 2000"
  "noise_ceiling_refresh|cpu|$PY guidance/cheminformatics/noise_ceiling.py --n_perm 20 --b 2000"
  "test_suite|cpu|$PY guidance/run_tests.py"

  # --- APPEND HERE as each todo item's script lands:
  #   todo 2  PLIP typed interactions as the 8th representation block
  #   todo 3  conditional conformal coverage by novelty tier + protein family
  #   todo 4  within-target y-randomisation for the EGNN            (gpu)
  #   todo 5  MoleculeACE-style activity-cliff metric
  #   todo 6  matched molecular pair analysis
  #   todo 7  protein-family error stratification
  #   todo 1  positive control: guidance with Vina / PLIP           (gpu, scope first)
)
