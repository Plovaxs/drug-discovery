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

  # --- RQ2 positive control (todo 1). Pre-registered BEFORE this ran:
  #     guidance/RQ2_POSITIVE_CONTROL_PREREGISTRATION.md. Design: best-of-N Vina rerank, not
  #     backprop through Vina (not differentiable) -- oversample an unguided pool 2x, select the
  #     best half by cheap Vina score_only (~0.57s/molecule, no search), evaluate the KEPT set with
  #     an INDEPENDENT full Vina 'dock' pass (same mode/exhaustiveness=16 as baseline/affinity_only,
  #     so the comparison is not circular -- see the pre-registration Sec.3 for why that separation
  #     matters). Implementation: variant 'vina_rerank' in guidance/run_task_f.py, a sibling
  #     function that never touches run_triple() -- zero regression risk to the 600 baseline/
  #     affinity_only trials already on disk.
  #
  #     Pilot first, 6 pockets (stratified by measured wall-time across the existing 20 -- see the
  #     pre-registration Sec.6) x 1 seed, so a mechanical or statistical surprise costs ~1h of GPU
  #     time, not ~12h. rq2_full then covers all (pocket, seed) pairs with the SAME out_dir, so its
  #     60 triples -6 the pilot already did = 54 remaining -- run_task_f.py's DONE-marker
  #     resumability means the pilot's results are reused, not recomputed.
  "rq2_pilot|gpu|$PY guidance/run_task_f.py --variants vina_rerank --seeds 1 --samples_per_pocket 10 --oversample_k 2 --num_steps 1000 --batch_size 5 --pockets_file ./guidance/task_f_pockets_pilot6.json"
  "rq2_full|gpu|$PY guidance/run_task_f.py --variants vina_rerank --seeds 1 2 3 --samples_per_pocket 10 --oversample_k 2 --num_steps 1000 --batch_size 5"
  # Regenerates guidance/task_f_analysis.json with the (now three-way) comparisons dict --
  # strictly additive: baseline_vs_affinity_only's numbers are unchanged (verified against the
  # pre-existing file before this step was added), baseline_vs_vina_rerank is new.
  "rq2_analyze|cpu|$PY guidance/analyze_task_f.py"

  # --- todo item 5: MoleculeACE-style activity cliffs (guidance/cheminformatics/activity_cliffs.py).
  #     Correctness verified on a tiny hand-computed example BEFORE this was queued (4 tests in
  #     test_chem.py); the real computation (up to ~1M pairs for the single largest test target) was
  #     deliberately queued rather than run inline because it was written while the CPU was already
  #     the bottleneck for rq2_pilot/rq2_full -- no reason to compete with the run the user asked to
  #     prioritise.
  "activity_cliffs|cpu|$PY guidance/cheminformatics/activity_cliffs.py --b 2000"

  # --- todo item 4: within-target y-randomisation for the EGNN itself (not just the descriptor
  #     pipeline). Pre-emptively verified: 5 tests on a synthetic anchor table pin val/test staying
  #     untouched, the per-target label MULTISET being preserved, determinism, and seed-sensitivity --
  #     see guidance/lp_split/tests/test_withintarget_yrand.py. Same architecture/hyperparameters/split
  #     as the deployed Stage 0 checkpoint; the only difference is which ligand each TRAIN target's own
  #     label is assigned to. Compare its test R2 against the already-measured descriptor-pipeline
  #     benchmark, 0.2468 +/- 0.0019 (guidance/cheminformatics/noise_ceiling_results.json).
  "withintarget_yrand|gpu|$PY guidance/lp_split/train_egnn_stage0_withintarget_yrand.py configs/prop/crossdocked_affinity_egnn.yml --seed 2021"

  # --- todo item 8: Arm A (pure-CrossDocked, matched total row count) + Arm C (production BindingNet,
  #     MUST use the compound-leakage-filtered pool -- train_surrogate_arm.py's compound_filter=True
  #     default already enforces this in-code, verified by reading it before this was queued; CMP_LEAK_
  #     SUMMARY=bindingnet_v1_clean_summary.json exists on disk). Same bs=1/no-AMP regime as the
  #     validated Arm B run -- bs=4+AMP was REJECTED (Arm 0 gate failure, oscillating R2, see TASKS.md).
  #     Arm C mirrors Arm B's exact row composition (cd_rows=6000, bn_rows=14000, --bn_match_pk) MINUS
  #     --no_compound_filter -- that flag existed only so Arm B's seed 2021 (trained pre-filter) stayed
  #     comparable across its own 3 seeds; Arm C has no such constraint and must use the clean pool.
  #     Arm A uses cd_rows=20000 (pure CD, no BN) as the matched-total-row-count comparator, per TASKS.md
  #     item 8's "20.000 CD, 3 seed" spec. Estimated ~16h per arm (3 seeds) at the bs=1 throughput
  #     already measured (11.6 rows/sec) -- ~32h combined, confirmed with the user before queuing given
  #     its scale (guidance/ARM_A_C_SCOPE.md has no separate file; this comment IS the record).
  "armA_s2021|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armA --cd_rows 20000 --batch_size 1 --seed 2021"
  "armA_s2022|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armA --cd_rows 20000 --batch_size 1 --seed 2022"
  "armA_s2023|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armA --cd_rows 20000 --batch_size 1 --seed 2023"
  "armC_s2021|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armC --cd_rows 6000 --bn_rows 14000 --bn_match_pk --batch_size 1 --seed 2021"
  "armC_s2022|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armC --cd_rows 6000 --bn_rows 14000 --bn_match_pk --batch_size 1 --seed 2022"
  "armC_s2023|gpu|$PY guidance/surrogate_data/train_surrogate_arm.py $CFG --arm_tag armC --cd_rows 6000 --bn_rows 14000 --bn_match_pk --batch_size 1 --seed 2023"
  # ONE aggregate call, not one per arm: aggregate_results.py auto-groups every run it finds under
  # logs_surrogate_arms by its own embedded tag (arm0/armBmatch/armA/armC all separately, in one pass).
  # Calling it twice with different --json names would have produced two IDENTICAL files, each
  # misleadingly named as if it only covered one arm.
  "armAC_aggregate|cpu|$PY guidance/aggregate_results.py logs_surrogate_arms --metric R2 --min_epochs 2 --json guidance/surrogate_data/armAC_table.json"

  # --- APPEND HERE as each remaining todo item's script lands: (none left from the original 11-item list)
)
