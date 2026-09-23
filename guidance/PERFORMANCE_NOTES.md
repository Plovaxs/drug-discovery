# Performance Notes: Track C Optimization Batch (2026-09-18/19)

Records the speedups measured and adopted for Track C's remaining run
(and any future Track-C-scale generation+docking work), per the
optimization-tracking plan.

## Item 4: docking-side CPU parallelism (`eval/honest_eval_parallel.py`)

**Measured baseline**: a single Vina docking call at this project's
standard `exhaustiveness=8` uses ~810% CPU on this 20-core machine (~8
cores), leaving ~12 cores (60%) idle. Exhaustiveness caps how many cores
*one* docking call can use; nothing previously ran multiple molecules'
docking concurrently to use the rest.

**Change**: `evaluate_molecules_parallel` chunks the molecule list across
`n_workers` processes, each running the existing, unmodified
`eval.honest_eval.evaluate_molecules` on its chunk -- each individual
molecule's docking call is byte-for-byte identical to the sequential
version (same exhaustiveness, same Vina invocation). `n_workers=2`
adopted (targets ~16/20 cores, leaving 4 for the OS/GPU-side process).

**Verified correctness**: smoke test with `docking_n_workers=2` produced
identical columns/format to the sequential version, `vina_dock` values in
the expected range, no errors.

## Item 1: mixed precision (AMP, bf16 autocast)

**Change**: `guidance/guided_sampling.py`'s `sample_diffusion_ligand_guided`
gained a `use_amp` flag (default `False`, preserving every prior stage's
exact fp32 behavior). When enabled, only the diffusion core's own forward
pass (`model(...)`) runs under `torch.autocast(dtype=bfloat16)` -- the
posterior math right after (`_predict_x0_from_eps`, `q_pos_posterior`,
etc.) stays fp32 regardless, and the model's output is explicitly cast
back to fp32 before those steps to avoid ambiguous type promotion.

**Measured speedup**: sampling throughput went from ~4.4 it/s (fp32
baseline) to ~5.0-5.02 it/s (bf16 autocast) on this pocket -- **~14%
faster** for the sampling phase specifically.

**Quality verification** (n=16 smoke test, pocket BSD_ASPTE, vs. the two
existing fp32 baseline pools on the same pocket, n≈300 each):

| | n | vina_dock mean | vina_dock std | pb_valid rate |
|---|---|---|---|---|
| AMP + parallel docking | 16 | -7.946 | 1.082 | 0.312 |
| fp32 baseline (seed2021) | 299 | -7.817 | 1.332 | 0.355 |
| fp32 baseline (seed2022) | 295 | -7.840 | 1.135 | 0.346 |

The AMP-vs-fp32 gap (~0.1-0.13 mean shift) is well within the natural
run-to-run variation already present between the two fp32 baselines
themselves at this much smaller sample size (n=16 vs n≈300) -- no
concerning shift. **Adopted for the remainder of Track C.**

## Item 2: batch size re-tuning

Not changed. Direct benchmark (100 steps, batch_size 4/8/16) showed only
~3% per-sample throughput improvement scaling batch size up, confirming
the sampling loop is compute-bound on this GPU, not overhead-bound.
Combined with AMP's modest memory savings, this was not revisited further
-- the ~3% ceiling doesn't justify the added OOM risk at larger batch
sizes on a 4GB card running two processes at times.

## Item 3: `torch.compile()`

**Not attempted.** Per the addendum's explicit risk/effort ordering, this
carries real risk of graph-break issues (this diffusion sampling loop has
per-step conditional branching on `model_mean_type`, timestep windows,
etc.) and extended troubleshooting time -- deferred indefinitely unless a
future need specifically motivates revisiting it.

## Combined effect on Track C's remaining timeline

Per-pool time reduction: sampling ~4.5h -> ~3.95h (AMP, ~14% faster),
docking ~1.25h -> ~0.625h (2-worker parallelism, ~2x faster) => per-pool
total ~5.75h -> ~4.57h, roughly a **20% reduction** in per-pool wall-clock
time, on top of the separate scope reduction (60 -> 30 total pools, of
which 5 were already complete and retained -- see
`guidance/STAGE2_PLUS_EXPERIMENT_LOG.md`'s Track C scope-change entry for
that decision's own rationale).
