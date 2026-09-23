"""Draft, NOT YET deployed to any live run -- prepared per the "batch of
speed optimizations, test at next natural pause" addendum (item 4:
docking-side CPU parallelism).

Measured finding (2026-09-18): a single Vina docking call at this
project's standard exhaustiveness=8 uses ~810% CPU on a 20-core machine
(~8 cores), leaving ~12 cores (60%) idle -- exhaustiveness caps how many
cores ONE docking call can use, but nothing currently runs multiple
molecules' docking calls concurrently to use the rest. This wrapper
parallelizes across MOLECULES (each individual molecule's docking call is
byte-for-byte identical to the sequential version -- same exhaustiveness,
same Vina invocation, same everything) by chunking the molecule list
across worker processes, each running the existing, unmodified
eval.honest_eval.evaluate_molecules on its chunk.

Worker count is deliberately conservative (2 by default) rather than
maximal (e.g. 20/8=2.5) to leave headroom for the OS and any concurrent
GPU-side Python process (e.g. the sampling loop that produced these
molecules, if not yet finished) -- tune upward only after confirming no
regression at the pause-time test, per the addendum's protocol.

NOT tested against a live run yet -- do not import/use this in any
production script until the batched pause-time test (items 1-4 together)
has verified correctness and measured speedup, per the addendum's working
rule.
"""
import multiprocessing as mp

import pandas as pd

from eval.honest_eval import evaluate_molecules


def _worker(args):
    mols_chunk, protein_path, docking_mode, exhaustiveness, reference_sdf, run_posebusters, verbose, skip_score_and_minimize = args
    return evaluate_molecules(
        mols_chunk, protein_path, docking_mode=docking_mode, exhaustiveness=exhaustiveness,
        reference_sdf=reference_sdf, run_posebusters=run_posebusters, verbose=verbose,
        skip_score_and_minimize=skip_score_and_minimize,
    )


def evaluate_molecules_parallel(mols, protein_path, docking_mode='vina_dock', exhaustiveness=8,
                                reference_sdf=None, run_posebusters=True, verbose=False,
                                skip_score_and_minimize=False, n_workers=2):
    """Same signature/return shape as eval.honest_eval.evaluate_molecules
    (a DataFrame with one row per molecule, same columns, same values --
    each molecule's own docking is byte-for-byte identical to the
    sequential version) but splits `mols` across `n_workers` processes to
    use CPU cores that a single exhaustiveness=8 Vina call leaves idle.

    n_workers: keep at 2 (the addendum's conservative default) unless the
    pause-time test confirms headroom for more -- exhaustiveness=8 uses
    ~8 cores per call, so n_workers=2 targets ~16/20 cores, leaving 4 for
    the OS and any concurrent process.
    """
    if len(mols) == 0:
        return evaluate_molecules(mols, protein_path, docking_mode, exhaustiveness,
                                  reference_sdf, run_posebusters, verbose, skip_score_and_minimize)

    n_workers = min(n_workers, len(mols))
    chunks = [mols[i::n_workers] for i in range(n_workers)]
    chunks = [c for c in chunks if len(c) > 0]

    args_list = [
        (chunk, protein_path, docking_mode, exhaustiveness, reference_sdf, run_posebusters, verbose,
         skip_score_and_minimize)
        for chunk in chunks
    ]

    with mp.get_context('spawn').Pool(processes=len(chunks)) as pool:
        results = pool.map(_worker, args_list)

    return pd.concat(results, ignore_index=True)
