"""Track A FULL-TIER analysis (A2-full-checkpoint): the culminating
gradient-informativeness test of the whole Stage 2+ phase. Compares each
lambda (0.1, 0.3, 1.0) against that pocket's own unguided baseline, across
all 8 pockets (the same diverse pocket set used throughout this phase) --
8 pockets x 3 lambdas = 24 tests, BH-corrected together as one
checkpoint's worth of tests, per the addendum's Sec 1.4.

This determines which of the addendum's Part 3 decision bands applies:
3.1 (clean null, matching Track B's 3/3 null result -- completes the
dual-falsification requirement), 3.2 (genuine robust positive -- requires
extensive re-verification before trusting), or 3.3 (ambiguous/mixed).
"""
import pandas as pd

from guidance.gradient_informativeness_test import run_checkpoint_batch

RESULTS_DIR = './guidance/track_a_full_results'
POCKETS = [
    (0, 'BSD_ASPTE_1_130_0'),
    (5, 'HDAC8_HUMAN_1_377_0'),
    (10, 'CD38_HUMAN_44_300_0'),
    (17, 'MENE_BACSU_2_486_0'),
    (20, 'NQO1_HUMAN_2_274_0'),
    (25, 'PAK4_HUMAN_291_591_ATP_0'),
    (30, 'PNTM_STRAE_2_398_0'),
    (35, 'RIBB_VIBCH_2_218_0'),
]
LAMBDAS = [0.1, 0.3, 1.0]


def load(condition, pocket_id):
    return pd.read_csv(f'{RESULTS_DIR}/{condition}/pocket{pocket_id}/honest_eval.csv')


def main():
    comparisons = []
    for pocket_id, pocket_target in POCKETS:
        unguided_df = load('unguided', pocket_id)
        for lam in LAMBDAS:
            guided_df = load(f'lambda_{lam}', pocket_id)
            label = f'{pocket_target}:lambda_{lam}'
            comparisons.append((label, unguided_df, guided_df))

    result = run_checkpoint_batch(comparisons, alpha=0.05)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print(result.to_string(index=False))
    result.to_csv(f'{RESULTS_DIR}/gradient_informativeness_results.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/gradient_informativeness_results.csv')

    n_sig = result['significant_after_correction'].sum()
    print(f'\n{n_sig}/{len(result)} comparisons significant after BH correction.')
    if n_sig > 0:
        sig_rows = result[result['significant_after_correction']]
        print(sig_rows[['label', 'ks_p_bh_corrected', 'direction', 'ligand_efficiency_degraded', 'pb_degraded']].to_string(index=False))


if __name__ == '__main__':
    main()
