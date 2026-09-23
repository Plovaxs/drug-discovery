"""Track A exploratory-tier analysis: applies the mandatory
gradient-informativeness gate (guidance/gradient_informativeness_test.py)
to the GIGN+PIGNet2 checkpoint's results
(guidance/track_a/track_a_exploratory_checkpoint.py, phase=main).

Per pocket, compares each lambda (0.1, 0.3, 1.0) against that pocket's own
unguided baseline (same seed). All KS tests across 3 pockets x 3 lambdas
(9 tests) are BH-corrected together as one checkpoint's worth of tests,
per the addendum's Sec 1.4.
"""
import pandas as pd

from guidance.gradient_informativeness_test import run_checkpoint_batch

RESULTS_DIR = './guidance/track_a_exploratory_results'
POCKETS = [
    (0, 'BSD_ASPTE_1_130_0'),
    (5, 'HDAC8_HUMAN_1_377_0'),
    (10, 'CD38_HUMAN_44_300_0'),
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
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(result.to_string(index=False))
    result.to_csv(f'{RESULTS_DIR}/gradient_informativeness_results.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/gradient_informativeness_results.csv')


if __name__ == '__main__':
    main()
