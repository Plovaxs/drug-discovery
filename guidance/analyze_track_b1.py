"""Track B1 (gradient-norm normalization) analysis: the mandatory
three-way design (addendum Sec 1.6-1.7) -- per pocket, compares
raw-vs-unguided, normalized-vs-unguided, AND normalized-vs-raw. The
decision criterion hinges specifically on normalized-vs-raw: the
gradient-norm-normalization hypothesis is supported only if that
comparison shows a materially different result, not merely if either
condition differs from the unguided baseline alone.

All 3 pockets x 3 comparison-pairs = 9 tests are BH-corrected together as
one checkpoint's worth of tests, per Sec 1.4.
"""
import pandas as pd

from guidance.gradient_informativeness_test import run_checkpoint_batch

RESULTS_DIR = './guidance/track_b1_results'
POCKETS = [
    (0, 'BSD_ASPTE_1_130_0'),
    (5, 'HDAC8_HUMAN_1_377_0'),
    (10, 'CD38_HUMAN_44_300_0'),
]


def load(condition, pocket_id):
    return pd.read_csv(f'{RESULTS_DIR}/{condition}/pocket{pocket_id}/honest_eval.csv')


def main():
    comparisons = []
    for pocket_id, pocket_target in POCKETS:
        unguided_df = load('unguided', pocket_id)
        raw_df = load('raw', pocket_id)
        normalized_df = load('normalized', pocket_id)

        comparisons.append((f'{pocket_target}:raw_vs_unguided', unguided_df, raw_df))
        comparisons.append((f'{pocket_target}:normalized_vs_unguided', unguided_df, normalized_df))
        # The decisive comparison per the addendum's Sec 1.7: does
        # normalization change the result relative to the raw gradient,
        # not just relative to doing nothing.
        comparisons.append((f'{pocket_target}:normalized_vs_raw', raw_df, normalized_df))

    result = run_checkpoint_batch(comparisons, alpha=0.05)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print(result.to_string(index=False))
    result.to_csv(f'{RESULTS_DIR}/gradient_informativeness_results.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/gradient_informativeness_results.csv')


if __name__ == '__main__':
    main()
