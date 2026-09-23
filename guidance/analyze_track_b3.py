"""Track B3 analysis: applies the mandatory gradient-informativeness gate
(guidance/gradient_informativeness_test.py) to the timestep-windowed
guidance results produced by guidance/track_b3_timestep_window.py.

Per pocket, compares each windowed condition (uniform, late_third,
middle_third, early_third) against that pocket's own unguided baseline
(same seed, so this isolates the guidance-window intervention). All
KS tests across both pockets and all 4 conditions (8 tests total) are
BH-corrected together as one checkpoint's worth of tests, per the
addendum's Sec 1.4.
"""
import pandas as pd

from guidance.gradient_informativeness_test import run_checkpoint_batch

RESULTS_DIR = './guidance/track_b3_results'
POCKETS = [
    (0, 'BSD_ASPTE_1_130_0'),
    (5, 'HDAC8_HUMAN_1_377_0'),
]
CONDITIONS = ['uniform', 'late_third', 'middle_third', 'early_third']


def load(condition, pocket_id):
    return pd.read_csv(f'{RESULTS_DIR}/{condition}/pocket{pocket_id}/honest_eval.csv')


def main():
    comparisons = []
    for pocket_id, pocket_target in POCKETS:
        unguided_df = load('unguided', pocket_id)
        for condition in CONDITIONS:
            guided_df = load(condition, pocket_id)
            label = f'{pocket_target}:{condition}'
            comparisons.append((label, unguided_df, guided_df))

    result = run_checkpoint_batch(comparisons, alpha=0.05)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    print(result.to_string(index=False))
    result.to_csv(f'{RESULTS_DIR}/gradient_informativeness_results.csv', index=False)
    print(f'\nSaved to {RESULTS_DIR}/gradient_informativeness_results.csv')


if __name__ == '__main__':
    main()
