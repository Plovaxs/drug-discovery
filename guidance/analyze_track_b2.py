"""Track B2 (classifier-guidance reformulation) analysis. Compares the
newly-generated 'classifier' condition against Track B1's already-
generated 'unguided' and 'raw' (Stage 0 regression guidance) conditions
for the same 3 pockets/seed/settings -- reused directly rather than
regenerated (see guidance/track_b2_classifier.py's module docstring).

3 pockets x 2 comparisons (classifier-vs-unguided, classifier-vs-raw) = 6
tests, BH-corrected together as one checkpoint's worth of tests.
"""
import pandas as pd

from guidance.gradient_informativeness_test import run_checkpoint_batch

B1_RESULTS_DIR = './guidance/track_b1_results'
B2_RESULTS_DIR = './guidance/track_b2_results'
POCKETS = [
    (0, 'BSD_ASPTE_1_130_0'),
    (5, 'HDAC8_HUMAN_1_377_0'),
    (10, 'CD38_HUMAN_44_300_0'),
]


def main():
    comparisons = []
    for pocket_id, pocket_target in POCKETS:
        unguided_df = pd.read_csv(f'{B1_RESULTS_DIR}/unguided/pocket{pocket_id}/honest_eval.csv')
        raw_df = pd.read_csv(f'{B1_RESULTS_DIR}/raw/pocket{pocket_id}/honest_eval.csv')
        classifier_df = pd.read_csv(f'{B2_RESULTS_DIR}/classifier/pocket{pocket_id}/honest_eval.csv')

        comparisons.append((f'{pocket_target}:classifier_vs_unguided', unguided_df, classifier_df))
        comparisons.append((f'{pocket_target}:classifier_vs_raw', raw_df, classifier_df))

    result = run_checkpoint_batch(comparisons, alpha=0.05)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print(result.to_string(index=False))
    result.to_csv(f'{B2_RESULTS_DIR}/gradient_informativeness_results.csv', index=False)
    print(f'\nSaved to {B2_RESULTS_DIR}/gradient_informativeness_results.csv')


if __name__ == '__main__':
    main()
