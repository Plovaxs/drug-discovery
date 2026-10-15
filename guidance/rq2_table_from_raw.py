"""Regenerate the positive-control (RQ2) results table from the per-molecule result files.

This is the most consequential table in the thesis: it is the one result where guidance-like
intervention produced a significant, size-normalised improvement, and the interpretation of several
chapters rests on it. It was originally assembled by hand, and a later audit found that four of its
rows named the wrong statistical test (they were labelled "KS" when the quoted p-values came from a
pocket-paired Wilcoxon signed-rank test, the only paired test the analysis ever computed). The values
themselves were correct, but the methodology column was not.

Emitting the table from the raw files removes the hand-assembly step that allowed that. The output
JSON feeds guidance/thesis/check_numbers_against_sources.py, so a future edit that retypes a figure
fails the build rather than reaching a reader.

Run:  PYTHONPATH=. python guidance/rq2_table_from_raw.py
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

RESULTS = 'guidance/task_f_results'
METRICS = ['vina_dock', 'sa', 'ra_score', 'qed', 'ligand_efficiency', 'heavy_atoms', 'pb_valid']


def load(variant):
    files = sorted(glob.glob(os.path.join(RESULTS, variant, '*', 'honest_eval.csv')))
    if not files:
        raise SystemExit(f'no result files for variant {variant!r} under {RESULTS}')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired_by_pocket(a, b, metric):
    """Wilcoxon on per-pocket means. The pocket is the unit of replication throughout this thesis, so
    the paired test is the one quoted; the unpaired Mann-Whitney is reported alongside, not instead."""
    ka = a.groupby('pocket_target')[metric].mean()
    kb = b.groupby('pocket_target')[metric].mean()
    common = sorted(set(ka.index) & set(kb.index))
    if len(common) < 3:
        return None, len(common)
    stat = wilcoxon(ka.loc[common].values, kb.loc[common].values)
    return float(stat.pvalue), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='./guidance/rq2_table.json')
    args = ap.parse_args()

    base, rer = load('baseline'), load('vina_rerank')
    out = {'n_baseline': int(len(base)), 'n_reranked': int(len(rer)), 'metrics': {}}
    for m in METRICS:
        if m not in base.columns:
            continue
        bb, rr = base[m].dropna(), rer[m].dropna()
        mw = mannwhitneyu(bb, rr, alternative='two-sided').pvalue
        wp, npock = paired_by_pocket(base, rer, m)
        out['metrics'][m] = {
            'baseline_mean': round(float(bb.mean()), 3),
            'baseline_sd': round(float(bb.std()), 3),
            'reranked_mean': round(float(rr.mean()), 3),
            'reranked_sd': round(float(rr.std()), 3),
            'mannwhitney_p': float(mw),
            'wilcoxon_paired_p': wp,
            'n_pockets_paired': int(npock),
            'test_quoted_in_thesis': 'Wilcoxon (paired by pocket)',
        }

    print(f"{'metric':18s} {'baseline':>18s} {'reranked':>18s} {'wilcoxon_p':>12s}")
    for m, v in out['metrics'].items():
        wp = v['wilcoxon_paired_p']
        print(f"{m:18s} {v['baseline_mean']:9.3f}±{v['baseline_sd']:<8.3f} "
              f"{v['reranked_mean']:9.3f}±{v['reranked_sd']:<8.3f} "
              f"{(f'{wp:.3g}' if wp is not None else 'n/a'):>12s}")
    print("\nEvery p-value above is a pocket-paired Wilcoxon signed-rank test. The analysis computed no "
          "Kolmogorov-Smirnov test for these metrics; any table describing them as KS is mislabelled.")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
