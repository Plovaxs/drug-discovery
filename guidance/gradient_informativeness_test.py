"""The mandatory gradient-informativeness gate, per
guidance/FLAGSHIP_ARCHITECTURE_RESEARCH.md's Stage 2+ addendum (Sec 1).

Detects, cheaply, whether a (predictor, guidance-mechanism) pairing shifts
the *distribution* of real docking outcomes -- an early-warning/triage
signal, not a replacement for a full Task-F-style significance-tested
ablation (Sec 1.7). Applied identically at every checkpoint from Stage 2
onward and to every Track B variant so results stay comparable.

Reports (Sec 1.3):
  - two-sample Kolmogorov-Smirnov test (guided vs unguided real Vina Dock
    scores), per (pocket, lambda) or (pocket, condition) combination.
  - directional effect size (mean/median shift), since KS alone does not
    say which way a detected difference goes.
  - ligand-efficiency and PoseBusters-pass-rate shift, to distinguish a
    genuine guidance effect from a Vina-hacking-pattern artifact.
Applies Benjamini-Hochberg FDR correction (Sec 1.4) across all tests run
within one checkpoint before any p-value is interpreted as significant.
"""
import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns (corrected_pvals, reject) arrays, same order as input.
    Standard BH step-up procedure."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    corrected_sorted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (step-up)
    corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
    corrected_sorted = np.clip(corrected_sorted, 0, 1)
    corrected = np.empty(n)
    corrected[order] = corrected_sorted
    reject = corrected <= alpha
    return corrected, reject


def compare_distributions(unguided_df, guided_df, label=''):
    """unguided_df / guided_df: honest_eval-style DataFrames (columns incl.
    vina_dock, ligand_efficiency, pb_valid). Returns a dict with the raw
    (uncorrected) KS p-value and all the auxiliary quality-degradation
    signals for one (pocket, condition) comparison -- correction across
    multiple such dicts happens one level up, in run_checkpoint_batch."""
    u = unguided_df['vina_dock'].dropna().to_numpy()
    g = guided_df['vina_dock'].dropna().to_numpy()
    if len(u) < 2 or len(g) < 2:
        return {'label': label, 'n_unguided': len(u), 'n_guided': len(g), 'ks_p': None,
               'mean_shift': None, 'median_shift': None, 'direction': 'insufficient_data'}

    ks_stat, ks_p = stats.ks_2samp(u, g)
    mean_shift = float(np.mean(g) - np.mean(u))  # negative = guided more negative = "better" binding
    median_shift = float(np.median(g) - np.median(u))
    direction = 'correct (more negative/better)' if mean_shift < 0 else (
        'wrong (more positive/worse)' if mean_shift > 0 else 'no_shift')

    le_u = unguided_df['ligand_efficiency'].dropna()
    le_g = guided_df['ligand_efficiency'].dropna()
    le_shift = float(le_g.mean() - le_u.mean()) if len(le_u) and len(le_g) else None
    le_degraded = bool(le_shift is not None and le_shift > 0)  # more positive (less negative) = worse efficiency

    pb_u = unguided_df['pb_valid'].fillna(False).mean() if 'pb_valid' in unguided_df else None
    pb_g = guided_df['pb_valid'].fillna(False).mean() if 'pb_valid' in guided_df else None
    pb_degraded = bool(pb_u is not None and pb_g is not None and pb_g < pb_u - 0.05)

    return {
        'label': label, 'n_unguided': len(u), 'n_guided': len(g),
        'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
        'mean_shift': mean_shift, 'median_shift': median_shift, 'direction': direction,
        'ligand_efficiency_shift': le_shift, 'ligand_efficiency_degraded': le_degraded,
        'pb_valid_rate_unguided': pb_u, 'pb_valid_rate_guided': pb_g, 'pb_degraded': pb_degraded,
    }


def run_checkpoint_batch(comparisons, alpha=0.05):
    """comparisons: list of (label, unguided_df, guided_df). Runs
    compare_distributions on each, then applies BH correction across all
    KS p-values in this batch (one checkpoint's worth of tests) before
    returning. Returns a DataFrame, one row per comparison, with both raw
    and BH-corrected p-values and a per-row verdict."""
    rows = [compare_distributions(u, g, label=lbl) for lbl, u, g in comparisons]
    df = pd.DataFrame(rows)

    valid = df['ks_p'].notna()
    if valid.sum() > 0:
        corrected, reject = benjamini_hochberg(df.loc[valid, 'ks_p'].to_numpy(), alpha=alpha)
        df.loc[valid, 'ks_p_bh_corrected'] = corrected
        df.loc[valid, 'significant_after_correction'] = reject
    else:
        df['ks_p_bh_corrected'] = None
        df['significant_after_correction'] = False

    def verdict(row):
        if not row.get('significant_after_correction', False):
            return 'no_signal'
        if row['direction'].startswith('wrong'):
            return 'signal_wrong_direction'
        if row.get('ligand_efficiency_degraded') or row.get('pb_degraded'):
            return 'signal_but_quality_degraded'
        return 'clear_signal'
    df['verdict'] = df.apply(verdict, axis=1)
    return df
