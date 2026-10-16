"""Generate the complete effect-size register (Appendix F) from the raw per-comparison results.

Audit item F049 asked for every effect size to be published, not only the count of comparisons
that reached significance. The distinction matters for a thesis whose central result is a null:
"none of 24 comparisons was significant" is compatible with 24 effects that are all tiny and with
24 that are large but noisily measured, and only the effect sizes tell the two apart. Publishing
them also lets a later reader pool these comparisons into a meta-analysis, which a table of
verdicts does not permit.

The register is generated rather than written by hand. It restates 71 comparisons that already
exist in machine-readable form, and a hand-copied table of that size would be a near-certain
source of transcription error; this project has already found four such errors in much smaller
tables. The output file carries a header saying it is generated, and regenerating it is part of
the ordinary figure/table refresh.

Run:  PYTHONPATH=. python guidance/thesis/make_effect_size_appendix.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'chapters', '11z_effect_sizes.txt')

#: Each gradient-informativeness run, in the order the thesis presents them.
GRADIENT_RUNS = [
    ('Track A, exploratory tier (Section V.2.3)', 'guidance/track_a_exploratory_results'),
    ('Track A, full tier (Section V.2.4)', 'guidance/track_a_full_results'),
    ('Track B1, gradient-norm normalisation (Section V.3.2)', 'guidance/track_b1_results'),
    ('Track B2, classifier-guidance reformulation (Section V.3.3)', 'guidance/track_b2_results'),
    ('Track B3, timestep-windowed guidance (Section V.3.1)', 'guidance/track_b3_results'),
]

CONDITION_NAMES = {
    'raw_vs_unguided': 'raw vs unguided',
    'normalized_vs_unguided': 'normalised vs unguided',
    'normalized_vs_raw': 'normalised vs raw',
    'classifier_vs_unguided': 'classifier vs unguided',
    'classifier_vs_raw': 'classifier vs raw',
    'uniform': 'uniform window',
    'late_third': 'late third',
    'middle_third': 'middle third',
    'early_third': 'early third',
}


def split_label(label):
    """'BSD_ASPTE_1_130_0:lambda_0.3' -> ('BSD', 'lambda = 0.3')."""
    pocket, _, cond = label.partition(':')
    short = pocket.split('_')[0]
    if cond.startswith('lambda_'):
        return short, 'λ = ' + cond[len('lambda_'):]
    return short, CONDITION_NAMES.get(cond, cond.replace('_', ' '))


def fmt(x, nd=3, signed=False):
    if x is None or x == '':
        return '—'
    v = float(x)
    s = f'{v:+.{nd}f}' if signed else f'{v:.{nd}f}'
    return s.replace('-', '−')


def fmt_p(x):
    v = float(x)
    if v < 1e-4:
        mant, exp = f'{v:.1e}'.split('e')
        return f'{mant} × 10{superscript(int(exp))}'
    return f'{v:.4f}'


def superscript(n):
    digits = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³',
              '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸',
              '9': '⁹'}
    return ''.join(digits[c] for c in str(n))


def gradient_rows():
    out, n_sig = [], 0
    for title, d in GRADIENT_RUNS:
        rows = list(csv.DictReader(open(os.path.join(d, 'gradient_informativeness_results.csv'))))
        for r in rows:
            pocket, cond = split_label(r['label'])
            sig = r['significant_after_correction'] == 'True'
            n_sig += sig
            out.append((title, [
                pocket, cond,
                f"{r['n_unguided']} / {r['n_guided']}",
                fmt(r['ks_stat']),
                fmt(r['mean_shift'], signed=True),
                fmt(r['median_shift'], signed=True),
                fmt(r['ligand_efficiency_shift'], nd=4, signed=True),
                f"{float(r['pb_valid_rate_unguided']):.2f} → {float(r['pb_valid_rate_guided']):.2f}",
                fmt_p(r['ks_p']),
                fmt_p(r['ks_p_bh_corrected']) + ('*' if sig else ''),
            ]))
    return out, n_sig


def track_c_rows():
    d = json.load(open('guidance/track_c_analysis/track_c_analysis_results.json'))
    out, n_sig = [], 0
    for pocket, v in d['per_pocket_primary'].items():
        lo, hi = v['effect_size_ci95']
        sig = bool(v.get('ks_reject_bh'))
        n_sig += sig
        sc, vh = v['size_confound'], v['vina_hacking_check']
        out.append([
            pocket.split('_')[0], str(v['n']), str(v['k']),
            fmt(v['full_pool_mean_vina_dock'], 2), fmt(v['real_topk_mean_vina_dock'], 2),
            f"{fmt(v['effect_size'], signed=True)}  [{fmt(lo, signed=True)}, {fmt(hi, signed=True)}]",
            f"{sc['mean_heavy_atoms_full']:.1f} → {sc['mean_heavy_atoms_topk']:.1f}",
            f"{vh['pb_valid_rate_full']:.2f} → {vh['pb_valid_rate_topk']:.2f}",
            fmt_p(v['ks_pvalue']) + ('*' if sig else ''),
        ])
    return out, n_sig


def main(out_path=None):
    out_path = out_path or OUT
    grad, grad_sig = gradient_rows()
    tc, tc_sig = track_c_rows()
    n_grad = len(grad)

    L = []
    w = L.append
    w('@appendix F | COMPLETE EFFECT-SIZE REGISTER')
    w('')
    w('@sec Why This Register Exists')
    w('')
    w('The results chapters report how many comparisons reached significance after correction, '
      'which is the quantity the decision rule of Section III.6 acts on but is not, on its own, '
      'enough to judge a null result. A count of zero significant comparisons is equally '
      'consistent with effects that are genuinely negligible and with effects that are '
      'substantial but measured too noisily to resolve, and only the effect sizes distinguish '
      'the two. This appendix therefore publishes every comparison behind Chapters V and VI '
      f'individually: all {n_grad} gradient-guidance comparisons in {{tab:esgrad}} and all '
      f'{len(tc)} rejection-sampling pockets in {{tab:estrackc}}. A reader who wishes to pool '
      'these into a meta-analysis, or to check that the null is a null of small effects rather '
      'than of wide intervals, has the per-comparison numbers to do so here.')
    w('')
    w('Both tables are generated directly from the stored result files by '
      '`guidance/thesis/make_effect_size_appendix.py`, not transcribed. At this size a '
      'hand-copied table would be a near-certain source of transcription error, and four such '
      'errors have already been found and corrected in much smaller tables in this thesis.')
    w('')
    w('@sec Gradient Guidance: Every Comparison')
    w('')
    w(f'{{tab:esgrad}} lists all {n_grad} comparisons of Tracks A and B. The shift columns are in '
      'kcal/mol and are oriented so that a negative value is a better (more negative) docking '
      f'score under guidance. {grad_sig} of the {n_grad} comparisons is significant after '
      'Benjamini–Hochberg correction within its own run. The full tier of Track A is also '
      'presented, with its own commentary, in Appendix C; it is repeated here so that this '
      'register is complete on its own.')
    w('')
    w('@tab esgrad | Every gradient-guidance comparison, with effect sizes')
    w('| Run | Pocket | Condition | n (un / gu) | KS D | Mean shift | Median shift | '
      'LE shift | PoseBusters | p (raw) | p (BH) |')
    last = None
    for title, cells in grad:
        run_cell = title.split(',')[0] + (', ' + title.split(',')[1].split('(')[0].strip()
                                          if ',' in title else '')
        w('| ' + ' | '.join([run_cell if title != last else '″'] + cells) + ' |')
        last = title
    w('@note Shifts are guided minus unguided, so a negative mean shift is an improvement. '
      '"LE shift" is the change in ligand efficiency (kcal/mol per heavy atom). "PoseBusters" '
      'gives the pass rate unguided → guided. p (BH) is corrected within each run, which is '
      'the unit the decision rule of Section III.6 applies to; an asterisk marks a comparison '
      'significant at that level. A repeated run name is shown as ″.')
    w('@endtab')
    w('')
    w('@sec Rejection Sampling: Every Pocket')
    w('')
    w(f'{{tab:estrackc}} lists all {len(tc)} pockets of Track C (Chapter VI) with the effect size '
      'and its 95% bootstrap interval, alongside the two quantities that determine how the effect '
      'should be read: the change in mean heavy-atom count, which is the size confound, and the '
      'change in PoseBusters pass rate. The effect size is positive when the selected top 10% has '
      'a *worse* (less negative) mean docking score than the full pool.')
    w('')
    w('@tab estrackc | Every Track C pocket, with effect size, interval and confounds')
    w('| Pocket | n pool | k selected | Vina (pool) | Vina (top 10%) | Effect size [95% CI] | '
      'Heavy atoms | PoseBusters | KS p |')
    for cells in tc:
        w('| ' + ' | '.join(cells) + ' |')
    w('@note Vina Dock in kcal/mol, lower is better. The effect size is the difference in mean '
      'Vina Dock between the selected subset and the full pool, in kcal/mol, with a bootstrap 95% '
      'interval. Heavy atoms and PoseBusters are given as full pool → selected subset; the '
      'systematic rise in the first and fall in the second are the confounds analysed in Section '
      f'VI.2.2. An asterisk marks a pocket significant after correction ({tc_sig} of {len(tc)}).')
    w('@endtab')
    w('')

    header = ('# GENERATED FILE -- do not edit by hand.\n'
              '# Regenerate with: PYTHONPATH=. python guidance/thesis/make_effect_size_appendix.py\n'
              '# Source: the gradient_informativeness_results.csv of each track directory and\n'
              '#         guidance/track_c_analysis/track_c_analysis_results.json\n')
    body = '\n'.join(L) + '\n'
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(header + body)
    print(f'wrote {out_path}: {n_grad} gradient comparisons ({grad_sig} significant), '
          f'{len(tc)} Track C pockets ({tc_sig} significant)')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='write here instead of the chapter file')
    main(ap.parse_args().out)
