"""Audit item F051: write the expanded (n=20) ChEMBL cross-validation table into the appendix.

The four-target check finds one real discrepancy (CD38) and three agreements. With four targets
that is not enough to say whether CD38 is a pattern or a one-off. This generates the follow-up
section from `guidance/chembl_crossvalidation_expanded_results.json`, built by
`guidance/chembl_f051_sample.py` (the local, reproducible half: which 20 targets were drawn) and a
manual ChEMBL retrieval recorded in that file's own metadata (the half that cannot be re-run from
inside this repository as a script).

Run:  PYTHONPATH=. python guidance/thesis/make_chembl_expansion.py
"""
import json

OUT = 'guidance/thesis/fragments/chembl_expansion.txt'  # NOT under chapters/ -- see module docstring
SRC = 'guidance/chembl_crossvalidation_expanded_results.json'


def fmt_pk(lo, hi, n):
    return f'{lo:.2f}–{hi:.2f} ({n})'


def fmt_chembl(row):
    if row['status'] == 'resolved_zero_ki_data':
        return 'no Ki-type activity in ChEMBL'
    if row['status'] == 'organism_or_gene_not_confidently_resolved':
        return 'not resolved (excluded)'
    return (f"{row['chembl_pchembl_min']:.2f}–{row['chembl_pchembl_max']:.2f} "
            f"({row['chembl_ki_n_pchembl']} / {row['chembl_ki_n_returned']})")


def verdict(row):
    return {'overlap': 'Overlap', 'no_overlap': 'No overlap',
            'n/a': '—'}.get(row['agreement'], row['agreement'])


def main(out_path=None):
    out_path = out_path or OUT
    d = json.load(open(SRC))
    rows = d['rows']
    s = d['summary']

    L = []
    w = L.append
    w('@sub Was CD38 a pattern or a one-off? An expanded sample')
    w('')
    w(f"The four-target check above is not a sample; the targets were chosen because they recur "
      f"throughout this appendix, and four points cannot say whether the one disagreement found "
      f"(CD38) is a pattern or a one-off. A further {d['n_sampled']} targets were drawn by seeded "
      f"random sampling from the {d['n_targets_in_pool']}-target leakage-safe test set of Chapter "
      f"IV (seed {d['seed']}, `guidance/chembl_f051_sample.py`) and checked the same way: this "
      f"project's own pK range for the target against ChEMBL's independently curated Ki range, "
      f"retrieved {d['retrieval_date']}, via {d['retrieval_method']}.")
    w('')
    w(f"Of the {d['n_sampled']} targets drawn, {s['n_not_resolved']} could not be mapped to a "
      f"single ChEMBL target with confidence and were excluded rather than guessed at: one "
      f"carried an organism code this check could not resolve to a specific species without risking "
      f"a cross-species mismatch, and two returned no plausible ChEMBL target at all under a "
      f"gene-symbol or name search. Of the {s['n_resolved_with_ki_data'] + s['n_resolved_zero_ki_data']} "
      f"that resolved, {s['n_resolved_zero_ki_data']} have no Ki-type measurement in ChEMBL at all, "
      f"which is itself a finding about coverage rather than a failure of this check: ChEMBL's "
      f"curation of some targets, such as a lectin typically assayed by fluorescence polarisation "
      f"rather than classical binding, or targets with little medicinal-chemistry history, is "
      f"sparse for the specific activity type (Ki) this comparison requires, independent of "
      f"whether the target is well studied by other measures.")
    w('')
    w(f"Of the {s['n_resolved_with_ki_data']} targets with ChEMBL Ki data, "
      f"{s['n_overlap']} of {s['n_resolved_with_ki_data']} show this project's pK range overlapping "
      f"ChEMBL's independently curated range. CD38 is not a one-off: a second, independent "
      f"disagreement of the same shape appears in {rows['RNAS2_HUMAN_27_161_0']['mnemonic']} "
      f"(non-secretory ribonuclease / RNase 2), where this project's labels are uniformly weak "
      f"(pK {rows['RNAS2_HUMAN_27_161_0']['our_pk_min']:.2f}–"
      f"{rows['RNAS2_HUMAN_27_161_0']['our_pk_max']:.2f}, n = "
      f"{rows['RNAS2_HUMAN_27_161_0']['our_n']}) while ChEMBL's only two Ki entries for this "
      f"target, from the same published inhibitor series, are both sub-micromolar "
      f"(pChEMBL {rows['RNAS2_HUMAN_27_161_0']['chembl_pchembl_min']:.2f}), "
      f"shown in {{fig:chemblexp}}. The explanation "
      f"proposed for CD38 fits here too and is strengthened by recurring: a crystallised complex's "
      f"co-crystallisation ligand need not be the most potent compound known against that target, "
      f"while ChEMBL's curation favours published medicinal-chemistry series that report their "
      f"best analogues. At {s['n_overlap']} of {s['n_resolved_with_ki_data']} targets overlapping, "
      f"the large majority of this project's labels agree with an independent source; the "
      f"PDBBind/crystallography-versus-ChEMBL/medicinal-chemistry divergence is real but occasional, "
      f"consistent with the single occurrence found in the four-target check and the same single "
      f"additional occurrence found here.")
    w('')
    w('@tab fuchemblexp | Expanded ChEMBL cross-validation: all resolved targets of the 20-target random sample')
    w('| Target (CrossDocked mnemonic) | Our pK range (n) | ChEMBL pChEMBL range, Ki (n with Ki / n returned) | Agreement |')
    order = sorted(rows, key=lambda t: (rows[t]['status'] != 'resolved_with_data', t))
    for t in order:
        r = rows[t]
        w(f"| {r['mnemonic']} | {fmt_pk(r['our_pk_min'], r['our_pk_max'], r['our_n'])} | "
          f"{fmt_chembl(r)} | {verdict(r)} |")
    w(f"@note Source: `{SRC}`, generated by `guidance/chembl_f051_sample.py` (local sampling, "
      f"reproducible) combined with a ChEMBL Ki retrieval on {d['retrieval_date']} (not re-runnable "
      f"from inside this repository as a script; target ChEMBL IDs are recorded in the source file "
      f"for direct re-query). ChEMBL ranges for the four best-studied targets (BTK, Factor XI, "
      f"KIT, ROCK1) are drawn from the first 1,000 Ki records ChEMBL returns and may understate the "
      f"true range if more than 1,000 exist; this can only narrow, not manufacture, an apparent "
      f"disagreement. ‘No Ki-type activity’ means ChEMBL returned zero records of activity "
      f"type Ki for that target specifically, not that the target is absent from ChEMBL or unstudied.")
    w('@endtab')
    w('')
    w('@fig chemblexp | guidance/thesis/figures/fig_chembl_expansion.png | 6.2 | '
      "This project's pK range (blue) against ChEMBL's independently curated Ki range "
      '(grey, or red where the two ranges fail to overlap), for every target of '
      '{tab:fuchemblexp} with ChEMBL Ki data, sorted by this project\'s own minimum. '
      'A range with a single measurement, or several identical ones, is drawn as a dot '
      'rather than a disappearing zero-length line. Twelve of the thirteen ranges touch; '
      'RNAS2 is the exception, and it fails to overlap by nearly two pK units, the same '
      "scale and the same direction (this project reads weak, ChEMBL reads potent) as "
      "the CD38 discrepancy of {fig:fuchembl}.")
    w('')

    body = '\n'.join(L) + '\n'
    header = ('# GENERATED FILE -- do not edit by hand.\n'
              '# Regenerate with: PYTHONPATH=. python guidance/thesis/make_chembl_expansion.py\n'
              f'# Source: {SRC}\n')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(header + body)
    print(f'wrote {out_path}: {s}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    main(ap.parse_args().out)
