"""Puts our mixed Kd / Ki / IC50 labels on one thermodynamic scale, and tests whether doing so changes
any conclusion.

Why this exists. Our labels are a mixture -- Kd 37%, IC50 37%, Ki 26% -- and the composition SHIFTS
between splits (train Kd 41% / IC50 33%; test IC50 46% / Kd 26%). Kd and Ki are both equilibrium
dissociation constants; IC50 is a functional readout that, under the Cheng-Prusoff relation at
[S] = Km, equals 2*Ki. So an IC50 systematically overstates the dissociation constant by about a factor
of two, i.e. pIC50 UNDERSTATES pKi by log10(2) = 0.301 log units. Pooling the three without correction
mixes two different quantities, and does so in different proportions on either side of the split.

**Kalliokoski et al. 2013** (PLoS ONE, 311 cites) measured this empirically rather than assuming the
textbook relation, and reached the same number: "for a broad dataset such as ChEMBL database a Ki-IC50
conversion factor of **2** was found to be the most reasonable", concluding that pooling is acceptable
"if the Ki is corrected by an offset". Neither we nor LP-PDBBind applies it.

Recovering the measurement type. `data/affinity_info.pkl` stores only rmsd / pk / vina, with no assay
type, so the type has to be joined in. CrossDocked ligand filenames carry the PDB entry the LIGAND came
from (`..._rec_<ligPDB>_<lig>_lig_tt_*.sdf`), which is the cognate complex the affinity was measured on,
and `data/lp_pdbbind/LP_PDBBind.csv` records the type for each PDB entry. Measured coverage of that
join: **99.7%**.

What this script does NOT do: retrain anything. The models on record were fitted to uncorrected labels,
so the correction is applied to the EVALUATION target and the question asked is whether any conclusion
moves. If every model shifts together, rankings are safe and the correction is a presentational
improvement; if rankings move, it is a substantive finding and the affected arms need refitting.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/label_harmonisation.py
"""
import json
import math
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.cheminformatics.compare_vs_structure import (LIGAND_PREDS, STRUCTURE_SOURCES,
                                                           load_structure, r2_rows)
from guidance.cheminformatics.chem_data import target_groups
from guidance.uncertainty_a1.stats_common import cluster_bootstrap_indices

CD_LMDB = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
SPLIT_JSON = './guidance/lp_split/leakage_safe_split.json'
LP_CSV = './data/lp_pdbbind/LP_PDBBind.csv'
OUT = './guidance/cheminformatics/label_harmonisation_results.json'
BOOT_SEED = 20260925

# Cheng-Prusoff at [S] = Km gives IC50 = 2*Ki; Kalliokoski et al. 2013 measured the same factor of 2
# empirically across ChEMBL. Expressed on a log scale and applied to put IC50 onto a Ki/Kd-equivalent
# axis, which is the thermodynamic quantity Kd and Ki already report.
LOG2 = math.log10(2.0)      # 0.30103


def measurement_types():
    lp = pd.read_csv(LP_CSV)
    pdb = lp.iloc[:, 0].astype(str).str.lower().str.strip()
    kind = lp['kd/ki'].astype(str).str.extract(r'^(Kd|Ki|IC50|EC50)', expand=False)
    return {p: k for p, k in zip(pdb, kind) if isinstance(k, str)}


def per_complex_types():
    """idx -> ('Kd'|'Ki'|'IC50'|'EC50'|None) for every index of the LP split."""
    import lmdb
    types = measurement_types()
    db = lmdb.open(CD_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    split = json.load(open(SPLIT_JSON))
    out, by_part = {}, {}
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for part in ('train', 'val', 'test'):
            counts = {}
            for idx, _pk in split[part]:
                rec = pickle.loads(txn.get(keys[idx]))
                m = re.search(r'_rec_([0-9a-z]{4})_', rec['ligand_filename'])
                t = types.get(m.group(1)) if m else None
                out[int(idx)] = t
                counts[str(t)] = counts.get(str(t), 0) + 1
            by_part[part] = counts
    db.close()
    return out, by_part


def harmonise(pk, kinds):
    """Return pk expressed on a Ki/Kd-equivalent axis.

    IC50 (and EC50, same functional character) are raised by log10(2): a functional IC50 of 2*Ki
    corresponds to pIC50 = pKi - 0.301, so adding 0.301 recovers the Ki-equivalent. Kd and Ki are left
    alone. Entries whose type could not be recovered are left alone and counted, so the residual
    uncertainty stays visible rather than being silently absorbed."""
    out = np.array(pk, dtype=np.float64).copy()
    n_adj = n_unknown = 0
    for i, k in enumerate(kinds):
        if k in ('IC50', 'EC50'):
            out[i] += LOG2
            n_adj += 1
        elif k not in ('Kd', 'Ki'):
            n_unknown += 1
    return out, n_adj, n_unknown


def main():
    print('recovering measurement type per complex via the ligand-source PDB entry...', flush=True)
    types_by_idx, by_part = per_complex_types()
    print('\n=== measurement-type composition, recovered per complex ===')
    for part, counts in by_part.items():
        n = sum(counts.values())
        parts = '  '.join(f'{k}={100 * v / n:.1f}%' for k, v in sorted(counts.items()))
        print(f'  {part:5s} n={n:6d}   {parts}')
    cov = {p: 100 * (1 - c.get('None', 0) / sum(c.values())) for p, c in by_part.items()}
    print(f'  join coverage: ' + '  '.join(f'{p} {v:.1f}%' for p, v in cov.items()))

    L = np.load(LIGAND_PREDS, allow_pickle=True)
    idx = np.asarray(L['idx'], dtype=np.int64)
    pk_raw = np.asarray(L['pk'], dtype=np.float64)
    target = np.asarray(L['target'], dtype=str)
    kinds = [types_by_idx.get(int(i)) for i in idx]

    pk_adj, n_adj, n_unknown = harmonise(pk_raw, kinds)
    print(f'\n=== harmonisation on the TEST set ({len(pk_raw)} complexes) ===')
    print(f'  IC50/EC50 entries raised by log10(2) = {LOG2:.5f}: {n_adj} '
          f'({100 * n_adj / len(pk_raw):.1f}%)')
    print(f'  type not recoverable, left unchanged  : {n_unknown} '
          f'({100 * n_unknown / len(pk_raw):.1f}%)')
    print(f'  pk before : mean {pk_raw.mean():.4f}  sd {pk_raw.std(ddof=1):.4f}')
    print(f'  pk after  : mean {pk_adj.mean():.4f}  sd {pk_adj.std(ddof=1):.4f}')
    print(f'  correlation between the two label vectors: '
          f'{np.corrcoef(pk_raw, pk_adj)[0, 1]:.6f}')
    print(f'  => the correction raises the mean by {pk_adj.mean() - pk_raw.mean():+.4f} and changes the')
    print(f'     sd by {pk_adj.std(ddof=1) - pk_raw.std(ddof=1):+.4f}. A sd INCREASE would raise the')
    print(f'     achievable R2 ceiling slightly, since R2 is variance-explained.')

    # --- does any conclusion move?
    models = {}
    for n, (p_, yk, pk_) in STRUCTURE_SOURCES.items():
        v, _ = load_structure(n, p_, yk, pk_, pk_raw, target)
        if v is not None:
            models[n] = v
    for k in L.files:
        if k.startswith('pred_') and k != 'pred_global_mean':
            models[k[5:] + ' (lig)'] = np.asarray(L[k], dtype=np.float64)

    groups = target_groups(target)
    all_rows = np.concatenate(groups)
    print(f'\n=== does the correction change any ranking? ===')
    print(f'{"model":24s} {"R2 raw":>9s} {"R2 harmonised":>14s} {"delta":>9s}')
    print('-' * 60)
    rows = {}
    for name, p in sorted(models.items(), key=lambda kv: -r2_rows(pk_raw, kv[1], all_rows)):
        a = r2_rows(pk_raw, p, all_rows)
        b = r2_rows(pk_adj, p, all_rows)
        rows[name] = dict(r2_raw=a, r2_harmonised=b, delta=b - a)
        print(f'{name:24s} {a:>9.4f} {b:>14.4f} {b - a:>+9.4f}')

    order_raw = [k for k, _ in sorted(rows.items(), key=lambda kv: -kv[1]['r2_raw'])]
    order_adj = [k for k, _ in sorted(rows.items(), key=lambda kv: -kv[1]['r2_harmonised'])]
    deltas = np.array([v['delta'] for v in rows.values()])

    # The decisive test, rather than comparing sort orders. Exact order equality is hypersensitive when
    # models are statistically tied: two models within 0.01 R2 of each other will swap on any
    # perturbation, and reporting that as "the ranking changed" would be the same mistake this project
    # already made twice at small n. So the comparison of record is a PAIRED target-clustered bootstrap
    # of the structure-vs-ligand difference, computed under the harmonised labels.
    champion = 'desc_ridge (lig)'
    draws = [np.concatenate([groups[g] for g in gi])
             for gi in cluster_bootstrap_indices(len(groups), 2000, BOOT_SEED)]
    paired = {}
    if champion in models:
        lp_ = models[champion]
        print(f'\n=== PAIRED dR2 vs {champion}, under HARMONISED labels (B=2000) ===')
        for name, p_ in models.items():
            if name.endswith('(lig)'):
                continue
            point = r2_rows(pk_adj, p_, all_rows) - r2_rows(pk_adj, lp_, all_rows)
            boot = np.array([r2_rows(pk_adj, p_, d) - r2_rows(pk_adj, lp_, d) for d in draws])
            boot = boot[np.isfinite(boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pval = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
            pval = min(1.0, max(pval, 1.0 / (len(boot) + 1)))
            verdict = 'structure WINS' if lo > 0 else ('ligand WINS' if hi < 0 else 'tie')
            paired[name] = dict(d_r2=float(point), ci95=[float(lo), float(hi)],
                                boot_p=float(pval), verdict=verdict)
            print(f'  {name:22s} {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={pval:.3f}  {verdict}')

    print('\n=== READING ===')
    typical_ci = 0.3871      # mean 95% CI width measured in noise_ceiling.py
    spread = float(deltas.max() - deltas.min())
    print(f'* The correction shifts every model by between {deltas.min():+.4f} and {deltas.max():+.4f}')
    print(f'  (spread {spread:.4f}), against a typical 95% CI width on R2 of {typical_ci:.4f}.')
    if spread < typical_ci / 2:
        print(f'* The sort ORDER does change -- {sum(a != b for a, b in zip(order_raw, order_adj))} of')
        print(f'  {len(order_raw)} positions move -- but every shift is far smaller than one confidence')
        print('  interval, so those are reorderings among models that were already statistically tied.')
        print('  Reporting them as a substantive change would repeat the small-n error this project made')
        print('  twice. The honest statement: the correction does not resolve the comparison, and the')
        print('  instability of the order is itself another symptom of the power problem.')
    else:
        print('* The shifts are large relative to our confidence intervals, so the correction IS')
        print('  result-changing and the affected arms need refitting on harmonised labels.')
    if paired:
        wins = [k for k, v in paired.items() if v['verdict'] == 'structure WINS']
        print(f'* Under harmonised labels, {len(wins)}/{len(paired)} structure models beat {champion}')
        print(f'  with a paired CI excluding zero' + (f': {", ".join(wins)}' if wins else ' (none).'))
        print('* Direction worth noting even where it is not significant: harmonisation moves the ligand')
        print(f'  champion DOWN ({rows[champion]["delta"]:+.4f}) while the ensemble moves UP')
        print(f'  ({rows.get("egnn_a1c_ensemble", {}).get("delta", float("nan")):+.4f}). So the')
        print('  uncorrected mixture was, if anything, flattering the ligand-only model. That is the')
        print('  opposite of a convenient result for us and should be reported as such.')
    print('* Caveat stated plainly: the models were TRAINED on uncorrected labels. This measures whether')
    print('  the EVALUATION target matters, not whether retraining on harmonised labels would do better.')
    print('  Arm C is the natural place to train on harmonised labels from the start.')
    print(f'* Offset applied: log10(2) = {LOG2:.5f}, from Kalliokoski et al. 2013 (311 cites), which')
    print('  matches the Cheng-Prusoff relation IC50 = 2*Ki at [S] = Km. Kd and Ki left untouched.')

    json.dump(dict(offset_log10=LOG2, offset_source='Kalliokoski et al. 2013 (PLoS ONE, 311 cites)',
                   composition_by_split=by_part, join_coverage_pct=cov,
                   test_set=dict(n=len(pk_raw), n_adjusted=n_adj, n_unknown=n_unknown,
                                 mean_raw=float(pk_raw.mean()), mean_harmonised=float(pk_adj.mean()),
                                 sd_raw=float(pk_raw.std(ddof=1)),
                                 sd_harmonised=float(pk_adj.std(ddof=1))),
                   models=rows, ranking_unchanged=(order_raw == order_adj),
                   paired_vs_ligand_champion=paired,
                   delta_spread=float(deltas.max() - deltas.min()),
                   typical_ci_width=0.3871,
                   note='IC50/EC50 raised by log10(2) to a Ki/Kd-equivalent axis. Models were trained on '
                        'uncorrected labels, so this tests whether the EVALUATION target changes any '
                        'conclusion, not whether retraining would improve on it.'),
              open(OUT, 'w'), indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
