"""The complete representation ladder: what each SOURCE of information is worth on its own, and whether
combining them adds anything.

ligand_only_baseline.py asked "can the ligand alone do it?" and answered yes. That leaves the obvious
rebuttal open -- perhaps the network uses the protein in a way a ligand model cannot imitate. Volkov et
al. 2022 closed it by testing protein descriptors too, and this script does the same on our split, plus
an interaction block that tests a second published claim.

The blocks:
  ligand       14 RDKit descriptors                        -- the small molecule alone
  pocket       38 pocket descriptors (AA composition,      -- the binding site alone, NO ligand
               physicochemical groups, element content,
               geometry)
  ligand+pocket                                            -- do the two marginals add up?
  ecif_raw     72 (protein element x ligand element x      -- interaction counts
               distance shell) contact counts
  ecif_norm    the same, divided by ligand heavy atoms     -- interaction DENSITY
  ecif_norm+size                                           -- density plus one size feature
  plip_raw     4 (hbond/hydrophobic/pi-stack/salt-bridge)  -- TYPED interaction counts (PLIP)
               counts, summed over ligand atoms
  plip_norm    the same, divided by ligand heavy atoms     -- typed interaction DENSITY
  plip_norm+size                                           -- density plus one size feature
  all          ligand + pocket + ecif_norm                 -- descriptor-level ceiling

TODO item 2 (added 2026-10-13): ecif's contact counts are untyped (element x element x distance shell,
no chemistry). PLIP typed interactions test the sharper version of the same question: if BERTYPED
contact counts -- "this is specifically an h-bond", not just "these elements are within 3.5 Angstrom" --
still add nothing beyond the marginals, that strengthens the redundancy claim considerably, since typed
interactions are the more informative representation a structure-based model's advantage is usually
argued to come from. If they DO add something ecif_norm did not, that is a genuinely new positive result
and should be reported as cleanly as the AVE-bias finding was.

Classes used: all four PLIP_CLASSES_ALL (hbond, hydrophobic, pi_stack, salt_bridge) from
guidance/track_e/core.py's PlipLabelStore, cached from guidance/track_e/build_plip_cache.py -- verified
100% coverage across train/val/test (64,888/64,888 LP-split rows) before this was written, so there is no
missing-row exclusion logic to get wrong. Track E's own prediction head (decision D2, same file) drops
pi_stack from what it tries to PREDICT; that decision is about head stability for a different task
(predicting labels from structure) and is not reused here uncritically -- no evidence was found that
pi_stack counts are unreliable AS a feature for affinity regression specifically, so all four are kept,
and this choice is stated rather than silently inherited.

The raw/norm pair mirrors ecif_raw/ecif_norm for exactly the reason stated two paragraphs below: raw
counts of ANY kind scale with ligand size, and this project already found that mistake once (ecif_raw
dR2 was +0.1625 over ecif_norm, "working mainly by re-encoding ligand size" -- see this file's own
ecif section). Running plip_raw without plip_norm alongside it would risk repeating that exact finding
under a different feature name instead of learning from it.

Two claims are under test.

1. Volkov et al. 2022: an explicit description of noncovalent interactions gives no advantage over
   ligand OR protein descriptors. Testing the pocket block completes that replication.

2. Brown 2025 (PNAS, CORDIAL): models should receive distance-dependent interaction signatures while
   being DENIED direct parameterisation of protein and ligand structure, because the latter is what
   lets them learn structural shortcuts; they report this bias survives leave-superfamily-out validation
   where conventional models fail. Our split is target-disjoint, which is the regime that claim is about,
   so ecif_norm versus ligand is a fair test of it.

The ecif_raw / ecif_norm pair is a confound control, not redundancy. Raw contact counts scale with ligand
size, and ligand size alone already reaches R2 0.307 here -- so a raw-count model can look like it has
learned interactions while mostly re-encoding heavy-atom count. The normalised block removes that
channel. Reporting only the raw version is the same mistake the scaffold audit was already caught making.

Statistics: cluster bootstrap over the 127 test targets (B=2000, seed 20260925) for every CI; paired
bootstrap for every head-to-head; Benjamini-Hochberg over all paired tests actually run.

CPU only.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/representation_ladder.py
  PYTHONPATH=. python guidance/cheminformatics/representation_ladder.py --b 500
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guidance.cheminformatics.chem_data import build_features, split_view, target_groups
from guidance.cheminformatics.pocket_features import build as build_pocket, split_view_pocket
from guidance.cheminformatics.compare_vs_structure import (LIGAND_PREDS, STRUCTURE_SOURCES,
                                                           load_structure, r2_rows)
from guidance.track_e.core import PLIP_CLASSES_ALL, PlipLabelStore
from guidance.uncertainty_a1.stats_common import bh, cluster_bootstrap_indices

OUT = './guidance/cheminformatics/representation_ladder_results.json'
BOOT_SEED = 20260925


def plip_features(idx_array, n_lig_array, store=None):
    """(raw, norm) typed-interaction feature matrices, row-aligned to idx_array. raw[i, c] is the COUNT
    of PLIP class c (PLIP_CLASSES_ALL order) over complex i's ligand atoms; norm is the same divided by
    that complex's heavy-atom count -- the ecif_raw/ecif_norm convention, for the reason given in this
    module's docstring. Raises, not silently zero-fills, if any idx is missing from the store -- coverage
    was verified 100% before this was written, so a miss here means the cache went stale, not that a
    gap was always expected and should be papered over."""
    store = store or PlipLabelStore()
    missing = [int(i) for i in idx_array if i not in store]
    if missing:
        raise SystemExit(f'ABORT: {len(missing)} / {len(idx_array)} rows have no PLIP labels (first few: '
                         f'{missing[:5]}) -- coverage was 100% when this was written; rebuild the PLIP '
                         f'cache (guidance/track_e/build_plip_cache.py) before trusting this block.')
    raw = np.array([store.get(int(i), classes=PLIP_CLASSES_ALL).sum(axis=0) for i in idx_array],
                   dtype=np.float64)
    norm = raw / np.maximum(n_lig_array.astype(np.float64)[:, None], 1.0)
    return raw, norm


def blocks(chem, pock, plip_store=None):
    """Feature matrices per information source, all row-aligned to the split."""
    size = chem['n_lig'].astype(np.float64)[:, None]
    lig = np.nan_to_num(chem['desc'])
    plip_raw, plip_norm = plip_features(chem['idx'], chem['n_lig'], store=plip_store)
    return {
        'ligand': lig,
        'pocket': pock['pocket'],
        'ligand+pocket': np.hstack([lig, pock['pocket']]),
        'ecif_raw': pock['ecif_raw'],
        'ecif_norm': pock['ecif_norm'],
        'ecif_norm+size': np.hstack([pock['ecif_norm'], size]),
        'plip_raw': plip_raw,
        'plip_norm': plip_norm,
        'plip_norm+size': np.hstack([plip_norm, size]),
        # 'all' is UNCHANGED (ligand + pocket + ecif_norm, no plip) -- this is an already-reported block
        # and folding plip in would silently change its number. A plip-inclusive ceiling, if wanted, is
        # a separate deliberate addition, not an implicit side effect of adding the plip blocks above.
        'all': np.hstack([lig, pock['pocket'], pock['ecif_norm']]),
    }


def fit_block(Xtr, ytr, Xva, yva, Xte):
    """Ridge with alpha chosen on val, and gradient boosting; the better-on-VAL of the two is returned.
    Selection on val only -- test is scored once. Both model families are tried for every block so a
    block is not penalised for suiting one of them: the question is what the INFORMATION is worth, not
    which regressor happens to fit it."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
        pv = m.predict(sc.transform(Xva))
        s = 1 - ((yva - pv) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()
        if best is None or s > best[0]:
            best = (s, f'ridge(a={alpha})', m.predict(sc.transform(Xte)))
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
    pv = m.predict(Xva)
    s = 1 - ((yva - pv) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()
    if s > best[0]:
        best = (s, 'hgb', m.predict(Xte))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=2000)
    args = ap.parse_args()

    chem = build_features()
    pock = build_pocket()
    tr = split_view('train', chem); va = split_view('val', chem); te = split_view('test', chem)
    ptr = split_view_pocket('train', pock); pva = split_view_pocket('val', pock)
    pte = split_view_pocket('test', pock)

    plip_store = PlipLabelStore()      # loaded once (7.5MB), passed to all three blocks() calls below
    Btr = blocks(tr, ptr, plip_store=plip_store)
    Bva = blocks(va, pva, plip_store=plip_store)
    Bte = blocks(te, pte, plip_store=plip_store)
    groups = target_groups(te['target'])
    all_rows = np.concatenate(groups)
    print(f"train {len(tr['pk'])} | val {len(va['pk'])} | test {len(te['pk'])} over {len(groups)} "
          f"target clusters | B={args.b}\n")

    draws = [np.concatenate([groups[g] for g in gi])
             for gi in cluster_bootstrap_indices(len(groups), args.b, BOOT_SEED)]

    preds, rows_out = {}, {}
    print(f'{"block":16s} {"dim":>5s} {"model":>14s} {"val R2":>8s} {"test R2":>9s} {"95% CI":>22s}')
    print('-' * 80)
    for name in Btr:
        vs, model, pt = fit_block(Btr[name], tr['pk'], Bva[name], va['pk'], Bte[name])
        preds[name] = pt
        point = r2_rows(te['pk'], pt, all_rows)
        boot = np.array([r2_rows(te['pk'], pt, d) for d in draws])
        boot = boot[np.isfinite(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows_out[name] = dict(dim=int(Btr[name].shape[1]), model=model, val_r2=float(vs),
                              test_r2=float(point), ci95=[float(lo), float(hi)])
        print(f'{name:16s} {Btr[name].shape[1]:>5d} {model:>14s} {vs:>8.4f} {point:>9.4f} '
              f'[{lo:+.4f}, {hi:+.4f}]')

    # Structure models, for context on the same rows.
    L = np.load(LIGAND_PREDS, allow_pickle=True)
    pk_ref = np.asarray(L['pk'], dtype=np.float64)
    tgt_ref = np.asarray(L['target'], dtype=str)
    struct = {}
    for n, (p_, yk, pk_) in STRUCTURE_SOURCES.items():
        v, _ = load_structure(n, p_, yk, pk_, pk_ref, tgt_ref)
        if v is not None:
            struct[n] = v
    print('\nstructure models on the same test rows:')
    for n, v in struct.items():
        print(f'  {n:22s} {r2_rows(pk_ref, v, all_rows):+.4f}')

    # The head-to-heads that answer the two claims. Paired, same draws, BH at the end.
    pairs = [('ligand+pocket', 'ligand'), ('ligand+pocket', 'pocket'), ('pocket', 'ligand'),
             ('ecif_norm', 'ligand'), ('ecif_norm', 'pocket'), ('ecif_raw', 'ecif_norm'),
             ('ecif_norm+size', 'ecif_norm'), ('all', 'ligand'), ('all', 'ligand+pocket'),
             ('all', 'ecif_norm'),
             # item 2's actual questions: does the TYPED (plip_norm) representation beat the
             # marginals the same way ecif_norm was tested against them, does it beat UNTYPED ecif_norm
             # specifically, and does plip_raw/plip_norm show the same size-confound ecif_raw/ecif_norm
             # already did.
             ('plip_norm', 'ligand'), ('plip_norm', 'pocket'), ('plip_norm', 'ecif_norm'),
             ('plip_raw', 'plip_norm'), ('plip_norm+size', 'plip_norm')]
    for sname in struct:
        pairs.append((('STRUCT', sname), 'all'))
        pairs.append((('STRUCT', sname), 'ecif_norm'))

    def vec(spec):
        if isinstance(spec, tuple):
            return struct[spec[1]], pk_ref
        return preds[spec], te['pk']

    comp = {}
    print(f'\n=== PAIRED dR2 (a - b), {args.b} target-cluster resamples ===')
    for a, b in pairs:
        pa, ya = vec(a); pb, _ = vec(b)
        point = r2_rows(ya, pa, all_rows) - r2_rows(ya, pb, all_rows)
        boot = np.array([r2_rows(ya, pa, d) - r2_rows(ya, pb, d) for d in draws])
        boot = boot[np.isfinite(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pval = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
        pval = min(1.0, max(pval, 1.0 / (len(boot) + 1)))
        label_a = a[1] if isinstance(a, tuple) else a
        comp[f'{label_a}__minus__{b}'] = dict(d_r2=float(point), ci95=[float(lo), float(hi)],
                                              boot_p=float(pval))
        print(f'  {label_a:22s} - {b:16s} {point:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={pval:.3f}')

    keys = list(comp)
    qs = bh([comp[k]['boot_p'] for k in keys])
    print(f'\n=== after Benjamini-Hochberg over {len(keys)} paired tests ===')
    for k, q in zip(keys, qs):
        comp[k]['boot_p_bh'] = float(q)
        verdict = ('a > b' if comp[k]['ci95'][0] > 0 else
                   'b > a' if comp[k]['ci95'][1] < 0 else 'tie')
        if verdict != 'tie' and q >= 0.05:
            verdict = 'tie (BH)'
        comp[k]['verdict'] = verdict
        print(f'  {k:46s} dR2 {comp[k]["d_r2"]:+.4f}  q={q:.3f}  {verdict}')

    def v(k):
        return comp[k]['verdict'] if k in comp else 'n/a'

    print('\n=== WHAT THIS SETTLES ===')
    print(f'1. Volkov et al. 2022 replication, protein side: pocket-only reaches R2 '
          f'{rows_out["pocket"]["test_r2"]:+.4f}\n'
          f'   on UNSEEN targets, versus ligand-only {rows_out["ligand"]["test_r2"]:+.4f}. '
          f'pocket vs ligand: {v("pocket__minus__ligand")}.')
    print(f'2. Do the two sources add up? ligand+pocket vs ligand: '
          f'{v("ligand+pocket__minus__ligand")};\n'
          f'   vs pocket: {v("ligand+pocket__minus__pocket")}. If both are ties, neither source carries\n'
          f'   information the other lacks at descriptor level.')
    print(f'3. CORDIAL claim (Brown 2025): interaction-only should beat structure-parameterised models\n'
          f'   on unseen targets. ecif_norm R2 {rows_out["ecif_norm"]["test_r2"]:+.4f}; '
          f'vs ligand: {v("ecif_norm__minus__ligand")}; vs pocket: {v("ecif_norm__minus__pocket")}.')
    print(f'4. Size confound in interaction counts: ecif_raw R2 '
          f'{rows_out["ecif_raw"]["test_r2"]:+.4f} vs ecif_norm '
          f'{rows_out["ecif_norm"]["test_r2"]:+.4f};\n   raw vs norm: {v("ecif_raw__minus__ecif_norm")}. '
          f'A large positive gap means the raw counts were largely\n   re-encoding ligand size.')
    print(f'5. Descriptor-level ceiling: all R2 {rows_out["all"]["test_r2"]:+.4f}; '
          f'vs ligand alone: {v("all__minus__ligand")}.')
    st = [k for k in comp if k.split('__')[0] in struct]
    beat = [k for k in st if comp[k]['verdict'] == 'a > b']
    print(f'6. Structure models vs the descriptor ceiling: {len(beat)}/{len(st)} pairings where a\n'
          f'   3D GNN beats descriptors after BH.')

    json.dump(dict(bootstrap=dict(b=args.b, seed=BOOT_SEED, n_target_clusters=len(groups)),
                   blocks=rows_out,
                   structure_models={n: float(r2_rows(pk_ref, v_, all_rows)) for n, v_ in struct.items()},
                   paired=comp,
                   note='Blocks isolate INFORMATION SOURCES, not architectures: both ridge and gradient '
                        'boosting are fitted for every block and the better-on-val is reported, so a '
                        'block is not penalised for suiting one regressor. ecif_raw vs ecif_norm is a '
                        'ligand-size confound control, since raw contact counts scale with heavy-atom '
                        'count and size alone reaches R2 0.307 on this task.'),
              open(OUT, 'w'), indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
