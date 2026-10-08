"""Checks how much of the clean, filtered BindingDB set (guidance/surrogate_data/bindingdb_index_filtered.csv,
already excludes val+test leakage) targets a protein we ALREADY have a pocket structure for in our TRAIN
split. If a BindingDB measurement's target protein matches one of our 779 train target proteins, we can pair
it with a structure we already have (CrossDocked pocket) instead of needing BindingNet's Crystal_Templates
or any other new structure source -- this is a path to more training signal with zero new structure downloads,
complementary to (not a replacement for) the BindingNet/LP-PDBBind structure-pairing work.

This resolves our 779 train target directory names to UniProt accessions the same way
build_target_crossref.py did for val+test (first two underscore-separated tokens = UniProt entry name),
caches to disk, and intersects with BindingDB's per-row accession columns.

Usage:
  python guidance/surrogate_data/bindingdb_train_overlap.py
"""
import json
import os

import pandas as pd

from guidance.surrogate_data.build_target_crossref import fetch_json

TRAIN_TARGETS = './guidance/surrogate_data/train_targets.json'
CACHE = './guidance/surrogate_data/train_target_uniprot.json'
BINDINGDB_FILTERED = './guidance/surrogate_data/bindingdb_index_filtered.csv'
OUT = './guidance/surrogate_data/bindingdb_train_reusable.csv'


def resolve(target_dirs):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    unresolved = []
    resolved_keys = {k for k, v in cache.items() if v is not None}
    todo = sorted({'_'.join(t.split('_')[:2]) for t in target_dirs} - resolved_keys)
    for i, entry_name in enumerate(todo):
        try:
            d = fetch_json(f'https://rest.uniprot.org/uniprotkb/{entry_name}.json')
            cache[entry_name] = d.get('primaryAccession') if d else None
        except Exception as e:
            print(f'  lookup failed for {entry_name}: {e} -- leaving unresolved, will retry next run', flush=True)
            cache[entry_name] = None
        if cache[entry_name] is None:
            unresolved.append(entry_name)
        if (i + 1) % 50 == 0:
            json.dump(cache, open(CACHE, 'w'), indent=2)
            print(f'...resolved {i + 1}/{len(todo)}, checkpoint saved', flush=True)
    json.dump(cache, open(CACHE, 'w'), indent=2)
    acc_to_targets = {}
    for t in target_dirs:
        entry_name = '_'.join(t.split('_')[:2])
        acc = cache.get(entry_name)
        if acc:
            acc_to_targets.setdefault(acc, []).append(t)
    print(f'Resolved {len(acc_to_targets)}/{len(cache)} unique entry-name guesses to a UniProt accession '
          f'covering {len(target_dirs)} train target dirs.')
    if unresolved:
        print(f'UNRESOLVED ({len(unresolved)}): {unresolved}')
    return acc_to_targets


def main():
    train_targets = json.load(open(TRAIN_TARGETS))
    acc_to_targets = resolve(train_targets)
    train_accessions = set(acc_to_targets)

    df = pd.read_csv(BINDINGDB_FILTERED, low_memory=False)
    acc_cols = [c for c in df.columns if c.startswith('UniProt (SwissProt) Primary ID of Target Chain')]
    match_acc = pd.Series([None] * len(df), index=df.index, dtype=object)
    for c in acc_cols:
        hit = df[c].isin(train_accessions)
        match_acc = match_acc.mask(hit & match_acc.isna(), df[c])

    reusable = df.loc[match_acc.notna()].copy()
    reusable['matched_accession'] = match_acc.loc[reusable.index]
    reusable['matched_train_targets'] = reusable['matched_accession'].map(
        lambda a: ';'.join(acc_to_targets.get(a, [])))
    reusable.to_csv(OUT, index=False)

    print(f'BindingDB rows whose target protein matches a train-set accession (reusable structure, '
          f'no new download needed): {len(reusable)} / {len(df)}')
    print(f'Distinct matched train-set accessions: {reusable["matched_accession"].nunique()} / {len(train_accessions)} '
          f'train accessions have at least one BindingDB measurement.')
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
