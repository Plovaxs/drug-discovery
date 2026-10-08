"""Cross-references ChEMBL target IDs (BindingNet v1's target identity) against this project's
UniProt-entry-name-based target directories (val+test, from guidance/surrogate_data/val_test_exclusion_set.json),
via UniProt accession as the common key. Needed because guidance/surrogate_data/build_bindingnet_index.py's
PDB-ID filter only catches exact structure reuse, not "same protein, different PDB entry" leakage.

Our target directory names (e.g. "RXRA_HUMAN_222_461_ligBind_0") are CrossDocked2020's convention:
GENE_SPECIES (a UniProt entry name / mnemonic) followed by a residue range and an index -- the first two
underscore-separated tokens are always the UniProt entry name (verified against UniProt's REST API below;
any directory whose first-two-token guess fails to resolve is reported, not silently dropped).

ChEMBL target -> UniProt accession(s): ChEMBL REST API, batched (target_chembl_id__in, chunks of 100).
Our target name -> UniProt accession: UniProt REST API, one lookup per unique entry-name guess (192 of them,
small, cached to a local JSON so repeat runs don't re-hit the network).

Usage:
  python guidance/surrogate_data/build_target_crossref.py
"""
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

EXCLUSION = './guidance/surrogate_data/val_test_exclusion_set.json'
FILTERED_IN = './guidance/surrogate_data/bindingnet_v1_index_filtered.csv'
FILTERED_OUT = './guidance/surrogate_data/bindingnet_v1_index_filtered2.csv'
UNIPROT_CACHE = './guidance/surrogate_data/val_test_target_uniprot.json'
CHEMBL_CACHE = './guidance/surrogate_data/bindingnet_chembl_target_accessions.json'


def fetch_json(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f'failed after {retries} retries: {url}')


def resolve_our_targets(target_dirs):
    if os.path.exists(UNIPROT_CACHE):
        cache = json.load(open(UNIPROT_CACHE))
    else:
        cache = {}
    unresolved = []
    for t in sorted(target_dirs):
        entry_name = '_'.join(t.split('_')[:2])
        if entry_name in cache:
            continue
        d = fetch_json(f'https://rest.uniprot.org/uniprotkb/{entry_name}.json')
        cache[entry_name] = d.get('primaryAccession') if d else None
        if cache[entry_name] is None:
            unresolved.append((t, entry_name))
    json.dump(cache, open(UNIPROT_CACHE, 'w'), indent=2)
    acc_set = {acc for acc in cache.values() if acc}
    print(f'Resolved {len(acc_set)}/{len(cache)} unique entry-name guesses to a UniProt accession.')
    if unresolved:
        print(f'UNRESOLVED ({len(unresolved)}), needs manual check: {unresolved}')
    return acc_set


def chembl_target_accessions(chembl_ids):
    if os.path.exists(CHEMBL_CACHE):
        cache = json.load(open(CHEMBL_CACHE))
    else:
        cache = {}
    todo = [c for c in chembl_ids if c not in cache]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        url = ('https://www.ebi.ac.uk/chembl/api/data/target.json?limit=1000&target_chembl_id__in='
               + ','.join(chunk))
        d = fetch_json(url)
        found = {t['target_chembl_id']: [c['accession'] for c in t['target_components']] for t in d['targets']}
        for c in chunk:
            cache[c] = found.get(c, [])
        print(f'...chembl targets resolved: {min(i + 100, len(todo))}/{len(todo)}', flush=True)
    json.dump(cache, open(CHEMBL_CACHE, 'w'), indent=2)
    return cache


def main():
    excl = json.load(open(EXCLUSION))
    our_accessions = resolve_our_targets(excl['val_test_targets'])

    df = pd.read_csv(FILTERED_IN)
    chembl_acc = chembl_target_accessions(sorted(df['chembl_target'].unique()))

    leaking_chembl_targets = {c for c, accs in chembl_acc.items() if our_accessions.intersection(accs)}
    print(f'ChEMBL targets whose UniProt accession matches a val+test target: {len(leaking_chembl_targets)} '
          f'/ {df["chembl_target"].nunique()}')
    leaked = df['chembl_target'].isin(leaking_chembl_targets)
    print(f'Additional rows dropped by target-identity filter: {leaked.sum()} / {len(df)}')
    df_clean = df.loc[~leaked].reset_index(drop=True)
    df_clean.to_csv(FILTERED_OUT, index=False)
    print(f'Saved fully-filtered index ({len(df_clean)} entries) to {FILTERED_OUT}')


if __name__ == '__main__':
    main()
