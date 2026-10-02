"""Cached AlphaFold DB (EBI) client, mirroring uniprot_client.py's pattern:
fetches the precomputed AlphaFold model (not a fresh inference run -- see
module docstring of guidance/alphafold_pocket_robustness.py for why running
AlphaFold2 from scratch is infeasible on this 4GB GPU, and why the DB
suffices for every target in our dataset since they are all real PDB
entries, almost certainly already modeled in AFDB via their UniProt
accession) plus its per-residue PDB structure file, for a given UniProt
accession.

Bare requests without a User-Agent are not required by this API (unlike
UniProt's REST endpoint), but one is sent anyway to be a polite, identifiable
client.
"""
import json
import os
import time
import urllib.error
import urllib.request

CACHE_PATH = './guidance/lp_split/alphafold_cache.json'
STRUCT_DIR = './guidance/lp_split/alphafold_structures'
USER_AGENT = 'targetdiff-thesis-research/1.0 (academic, non-commercial)'
RATE_LIMIT_SECONDS = 0.34  # ~3 req/sec, polite default


def _load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_cache(cache, cache_path):
    tmp = cache_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)


def _fetch_one(accession):
    url = f'https://alphafold.ebi.ac.uk/api/prediction/{accession}'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        if not data:
            return {'found': False}
        entry = data[0]  # first fragment/model; all our targets are single-fragment-sized
        return {
            'found': True,
            'model_id': entry['modelEntityId'],
            'global_plddt': entry['globalMetricValue'],
            'sequence': entry['sequence'],
            'pdb_url': entry['pdbUrl'],
            'plddt_doc_url': entry['plddtDocUrl'],
        }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'found': False}
        raise
    except Exception as e:
        return {'found': False, 'error': str(e)}


def _download_structure(accession, pdb_url, struct_dir):
    os.makedirs(struct_dir, exist_ok=True)
    out_path = os.path.join(struct_dir, f'{accession}.pdb')
    if os.path.exists(out_path):
        return out_path
    req = urllib.request.Request(pdb_url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        content = r.read()
    with open(out_path, 'wb') as f:
        f.write(content)
    return out_path


def fetch_records(accessions, cache_path=CACHE_PATH, struct_dir=STRUCT_DIR,
                   download_structures=True, log=print):
    """accessions: iterable of UniProt accessions (e.g. 'P31946').
    Returns {accession: record_dict}, where record_dict includes a
    'struct_path' key (local PDB file path) when download_structures=True
    and the entry was found.
    """
    cache = _load_cache(cache_path)
    accessions = list(dict.fromkeys(accessions))  # de-dup, keep order
    n_new = 0
    for i, acc in enumerate(accessions, 1):
        if acc in cache and 'found' in cache[acc]:
            continue
        rec = _fetch_one(acc)
        cache[acc] = rec
        n_new += 1
        log(f'[{i}/{len(accessions)}] {acc}: found={rec.get("found")}')
        time.sleep(RATE_LIMIT_SECONDS)
        if n_new % 25 == 0:
            _save_cache(cache, cache_path)
    if n_new:
        _save_cache(cache, cache_path)

    if download_structures:
        for acc in accessions:
            rec = cache.get(acc, {})
            if rec.get('found') and 'struct_path' not in rec:
                try:
                    path = _download_structure(acc, rec['pdb_url'], struct_dir)
                    rec['struct_path'] = path
                    cache[acc] = rec
                except Exception as e:
                    rec['struct_download_error'] = str(e)
                    cache[acc] = rec
                time.sleep(RATE_LIMIT_SECONDS)
        _save_cache(cache, cache_path)

    return {acc: cache.get(acc, {'found': False}) for acc in accessions}
