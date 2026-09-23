"""Cached UniProt REST client for Stage 0's protein categorization and
sequence retrieval. CrossDocked2020's target directory naming convention
(e.g. "AKT1_HUMAN_1_137_0") has a leading "<GENE>_<SPECIES>" token that is
already a valid UniProt mnemonic entry name -- confirmed by direct query
(guidance/FLAGSHIP_ARCHITECTURE_RESEARCH.md's environment feasibility
notes) -- so no separate ID-mapping step is needed.

Bare requests to rest.uniprot.org without a User-Agent header return 403;
this is a header requirement, not a network block.
"""
import json
import os
import time
import urllib.error
import urllib.request

CACHE_PATH = './guidance/lp_split/uniprot_cache.json'
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


def _fetch_one(entry_name):
    url = f'https://rest.uniprot.org/uniprotkb/{entry_name}.json'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'found': False}
        raise

    seq = data.get('sequence', {}).get('value')
    pfam_ids = [x['id'] for x in data.get('uniProtKBCrossReferences', []) if x.get('database') == 'Pfam']
    similarity_family = None
    for c in data.get('comments', []):
        if c.get('commentType') == 'SIMILARITY':
            texts = c.get('texts', [])
            if texts:
                similarity_family = texts[0]['value']
                break
    return {
        'found': True,
        'accession': data.get('primaryAccession'),
        'sequence': seq,
        'pfam_ids': pfam_ids,
        'similarity_family': similarity_family,
    }


def fetch_records(entry_names, cache_path=CACHE_PATH, verbose=True):
    """Returns {entry_name: record_dict}. Caches to disk so repeated calls
    (across Stage 0 re-runs) don't re-hit the API for already-fetched
    entries."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = _load_cache(cache_path)
    to_fetch = [e for e in entry_names if e not in cache]
    if verbose:
        print(f'UniProt cache: {len(cache)} cached, {len(to_fetch)} to fetch')

    for i, entry_name in enumerate(to_fetch):
        try:
            cache[entry_name] = _fetch_one(entry_name)
        except Exception as e:
            cache[entry_name] = {'found': False, 'error': str(e)}
        if verbose and (i + 1) % 50 == 0:
            print(f'  fetched {i + 1}/{len(to_fetch)}')
            _save_cache(cache, cache_path)  # periodic checkpoint
        time.sleep(RATE_LIMIT_SECONDS)

    if to_fetch:
        _save_cache(cache, cache_path)
    return {e: cache[e] for e in entry_names}


def get_category(record):
    """Returns a category label for clustering. Pfam ID is the most
    specific standard family classification; falls back to the free-text
    SIMILARITY family; falls back to a singleton per-entry category
    (never merges unclassifiable proteins into a shared 'unknown' bucket,
    which would be a worse leakage risk than treating them as each their
    own category -- see FLAGSHIP_ARCHITECTURE_RESEARCH.md)."""
    if not record.get('found'):
        return None  # caller assigns a singleton fallback using the entry name
    if record.get('pfam_ids'):
        return 'pfam:' + '+'.join(sorted(record['pfam_ids']))
    if record.get('similarity_family'):
        return 'family:' + record['similarity_family']
    return None
