"""Compute PLIP ligand-atom labels for every LP-split complex once, offline, in resumable chunks.
Output: guidance/track_e/cache/plip/chunk_XXXX.npz (idx, lens, flat uint8) + failures.json.
Re-running skips finished chunks."""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from guidance.track_e.core import CACHE_DIR, build_anchor_table
from guidance.track_e.plip_label_extraction import ligand_atom_labels, build_complex_pdb

DATA = './data/crossdocked_v1.1_rmsd1.0_pocket10'
_base = None
OUT = os.path.join(CACHE_DIR, 'plip')


def _init():
    global _base
    from datasets.pl_pair_dataset import PocketLigandPairDataset
    _base = PocketLigandPairDataset(DATA)


def _one(idx):
    d = _base[idx]
    elems = d.ligand_element.tolist()
    tmp = f'/tmp/track_e_plip_{os.getpid()}.pdb'
    try:
        txt, ls = build_complex_pdb(d)
        lab, mis = ligand_atom_labels(txt, ls, elems, tmp)
        return idx, lab, mis, None
    except Exception as e:
        return idx, np.zeros((len(elems), 4), np.uint8), 0, repr(e)[:150]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = build_anchor_table()
    all_idx = [int(i) for p in ('train', 'val', 'test') for i in t[p][:, 0]]
    chunks = [all_idx[i:i + a.chunk] for i in range(0, len(all_idx), a.chunk)]
    fail_path = os.path.join(OUT, 'failures.json')
    failures = json.load(open(fail_path)) if os.path.exists(fail_path) else {}
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        for k, ids in enumerate(chunks):
            path = os.path.join(OUT, f'chunk_{k:04d}.npz')
            if os.path.exists(path):
                continue
            res = pool.map(_one, ids, chunksize=8)
            for i, lab, mis, err in res:
                if err or mis:
                    failures[str(i)] = err or f'element_mismatch={mis}'
            np.savez(path + '.tmp.npz', idx=np.array([r[0] for r in res]), lens=np.array([r[1].shape[0] for r in res]),
                     flat=np.concatenate([r[1].reshape(-1) for r in res]))
            os.replace(path + '.tmp.npz', path)
            json.dump(failures, open(fail_path, 'w'))
            print(f'chunk {k + 1}/{len(chunks)} done, elapsed {time.time() - t0:.0f}s, failures so far {len(failures)}', flush=True)
    print('PLIP cache complete')


if __name__ == '__main__':
    main()
