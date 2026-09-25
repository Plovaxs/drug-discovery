"""Track E planning-phase feasibility check (NOT training).

On N random leakage-safe TRAIN complexes: PLIP throughput with a worker pool, zero-interaction rate,
ligand-atom label prevalence, the ligand-index element check, and a pose-jitter stability test
(Gaussian noise on ligand coordinates) as a first estimate of PLIP label noise.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from guidance.track_e.plip_label_extraction import CLASSES, build_complex_pdb, ligand_atom_labels

DATA = './data/crossdocked_v1.1_rmsd1.0_pocket10'
_base = None


def _init():
    global _base
    from datasets.pl_pair_dataset import PocketLigandPairDataset
    _base = PocketLigandPairDataset(DATA)


def _work(args):
    idx, sigma, tmp_dir, seed = args
    d = _base[idx]
    tmp = os.path.join(tmp_dir, f'w{os.getpid()}.pdb')
    elems = d.ligand_element.tolist()
    out = {'idx': idx, 'n_lig': len(elems), 'n_prot': int(d.protein_pos.shape[0])}
    try:
        t = time.time()
        txt, ls = build_complex_pdb(d)
        lab, mis = ligand_atom_labels(txt, ls, elems, tmp)
        out.update(t=time.time() - t, mismatches=mis, lab=lab.tolist())
        if sigma > 0:
            rng = np.random.RandomState(seed + idx)
            jp = d.ligand_pos.numpy() + rng.normal(0, sigma, size=d.ligand_pos.shape)
            txt2, ls2 = build_complex_pdb(d, ligand_pos=jp)
            lab2, _ = ligand_atom_labels(txt2, ls2, elems, tmp)
            out['lab_jit'] = lab2.tolist()
    except Exception as e:
        out['error'] = repr(e)[:200]
    return out


def jaccard(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return None if u == 0 else float((a & b).sum() / u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--n_jitter', type=int, default=100)
    ap.add_argument('--sigma', type=float, default=0.2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--out', default='./guidance/track_e/plip_scale_check_results.json')
    ap.add_argument('--tmp', default='/tmp/track_e_plip')
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    split = json.load(open('./guidance/lp_split/leakage_safe_split.json'))
    train_idx = [x[0] for x in split['train']]
    rng = np.random.RandomState(a.seed)
    pick = rng.choice(len(train_idx), size=a.n, replace=False)
    jobs = [(train_idx[i], a.sigma if k < a.n_jitter else 0.0, a.tmp, a.seed) for k, i in enumerate(pick)]

    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        res = pool.map(_work, jobs, chunksize=4)
    wall = time.time() - t0

    ok = [r for r in res if 'error' not in r]
    errs = [r for r in res if 'error' in r]
    labs = [np.array(r['lab'], dtype=np.uint8) for r in ok]
    complex_any = [int(l.any()) for l in labs]
    per_class_atom_rate = np.concatenate(labs).mean(0)
    per_class_complex_rate = np.mean([l.any(0) for l in labs], axis=0)
    summary = {
        'n_requested': a.n, 'n_ok': len(ok), 'n_errors': len(errs), 'errors': [e['error'] for e in errs][:5],
        'wall_s': wall, 'workers': a.workers, 'complexes_per_s': len(ok) / wall,
        'mean_plip_s_per_complex': float(np.mean([r['t'] for r in ok])),
        'zero_interaction_complex_frac': 1 - float(np.mean(complex_any)),
        'ligand_atom_positive_rate': dict(zip(CLASSES, map(float, per_class_atom_rate))),
        'complex_has_class_rate': dict(zip(CLASSES, map(float, per_class_complex_rate))),
        'index_element_mismatches_total': int(sum(r['mismatches'] for r in ok)),
    }
    jit = [r for r in ok if 'lab_jit' in r]
    if jit:
        per_cls = {c: [] for c in CLASSES}
        agree_flag = {c: [] for c in CLASSES}
        for r in jit:
            l0, l1 = np.array(r['lab'], dtype=np.uint8), np.array(r['lab_jit'], dtype=np.uint8)
            for ci, c in enumerate(CLASSES):
                j = jaccard(l0[:, ci], l1[:, ci])
                if j is not None:
                    per_cls[c].append(j)
                agree_flag[c].append(int(l0[:, ci].any() == l1[:, ci].any()))
        summary['jitter'] = {
            'sigma_A': a.sigma, 'n': len(jit),
            'atom_level_jaccard_mean_over_complexes_with_label': {c: (float(np.mean(v)) if v else None) for c, v in per_cls.items()},
            'n_complexes_with_label': {c: len(v) for c, v in per_cls.items()},
            'complex_level_presence_agreement': {c: float(np.mean(v)) for c, v in agree_flag.items()},
        }
    json.dump({'summary': summary}, open(a.out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
