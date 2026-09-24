"""Track C follow-up (§6b of TRACK_C_REJECTION_SAMPLING_REPORT.md):
does pocket size predict how badly the top-10% affinity-ranker selection
damages PoseBusters structural validity?

Exploratory, post-hoc (run after observing the wide 5.7-52.7pp spread in
the per-pocket PoseBusters drop already reported in the main analysis) --
not BH-corrected against the rest of that report's tests. Reported as a
plausible mechanistic hypothesis for future work, not a confirmed causal
claim (see the caveats in §6b).

Pocket size = protein atom count in the CrossDocked2020 pocket10 crop
(the same 10A pocket definition used throughout this project), read
directly from the diffusion model's own dataset loader -- not a new
label, no risk of a size proxy inconsistent with what the model actually
sees.
"""
import json

import numpy as np
import torch
from scipy import stats
from torch_geometric.transforms import Compose

import utils.transforms as trans
from datasets import get_dataset

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
POCKETS_FILE = './guidance/task_f_pockets.json'
TRACK_C_RESULTS = './guidance/track_c_analysis/track_c_analysis_results.json'
N_BOOTSTRAP = 5000
SEED = 42


def load_pocket_sizes(n_pockets=15):
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(ckpt['config'].data.transform.ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()])
    dataset, subsets = get_dataset(config=ckpt['config'].data, transform=transform)
    test_set = subsets['test']

    with open(POCKETS_FILE) as f:
        pockets = json.load(f)[:n_pockets]

    sizes = {}
    for p in pockets:
        data = test_set[p['data_id']]
        sizes[p['target']] = int(data.protein_pos.shape[0])
    return sizes


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    r_obs, p_obs = stats.spearmanr(x, y)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(idx)) < 3:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rs.append(r)
    ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
    return r_obs, p_obs, ci_lo, ci_hi


def leave_one_out(x, y, targets):
    print('\nLeave-one-out robustness check:')
    for i, t in enumerate(targets):
        mask = np.arange(len(x)) != i
        r, p = stats.spearmanr(x[mask], y[mask])
        print(f'  excl {t:28s} r={r:+.3f} p={p:.4f}')


def main():
    sizes = load_pocket_sizes()
    d = json.load(open(TRACK_C_RESULTS))
    primary = d['per_pocket_primary']
    targets = list(primary.keys())

    pocket_size = np.array([sizes[t] for t in targets])
    pb_drop = np.array([(primary[t]['vina_hacking_check']['pb_valid_rate_full']
                          - primary[t]['vina_hacking_check']['pb_valid_rate_topk']) * 100
                         for t in targets])
    full_pool_heavy = np.array([primary[t]['size_confound']['mean_heavy_atoms_full'] for t in targets])
    le_effect = np.array([primary[t]['vina_hacking_check']['ligand_efficiency_effect'] for t in targets])
    heavy_diff = np.array([primary[t]['size_confound']['mean_heavy_atoms_topk']
                            - primary[t]['size_confound']['mean_heavy_atoms_full'] for t in targets])

    print(f'N pockets = {len(targets)}\n')
    for name, y in [
        ('PB valid-rate drop (pp)', pb_drop),
        ('Baseline (full-pool) mean heavy atoms', full_pool_heavy),
        ('Ligand efficiency effect', le_effect),
        ('Heavy-atom diff (topk - full)', heavy_diff),
    ]:
        r, p, lo, hi = bootstrap_spearman(pocket_size, y)
        sig = 'SIG (CI excludes 0)' if not (lo < 0 < hi) else 'n.s.'
        print(f'Pocket size vs {name}: Spearman r={r:+.3f}, raw p={p:.4f}, '
              f'95% bootstrap CI=[{lo:+.3f},{hi:+.3f}] -> {sig}')

    print(f'\nPearson (linear) pocket size vs PB drop: {stats.pearsonr(pocket_size, pb_drop)}')
    leave_one_out(pocket_size, pb_drop, targets)


if __name__ == '__main__':
    main()
