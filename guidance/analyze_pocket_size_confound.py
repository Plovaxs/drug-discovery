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
import pandas as pd
import torch
from scipy import stats
from torch_geometric.transforms import Compose
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

import utils.transforms as trans
from datasets import get_dataset

CHECKPOINT = './pretrained_models/pretrained_diffusion.pt'
POCKETS_FILE = './guidance/task_f_pockets.json'
TRACK_C_RESULTS = './guidance/track_c_analysis/track_c_analysis_results.json'
TRACK_C_POOL_DIR = './guidance/track_c_pools'
N_BOOTSTRAP = 5000
SEED = 42


def pool_diversity_and_spread(smiles_list, max_n=600):
    """Mean 1-Tanimoto diversity (same metric Task F used) and heavy-atom
    count std, for one pocket's combined-seed pool."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list[:max_n]]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return np.nan, np.nan
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    sims = []
    for i in range(len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    diversity = 1 - np.mean(sims)
    heavy_std = np.std([m.GetNumHeavyAtoms() for m in mols])
    return diversity, heavy_std


def load_pool_diversity(pockets, seeds=(2021, 2022)):
    out = {}
    for p in pockets:
        smiles_all = []
        for seed in seeds:
            path = f'{TRACK_C_POOL_DIR}/pocket{p["data_id"]}/seed{seed}/honest_eval.csv'
            df = pd.read_csv(path)
            smiles_all.extend(df['smiles'].dropna().tolist())
        div, heavy_std = pool_diversity_and_spread(smiles_all)
        out[p['target']] = {'diversity': div, 'heavy_atom_std': heavy_std}
    return out


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


def _residualize(y, x):
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)


def bootstrap_partial_spearman(y, x, control, n_boot=N_BOOTSTRAP, seed=SEED):
    """Partial Spearman correlation of x with y, controlling for `control`,
    via linear residualization -- with a case-resampling bootstrap CI
    (not just a parametric p-value on the point estimate)."""
    r_obs, p_obs = stats.spearmanr(_residualize(y, control), _residualize(x, control))
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(set(idx)) < 4:
            continue
        try:
            r, _ = stats.spearmanr(_residualize(y[idx], control[idx]), _residualize(x[idx], control[idx]))
        except Exception:
            continue
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
    with open(POCKETS_FILE) as f:
        pockets = json.load(f)[:15]
    sizes = load_pocket_sizes()
    pool_div = load_pool_diversity(pockets)

    d = json.load(open(TRACK_C_RESULTS))
    primary = d['per_pocket_primary']
    targets = list(primary.keys())

    pocket_size = np.array([sizes[t] for t in targets])
    diversity = np.array([pool_div[t]['diversity'] for t in targets])
    heavy_std = np.array([pool_div[t]['heavy_atom_std'] for t in targets])
    pb_drop = np.array([(primary[t]['vina_hacking_check']['pb_valid_rate_full']
                          - primary[t]['vina_hacking_check']['pb_valid_rate_topk']) * 100
                         for t in targets])
    full_pool_heavy = np.array([primary[t]['size_confound']['mean_heavy_atoms_full'] for t in targets])
    le_effect = np.array([primary[t]['vina_hacking_check']['ligand_efficiency_effect'] for t in targets])
    heavy_diff = np.array([primary[t]['size_confound']['mean_heavy_atoms_topk']
                            - primary[t]['size_confound']['mean_heavy_atoms_full'] for t in targets])
    raw_effect = np.array([primary[t]['effect_size'] for t in targets])

    print(f'N pockets = {len(targets)}\n')
    print('=== Primary: pocket size vs outcomes ===')
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

    print('\n=== Alternative explanations tested: pool diversity / heavy-atom spread ===')
    for name, x in [('pool diversity (1-Tanimoto)', diversity), ('pool heavy-atom std', heavy_std)]:
        for yname, y in [('PB drop', pb_drop), ('raw effect size', raw_effect)]:
            r, p, lo, hi = bootstrap_spearman(x, y)
            sig = 'SIG' if not (lo < 0 < hi) else 'n.s.'
            print(f'{name} vs {yname}: r={r:+.3f} p={p:.4f} CI=[{lo:+.3f},{hi:+.3f}] -> {sig}')

    r_sanity, p_sanity, lo_s, hi_s = bootstrap_spearman(pocket_size, diversity)
    print(f'\nSanity: pocket_size vs pool diversity: r={r_sanity:+.3f} p={p_sanity:.4f} '
          f'CI=[{lo_s:+.3f},{hi_s:+.3f}]')

    r_partial, p_partial, lo_p, hi_p = bootstrap_partial_spearman(pb_drop, heavy_std, pocket_size)
    sig = 'SIG (CI excludes 0)' if not (lo_p < 0 < hi_p) else 'n.s.'
    print(f'Partial corr (heavy_std vs PB_drop, controlling pocket_size): '
          f'r={r_partial:+.3f} raw p={p_partial:.4f} 95% bootstrap CI=[{lo_p:+.3f},{hi_p:+.3f}] -> {sig}')


if __name__ == '__main__':
    main()
