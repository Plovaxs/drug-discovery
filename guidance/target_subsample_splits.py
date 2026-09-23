"""Target-level training-size learning curve (Diagnostic 1): builds
train/val splits restricted to a FRACTION of the 507 distinct protein
targets used in the main sweep's 6000-example training pool, instead of
subsampling individual examples -- the causal variable under test is
target diversity, not raw example count (which follows automatically,
~11.8 examples/target on average).

The 27-pocket test set is always the untouched official-test-split pool
(datasets/crossdocked_affinity.py's `test_final`) -- identical across
every point on the curve, and by construction (official train/test split
has 0 target overlap) disjoint from every training-target subset built
here. `verify_no_test_overlap` re-checks this explicitly per the
addendum's request for one exercised safety check before spending GPU
time on this diagnostic.
"""
import pickle

import torch

from datasets.pl_pair_dataset import PocketLigandPairDataset
from datasets.crossdocked_affinity import _ligand_key, CrossDockedAffinityDataset


def build_target_fraction_splits(fraction, reference_seed=2021,
                                 subsample_seed=2021,
                                 root='./data/crossdocked_v1.1_rmsd1.0_pocket10',
                                 split_path='./data/crossdocked_pocket10_pose_split.pt',
                                 affinity_path='./data/affinity_info.pkl',
                                 val_fraction=0.1,
                                 reference_train_n=6000, reference_val_n=4061,
                                 base=None):
    """Returns (base, {'train': [...], 'val': [...], 'test': [...]}, pk_by_idx,
    selected_targets), like build_labeled_splits, except train+val are
    restricted to `fraction` of the 507-target pool used by the main
    sweep's train_subsample=6000 run (same `reference_seed` reproduces
    that exact 507-target pool), before re-deriving a fresh 90/10 val
    split from just those targets' available labeled examples.

    Total train/val example counts are additionally capped at
    round(reference_train_n * fraction) / round(reference_val_n * fraction)
    -- CrossDocked2020 has many more poses available per target overall
    than the 507-target/6000-example reference subsample used, so without
    this cap a target-fraction restriction would NOT proportionally
    shrink example count (data-rich targets can each contribute far more
    than ~11.8 poses when not also subsampled by example count) --
    breaking both epoch-time comparability and the intent of isolating
    target-diversity as the single manipulated variable.

    Pass a pre-built `base` (PocketLigandPairDataset) to avoid re-opening
    its LMDB env when calling this repeatedly in one process.
    """
    if base is None:
        base = PocketLigandPairDataset(root)
    split = torch.load(split_path, weights_only=False)
    with open(affinity_path, 'rb') as f:
        affinity_info = pickle.load(f)

    def labeled(indices):
        out = []
        for idx in indices:
            key = _ligand_key(base[idx].ligand_filename)
            entry = affinity_info.get(key)
            if entry is not None and entry['pk'] != 0.0:
                out.append((idx, float(entry['pk'])))
        return out

    train_labeled = labeled(split['train'])
    test_labeled = labeled(split['test'])

    # Reproduce the exact 507-target pool the main sweep's train_subsample=6000
    # run used, by replaying the same subsample draw.
    g_ref = torch.Generator().manual_seed(reference_seed)
    perm_ref = torch.randperm(len(train_labeled), generator=g_ref).tolist()
    n_val_ref = max(1, int(len(train_labeled) * val_fraction))
    val_idx_pos_ref = set(perm_ref[:n_val_ref])
    train_final_ref = [train_labeled[i][0] for i in range(len(train_labeled)) if i not in val_idx_pos_ref]
    sub_perm_ref = torch.randperm(len(train_final_ref), generator=g_ref).tolist()[:6000]
    reference_6000 = [train_final_ref[i] for i in sub_perm_ref]
    target_pool_507 = sorted(set(base[idx].ligand_filename.split('/')[0] for idx in reference_6000))

    g_frac = torch.Generator().manual_seed(subsample_seed)
    n_targets = max(1, round(len(target_pool_507) * fraction))
    perm_targets = torch.randperm(len(target_pool_507), generator=g_frac).tolist()[:n_targets]
    selected_targets = set(target_pool_507[i] for i in perm_targets)

    # All labeled examples (train side of the official split) belonging to
    # the selected targets -- not limited to the reference 6000, so smaller
    # fractions still get whatever real data exists for their targets.
    restricted_pool = [(idx, pk) for idx, pk in train_labeled
                       if base[idx].ligand_filename.split('/')[0] in selected_targets]

    g_val = torch.Generator().manual_seed(subsample_seed)
    perm_val = torch.randperm(len(restricted_pool), generator=g_val).tolist()
    n_val = max(1, int(len(restricted_pool) * val_fraction))
    val_idx_pos = set(perm_val[:n_val])

    train_final, val_final = [], []
    pk_by_idx = {}
    for i, (idx, pk) in enumerate(restricted_pool):
        pk_by_idx[idx] = pk
        (val_final if i in val_idx_pos else train_final).append(idx)
    test_final = []
    for idx, pk in test_labeled:
        pk_by_idx[idx] = pk
        test_final.append(idx)

    # Cap example counts so target-count is the sole manipulated variable
    # (see docstring) -- shuffle deterministically before capping.
    g_cap = torch.Generator().manual_seed(subsample_seed)
    train_cap = round(reference_train_n * fraction)
    if len(train_final) > train_cap:
        keep = torch.randperm(len(train_final), generator=g_cap).tolist()[:train_cap]
        train_final = [train_final[i] for i in keep]
    val_cap = round(reference_val_n * fraction)
    if len(val_final) > val_cap:
        keep = torch.randperm(len(val_final), generator=g_cap).tolist()[:val_cap]
        val_final = [val_final[i] for i in keep]

    return base, {'train': train_final, 'val': val_final, 'test': test_final}, pk_by_idx, selected_targets


def verify_no_test_overlap(base, splits, selected_targets):
    test_targets = set(base[idx].ligand_filename.split('/')[0] for idx in splits['test'])
    overlap = selected_targets & test_targets
    if overlap:
        raise RuntimeError(f'Target overlap between training subsample and held-out test set: {overlap}')
    return True


if __name__ == '__main__':
    base = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')
    for fraction in [0.25, 0.5, 0.75]:
        _, splits, pk_by_idx, selected_targets = build_target_fraction_splits(fraction, base=base)
        verify_no_test_overlap(base, splits, selected_targets)
        print(f'fraction={fraction}: n_targets={len(selected_targets)} '
             f'train={len(splits["train"])} val={len(splits["val"])} test={len(splits["test"])} '
             f'-- no test overlap OK')
