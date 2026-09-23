"""Adapts guidance/lp_split/leakage_safe_split.json (Stage 0's LP-PDBBind-
style split) to the same (base_dataset, {'train','val','test'}, pk_by_idx)
interface build_labeled_splits/build_target_fraction_splits already use,
so existing training code (CrossDockedAffinityDataset) needs no changes.

Train is optionally subsampled to `train_subsample` examples (default
6000, matching this project's established convention for keeping epoch
time tractable on a 4GB card -- see crossdocked_affinity.py's docstring)
-- val and test are used in full, since Stage 0's whole point is a
rigorous, complete evaluation on the leakage-safe held-out set.
"""
import json

import torch

from datasets.pl_pair_dataset import PocketLigandPairDataset


def build_lp_splits(split_json='./guidance/lp_split/leakage_safe_split.json',
                    root='./data/crossdocked_v1.1_rmsd1.0_pocket10',
                    train_subsample=6000, seed=2021):
    with open(split_json) as f:
        d = json.load(f)

    base = PocketLigandPairDataset(root)
    pk_by_idx = {}
    splits = {}
    for part in ('train', 'val', 'test'):
        indices = []
        for idx, pk in d[part]:
            pk_by_idx[idx] = pk
            indices.append(idx)
        splits[part] = indices

    if train_subsample is not None and train_subsample < len(splits['train']):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(splits['train']), generator=g).tolist()[:train_subsample]
        splits['train'] = [splits['train'][i] for i in perm]

    return base, splits, pk_by_idx
