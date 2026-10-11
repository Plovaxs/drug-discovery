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
    seen_in = {}
    for part in ('train', 'val', 'test'):
        indices = []
        for idx, pk in d[part]:
            # Fail closed on any index that appears twice. `pk_by_idx[idx] = pk` keeps whichever entry
            # came last, so without this check an index present in both train and test -- which IS
            # train/test leakage -- would be silently collapsed to one label by the very structure that
            # exists to prevent leakage. 37 modules import this function, so a masked overlap here would
            # travel into every number in the thesis.
            #
            # The shipped leakage_safe_split.json is clean: 46,964 / 6,069 / 11,855 indices, pairwise
            # disjoint, no internal duplicates, no conflicting pK, and its 779 / 65 / 127 targets are
            # disjoint too. The guard costs nothing today and keeps that true if the split is rebuilt.
            prev = seen_in.get(idx)
            if prev is not None:
                where = (f'twice inside {part!r}' if prev == part
                         else f'in both {prev!r} and {part!r}')
                raise ValueError(
                    f'ABORT: index {idx} appears {where} of {split_json}. '
                    + ('A duplicated index inflates that partition and makes its pK ambiguous.'
                       if prev == part else
                       'That is train/test leakage; pk_by_idx would have hidden it by keeping only the '
                       'last value.')
                    + ' Rebuild the split before training or evaluating anything on it.')
            seen_in[idx] = part
            pk_by_idx[idx] = pk
            indices.append(idx)
        splits[part] = indices

    if train_subsample is not None and train_subsample < len(splits['train']):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(splits['train']), generator=g).tolist()[:train_subsample]
        splits['train'] = [splits['train'][i] for i in perm]

    return base, splits, pk_by_idx
