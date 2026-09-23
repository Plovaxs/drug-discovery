"""Track D, Step D.2a: builds a leakage-safe train/val/test split for the
synthesizability (RA-score) guidance model, reusing the SAME target-level
leakage-safe assignment already computed for the affinity side
(leakage_safe_split.json) -- no new RA-score computation, no new
similarity/leakage analysis needed, since leakage-safety is a property of
which POCKETS/TARGETS land in which split, independent of which label
(pk vs. RA-score) is attached to entries within them.

Fixes the gap found in Track D's D.2 status re-establishment
(guidance/TRACK_D_STATUS_D2.md): datasets/synth_dataset.py's
build_synth_splits() does a plain random 90/10 train/val split directly
on the RA-score-labeled pool, with NO held-out test set and NO leakage
check at all -- the exact category of problem Stage 0 was built to fix
on the affinity side.
"""
import json

import torch

from datasets.pl_pair_dataset import PocketLigandPairDataset


def build_synth_lp_splits(split_json='./guidance/lp_split/leakage_safe_split.json',
                          labels_path='./data/synth_ra_labels.pkl',
                          root='./data/crossdocked_v1.1_rmsd1.0_pocket10'):
    """Returns (base_dataset, {'train':[...], 'val':[...], 'test':[...]},
    {idx: ra_score}) -- same interface shape as
    guidance/lp_split/lp_split_loader.build_lp_splits, so
    CrossDockedAffinityDataset (datasets/crossdocked_affinity.py) can be
    reused unchanged, just with RA-score labels instead of pk labels
    (that class only cares that `.y` is a float, not what it means).
    """
    import pickle

    with open(split_json) as f:
        d = json.load(f)
    train_targets, val_targets, test_targets = (
        set(d['train_targets']), set(d['val_targets']), set(d['test_targets']))

    with open(labels_path, 'rb') as f:
        ra_labels = pickle.load(f)  # {idx: ra_score}

    base = PocketLigandPairDataset(root)

    splits = {'train': [], 'val': [], 'test': []}
    ra_by_idx = {}
    discarded = 0
    for idx, ra_score in ra_labels.items():
        target = base[idx].ligand_filename.split('/')[0]
        if target in train_targets:
            splits['train'].append(idx)
        elif target in val_targets:
            splits['val'].append(idx)
        elif target in test_targets:
            splits['test'].append(idx)
        else:
            discarded += 1
            continue
        ra_by_idx[idx] = float(ra_score)

    return base, splits, ra_by_idx, discarded


if __name__ == '__main__':
    base, splits, ra_by_idx, discarded = build_synth_lp_splits()
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  "
         f"Test: {len(splits['test'])}  Discarded: {discarded}")
