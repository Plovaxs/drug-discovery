"""Affinity-labeled subset of CrossDocked2020, for Task C's affinity guidance
model.

The repo's own PDBBindDataset (datasets/pdbbind.py) expects a raw PDBBind
refined/general-set download (pocket_fn/ligand_fn pairs listed in an
index.pkl) which we do not have -- PDBBind requires a registered download
from pdbbind.org.cn. Substituting for it: data/affinity_info.pkl (already
present in the repo, 184,087 entries) carries experimentally measured
pKd/pKi/pIC50 values ('pk') keyed by the same
"<TARGET>/<pdbid>_<chain>_rec_..._lig_..." identifiers CrossDocked2020's
ligand_filename uses (minus the .sdf suffix) -- so real binding-affinity
labels can be attached directly to CrossDocked2020 poses already downloaded
for the diffusion model, with no separate PDBBind download needed. Of
184,087 entries, ~76,800 have a real (nonzero) pk; cross-referenced against
the official train/test split (crossdocked_pocket10_pose_split.pt) that
gives 40,617 labeled training-split entries.

This is a deliberate substitution for literal PDBBind, documented per Task
C's instruction to note it clearly: labels originate from PDBBind-derived
affinity annotations mapped onto CrossDocked2020 docked/minimized poses,
not from PDBBind's own crystal structures directly.
"""
import pickle

import torch
from torch.utils.data import Dataset

from datasets.pl_pair_dataset import PocketLigandPairDataset


def _ligand_key(ligand_filename):
    assert ligand_filename.endswith('.sdf')
    return ligand_filename[:-4]


def build_labeled_splits(root='./data/crossdocked_v1.1_rmsd1.0_pocket10',
                          split_path='./data/crossdocked_pocket10_pose_split.pt',
                          affinity_path='./data/affinity_info.pkl',
                          val_fraction=0.1, seed=2021, train_subsample=None):
    """Returns (base_dataset, {'train': [...], 'val': [...], 'test': [...]},
    {idx: pk}) — index lists into base_dataset, restricted to entries with a
    real (nonzero) experimental pk, and pk values for those indices.

    'val' is carved out of the official train split (90/10) since the
    official split has no val subset and only 27 of the official 100 test
    pockets carry a real affinity label (too few to serve alone as a
    reliable held-out affinity test set) — the 'test' split returned here
    is that small 27-pocket set, kept separate and clearly small, while
    'val' does the actual model-selection work during training.

    `train_subsample`, if given, randomly subsamples the training pool to
    this many entries (post train/val split) — trades off how much of the
    labeled data an epoch sees against how many epochs (and therefore how
    much validation-loss signal for early stopping) fit in a given time
    budget. On an RTX 3050, batch_size=1, ~11 it/s: 36,556 unsubsampled
    training entries is ~55 min/epoch; subsampling to e.g. 6,000 brings
    that to ~9 min/epoch. Val/test are never subsampled.
    """
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

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_labeled), generator=g).tolist()
    n_val = max(1, int(len(train_labeled) * val_fraction))
    val_idx_pos = set(perm[:n_val])

    train_final, val_final = [], []
    pk_by_idx = {}
    for i, (idx, pk) in enumerate(train_labeled):
        pk_by_idx[idx] = pk
        (val_final if i in val_idx_pos else train_final).append(idx)
    test_final = []
    for idx, pk in test_labeled:
        pk_by_idx[idx] = pk
        test_final.append(idx)

    if train_subsample is not None and train_subsample < len(train_final):
        sub_perm = torch.randperm(len(train_final), generator=g).tolist()[:train_subsample]
        train_final = [train_final[i] for i in sub_perm]

    return base, {'train': train_final, 'val': val_final, 'test': test_final}, pk_by_idx


class CrossDockedAffinityDataset(Dataset):
    """Wraps PocketLigandPairDataset, attaching `.y` (pk) to each item."""

    def __init__(self, base_dataset, indices, pk_by_idx, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pk_by_idx = pk_by_idx
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        data = self.base_dataset[idx].clone()
        data.y = torch.tensor(self.pk_by_idx[idx], dtype=torch.float)
        data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data
