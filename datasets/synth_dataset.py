"""Dataset wrapper for Task D: CrossDocked2020 ligands labeled with real
RA-score (see guidance/build_synth_dataset.py for label generation).
Ligand-only -- no protein featurization needed since synthesizability, per
guidance/synth_model.py's design, is a property of the molecule alone.
"""
import pickle

import torch
from torch.utils.data import Dataset

from datasets.pl_pair_dataset import PocketLigandPairDataset


def build_synth_splits(root='./data/crossdocked_v1.1_rmsd1.0_pocket10',
                        labels_path='./data/synth_ra_labels.pkl',
                        val_fraction=0.1, seed=2021):
    base = PocketLigandPairDataset(root)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)  # {idx: ra_score}

    indices = list(labels.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(indices), generator=g).tolist()
    n_val = max(1, int(len(indices) * val_fraction))
    val_pos = set(perm[:n_val])

    train_idx, val_idx = [], []
    for i, idx in enumerate(indices):
        (val_idx if i in val_pos else train_idx).append(idx)

    return base, {'train': train_idx, 'val': val_idx}, labels


class SynthDataset(Dataset):
    def __init__(self, base_dataset, indices, label_by_idx, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_by_idx = label_by_idx
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        data = self.base_dataset[idx].clone()
        data.y = torch.tensor(self.label_by_idx[idx], dtype=torch.float)
        data.id = idx
        if self.transform is not None:
            data = self.transform(data)
        return data
