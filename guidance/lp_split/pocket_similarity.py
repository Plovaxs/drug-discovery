"""Binding-site residue-composition similarity: the documented substitute
for ProBiS-style pocket structural alignment (unavailable in this
environment -- see FLAGSHIP_ARCHITECTURE_RESEARCH.md's Stage 0 section).

For each target, aggregates a normalized 20-dim amino-acid-type histogram
over all atoms in its pocket10 crop(s) (`protein_atom_to_aa_type`, already
present in the processed CrossDocked2020 data -- no residue-index field
exists in the dataset, so this operates at atom-type-composition
granularity rather than true per-residue identity, a coarser signal than
ProBiS's structural alignment but directly available with no new
dependency). Pocket similarity between two targets is the cosine
similarity of these histograms.
"""
import numpy as np
import torch

N_AA_TYPES = 20


def compute_target_histogram(base_dataset, indices_for_target):
    """Aggregates the AA-type histogram across all given pocket instances
    (indices into base_dataset) belonging to one target."""
    hist = np.zeros(N_AA_TYPES, dtype=np.float64)
    for idx in indices_for_target:
        aa_types = base_dataset[idx].protein_atom_to_aa_type
        for t in aa_types.tolist():
            if 0 <= t < N_AA_TYPES:
                hist[t] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def build_target_histograms(base_dataset, target_to_indices, verbose=True):
    """target_to_indices: {target_dir_name: [idx, ...]}. Returns
    {target_dir_name: np.ndarray[20]}."""
    histograms = {}
    for i, (target, indices) in enumerate(target_to_indices.items()):
        histograms[target] = compute_target_histogram(base_dataset, indices)
        if verbose and (i + 1) % 200 == 0:
            print(f'  pocket histograms: {i + 1}/{len(target_to_indices)}')
    return histograms


def pocket_similarity(hist_a, hist_b):
    na, nb = np.linalg.norm(hist_a), np.linalg.norm(hist_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(hist_a, hist_b) / (na * nb))
