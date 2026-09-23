"""Track B1 (redefined, per the council-consultation addendum superseding
the original reference-affinity-subtraction framing): gradient-norm
normalization, a shared transform applicable to ANY guidance model's
grad_log_score output.

Motivation (empirically documented, not hypothetical): every guidance
model built in this project so far (Stage 0's EGNN, Track A's GIGN+PIGNet2
physics-anchored model) has shown a different, hard-to-predict-in-advance
safe lambda range and raw gradient magnitude -- Track A's physics model
fragments by lambda=3.0, roughly an order of magnitude below Stage 0's
EGNN (stable to ~10-30). This makes it impossible to know, without a fresh
sweep every time, whether "no guidance effect" reflects an uninformative
gradient DIRECTION or simply a badly-scaled raw magnitude that never gets
a fair chance to act before either doing nothing or breaking the molecule.
This module decouples "how strong is the push" (lambda, in a consistent
unit) from "how large does this particular model's raw gradient happen to
be" (an accident of that model's own training/loss scale).
"""
import torch
from torch_scatter import scatter_mean


def normalize_unit_per_atom(grad_pos, grad_v, batch_ligand, eps=1e-6):
    """Rescales grad_pos and grad_v so each SAMPLE's (not whole-dataset,
    not whole-batch) average per-atom gradient magnitude is 1.

    Per-SAMPLE (not per-whole-molecule-then-shared-across-all-molecules-in-
    a-batch) normalization matters specifically because it keeps guidance
    strength comparable across differently-sized ligands within the same
    batch/run -- a single whole-batch norm would implicitly make guidance
    strength depend on ligand size (a larger molecule's gradient would be
    spread thinner per atom under a shared/global norm), reintroducing a
    version of the same size-confound this project is trying to control
    for elsewhere (see the separate, standalone size-confound diagnostic,
    guidance/diag_size_confound.py -- NOT the same thing as this
    normalization, which only rescales magnitude, not direction).

    grad_pos: (n_atoms_in_batch, 3). grad_v: (n_atoms_in_batch, n_classes).
    batch_ligand: (n_atoms_in_batch,) long, sample index per atom.
    Position and categorical/type-logit components are normalized
    SEPARATELY (different scales/meaning), each independently to a
    per-sample average-per-atom norm of 1.
    """
    n_samples = int(batch_ligand.max().item()) + 1 if batch_ligand.numel() > 0 else 0

    pos_norm = grad_pos.norm(dim=-1)  # (n_atoms,)
    mean_pos_norm = scatter_mean(pos_norm, batch_ligand, dim=0, dim_size=n_samples)  # (n_samples,)
    grad_pos_normalized = grad_pos / (mean_pos_norm[batch_ligand].unsqueeze(-1) + eps)

    v_norm = grad_v.norm(dim=-1)  # (n_atoms,)
    mean_v_norm = scatter_mean(v_norm, batch_ligand, dim=0, dim_size=n_samples)  # (n_samples,)
    grad_v_normalized = grad_v / (mean_v_norm[batch_ligand].unsqueeze(-1) + eps)

    return grad_pos_normalized, grad_v_normalized
