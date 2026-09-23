"""Shared helper: converts the diffusion core's per-step ligand-type
estimate (v0, 'add_aromatic' mode logits) into the continuous ligand
feature vector the property-prediction-style guidance models (Task C's
AffinityGuidance, Task D's SynthGuidance) expect as input.

Two feature modes for the Degree/NumHs/Hybridization slots (element,
atomic number and aromaticity are always derived the same way -- from
v0's argmax, which is inherently non-differentiable regardless of mode):

- placeholder (default, `use_bond_aware=False`): a fixed "unknown"
  one-hot, as before. Kept as the default and fully intact so the
  originally-documented configuration (see
  guidance/SYNTH_GUIDANCE_FINDING.md) remains exactly reproducible.
- bond-aware (`use_bond_aware=True`, requires `pos`): estimates these
  fields from geometry via guidance/bond_estimator.py, differentiably
  w.r.t. `pos` -- see that module's docstring for the method and its
  approximations. This is Phase 1 of the architectural-upgrade addendum,
  commissioned specifically to address the train/inference feature
  mismatch SYNTH_GUIDANCE_FINDING.md traces the score-divergence problem
  to.
"""
import torch

from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC

ELEMENT_ORDER = [1, 6, 7, 8, 9, 15, 16, 17]  # must match FeaturizeLigandAtom.atomic_numbers
DEGREE_CLASSES = 6
NUM_HS_CLASSES = 6

_bond_estimator_cache = {}


def _get_bond_estimator(hybridization_dim, device):
    key = (hybridization_dim, str(device))
    if key not in _bond_estimator_cache:
        from guidance.bond_estimator import BondAwareFeatureEstimator
        _bond_estimator_cache[key] = BondAwareFeatureEstimator(
            hybridization_dim, DEGREE_CLASSES, NUM_HS_CLASSES, device=device)
    return _bond_estimator_cache[key]


def v0_to_ligand_feature(v0, ligand_atom_feature_dim, pos=None, use_bond_aware=False):
    """v0: (N, 13) diffusion 'add_aromatic' logits.
    Returns (N, ligand_atom_feature_dim) matching utils.transforms_prop's
    FeaturizeLigandAtom output layout: [element one-hot(8), atomic_number/100,
    aromatic(0/1), degree(6), num_hs(6), hybridization(remaining dims)].

    `use_bond_aware=True` (requires `pos`, the same positions v0 was
    predicted from) estimates degree/num_hs/hybridization from geometry,
    differentiably w.r.t. `pos`; otherwise they are a fixed placeholder,
    and the whole function is non-differentiable (element/atomic-number/
    aromatic are always non-differentiable either way, from v0's argmax).
    """
    hybridization_dim = ligand_atom_feature_dim - (
        len(ELEMENT_ORDER) + 1 + 1 + DEGREE_CLASSES + NUM_HS_CLASSES)
    with torch.no_grad():
        idx = v0.argmax(dim=-1).tolist()
        atomic_numbers = torch.tensor(
            [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in idx], device=v0.device)
        is_aromatic = torch.tensor(
            [float(MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1]) for i in idx], device=v0.device)

        element_onehot = torch.stack(
            [atomic_numbers == z for z in ELEMENT_ORDER], dim=-1).float()
        atomic_number_feat = (atomic_numbers.float() / 100.).unsqueeze(-1)
        aromatic_feat = is_aromatic.unsqueeze(-1)

    if use_bond_aware:
        assert pos is not None, 'use_bond_aware=True requires pos'
        estimator = _get_bond_estimator(hybridization_dim, v0.device)
        degree_feat, numhs_feat, hybrid_feat = estimator(pos, atomic_numbers)
    else:
        n = len(idx)
        degree_feat = torch.zeros(n, DEGREE_CLASSES, device=v0.device)
        degree_feat[:, 0] = 1.0
        numhs_feat = torch.zeros(n, NUM_HS_CLASSES, device=v0.device)
        numhs_feat[:, 0] = 1.0
        hybrid_feat = torch.zeros(n, hybridization_dim, device=v0.device)
        hybrid_feat[:, 0] = 1.0

    feat = torch.cat([
        element_onehot, atomic_number_feat, aromatic_feat,
        degree_feat, numhs_feat, hybrid_feat,
    ], dim=-1)
    return feat
