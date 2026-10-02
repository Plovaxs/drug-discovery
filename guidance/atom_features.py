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

`use_soft_v=True` (added as a follow-up, post-thesis exploration branch):
v0_to_ligand_feature_soft() replaces the hard argmax (wrapped in
torch.no_grad() in v0_to_ligand_feature above, which makes grad_v
structurally zero regardless of what guided_sampling.py does with it --
this is not a small-magnitude effect, the computation graph is disconnected
from v0 entirely) with softmax-weighted expectations, so element identity,
atomic number and aromaticity all carry a real gradient back to v0. Degree/
NumHs/Hybridization are unaffected (still placeholder or bond-aware,
independent of this flag) since those already have no dependence on v0 to
begin with (they come from pos or a fixed constant).
"""
import torch

from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC

ELEMENT_ORDER = [1, 6, 7, 8, 9, 15, 16, 17]  # must match FeaturizeLigandAtom.atomic_numbers
DEGREE_CLASSES = 6
NUM_HS_CLASSES = 6

# Precomputed once: for each of the 13 'add_aromatic' logit indices, its
# atomic number and aromaticity flag (constants, not learned).
_INDEX_ATOMIC_NUMBER = None
_INDEX_IS_AROMATIC = None


def _index_tables(device, dtype):
    global _INDEX_ATOMIC_NUMBER, _INDEX_IS_AROMATIC
    if _INDEX_ATOMIC_NUMBER is None or _INDEX_ATOMIC_NUMBER.device != device:
        n_idx = len(MAP_INDEX_TO_ATOM_TYPE_AROMATIC)
        _INDEX_ATOMIC_NUMBER = torch.tensor(
            [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in range(n_idx)], device=device, dtype=dtype)
        _INDEX_IS_AROMATIC = torch.tensor(
            [float(MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1]) for i in range(n_idx)], device=device, dtype=dtype)
    return _INDEX_ATOMIC_NUMBER, _INDEX_IS_AROMATIC

_bond_estimator_cache = {}


def _get_bond_estimator(hybridization_dim, device):
    key = (hybridization_dim, str(device))
    if key not in _bond_estimator_cache:
        from guidance.bond_estimator import BondAwareFeatureEstimator
        _bond_estimator_cache[key] = BondAwareFeatureEstimator(
            hybridization_dim, DEGREE_CLASSES, NUM_HS_CLASSES, device=device)
    return _bond_estimator_cache[key]


def v0_to_ligand_feature(v0, ligand_atom_feature_dim, pos=None, use_bond_aware=False, use_soft_v=False):
    """v0: (N, 13) diffusion 'add_aromatic' logits.
    Returns (N, ligand_atom_feature_dim) matching utils.transforms_prop's
    FeaturizeLigandAtom output layout: [element one-hot(8), atomic_number/100,
    aromatic(0/1), degree(6), num_hs(6), hybridization(remaining dims)].

    `use_bond_aware=True` (requires `pos`, the same positions v0 was
    predicted from) estimates degree/num_hs/hybridization from geometry,
    differentiably w.r.t. `pos`; otherwise they are a fixed placeholder.

    `use_soft_v=False` (default): element/atomic-number/aromatic are read
    off v0's argmax inside torch.no_grad(), so the returned feature (and
    therefore any score computed from it) carries no gradient back to v0
    at all -- not a small gradient, a structurally absent one.
    `use_soft_v=True`: those three fields are instead softmax-weighted
    expectations over v0 (see v0_to_ligand_feature_soft), which do carry a
    real gradient back to v0.
    """
    if use_soft_v:
        return v0_to_ligand_feature_soft(v0, ligand_atom_feature_dim, pos=pos, use_bond_aware=use_bond_aware)

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


def v0_to_ligand_feature_soft(v0, ligand_atom_feature_dim, pos=None, use_bond_aware=False):
    """Differentiable-w.r.t.-v0 relaxation of v0_to_ligand_feature. Element
    identity, atomic number and aromaticity are softmax-weighted
    expectations over the 13 'add_aromatic' logit classes, instead of a
    hard argmax under no_grad -- this is the same style of relaxation
    DIAG1's expected_heaviness() uses for its size-confound diagnostic
    (guidance/diag_size_confound.py), applied here to build an actual
    forward-pass input instead of a post-hoc analysis quantity.

    A trained model's ligand-atom encoder was never shown this kind of
    soft, blended input at training time (real molecules have a hard
    element identity) -- this is an out-of-training-distribution input by
    construction, exactly as the fixed placeholder for Degree/NumHs/
    Hybridization already is. Disclosed, not hidden: this is why it is an
    exploratory follow-up branch, not a change adopted as a new default.
    """
    hybridization_dim = ligand_atom_feature_dim - (
        len(ELEMENT_ORDER) + 1 + 1 + DEGREE_CLASSES + NUM_HS_CLASSES)
    atomic_number_table, is_aromatic_table = _index_tables(v0.device, v0.dtype)
    probs = torch.softmax(v0, dim=-1)  # (N, 13)

    element_onehot = torch.stack(
        [(probs * (atomic_number_table == z).to(probs.dtype)).sum(dim=-1) for z in ELEMENT_ORDER],
        dim=-1)  # (N, 8), soft distribution over elements (sums to 1 per atom)
    atomic_number_feat = ((probs * atomic_number_table).sum(dim=-1) / 100.).unsqueeze(-1)
    aromatic_feat = (probs * is_aromatic_table).sum(dim=-1).unsqueeze(-1)

    # Degree/NumHs/Hybridization: same options as the hard path. With
    # use_bond_aware=True, atomic_numbers must be a hard per-atom scalar
    # for the estimator's element-pair lookup, so a detached argmax is
    # used there ONLY (does not affect the element/atomic-number/aromatic
    # gradient computed above).
    if use_bond_aware:
        assert pos is not None, 'use_bond_aware=True requires pos'
        with torch.no_grad():
            atomic_numbers_hard = atomic_number_table[v0.argmax(dim=-1)]
        estimator = _get_bond_estimator(hybridization_dim, v0.device)
        degree_feat, numhs_feat, hybrid_feat = estimator(pos, atomic_numbers_hard)
    else:
        n = v0.size(0)
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
