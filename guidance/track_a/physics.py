"""Physics energy-term functions, ported near-verbatim from PIGNet2
(github.com/ACE-KAIST/PIGNet2, src/models/physics.py) per
guidance/reference_code/IMPLEMENTATION_NOTES.md's recommendation to reuse
these close to verbatim rather than re-derive from the paper text (the
paper prose does not match the shipped formulas in several details, listed
in that notes file). Framework-agnostic (only needs torch), so this is a
straight copy with the ionic term and dvdw-radii correction removed since
the released pignet_morse.yaml config disables both by default
(include_ionic=false, dev_vdw_radii_coeff=0.0) and this project's
exploratory-tier build does not attempt to reproduce them.
"""
from typing import Union

import torch

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
BoolTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]


def interaction_edges(ligand_mask: BoolTensor, batch: LongTensor) -> LongTensor:
    """Uni-directional, fully-connected ligand->protein edges, per graph.
    Returns (2, node_pairs)."""
    device = ligand_mask.device
    nodes = torch.arange(ligand_mask.numel(), device=device)
    edges = torch.tensor([], dtype=torch.long, device=device)

    for i in range(int(batch.max().item()) + 1):
        batch_mask = batch == i
        ligand_nodes = nodes[ligand_mask & batch_mask]
        target_nodes = nodes[~ligand_mask & batch_mask]
        edges_per_graph = torch.cartesian_prod(ligand_nodes, target_nodes)
        edges = torch.cat((edges, edges_per_graph))

    return edges.t().contiguous()


def distances(pos: FloatTensor, edge_index: LongTensor) -> FloatTensor:
    pos1 = pos[edge_index[0]]
    pos2 = pos[edge_index[1]]
    D = torch.sqrt(torch.pow(pos1 - pos2, 2).sum(-1) + 1e-10)
    return D


def morse_potential(
    D: FloatTensor, R: FloatTensor, E: Union[float, FloatTensor],
    A: Union[float, FloatTensor], short_range_A: float = None,
) -> FloatTensor:
    """vdW term. E = well depth, A = width, R = pairwise vdW-radii sum."""
    energy = (1 - torch.exp(-A * (D - R))) ** 2 - 1
    if short_range_A is not None:
        energy2 = (1 - torch.exp(-short_range_A * (D - R))) ** 2 - 1
        energy = torch.where(D > R, energy, energy2)
    energy = energy.clamp(max=100.0)
    energy = energy * E
    return energy


def linear_potential(
    D: FloatTensor, R: FloatTensor, E: Union[float, FloatTensor],
    c1: float, c2: float,
) -> FloatTensor:
    """H-bond / metal-ligand / hydrophobic term: linear ramp of E between
    cutoffs c1, c2 (measured relative to D-R), clamped to [0, E]."""
    energy = (D - R - c2) / (c1 - c2)
    energy = energy.clamp(min=0.0, max=1.0)
    energy = energy * E
    return energy


def interaction_masks(
    metal_mask: BoolTensor, h_donor_mask: BoolTensor, h_acceptor_mask: BoolTensor,
    hydrophobic_mask: BoolTensor, edge_index: LongTensor,
) -> BoolTensor:
    """Returns (4, pairs): vdW / H-bond / metal-ligand / hydrophobic masks.
    Ionic term omitted (see module docstring)."""

    def combine(m1, m2=None):
        if m2 is None:
            return m1[edge_index[0]] & m1[edge_index[1]]
        m = m1[edge_index[0]] & m2[edge_index[1]]
        m = m | (m2[edge_index[0]] & m1[edge_index[1]])
        return m

    masks = torch.stack((
        combine(~metal_mask),
        combine(h_donor_mask & (~metal_mask), h_acceptor_mask & (~metal_mask)),
        combine(metal_mask, h_acceptor_mask & (~metal_mask)),
        combine(hydrophobic_mask),
    ))
    return masks


def soft_interaction_masks(
    metal_prob: FloatTensor, h_donor_prob: FloatTensor, h_acceptor_prob: FloatTensor,
    hydrophobic_prob: FloatTensor, edge_index: LongTensor,
) -> FloatTensor:
    """Continuous relaxation of interaction_masks, for use when atom identity
    is a soft distribution (guidance-time v0-hat) rather than a hard type --
    e.g. is_metal becomes P(atom is metal) in [0,1]. Products replace ANDs,
    "a+b-a*b" (probabilistic OR) replaces ORs. Not part of PIGNet2's own
    code -- a necessary addition for this project's guidance use case,
    documented as such. Returns (4, pairs) float in [0,1]."""

    def p_and(a, b):
        return a * b

    def p_or(a, b):
        return a + b - a * b

    def combine(m1, m2=None):
        if m2 is None:
            return p_and(m1[edge_index[0]], m1[edge_index[1]])
        term1 = p_and(m1[edge_index[0]], m2[edge_index[1]])
        term2 = p_and(m2[edge_index[0]], m1[edge_index[1]])
        return p_or(term1, term2)

    not_metal = 1.0 - metal_prob
    masks = torch.stack((
        combine(not_metal),
        combine(p_and(h_donor_prob, not_metal), p_and(h_acceptor_prob, not_metal)),
        combine(metal_prob, p_and(h_acceptor_prob, not_metal)),
        combine(hydrophobic_prob),
    ))
    return masks
