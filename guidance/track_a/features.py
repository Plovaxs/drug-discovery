"""Interaction-type masks, vdW radii, and intra/inter edge construction for
the GIGN+PIGNet2 hybrid, adapted to this project's CrossDocked-derived data
(which has no RDKit protein mol / SMARTS-matchable structure, unlike
PIGNet2's own PDBBind-based pipeline).

Documented departures from PIGNet2's src/data/data.py (which uses RDKit
SMARTS matching on a full protein+ligand mol pair -- not available here):

- Ligand masks are computed from the project's own raw per-atom features
  (`ligand_atom_feature`: [AtomicNumber, Aromatic, Degree, NumHs,
  Hybridization], see datasets/protein_ligand.py's ATOM_FEATS) via simple
  element+NumHs heuristics, not RDKit SMARTS matching. This is close to
  PIGNet2's H_DONOR_SMARTS ("any non-carbon heavy atom without a formal
  negative charge") / H_ACCEPTOR_SMARTS in spirit but coarser -- documented
  as a simplification appropriate to the exploratory tier, not a claim of
  exact parity.
- Protein atoms have no per-atom bond/hydrogen info in this project's
  pipeline at all (data.protein_element is the only available typing
  signal), so protein masks are pure element-based heuristics: N -> H-bond
  donor, O -> H-bond acceptor, C/S -> hydrophobic, no protein atom is ever
  flagged as a metal (pocket10 pockets rarely carry a genuine metal
  cofactor, and getting this exactly right would require re-parsing the
  raw PDB, out of scope for the exploratory tier).
- Protein "intra" (covalent-like) edges are NOT real covalent bonds -- this
  project's protein representation carries no explicit bond list -- but a
  short distance cutoff (PROTEIN_INTRA_CUTOFF) between protein atoms,
  approximating backbone/near-neighbor connectivity. Ligand intra edges use
  the project's real, RDKit-derived ligand_bond_index (exact, no
  approximation needed there).
"""
import torch

# PIGNet2's chem.VDW_RADII (github.com/ACE-KAIST/PIGNet2, src/data/chem.py),
# keyed by atomic number, kcal/mol-consistent Angstrom values used to
# reproduce their published numbers. Falls back to RDKit's periodic-table
# value for atomic numbers not in this curated set (matches their own
# get_vdw_radius's try/except fallback).
_VDW_RADII = {
    6: 1.90, 7: 1.8, 8: 1.7, 9: 1.5, 12: 1.2, 15: 2.1, 16: 2.0, 17: 1.8,
    20: 1.2, 25: 1.2, 26: 1.2, 27: 1.2, 28: 1.2, 29: 1.2, 30: 1.2, 35: 2.0, 53: 2.2,
}

PROTEIN_INTRA_CUTOFF = 2.0  # Angstrom; approximates covalent connectivity (see module docstring)
INTER_CUTOFF = 5.0  # Angstrom; matches GIGN's own dis_threshold default


def _rdkit_fallback_radius(atomic_number):
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetRvdw(int(atomic_number))


def vdw_radii_for(atomic_numbers: torch.Tensor) -> torch.Tensor:
    out = torch.empty(atomic_numbers.numel(), dtype=torch.float)
    for i, z in enumerate(atomic_numbers.tolist()):
        out[i] = _VDW_RADII.get(z, _rdkit_fallback_radius(z))
    return out


def ligand_interaction_masks(ligand_element: torch.Tensor, ligand_atom_feature: torch.Tensor):
    """ligand_atom_feature columns: [AtomicNumber, Aromatic, Degree, NumHs, Hybridization]
    (raw, pre-one-hot -- see datasets/protein_ligand.py get_ligand_atom_features)."""
    z = ligand_element
    num_hs = ligand_atom_feature[:, 3]
    is_n_or_o = (z == 7) | (z == 8)
    is_h_donor = is_n_or_o & (num_hs > 0)
    is_h_acceptor = is_n_or_o
    is_hydrophobic = (z == 6) | (z == 9) | (z == 16) | (z == 17) | (z == 35) | (z == 53)
    is_metal = torch.zeros_like(z, dtype=torch.bool)
    return is_metal, is_h_donor, is_h_acceptor, is_hydrophobic


def protein_interaction_masks(protein_element: torch.Tensor):
    z = protein_element
    is_h_donor = (z == 7)
    is_h_acceptor = (z == 8)
    is_hydrophobic = (z == 6) | (z == 16)
    is_metal = torch.zeros_like(z, dtype=torch.bool)
    return is_metal, is_h_donor, is_h_acceptor, is_hydrophobic


def build_intra_inter_edges(ligand_pos: torch.Tensor, protein_pos: torch.Tensor,
                            ligand_bond_index: torch.Tensor):
    """Returns (edge_index_intra, edge_index_inter), node ordering [ligand; protein]
    (ligand atoms first, indices 0..n_l-1; protein atoms n_l..n_l+n_p-1),
    matching GIGN's own dataset_GIGN.py convention."""
    n_l = ligand_pos.size(0)
    n_p = protein_pos.size(0)
    device = ligand_pos.device

    protein_dist = torch.cdist(protein_pos, protein_pos)
    pi, pj = torch.where((protein_dist < PROTEIN_INTRA_CUTOFF) & (protein_dist > 1e-6))
    protein_intra = torch.stack([pi + n_l, pj + n_l]) if pi.numel() > 0 else \
        torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index_intra = torch.cat([ligand_bond_index, protein_intra.to(device)], dim=1)

    inter_dist = torch.cdist(ligand_pos, protein_pos)
    li, pj2 = torch.where(inter_dist < INTER_CUTOFF)
    fwd = torch.stack([li, pj2 + n_l])
    bwd = torch.stack([pj2 + n_l, li])
    edge_index_inter = torch.cat([fwd, bwd], dim=1).to(device)

    return edge_index_intra, edge_index_inter
