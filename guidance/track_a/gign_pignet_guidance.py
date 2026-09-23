"""Frozen Track A guidance model: wraps the GIGN+PIGNet2 hybrid
(guidance/track_a/model.py, trained by train_gign_pignet_stage2.py) for use
inside the guided sampling loop (guidance/guided_sampling.py), following the
exact same GuidanceModel interface as guidance/affinity_guidance.py's
AffinityGuidance.

Design choices carried over unchanged from AffinityGuidance (not
re-derived, since they are already-established, already-documented scope
decisions for this project's guidance mechanism, not choices specific to
this architecture):

- grad_v is always zero -- guidance nudges ligand POSITION only, not
  composition. Same reasoning as AffinityGuidance's docstring: the
  diffusion core's v0 estimate only carries element identity + aromaticity,
  never bond connectivity, so there is no faithful differentiable path from
  v0 into the fields (Degree/NumHs/Hybridization) this model's ligand
  featurizer expects beyond that. guidance/atom_features.py's
  v0_to_ligand_feature (reused directly here) fills those with the same
  fixed placeholder AffinityGuidance already uses.

Known, documented train/inference featurization gap (same CATEGORY of
issue guidance/SYNTH_GUIDANCE_FINDING.md already flagged for Task D's synth
guidance -- disclosed here rather than silently repeated): training builds
ligand "intra" (covalent) edges from REAL RDKit-derived ligand_bond_index
(see guidance/track_a/features.py's build_intra_inter_edges), but at
guidance time no bonds exist yet -- the diffusion process only has atom
positions and a soft type distribution. This wrapper approximates ligand
intra edges at guidance time with the SAME short-distance-cutoff proxy
guidance/track_a/features.py already uses for protein intra edges (which
also have no real bond list available) -- one consistent approximation
used uniformly for both node types at guidance time, rather than two
different ad hoc mechanisms. This is a real train/inference mismatch,
disclosed rather than hidden; if Track A's exploratory checkpoint shows an
ambiguous or null result, this is one of the first things to revisit
(e.g. via guidance/bond_estimator.py's already-built differentiable soft
bond-order estimator, not used here to keep this exploratory-tier wrapper
simple and bounded in scope).
"""
import torch

from guidance.interfaces import GuidanceModel
from guidance.atom_features import v0_to_ligand_feature
from guidance.gradient_normalization import normalize_unit_per_atom
from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC
from guidance.track_a.model import GIGNPignetAffinity
from guidance.track_a.features import (
    protein_interaction_masks, vdw_radii_for, PROTEIN_INTRA_CUTOFF,
)

_PROTEIN_ELEMENT_ORDER = [1, 6, 7, 8, 16, 34]  # must match utils/transforms(_prop).FeaturizeProteinAtom


class GIGNPignetGuidance(GuidanceModel):
    def __init__(self, ckpt_path, device='cuda:0', normalize_gradient=False):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.model = GIGNPignetAffinity(
            protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
            ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
            hidden_dim=ckpt['hidden_dim'],
        ).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._ligand_atom_feature_dim = ckpt['ligand_atom_feature_dim']
        # Track B1 (gradient-norm normalization, guidance/gradient_normalization.py).
        self.normalize_gradient = normalize_gradient

    def _ligand_atomic_numbers(self, v0):
        with torch.no_grad():
            idx = v0.argmax(dim=-1).tolist()
            atomic_numbers = torch.tensor(
                [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in idx], device=v0.device)
        return atomic_numbers

    def _protein_atomic_numbers(self, protein_v):
        with torch.no_grad():
            elem_idx = protein_v[:, :len(_PROTEIN_ELEMENT_ORDER)].argmax(dim=-1)
            atomic_numbers = torch.tensor(_PROTEIN_ELEMENT_ORDER, device=protein_v.device)[elem_idx]
        return atomic_numbers

    def _build_edges_one_graph(self, ligand_pos, protein_pos):
        n_l = ligand_pos.size(0)
        with torch.no_grad():
            ligand_dist = torch.cdist(ligand_pos, ligand_pos)
            li, lj = torch.where((ligand_dist < PROTEIN_INTRA_CUTOFF) & (ligand_dist > 1e-6))
            protein_dist = torch.cdist(protein_pos, protein_pos)
            pi, pj = torch.where((protein_dist < PROTEIN_INTRA_CUTOFF) & (protein_dist > 1e-6))
            inter_dist = torch.cdist(ligand_pos, protein_pos)
            ii, jj = torch.where(inter_dist < 5.0)
        edge_index_intra = torch.cat([
            torch.stack([li, lj]),
            torch.stack([pi + n_l, pj + n_l]),
        ], dim=1)
        edge_index_inter = torch.cat([
            torch.stack([ii, jj + n_l]),
            torch.stack([jj + n_l, ii]),
        ], dim=1)
        return edge_index_intra, edge_index_inter

    def _build_batched_edges(self, ligand_pos, protein_pos, batch_ligand, batch_protein):
        n_graphs = int(batch_ligand.max().item()) + 1
        intra_list, inter_list = [], []
        l_offset, p_offset = 0, 0
        n_l_total = ligand_pos.size(0)
        for g in range(n_graphs):
            l_mask = batch_ligand == g
            p_mask = batch_protein == g
            n_l, n_p = int(l_mask.sum()), int(p_mask.sum())
            intra, inter = self._build_edges_one_graph(ligand_pos[l_mask], protein_pos[p_mask])
            # intra/inter are 0-indexed within this graph's own [ligand;protein]
            # block; shift ligand-side indices by l_offset and protein-side by
            # (n_l_total - n_l + p_offset) to land in the final concatenated
            # [all_ligand; all_protein] node ordering the model expects.
            intra_shifted = _shift_edges(intra, n_l, l_offset, n_l_total, p_offset)
            inter_shifted = _shift_edges(inter, n_l, l_offset, n_l_total, p_offset)
            intra_list.append(intra_shifted)
            inter_list.append(inter_shifted)
            l_offset += n_l
            p_offset += n_p
        edge_index_intra = torch.cat(intra_list, dim=1)
        edge_index_inter = torch.cat(inter_list, dim=1)
        return edge_index_intra, edge_index_inter

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        pos0_req = pos0.detach().clone().requires_grad_(True)
        protein_pos_det = protein_pos.detach()

        ligand_feat = v0_to_ligand_feature(
            v0, self._ligand_atom_feature_dim, pos=pos0_req, use_bond_aware=False)
        protein_feat = protein_v.detach()

        ligand_z = self._ligand_atomic_numbers(v0)
        protein_z = self._protein_atomic_numbers(protein_v)

        # Ligand NumHs is unknown at guidance time (same placeholder gap as
        # v0_to_ligand_feature's Degree/NumHs/Hybridization slots -- see
        # module docstring), so is_h_donor (which needs NumHs>0) is always
        # False here, not computed via features.ligand_interaction_masks
        # (which expects a real NumHs column this wrapper doesn't have).
        # Acceptor/hydrophobic/metal don't depend on NumHs and are computed
        # identically to the training-time featurizer.
        is_n_or_o = (ligand_z == 7) | (ligand_z == 8)
        l_donor = torch.zeros_like(ligand_z, dtype=torch.bool)
        l_acceptor = is_n_or_o
        l_hydrophobic = (ligand_z == 6) | (ligand_z == 9) | (ligand_z == 16) | (ligand_z == 17) | \
            (ligand_z == 35) | (ligand_z == 53)
        l_metal = torch.zeros_like(ligand_z, dtype=torch.bool)
        p_metal, p_donor, p_acceptor, p_hydrophobic = protein_interaction_masks(protein_z)

        is_metal = torch.cat([l_metal, p_metal])
        is_h_donor = torch.cat([l_donor, p_donor])
        is_h_acceptor = torch.cat([l_acceptor, p_acceptor])
        is_hydrophobic = torch.cat([l_hydrophobic, p_hydrophobic])
        vdw_radii = torch.cat([vdw_radii_for(ligand_z), vdw_radii_for(protein_z)]).to(self.device)

        edge_index_intra, edge_index_inter = self._build_batched_edges(
            pos0_req, protein_pos_det, batch_ligand, batch_protein)

        pred, energies = self.model(
            protein_pos=protein_pos_det, protein_feat=protein_feat,
            ligand_pos=pos0_req, ligand_feat=ligand_feat,
            edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
            batch_ligand=batch_ligand, batch_protein=batch_protein,
            vdw_radii=vdw_radii, is_metal=is_metal, is_h_donor=is_h_donor,
            is_h_acceptor=is_h_acceptor, is_hydrophobic=is_hydrophobic,
        )  # pred: (num_graphs,), higher = better predicted binding (pk-scale)
        score = pred.sum()
        grad_pos, = torch.autograd.grad(score, pos0_req)
        grad_v = torch.zeros_like(v0)
        if self.normalize_gradient:
            grad_pos, grad_v = normalize_unit_per_atom(grad_pos, grad_v, batch_ligand)
        return grad_pos, grad_v


def _shift_edges(edges, n_l, l_offset, n_l_total, p_offset):
    if edges.numel() == 0:
        return edges
    is_ligand_side = edges < n_l
    shift = torch.where(is_ligand_side, l_offset, n_l_total - n_l + p_offset)
    return edges + shift
