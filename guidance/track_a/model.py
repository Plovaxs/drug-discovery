"""Track A: GIGN heterogeneous backbone + PIGNet2 physics-decomposition
head, spliced together per guidance/reference_code/IMPLEMENTATION_NOTES.md's
"Practical implication" section (the two repos' backbones are genuinely
different code; PIGNet2's own backbone is a GatedGAT+GRU stack, not GIGN's
HIL layers -- this project substitutes GIGN's backbone in because it is far
cheaper for the 4GB VRAM budget, then feeds its output embeddings into
PIGNet2's physics head instead of PIGNet2's own GNN's).

Design choices not dictated by either source repo (documented since this is
a genuine splice, not a straight port of either):
- Input embedding follows this project's OWN validated pattern from
  models/property_pred/prop_model.py (PropPredNet): separate
  protein_atom_emb/ligand_atom_emb Linear layers projecting this project's
  existing feature sets into a shared hidden_dim, rather than GIGN's single
  lin_node (which assumes one unified input feature space that this
  project's protein/ligand featurizers don't share).
- The vdW-radius correction network (PIGNet2's nn_dvdw) is omitted entirely,
  matching the released pignet_morse.yaml's dev_vdw_radii_coeff=0.0 (i.e.
  the correction is disabled in the actual config that produced PIGNet2's
  published numbers, so it isn't reproduced here either).
- No rotor penalty (needs ligand rotatable-bond count, not computed by this
  project's existing featurizer) and no ionic term (disabled in PIGNet2's
  own released config) -- both omitted, consistent with the shipped config
  rather than the paper's full term list.
- Final prediction is the raw sum of the 3 remaining physics terms
  (vdW + H-bond + metal-ligand + hydrophobic), with NO extra learned
  output head -- this matches PIGNet2's own "scoring" task type
  (objective=regression directly against sample.y on the summed energies,
  per IMPLEMENTATION_NOTES.md), preserving the physical interpretability
  that is the entire point of Track A vs. the Stage 0 black-box EGNN.
"""
import torch
import torch.nn as nn

from guidance.track_a.hil import HIL
from guidance.track_a import physics


class GIGNPignetAffinity(nn.Module):
    def __init__(self, protein_atom_feature_dim, ligand_atom_feature_dim,
                hidden_dim=64, dim_mlp=64, n_layers=3, short_range_A=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.short_range_A = short_range_A

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, hidden_dim)

        self.hil_layers = nn.ModuleList([HIL(hidden_dim, hidden_dim) for _ in range(n_layers)])

        self.nn_vdw_epsilon = nn.Sequential(
            nn.Linear(hidden_dim * 2, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 1), nn.Sigmoid())
        self.nn_vdw_width = nn.Sequential(
            nn.Linear(hidden_dim * 2, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 1), nn.Sigmoid())

        # Initialized at PIGNet2's own released pignet_morse.yaml starting
        # values (0.714 / 1.0 / 0.216) -- a physically-motivated starting
        # point, not fit to this project's data yet; training adapts them.
        self.hbond_coeff = nn.Parameter(torch.tensor([0.714]))
        self.metal_ligand_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.216]))

        self.vdw_epsilon_scale = (0.0178, 0.0356)
        self.vdw_width_scale = (1.0, 2.0)
        self.hydrogen_bond_cutoffs = (-0.7, 0.0)
        self.metal_ligand_cutoffs = (-0.7, 0.0)
        self.hydrophobic_cutoffs = (0.5, 1.5)

    def _embed_and_convolve(self, protein_pos, protein_feat, ligand_pos, ligand_feat,
                            edge_index_intra, edge_index_inter):
        h_protein = self.protein_atom_emb(protein_feat)
        h_ligand = self.ligand_atom_emb(ligand_feat)
        x = torch.cat([h_ligand, h_protein], dim=0)
        pos = torch.cat([ligand_pos, protein_pos], dim=0)
        for layer in self.hil_layers:
            x = layer(x, edge_index_intra, edge_index_inter, pos)
        return x, pos

    def forward(self, protein_pos, protein_feat, ligand_pos, ligand_feat,
                edge_index_intra, edge_index_inter, batch_ligand, batch_protein,
                vdw_radii, is_metal, is_h_donor, is_h_acceptor, is_hydrophobic):
        """Single-graph or batched (pre-offset edge indices) forward pass.
        All the `is_*`/`vdw_radii` args are already concatenated in
        [ligand; protein] node order (see features.py)."""
        n_l = ligand_pos.size(0)
        x, pos = self._embed_and_convolve(
            protein_pos, protein_feat, ligand_pos, ligand_feat, edge_index_intra, edge_index_inter)

        batch = torch.cat([batch_ligand, batch_protein], dim=0)
        ligand_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        ligand_mask[:n_l] = True

        edge_index_i = physics.interaction_edges(ligand_mask, batch)
        D = physics.distances(pos, edge_index_i)

        x_cat = torch.cat((x[edge_index_i[0]], x[edge_index_i[1]]), dim=-1)
        R = vdw_radii[edge_index_i[0]] + vdw_radii[edge_index_i[1]]

        vdw_epsilon = self.nn_vdw_epsilon(x_cat).squeeze(-1)
        vdw_epsilon = vdw_epsilon * (self.vdw_epsilon_scale[1] - self.vdw_epsilon_scale[0]) + self.vdw_epsilon_scale[0]
        vdw_width = self.nn_vdw_width(x_cat).squeeze(-1)
        vdw_width = vdw_width * (self.vdw_width_scale[1] - self.vdw_width_scale[0]) + self.vdw_width_scale[0]

        energies_pairs = torch.zeros(4, D.numel(), device=x.device)
        energies_pairs[0] = physics.morse_potential(D, R, vdw_epsilon, vdw_width, self.short_range_A)

        minima_hbond = -(self.hbond_coeff ** 2)
        minima_metal = -(self.metal_ligand_coeff ** 2)
        minima_hydrophobic = -(self.hydrophobic_coeff ** 2)
        energies_pairs[1] = physics.linear_potential(D, R, minima_hbond, *self.hydrogen_bond_cutoffs)
        energies_pairs[2] = physics.linear_potential(D, R, minima_metal, *self.metal_ligand_cutoffs)
        energies_pairs[3] = physics.linear_potential(D, R, minima_hydrophobic, *self.hydrophobic_cutoffs)

        masks = physics.interaction_masks(is_metal, is_h_donor, is_h_acceptor, is_hydrophobic, edge_index_i)
        energies_pairs = energies_pairs * masks

        num_graphs = int(batch.max().item()) + 1
        pair_batch = batch[edge_index_i[0]]
        energies = torch.zeros(4, num_graphs, device=x.device)
        energies.scatter_add_(1, pair_batch.unsqueeze(0).expand(4, -1), energies_pairs)
        energies = energies.t().contiguous()  # (num_graphs, 4)
        # Physics energies are negative for favorable binding (deeper wells);
        # this project's regression target is pk (pKd/pKi/pIC50), positive
        # and INCREASING with binding strength -- same relationship as
        # Delta-G to pK (pK ~= -Delta-G / (RT ln 10), a positive
        # proportionality). Negating here aligns the two directions; the
        # network's free coefficients absorb whatever scale factor is
        # needed via training, same as PIGNet2's own coefficients are fit
        # to their own label scale rather than fixed to literal kcal/mol.
        pred = -energies.sum(-1)
        return pred, energies
