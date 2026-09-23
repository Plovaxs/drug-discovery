"""Shared helper for Track B2 (classifier-guidance reformulation): extracts
Stage 0's already-trained EGNN backbone's pooled graph embedding (the input
to its regression head, models/property_pred/prop_model.py's PropPredNet),
so a new, small classification head can be trained as a linear probe on
top of an ALREADY-LEARNED, frozen representation rather than training a
whole new backbone from scratch.

This replicates PropPredNet.forward()'s body up to (and not including)
`self.out_block`, calling the passed-in model's own submodules directly --
prop_model.py itself is not modified, since this is a read-only reuse of
an existing model's internals, not a change to its behavior.
"""
import torch
from torch_scatter import scatter

from models.common import compose_context_prop


def extract_pooled_embedding(prop_model, protein_pos, protein_atom_feature,
                             ligand_pos, ligand_atom_feature, batch_protein, batch_ligand):
    """prop_model: a PropPredNet instance (frozen or not -- caller decides;
    Track B2's training script wraps this call in torch.no_grad() and
    guidance uses it under torch.enable_grad() for the position gradient).
    Returns (num_graphs, hidden_dim) pooled embedding, i.e. PropPredNet's
    own `pre_out` -- exactly what its out_block would otherwise consume.
    """
    h_protein = prop_model.protein_atom_emb(protein_atom_feature)
    h_ligand = prop_model.ligand_atom_emb(ligand_atom_feature)
    h_ctx, pos_ctx, batch_ctx = compose_context_prop(
        h_protein=h_protein, h_ligand=h_ligand,
        pos_protein=protein_pos, pos_ligand=ligand_pos,
        batch_protein=batch_protein, batch_ligand=batch_ligand,
    )
    h_ctx = prop_model.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)
    pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')
    return pre_out
