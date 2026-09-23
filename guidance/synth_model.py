"""Ligand-only 3D EGNN regressor for Task D's synthesizability guidance.

Unlike Task C's PropPredNet (models/property_pred/prop_model.py), this has
no protein/pocket input at all -- RA-score-style synthesizability is a
property of the molecule alone (does it look like something a
retrosynthesis planner can route to?), not of how it sits in a pocket.
Reuses models/property_pred/prop_egnn.EnEquiEncoder unmodified (it already
operates on a generic point cloud: node features + positions + batch
index, with no assumption about what the points represent), wrapped with a
small linear embed + scatter-sum + MLP head, matching PropPredNet's own
structure but without compose_context_prop's protein/ligand merge.

Target is RA score in [0, 1] (see guidance/build_synth_dataset.py for how
labels are produced), so the head ends in a sigmoid.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter

from models.common import ShiftedSoftplus
from models.property_pred.prop_egnn import EnEquiEncoder


class SynthPredNet(nn.Module):
    def __init__(self, config, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_channels
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)
        self.encoder = EnEquiEncoder(
            num_layers=config.encoder.num_layers,
            edge_feat_dim=config.encoder.edge_dim,
            hidden_dim=config.encoder.hidden_dim,
            num_r_gaussian=config.encoder.num_r_gaussian,
            act_fn=config.encoder.act_fn,
            norm=config.encoder.norm,
            update_x=False,
            k=config.encoder.knn,
            cutoff=config.encoder.cutoff,
        )
        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, ligand_pos, ligand_atom_feature, batch_ligand):
        """Returns the raw logit (NOT passed through sigmoid) -- pair with
        BCEWithLogitsLoss for training and apply torch.sigmoid explicitly
        wherever a [0,1] probability is needed. An earlier version applied
        sigmoid here and trained with MSE, which saturated to a constant
        1.0 output within the first epoch: MSE's gradient through a
        saturated sigmoid vanishes (sigmoid'(x) -> 0 for large |x|), so
        once a bad early update pushed the logit high, training could
        never recover -- confirmed via per-sample predictions all reading
        exactly 1.000000 despite continuing to see different training
        batches. BCEWithLogitsLoss's gradient wrt the logit is
        (sigmoid(logit) - target), which does not vanish under saturation.
        """
        h = self.ligand_atom_emb(ligand_atom_feature)
        h = self.encoder(node_attr=h, pos=ligand_pos, batch=batch_ligand)
        pre_out = scatter(h, index=batch_ligand, dim=0, reduce='sum')
        logit = self.out_block(pre_out)
        return logit
