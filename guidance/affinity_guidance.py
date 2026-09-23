"""Frozen affinity guidance model, wrapping the Task C-trained PropPredNet
(models/property_pred/prop_model.py) checkpoint (see
guidance/train_affinity_model.py) for use inside the guided sampling loop
(guidance/guided_sampling.py).

Design simplification, documented rather than silently applied: guidance is
computed only with respect to ligand POSITION (pos0), not composition (v0)
-- grad_v is always zero. Reason: the diffusion core's atom-type space
('add_aromatic' mode, 13 classes encoding (atomic_number, is_aromatic)
pairs -- see utils/transforms.py MAP_ATOM_TYPE_AROMATIC_TO_INDEX) is a
strict subset of the affinity model's ligand feature space
(FeaturizeLigandAtom in utils/transforms_prop.py, which additionally uses
per-atom Degree, NumHs and Hybridization). Those three fields describe bond
connectivity, which the diffusion process does not have during sampling --
bonds are only decided post-hoc by utils/reconstruct.py after sampling
finishes. So there is no faithful, differentiable way to turn a mid-sampling
v0 estimate into the affinity model's full input space; this wrapper fills
Degree/NumHs/Hybridization with a fixed "unknown" placeholder (index 0) to
keep the affinity model's forward pass well-defined for computing a
position gradient, and does not attempt to differentiate through that
placeholder into v0. Ligand element identity and aromaticity (the two
fields the diffusion model actually predicts) ARE passed through faithfully.

This means affinity guidance in this implementation nudges WHERE the ligand
sits in the pocket, not WHAT it is made of -- a real scope-narrowing of the
"coupled affinity-synthesizability guidance" claim, worth stating plainly
in the thesis rather than glossing over.
"""
import torch

from guidance.interfaces import GuidanceModel
from guidance.atom_features import v0_to_ligand_feature
from guidance.gradient_normalization import normalize_unit_per_atom
from models.property_pred.prop_model import PropPredNet


class AffinityGuidance(GuidanceModel):
    def __init__(self, ckpt_path, device='cuda:0', use_bond_aware=False, normalize_gradient=False):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.model = PropPredNet(
            ckpt['config'].model,
            protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
            ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
            output_dim=1,
        ).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._ligand_atom_feature_dim = ckpt['ligand_atom_feature_dim']
        # Phase 1 (architectural-upgrade addendum): use geometry-derived
        # Degree/NumHs/Hybridization instead of the fixed placeholder --
        # see guidance/atom_features.py and guidance/bond_estimator.py.
        self.use_bond_aware = use_bond_aware
        # Track B1 (redefined as gradient-norm normalization, see
        # guidance/gradient_normalization.py): rescales the raw gradient to
        # a fixed per-sample per-atom norm before lambda is applied, so
        # lambda means a consistent "push strength" independent of this
        # particular model's raw output scale. Default False preserves
        # every prior stage's exact behavior unchanged.
        self.normalize_gradient = normalize_gradient

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        pos0_req = pos0.detach().clone().requires_grad_(True)
        # pos=pos0_req (not pos0) so the bond-aware path, when enabled,
        # differentiates connectivity estimation back into the same tensor
        # autograd.grad below takes the gradient with respect to.
        ligand_feat = v0_to_ligand_feature(
            v0, self._ligand_atom_feature_dim, pos=pos0_req, use_bond_aware=self.use_bond_aware)

        pred = self.model(
            protein_pos=protein_pos.detach(),
            protein_atom_feature=protein_v.detach(),
            ligand_pos=pos0_req,
            ligand_atom_feature=ligand_feat,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            output_kind=None,
        )  # (num_graphs, 1), predicted pKd/pKi/pIC50 -- higher is better binding
        score = pred.sum()
        grad_pos, = torch.autograd.grad(score, pos0_req)
        grad_v = torch.zeros_like(v0)
        if self.normalize_gradient:
            grad_pos, grad_v = normalize_unit_per_atom(grad_pos, grad_v, batch_ligand)
        return grad_pos, grad_v
