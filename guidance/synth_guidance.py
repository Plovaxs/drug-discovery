"""Frozen synthesizability guidance model, wrapping the Task D-trained
SynthPredNet checkpoint (guidance/synth_model.py, guidance/train_synth_model.py)
for use inside the guided sampling loop (guidance/guided_sampling.py).

Same position-only-gradient design as guidance/affinity_guidance.py, and
for the same reason: v0 (the diffusion core's per-step atom-type estimate)
doesn't carry the bond-connectivity information (Degree/NumHs/
Hybridization) the underlying feature space was trained on, so v0 is
converted to a fixed (non-differentiable) feature via
guidance/atom_features.py and grad_v is always zero. Unlike the affinity
model, this one has no protein/pocket input at all -- synthesizability is
a property of the ligand alone.
"""
import torch

from guidance.interfaces import GuidanceModel
from guidance.atom_features import v0_to_ligand_feature
from guidance.synth_model import SynthPredNet


class SynthGuidance(GuidanceModel):
    def __init__(self, ckpt_path, device='cuda:0', use_bond_aware=False):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.model = SynthPredNet(
            ckpt['config'].model, ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
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

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        pos0_req = pos0.detach().clone().requires_grad_(True)
        ligand_feat = v0_to_ligand_feature(
            v0, self._ligand_atom_feature_dim, pos=pos0_req, use_bond_aware=self.use_bond_aware)

        logit = self.model(
            ligand_pos=pos0_req, ligand_atom_feature=ligand_feat, batch_ligand=batch_ligand,
        )  # (num_graphs, 1), raw logit -- ascending it ascends predicted RA score
           # (sigmoid is monotonic), and avoids the saturated-gradient region
           # sigmoid-space would have near the top of its range.
        score = logit.sum()
        grad_pos, = torch.autograd.grad(score, pos0_req)
        grad_v = torch.zeros_like(v0)
        return grad_pos, grad_v
