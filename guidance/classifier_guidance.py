"""Track B2: classifier-guidance reformulation. Wraps Stage 0's frozen
EGNN backbone (guidance_models/affinity_egnn_lpsplit.pt) plus a small
trained linear classification head (guidance/train_classifier_head.py's
checkpoint) for use inside the guided sampling loop, following the same
GuidanceModel interface as guidance/affinity_guidance.py's AffinityGuidance.

The actual reformulation, relative to AffinityGuidance: the guidance
target is grad_x log sigmoid(logit) = grad_x log p(y=1|x) (classic
classifier guidance, Dhariwal & Nichol 2021's gradient target), not the
raw regression output's gradient. Same backbone, same input featurization
(guidance/atom_features.py's v0_to_ligand_feature, same placeholder
Degree/NumHs/Hybridization limitation, same position-only grad_v=0 scope
narrowing) as AffinityGuidance -- the ONLY thing that differs is the head
and the training objective it was fit with.
"""
import torch
import torch.nn.functional as F

from guidance.interfaces import GuidanceModel
from guidance.atom_features import v0_to_ligand_feature
from guidance.gradient_normalization import normalize_unit_per_atom
from guidance.frozen_backbone_features import extract_pooled_embedding
from models.property_pred.prop_model import PropPredNet


class ClassifierGuidance(GuidanceModel):
    def __init__(self, backbone_ckpt_path, head_ckpt_path, device='cuda:0', normalize_gradient=False):
        backbone_ckpt = torch.load(backbone_ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.backbone = PropPredNet(
            backbone_ckpt['config'].model,
            protein_atom_feature_dim=backbone_ckpt['protein_atom_feature_dim'],
            ligand_atom_feature_dim=backbone_ckpt['ligand_atom_feature_dim'],
            output_dim=1,
        ).to(device)
        self.backbone.load_state_dict(backbone_ckpt['model'])
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._ligand_atom_feature_dim = backbone_ckpt['ligand_atom_feature_dim']

        head_ckpt = torch.load(head_ckpt_path, map_location=device, weights_only=False)
        self.head = torch.nn.Linear(head_ckpt['hidden_dim'], 1).to(device)
        self.head.load_state_dict(head_ckpt['head_state_dict'])
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)

        self.normalize_gradient = normalize_gradient

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        pos0_req = pos0.detach().clone().requires_grad_(True)
        ligand_feat = v0_to_ligand_feature(
            v0, self._ligand_atom_feature_dim, pos=pos0_req, use_bond_aware=False)

        pooled = extract_pooled_embedding(
            self.backbone, protein_pos.detach(), protein_v.detach(),
            pos0_req, ligand_feat, batch_protein, batch_ligand)
        logit = self.head(pooled).view(-1)
        # grad_x log p(y=1|x) = grad_x log sigmoid(logit) -- the classic
        # classifier-guidance target, not the raw logit's gradient (which
        # would not correspond to a normalized probability's log-gradient).
        log_prob = F.logsigmoid(logit)
        score = log_prob.sum()
        grad_pos, = torch.autograd.grad(score, pos0_req)
        grad_v = torch.zeros_like(v0)
        if self.normalize_gradient:
            grad_pos, grad_v = normalize_unit_per_atom(grad_pos, grad_v, batch_ligand)
        return grad_pos, grad_v
