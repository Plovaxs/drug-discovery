"""ESM2-aware counterpart to guidance/affinity_guidance.py's AffinityGuidance,
wrapping a guidance/lp_split/train_egnn_stage0_esm2.py-trained PropPredNetESM2
checkpoint instead of the plain PropPredNet every other guidance checkpoint
in this project uses.

Why this needs its own class rather than a flag on AffinityGuidance:
PropPredNetESM2.forward() requires an extra `esm2_vec` argument (the
480-dim global protein embedding, projected internally) that plain
PropPredNet does not have. That vector is PER-TARGET (one pocket, looked
up once), not per-atom, so it is fixed at construction time for "whichever
pocket is currently being sampled" rather than threaded through
grad_log_score's existing per-call arguments -- every guided-sampling call
site (guided_sampling.py, DIAG1, lambda_sweep.py) already samples one
pocket at a time (n_samples molecules for that one pocket per call), so a
fixed per-instance vector is exactly as general as this project's existing
guidance call pattern requires; it is NOT safe to reuse across pockets --
construct a fresh instance (or call set_pocket) per pocket.

Same disclosed scope-narrowing as AffinityGuidance (guidance computed only
w.r.t. ligand position, grad_v always zero) -- this class does not attempt
the soft-v relaxation.
"""
import torch

from guidance.interfaces import GuidanceModel
from guidance.atom_features import v0_to_ligand_feature
from guidance.gradient_normalization import normalize_unit_per_atom
from guidance.lp_split.train_egnn_stage0_esm2 import PropPredNetESM2, build_target_to_esm2


class AffinityGuidanceESM2(GuidanceModel):
    def __init__(self, ckpt_path, device='cuda:0', normalize_gradient=False):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.device = device
        self.model = PropPredNetESM2(
            ckpt['config'].model,
            protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
            ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
            esm2_dim=ckpt['esm2_dim'], esm2_proj_dim=ckpt['esm2_proj_dim'],
        ).to(device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._ligand_atom_feature_dim = ckpt['ligand_atom_feature_dim']
        self.normalize_gradient = normalize_gradient
        self._vec_for_target, _, _ = build_target_to_esm2()
        self.esm2_vec = None  # set via set_pocket before sampling

    def set_pocket(self, target_name):
        """target_name: the '<GENE>_<SPECIES>_...' directory-name prefix,
        e.g. data.ligand_filename.split('/')[0]. Must be called before
        grad_log_score for a new pocket -- there is no default."""
        self.esm2_vec = self._vec_for_target(target_name).to(self.device)

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        if self.esm2_vec is None:
            raise RuntimeError('AffinityGuidanceESM2.set_pocket(target_name) must be '
                                'called before grad_log_score (no default pocket).')
        pos0_req = pos0.detach().clone().requires_grad_(True)
        ligand_feat = v0_to_ligand_feature(
            v0, self._ligand_atom_feature_dim, pos=pos0_req, use_bond_aware=False, use_soft_v=False)

        num_graphs = int(batch_ligand.max().item()) + 1
        esm2_batch = self.esm2_vec.unsqueeze(0).expand(num_graphs, -1)

        pred = self.model(
            protein_pos=protein_pos.detach(),
            protein_atom_feature=protein_v.detach(),
            ligand_pos=pos0_req,
            ligand_atom_feature=ligand_feat,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            esm2_vec=esm2_batch,
            output_kind=None,
        )
        score = pred.sum()
        grad_pos, = torch.autograd.grad(score, pos0_req)
        grad_v = torch.zeros_like(v0)
        if self.normalize_gradient:
            grad_pos, grad_v = normalize_unit_per_atom(grad_pos, grad_v, batch_ligand)
        return grad_pos, grad_v
