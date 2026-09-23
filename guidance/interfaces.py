"""Interface every frozen guidance model (affinity, synthesizability) implements.

Guidance is applied to the diffusion core's per-step *clean-data estimate*
(x0-hat: `pos0_from_e` / `v0_from_e` in `ScorePosNet3D.sample_diffusion`),
not to the noisy `x_t` directly — this is the standard diffusion-posterior-
sampling (DPS) style of guidance, and is necessary here because the frozen
affinity/synthesizability predictors (Task C/D) are trained on clean
ligand structures, not on noised intermediate states. Concretely, at each
reverse-diffusion step t:

    pos0_from_e, v0_from_e = diffusion_core(x_t, t)        # base score
    grad_pos, grad_v = guidance_model.grad_log_score(
        pos0_from_e, v0_from_e, batch_ligand, protein_pos, protein_v, batch_protein)
    guided_pos0 = pos0_from_e + lambda_ * grad_pos
    guided_v0   = v0_from_e   + lambda_ * grad_v
    # ... guided_pos0/guided_v0 feed into q_pos_posterior / q_v_posterior as usual

All guidance models must be frozen (`requires_grad_(False)` on parameters);
the gradient computed is only ever with respect to `pos0_from_e` /
`v0_from_e`, never the guidance model's own weights.
"""
from abc import ABC, abstractmethod

import torch


class GuidanceModel(ABC):
    """Base class for a frozen, differentiable guidance signal.

    Subclasses wrap a trained predictor (e.g. the PDBbind-trained EGNN
    affinity model from Task C, or the RA-score-style synthesizability
    classifier from Task D) and expose the gradient of its predicted score
    with respect to the diffusion core's clean-data estimate.
    """

    @abstractmethod
    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v,
                        batch_protein):
        """Return (grad_pos, grad_v), gradients of log predicted score wrt
        (pos0, v0). Shapes must match pos0 / v0 exactly. Must not modify
        pos0/v0 in place, and must leave the guidance model's own
        parameters untouched (frozen, no optimizer step).
        """
        raise NotImplementedError


class ZeroGuidance(GuidanceModel):
    """No-op guidance: grad is exactly zero.

    Used when a lambda is 0 or no guidance model is supplied, so the guided
    sampling loop can treat "no guidance" and "guidance with a real model"
    uniformly without branching — and, since the gradient is literally
    absent (not just multiplied by 0), this guarantees the loop reduces to
    the original unguided baseline exactly, not just numerically close to
    it.
    """

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v,
                        batch_protein):
        return torch.zeros_like(pos0), torch.zeros_like(v0)
