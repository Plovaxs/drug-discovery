"""Guided reverse-diffusion sampling loop.

Reimplements `ScorePosNet3D.sample_diffusion` (models/molopt_score_model.py)
step for step, so the frozen pretrained diffusion core in
pretrained_models/pretrained_diffusion.pt is used completely unmodified —
this file only adds a guidance term to the per-step clean-data estimate
before it is handed to the model's own (also unmodified) posterior
functions. The original `models/molopt_score_model.py` is never edited, so
the unguided baseline in scripts/sample_diffusion.py keeps working exactly
as before.

Design choice — where guidance is computed (documented, not left implicit):
Guidance gradients are computed with respect to the diffusion core's
per-step *clean-data estimate* (`pos0_from_e` / `v0_from_e`, i.e. x0-hat),
NOT backpropagated through the diffusion core's own weights back to x_t.
This is the same simplification used by e.g. classifier guidance
implementations that treat the x0-estimate as the effective guidance point
rather than differentiating through the full score network at every one of
up to 1000 steps. It is a deliberate compute/memory tradeoff for this
project's hardware budget (RTX 3050, 4GB VRAM): full backprop through the
diffusion transformer at every step would multiply both the memory and
runtime cost of sampling by roughly the depth of the network, which does
not fit in 4GB for batches of any practical size. Guidance models are
therefore trained on, and applied to, clean ligand structures — consistent
with how Task C/D train them on PDBbind — and the gradient is taken only
through the guidance model's own forward pass.
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.molopt_score_model import center_pos, index_to_log_onehot, log_sample_categorical
from guidance.interfaces import ZeroGuidance


def sample_diffusion_ligand_guided(
        model, protein_pos, protein_v, batch_protein,
        init_ligand_pos, init_ligand_v, batch_ligand,
        num_steps=None, center_pos_mode=None, pos_only=False,
        affinity_model=None, synth_model=None,
        lambda_affinity=0.0, lambda_synth=0.0,
        guidance_timestep_window=None,
        capture_timesteps=None, capture_list=None,
        use_amp=False):
    """Same signature/return shape as ScorePosNet3D.sample_diffusion, plus
    optional guidance models and their weights.

    With lambda_affinity == 0 and lambda_synth == 0 (the default), this
    function performs exactly the same computation as
    ScorePosNet3D.sample_diffusion — same forward calls, same posterior
    functions, same RNG consumption order, under the same `torch.no_grad()`
    — so outputs match the unguided baseline up to GPU floating-point
    nondeterminism (calling the unmodified baseline twice with the same
    seed already differs by ~1e-6, from scatter/atomic-add reduction order;
    it is not bit-exact even against itself). This is verified in
    guidance/verify_baseline_equivalence.py.

    guidance_timestep_window: optional (lo, hi) inclusive bounds on the raw
    diffusion timestep index `i` (0 = least noisy / final, finalizing step;
    num_timesteps-1 = most noisy / first, coarse-structure step) during
    which guidance is applied. None (default) applies guidance uniformly
    across all steps, matching every prior stage of this project. Used by
    Track B3's timestep-windowed guidance experiment
    (guidance/STAGE2_PLUS_EXPERIMENT_LOG.md) to test whether the affinity
    gradient is more informative in some phase of the trajectory than
    others, rather than assuming uniform applicability.

    capture_timesteps / capture_list: optional read-only instrumentation
    hook for guidance/diag_size_confound.py's standalone diagnostic (NOT
    used by any guidance mechanism itself, and inert -- zero effect on the
    sampled trajectory -- unless explicitly passed). When `i` (the raw
    timestep index) is in `capture_timesteps`, appends a detached
    (i, pos0_from_e, v0_from_e, batch_ligand) snapshot to `capture_list` BEFORE any
    guidance perturbation is applied that step, so the same unguided
    trajectory used everywhere else in this project can also be inspected
    for what the affinity model's gradient *would* look like at that
    state, without needing guidance active (guidance being active would
    itself perturb the very states being inspected).

    use_amp: wraps the diffusion core's forward pass (the compute-bound
    part confirmed by direct benchmark to dominate sampling wall-clock
    time) in torch.autocast(dtype=bfloat16). Default False preserves
    every prior stage's exact fp32 behavior. Only enabled for Track C's
    optimized remainder (guidance/PERFORMANCE_NOTES.md) after verifying
    honest-eval metrics on a small batch match the fp32 baseline within
    noise. Guidance's own gradient computation (the small guidance model,
    not the diffusion core) is left in fp32 regardless of this flag --
    autocast is only applied around the `model(...)` call below.
    """
    if num_steps is None:
        num_steps = model.num_timesteps
    num_graphs = batch_protein.max().item() + 1

    guidance_active = (
        (lambda_affinity != 0.0 and affinity_model is not None) or
        (lambda_synth != 0.0 and synth_model is not None)
    )
    if affinity_model is None:
        affinity_model = ZeroGuidance()
    if synth_model is None:
        synth_model = ZeroGuidance()

    protein_pos, init_ligand_pos, offset = center_pos(
        protein_pos, init_ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

    pos_traj, v_traj = [], []
    v0_pred_traj, vt_pred_traj = [], []
    ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
    time_seq = list(reversed(range(model.num_timesteps - num_steps, model.num_timesteps)))

    for i in tqdm(time_seq, desc='guided sampling', total=len(time_seq)):
        t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

        # The diffusion core's forward pass always runs under no_grad: guidance
        # never backprops through it (grad is taken only through the small
        # guidance model, on a detached copy of its output below), so
        # building an autograd graph for the full transformer here would
        # only cost memory for nothing -- this was a real bug in an earlier
        # version (enable_grad wrapped the whole model() call), which OOM'd
        # on the RTX 3050 the moment guidance was actually used.
        with torch.no_grad():
            # Only the diffusion core's own forward pass runs under
            # autocast (the compute-bound part) -- the posterior math
            # right after (_predict_x0_from_eps, q_pos_posterior, etc.)
            # stays fp32 regardless of use_amp, since those are cheap
            # relative to the transformer forward pass and more sensitive
            # to precision (small, compounding per-step updates).
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                preds = model(
                    protein_pos=protein_pos,
                    protein_v=protein_v,
                    batch_protein=batch_protein,
                    init_ligand_pos=ligand_pos,
                    init_ligand_v=ligand_v,
                    batch_ligand=batch_ligand,
                    time_step=t
                )
            preds = {k: (v.float() if torch.is_tensor(v) else v) for k, v in preds.items()}
            if model.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = model._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif model.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

        if capture_timesteps is not None and i in capture_timesteps and capture_list is not None:
            # protein_pos is captured too (even though it's constant across
            # the trajectory) since it's already in the SAME centered frame
            # (mode=center_pos_mode) as pos0_from_e at this point in the
            # function -- the caller only has the original, uncentered
            # data.protein_pos, which would silently mismatch pos0_from_e's
            # frame and corrupt any distance computation downstream.
            capture_list.append((i, pos0_from_e.detach().clone(), v0_from_e.detach().clone(),
                                 batch_ligand.detach().clone(), protein_pos.detach().clone()))

        step_guidance_active = guidance_active
        if guidance_timestep_window is not None:
            lo, hi = guidance_timestep_window
            step_guidance_active = step_guidance_active and (lo <= i <= hi)

        if step_guidance_active:
            with torch.enable_grad():
                pos0_detached = pos0_from_e.detach().requires_grad_(True)
                v0_detached = v0_from_e.detach().requires_grad_(True)

                grad_pos = torch.zeros_like(pos0_detached)
                grad_v = torch.zeros_like(v0_detached)

                if lambda_affinity != 0.0:
                    g_pos, g_v = affinity_model.grad_log_score(
                        pos0_detached, v0_detached, batch_ligand,
                        protein_pos, protein_v, batch_protein)
                    grad_pos = grad_pos + lambda_affinity * g_pos
                    grad_v = grad_v + lambda_affinity * g_v

                if lambda_synth != 0.0:
                    g_pos, g_v = synth_model.grad_log_score(
                        pos0_detached, v0_detached, batch_ligand,
                        protein_pos, protein_v, batch_protein)
                    grad_pos = grad_pos + lambda_synth * g_pos
                    grad_v = grad_v + lambda_synth * g_v

            pos0_from_e = (pos0_from_e + grad_pos).detach()
            v0_from_e = (v0_from_e + grad_v).detach()

        pos_model_mean = model.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
        pos_log_variance = model.posterior_logvar[t][batch_ligand].unsqueeze(-1)
        nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
        ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
        ligand_pos = ligand_pos_next

        if not pos_only:
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, model.num_classes)
            log_model_prob = model.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v_next = log_sample_categorical(log_model_prob)

            v0_pred_traj.append(log_ligand_v_recon.clone().detach().cpu())
            vt_pred_traj.append(log_model_prob.clone().detach().cpu())
            ligand_v = ligand_v_next

        ori_ligand_pos = ligand_pos + offset[batch_ligand]
        pos_traj.append(ori_ligand_pos.clone().detach().cpu())
        v_traj.append(ligand_v.clone().detach().cpu())

    ligand_pos = ligand_pos + offset[batch_ligand]
    return {
        'pos': ligand_pos.detach(),
        'v': ligand_v,
        'pos_traj': pos_traj,
        'v_traj': v_traj,
        'v0_traj': v0_pred_traj,
        'vt_traj': vt_pred_traj
    }
