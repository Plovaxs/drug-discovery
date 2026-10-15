"""DIAG2: the SIGNED direction of the affinity guidance gradient.

DIAG1 (guidance/diag_gradient_magnitude.py and Section V.4 of the thesis) correlated the gradient's
MAGNITUDE with each atom's distance to the pocket. A magnitude is a norm, and a norm carries no
direction: it cannot distinguish a large gradient aimed at a binding contact from an equally large one
aimed away from it. The thesis states this limit explicitly and leaves the measurement open. This
script closes it.

For every ligand atom of every captured x0-hat state it computes

    cos_theta = < unit(grad_pos) , unit(nearest_pocket_atom - atom_pos) >

Sign convention, verified against the sampler rather than assumed: guided_sampling.py applies
`pos0_from_e = pos0_from_e + grad_pos`, i.e. the gradient is ADDED to positions. So cos_theta > 0 means
guidance pushes that atom TOWARD its nearest pocket atom, and cos_theta < 0 means away. Under a
null of "direction unrelated to the pocket", E[cos_theta] = 0.

Two controls run alongside, because a direction result is worthless if the sign convention or the
pipeline is wrong:
  * random-direction control: the same statistic with grad replaced by an isotropic random vector.
    It must come out at ~0; if it does not, the geometry or the pairing is broken.
  * finite-difference check: step the atoms a small distance ALONG +grad and recompute the model's own
    predicted score. Because +grad is supposed to be the ascent direction of that score, the score must
    increase. If it decreases, the sign convention is inverted and every direction claim flips.

Read-only: no training, no generation beyond the unguided capture, one forward/backward pass per state.
"""
import argparse
import json

import numpy as np
import torch

from guidance.affinity_guidance import AffinityGuidance
from guidance.diag_size_confound import load_model_and_dataset, CAPTURE_TIMESTEPS, NUM_TIMESTEPS
from scripts.sample_diffusion_guided import sample_diffusion_ligand_guided_batched
import utils.misc as misc

EPS = 1e-12


def capture_states(model, test_set, pocket, n_samples, device, seed=2021):
    """Unguided x0-hat states, captured exactly as DIAG1 does so the two are comparable."""
    misc.seed_all(seed)
    data = test_set[pocket['data_id']]
    captures = []
    sample_diffusion_ligand_guided_batched(
        model, data, n_samples, batch_size=n_samples, device=device,
        num_steps=NUM_TIMESTEPS, pos_only=False, center_pos_mode='protein', sample_num_atoms='prior',
        affinity_model=None, synth_model=None, lambda_affinity=0.0, lambda_synth=0.0,
        capture_timesteps=CAPTURE_TIMESTEPS, capture_list=captures,
    )
    n_protein_atoms = data.protein_pos.size(0)
    protein_v_single = data.protein_atom_feature.float().to(device)
    per_graph = []
    for _i, pos0, v0, batch_ligand, protein_pos_all in captures:
        n_graphs = int(batch_ligand.max().item()) + 1
        for g in range(n_graphs):
            mask = batch_ligand == g
            pos0_g, v0_g = pos0[mask], v0[mask]
            if pos0_g.size(0) < 3:
                continue
            protein_pos_g = protein_pos_all[g * n_protein_atoms:(g + 1) * n_protein_atoms]
            per_graph.append((pos0_g, v0_g, protein_pos_g))
    return per_graph, protein_v_single


def _predicted_score(gm, pos, v0, batch_ligand, protein_pos, protein_v, batch_protein):
    from guidance.atom_features import v0_to_ligand_feature
    with torch.no_grad():
        feat = v0_to_ligand_feature(v0, gm._ligand_atom_feature_dim, pos=pos, use_bond_aware=gm.use_bond_aware)
        pred = gm.model(protein_pos=protein_pos, protein_atom_feature=protein_v,
                        ligand_pos=pos, ligand_atom_feature=feat,
                        batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
    return float(pred.sum().cpu().item())


def direction_stats(ckpt_path, per_graph, protein_v_single, device, fd_step=0.01, rng_seed=20261015):
    gm = AffinityGuidance(ckpt_path, device=device, normalize_gradient=False)
    rng = np.random.default_rng(rng_seed)
    cos_all, cos_rand_all, dist_all, mag_all, graph_id = [], [], [], [], []
    fd_delta = []
    for gi, (pos0_g, v0_g, protein_pos_g) in enumerate(per_graph):
        n_l = pos0_g.size(0)
        bl = torch.zeros(n_l, dtype=torch.long, device=device)
        bp = torch.zeros(protein_pos_g.size(0), dtype=torch.long, device=device)
        grad_pos, _ = gm.grad_log_score(pos0_g, v0_g, bl, protein_pos_g, protein_v_single, bp)
        grad = grad_pos.detach()

        # vector from each ligand atom to its NEAREST pocket atom
        d = protein_pos_g.unsqueeze(0) - pos0_g.unsqueeze(1)          # (n_l, n_p, 3)
        dist = d.norm(dim=-1)                                          # (n_l, n_p)
        nn_idx = dist.argmin(dim=1)                                    # (n_l,)
        nn_dist = dist.gather(1, nn_idx.unsqueeze(1)).squeeze(1)       # (n_l,)
        to_contact = d[torch.arange(n_l, device=device), nn_idx]       # (n_l, 3)
        to_contact = to_contact / (to_contact.norm(dim=-1, keepdim=True) + EPS)

        g_unit = grad / (grad.norm(dim=-1, keepdim=True) + EPS)
        cos = (g_unit * to_contact).sum(dim=-1)

        r = torch.from_numpy(rng.normal(size=(n_l, 3))).float().to(device)
        r = r / (r.norm(dim=-1, keepdim=True) + EPS)
        cos_rand = (r * to_contact).sum(dim=-1)

        cos_all.extend(cos.cpu().numpy().tolist())
        cos_rand_all.extend(cos_rand.cpu().numpy().tolist())
        dist_all.extend(nn_dist.cpu().numpy().tolist())
        mag_all.extend(grad.norm(dim=-1).cpu().numpy().tolist())
        graph_id.extend([gi] * n_l)

        # finite-difference validation of the sign convention
        s0 = _predicted_score(gm, pos0_g, v0_g, bl, protein_pos_g, protein_v_single, bp)
        step = grad / (grad.norm() + EPS) * fd_step * (n_l ** 0.5)
        s1 = _predicted_score(gm, pos0_g + step, v0_g, bl, protein_pos_g, protein_v_single, bp)
        fd_delta.append(s1 - s0)

    return (np.array(cos_all), np.array(cos_rand_all), np.array(dist_all),
            np.array(mag_all), np.array(graph_id), np.array(fd_delta))


def cluster_bootstrap_mean(x, clusters, b=2000, seed=20260925):
    """Mean with a CI resampled over CLUSTERS (pockets), not atoms: atoms within a pocket are not
    independent, and resampling them directly would understate the interval."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    out = np.empty(b)
    for i in range(b):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([x[clusters == c] for c in take])
        out[i] = vals.mean()
    return float(x.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    ap.add_argument('--n_pockets', type=int, default=3)
    ap.add_argument('--n_samples', type=int, default=16)
    ap.add_argument('--ckpts', type=str, nargs='+', required=True)
    ap.add_argument('--b', type=int, default=2000)
    ap.add_argument('--out', type=str, default='./guidance/diag_gradient_direction_results.json')
    args = ap.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]
    model, test_set = load_model_and_dataset(device)

    # Capture once, reuse for every checkpoint, exactly as DIAG1 does, so any difference between
    # checkpoints is attributable to the checkpoint and not to a different set of states.
    per_graph, pocket_of_graph = [], []
    for pi, pk in enumerate(pockets):
        pg, pv = capture_states(model, test_set, pk, args.n_samples, device)
        for item in pg:
            per_graph.append((item, pv))
            pocket_of_graph.append(pi)
        print(f'pocket {pi} ({pk.get("target", "?")}): {len(pg)} captured states', flush=True)

    results = {'n_pockets': len(pockets), 'n_samples': args.n_samples, 'checkpoints': {}}
    for ck in args.ckpts:
        cos_c, cos_r, dists, mags, gid, fd = [], [], [], [], [], []
        offset = 0
        # group graphs by pocket so each pocket's own protein features are used
        for pi in range(len(pockets)):
            sel = [per_graph[k][0] for k in range(len(per_graph)) if pocket_of_graph[k] == pi]
            pv = [per_graph[k][1] for k in range(len(per_graph)) if pocket_of_graph[k] == pi][0]
            c, cr, d, m, g, f = direction_stats(ck, sel, pv, device)
            cos_c.append(c); cos_r.append(cr); dists.append(d); mags.append(m)
            gid.append(np.full(len(c), pi)); fd.append(f)
        cos_c = np.concatenate(cos_c); cos_r = np.concatenate(cos_r)
        dists = np.concatenate(dists); mags = np.concatenate(mags)
        gid = np.concatenate(gid); fd = np.concatenate(fd)

        m_cos, lo, hi = cluster_bootstrap_mean(cos_c, gid, b=args.b)
        m_rnd, rlo, rhi = cluster_bootstrap_mean(cos_r, gid, b=args.b)
        far = dists > np.median(dists)
        res = {
            'n_atoms': int(len(cos_c)),
            'mean_cos_toward_contact': m_cos, 'ci95': [lo, hi],
            'frac_pointing_toward': float((cos_c > 0).mean()),
            'random_direction_control_mean_cos': m_rnd, 'random_ci95': [rlo, rhi],
            'mean_cos_near_half': float(cos_c[~far].mean()),
            'mean_cos_far_half': float(cos_c[far].mean()),
            'median_nn_distance_A': float(np.median(dists)),
            'mean_grad_magnitude': float(mags.mean()),
            'finite_difference_mean_score_change_along_+grad': float(fd.mean()),
            'finite_difference_frac_increasing': float((fd > 0).mean()),
        }
        results['checkpoints'][ck] = res
        print(json.dumps({ck: res}, indent=2), flush=True)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
