"""KEYSTONE EXPERIMENT: is the affinity predictor sensitive to the ligand's POSE at all?

Everything else in guidance/RESEARCH_PROGRAM.md is circumstantial for one claim: that the surrogate's
accuracy rides on channels that are not pose-specific, which is why its gradient carries no pose
information and why guidance fails. The circumstantial evidence is strong -- ligand descriptors alone
match the 3D GNN, pocket descriptors alone nearly do, the two are redundant, and interaction counts turn
out to encode ligand size -- but all of it is about what OTHER models can do. This script tests the
predictor itself, directly.

The experiment: hold the molecule and the pocket fixed, change only the ligand's rigid-body placement,
and see whether the prediction moves.

  translate_{0.5,1,2,4,8}   rigid translation along a random unit vector
  rotate_{15,45,90,180}     rigid rotation about the ligand centroid
  randomise                 random rotation plus a random 2 A translation
  recentre_pocket           ligand moved to the POCKET's centre of mass (a different, plausible pose)

The 8 A translation is the control that makes the result unambiguous. At 8 A the ligand is outside the
pocket it was crystallised in -- most of its protein contacts are gone. Any model that reads the
protein-ligand interface MUST change its prediction. A model that does not is not using the interface.

How to read the magnitude. A change of 0.1 pK is meaningless on its own; it has to be compared to
something. Two references are reported:
  * sigma_across_ligands -- the sd of the predictor's own native-pose predictions across the sampled
    complexes. This is the full dynamic range the model actually uses. If destroying the pose moves a
    prediction by far less than this, pose contributes almost nothing to what the model outputs.
  * the pK label sd, i.e. the scale of the quantity being predicted.
A pose-sensitivity ratio = mean|delta| / sigma_across_ligands near 0 means pose-blind; near or above 1
means pose is a first-order input.

Statistics: paired per complex (the same molecule before and after), bootstrap over TARGET clusters as
everywhere else in this project, so no target with many complexes dominates.

Runs on CPU by default so it can be piloted while the GPU is occupied; --device cuda and a larger -n for
the full run.

Usage:
  PYTHONPATH=. python guidance/pose_sensitivity.py -n 300                      # CPU pilot
  PYTHONPATH=. python guidance/pose_sensitivity.py -n 2000 --device cuda       # full
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.transforms import Compose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from models.property_pred.prop_model import PropPredNet
from guidance.uncertainty_a1.stats_common import cluster_bootstrap_ci

STAGE0_CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'
OUT = './guidance/pose_sensitivity_results.json'
BOOT_SEED = 20260925


def random_rotation(rng, angle_deg=None):
    """A rotation matrix: uniformly random if angle_deg is None, otherwise a random axis at that exact
    angle (so 'rotate_15' really is 15 degrees, not 'up to 15')."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(0, 2 * np.pi) if angle_deg is None else np.deg2rad(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def perturb(lig_pos, prot_pos, kind, rng):
    """Rigid-body only: the molecule's internal geometry is never touched, so a change in prediction
    cannot be attributed to the conformer becoming unphysical."""
    p = np.asarray(lig_pos, dtype=np.float64)
    centroid = p.mean(0)
    if kind.startswith('translate_'):
        d = float(kind.split('_')[1])
        v = rng.normal(size=3)
        return p + d * v / np.linalg.norm(v)
    if kind.startswith('rotate_'):
        a = float(kind.split('_')[1])
        return (p - centroid) @ random_rotation(rng, a).T + centroid
    if kind == 'randomise':
        q = (p - centroid) @ random_rotation(rng).T + centroid
        v = rng.normal(size=3)
        return q + 2.0 * v / np.linalg.norm(v)
    if kind == 'recentre_pocket':
        return p - centroid + np.asarray(prot_pos, dtype=np.float64).mean(0)
    raise ValueError(kind)


KINDS = ['translate_0.5', 'translate_1', 'translate_2', 'translate_4', 'translate_8',
         'rotate_15', 'rotate_45', 'rotate_90', 'rotate_180', 'randomise', 'recentre_pocket']

# Protein-side controls, which test something pose perturbation cannot. A pose perturbation keeps the
# same pocket; these change or remove the pocket while keeping the ligand. If replacing the protein
# entirely barely moves the prediction, the model is not using protein identity -- a stronger statement
# than pose-insensitivity, and the direct counterpart of the ligand-only baseline result.
# A factorial design, which is what separates 'uses the interface' from 'uses bulk protein statistics'.
# All three strip controls delete exactly 50% of the protein atoms, so the BULK change is identical and
# only the interface content differs:
#   strip_distant_half  interface INTACT   (the deleted atoms are the farthest from the ligand)
#   strip_near_half     interface DESTROYED (the deleted atoms are the ligand's actual contacts)
#   strip_random_half   interface PARTLY degraded, composition preserved
# If the three move the prediction by a similar amount, the model is responding to the atom count and
# gross composition, not to the contacts -- which is precisely the claim in RESEARCH_PROGRAM.md 2.
PROTEIN_KINDS = ['swap_pocket', 'strip_distant_half', 'strip_near_half', 'strip_random_half']


@torch.no_grad()
def predict(model, d, lig_pos, device, prot_pos=None, prot_feat=None):
    pp = d.protein_pos if prot_pos is None else prot_pos
    pf_ = d.protein_atom_feature if prot_feat is None else prot_feat
    bp = torch.zeros(pp.size(0), dtype=torch.long, device=device)
    bl = torch.zeros(lig_pos.size(0), dtype=torch.long, device=device)
    out = model(protein_pos=pp.to(device),
                protein_atom_feature=pf_.float().to(device),
                ligand_pos=lig_pos.to(device),
                ligand_atom_feature=d.ligand_atom_feature_full.float().to(device),
                batch_protein=bp, batch_ligand=bl, output_kind=None)
    return float(out.view(-1)[0])


def protein_control(d, other, kind):
    """(ligand_pos, protein_pos, protein_feat) for a protein-side control.

    swap_pocket        the ligand keeps its conformer but is moved to ANOTHER complex's pocket, which
                       replaces the original protein entirely. The ligand is placed at the new pocket's
                       centre of mass so the two are in contact rather than separated -- otherwise the
                       test would degenerate into 'ligand far from protein', which translate_8 already
                       covers.
    strip_distant_half the half of the protein atoms FARTHEST from the ligand is deleted. Those atoms
                       contribute little interface; a model reading the interface should barely notice,
                       so this is a specificity check in the opposite direction -- it should produce a
                       SMALL change even in a protein-sensitive model.
    """
    lig = d.ligand_pos
    if kind == 'swap_pocket':
        new_prot = other.protein_pos
        shift = new_prot.numpy().mean(0) - lig.numpy().mean(0)
        return (torch.tensor(lig.numpy() + shift, dtype=lig.dtype), new_prot,
                other.protein_atom_feature)
    if kind in ('strip_distant_half', 'strip_near_half', 'strip_random_half'):
        pp = d.protein_pos.numpy()
        half = max(1, len(pp) // 2)
        dist = np.linalg.norm(pp[:, None, :] - lig.numpy()[None, :, :], axis=2).min(1)
        order = np.argsort(dist)
        if kind == 'strip_distant_half':
            keep = order[:half]                       # keep the CLOSEST half: interface preserved
        elif kind == 'strip_near_half':
            keep = order[half:]                       # keep the FARTHEST half: interface destroyed
        else:
            keep = np.random.default_rng(len(pp)).permutation(len(pp))[:half]
        return lig, d.protein_pos[keep], d.protein_atom_feature[keep]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=300, help='test complexes to sample')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--b', type=int, default=2000)
    ap.add_argument('--ckpt', default=STAGE0_CKPT)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'checkpoint {args.ckpt} (epoch {ckpt["epoch"]})  device {args.device}')

    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    ds = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, Compose([pf, lf]))
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    print(f'sampling {len(sel)} of {len(ds)} test complexes\n', flush=True)

    native, labels, targets = [], [], []
    deltas = {k: [] for k in KINDS + PROTEIN_KINDS}
    t0 = time.time()
    for c, i in enumerate(sel, 1):
        d = ds[int(i)]
        lig = d.ligand_pos
        y0 = predict(model, d, lig, args.device)
        native.append(y0)
        labels.append(float(d.y))
        targets.append(d.protein_filename.split('/')[0])
        for kind in KINDS:
            pp = perturb(lig.numpy(), d.protein_pos.numpy(), kind, rng)
            y1 = predict(model, d, torch.tensor(pp, dtype=lig.dtype), args.device)
            deltas[kind].append(y1 - y0)
        # Protein-side controls need a donor complex for swap_pocket; drawn from the sample so no extra
        # dataset access is needed, and never the complex itself.
        donor_i = int(sel[(c - 1 + 1 + len(sel) // 2) % len(sel)])
        donor = ds[donor_i] if donor_i != int(i) else ds[int(sel[(c) % len(sel)])]
        for kind in PROTEIN_KINDS:
            lp, pp_, pf2 = protein_control(d, donor, kind)
            y1 = predict(model, d, lp, args.device, prot_pos=pp_, prot_feat=pf2)
            deltas[kind].append(y1 - y0)
        if c % 50 == 0:
            print(f'  {c}/{len(sel)}  ({time.time() - t0:.0f}s)', flush=True)

    native = np.array(native); labels = np.array(labels)
    sigma_across = float(native.std(ddof=1))
    label_sd = float(labels.std(ddof=1))
    print(f'\nnative-pose predictions: mean {native.mean():.3f}, sd {sigma_across:.3f}')
    print(f'pK labels of the sample : mean {labels.mean():.3f}, sd {label_sd:.3f}')
    print(f'(sd of the predictions is the full dynamic range the model actually uses)\n')

    groups = {}
    for r, t in enumerate(targets):
        groups.setdefault(t, []).append(r)
    groups = [np.array(v) for v in groups.values()]
    print(f'{len(groups)} target clusters in the sample; bootstrap B={args.b}\n')

    print(f'{"perturbation":18s} {"mean|dpred|":>12s} {"95% CI":>22s} {"ratio to sigma":>15s} '
          f'{"max|dpred|":>11s}')
    print('-' * 84)
    results = {}
    for kind in KINDS + PROTEIN_KINDS:
        dv = np.array(deltas[kind])
        point, ci, _ = cluster_bootstrap_ci(lambda rows: float(np.abs(dv[rows]).mean()),
                                            groups, b=args.b, seed=BOOT_SEED)
        ratio = point / sigma_across if sigma_across > 0 else float('nan')
        results[kind] = dict(mean_abs_delta=float(point), ci95=[float(ci[0]), float(ci[1])],
                             ratio_to_sigma=float(ratio), max_abs_delta=float(np.abs(dv).max()),
                             mean_signed_delta=float(dv.mean()))
        print(f'{kind:18s} {point:>12.4f} [{ci[0]:>8.4f}, {ci[1]:>8.4f}] {ratio:>15.3f} '
              f'{np.abs(dv).max():>11.4f}')

    worst = 'translate_8'
    w = results[worst]
    print('\n=== READING ===')
    print(f'The control that settles it is `{worst}`: an 8 A rigid translation puts the ligand outside\n'
          f'the pocket it was crystallised in, destroying most of its protein contacts.')
    print(f'  mean |change in prediction| = {w["mean_abs_delta"]:.4f} pK  '
          f'(CI [{w["ci95"][0]:.4f}, {w["ci95"][1]:.4f}])')
    print(f'  as a fraction of the range the model uses = {w["ratio_to_sigma"]:.3f}')
    if w['ratio_to_sigma'] < 0.1:
        print(f'\n  POSE-BLIND. Moving the ligand 8 A out of the pocket changes the prediction by\n'
              f'  {100 * w["ratio_to_sigma"]:.1f}% of the model\'s own dynamic range. The predictor is\n'
              f'  not reading the protein-ligand interface, so it HAS no pose-specific gradient to\n'
              f'  supply. This is the direct cause of the guidance failure, and it explains why the\n'
              f'  gradient magnitude correlates positively with distance from the pocket (r = +0.24).')
    elif w['ratio_to_sigma'] < 0.5:
        print(f'\n  WEAKLY POSE-SENSITIVE. Destroying the pose moves the prediction by only\n'
              f'  {100 * w["ratio_to_sigma"]:.1f}% of the model\'s dynamic range -- present but small\n'
              f'  relative to what the model varies over across ligands.')
    else:
        print(f'\n  POSE-SENSITIVE. The claim in RESEARCH_PROGRAM.md section 2 is NOT supported: the\n'
              f'  predictor does respond substantially to pose, so the guidance failure must be\n'
              f'  explained some other way. This is the outcome that falsifies the programme, and it\n'
              f'  is more informative than a confirmation.')
    # The model's own error is the reference a practitioner cares about: a perturbation that moves the
    # prediction by less than the model's typical error is invisible in use.
    rmse = float(np.sqrt(((labels - native) ** 2).mean()))
    print(f'\n  reference scales: model RMSE on this sample {rmse:.4f} pK, prediction sd '
          f'{sigma_across:.4f}, label sd {label_sd:.4f}')
    print(f'  destroying the pose (8 A) moves the prediction by '
          f'{w["mean_abs_delta"] / rmse:.2f}x the model\'s own RMSE')

    sw = results.get('swap_pocket')
    if sw:
        print(f'\n  PROTEIN-side control -- replacing the pocket with a DIFFERENT protein entirely:')
        print(f'    mean |dpred| = {sw["mean_abs_delta"]:.4f} pK '
              f'(CI [{sw["ci95"][0]:.4f}, {sw["ci95"][1]:.4f}]), '
              f'{sw["ratio_to_sigma"]:.3f} of the model\'s range, '
              f'{sw["mean_abs_delta"] / rmse:.2f}x its RMSE')
        if sw['ratio_to_sigma'] < 0.25:
            print(f'    Swapping in a completely different protein changes the prediction by only\n'
                  f'    {100 * sw["ratio_to_sigma"]:.1f}% of the range the model uses. The predictor is\n'
                  f'    largely protein-agnostic, which is the mechanism behind the ligand-only\n'
                  f'    baseline matching it (RESEARCH_PROGRAM.md section 3.2).')
        else:
            print(f'    The identity of the protein does move the prediction substantially, so the\n'
                  f'    predictor is not protein-agnostic and section 3.2 needs a different mechanism.')
    far = results.get('strip_distant_half')     # interface kept
    near = results.get('strip_near_half')       # interface destroyed
    rand = results.get('strip_random_half')
    if far and near:
        print(f'\n  THE FACTORIAL: all three delete exactly 50% of protein atoms, so the bulk change is\n'
              f'  identical and only the interface content differs.')
        print(f'    keep closest half  (interface INTACT)    |dpred| = {far["mean_abs_delta"]:.4f} pK'
              f'  [{far["ci95"][0]:.4f}, {far["ci95"][1]:.4f}]')
        print(f'    keep farthest half (interface DESTROYED) |dpred| = {near["mean_abs_delta"]:.4f} pK'
              f'  [{near["ci95"][0]:.4f}, {near["ci95"][1]:.4f}]')
        if rand:
            print(f'    keep a random half                       |dpred| = '
                  f'{rand["mean_abs_delta"]:.4f} pK'
                  f'  [{rand["ci95"][0]:.4f}, {rand["ci95"][1]:.4f}]')
        gap = near['mean_abs_delta'] - far['mean_abs_delta']
        rel = gap / max(1e-9, far['mean_abs_delta'])
        print(f'    interface effect (destroyed - intact) = {gap:+.4f} pK ({100 * rel:+.0f}% relative)')
        if abs(rel) < 0.5:
            print(f'    The two are within {100 * abs(rel):.0f}% of each other. Deleting the ligand\'s\n'
                  f'    ACTUAL CONTACTS costs about the same as deleting atoms that touch nothing, so\n'
                  f'    the response is driven by how much protein is present, not by which protein is\n'
                  f'    in contact. The predictor reads BULK protein statistics, not the interface.\n'
                  f'    That is the mechanism: there is no interface signal for a gradient to carry.')
        else:
            print(f'    Destroying the contacts costs substantially more than deleting distant atoms,\n'
                  f'    so the model does read the interface and RESEARCH_PROGRAM.md 2 needs revision.')

    mono = [results[k]['mean_abs_delta'] for k in
            ('translate_0.5', 'translate_1', 'translate_2', 'translate_4', 'translate_8')]
    print(f'\n  translation dose-response (0.5/1/2/4/8 A): ' + ' -> '.join(f'{m:.4f}' for m in mono))
    print(f'  a pose-reading model should rise steeply here; flatness is itself the finding.')

    json.dump(dict(checkpoint=args.ckpt, n_complexes=int(len(sel)), device=args.device,
                   seed=args.seed, bootstrap=dict(b=args.b, seed=BOOT_SEED,
                                                  n_target_clusters=len(groups)),
                   sigma_across_ligands=sigma_across, label_sd=label_sd,
                   model_rmse_on_sample=float(np.sqrt(((labels - native) ** 2).mean())),
                   native_pred_mean=float(native.mean()), perturbations=results,
                   note='Rigid-body perturbations only: the ligand conformer is never altered, so a '
                        'change in prediction cannot be attributed to unphysical internal geometry. '
                        'ratio_to_sigma divides the mean absolute prediction change by the sd of the '
                        'model\'s own native-pose predictions across complexes, i.e. by the dynamic '
                        'range the model actually uses.'),
              open(OUT, 'w'), indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
