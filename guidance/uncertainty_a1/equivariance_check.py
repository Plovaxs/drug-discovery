"""Does ternary quantization damage the affinity model's E(3) invariance?

WHY THIS EXISTS. Our reported claim is that ternary QAT is indistinguishable from FP32 on this project's
EGNN (R2 0.3437 +/- 0.0377 vs 0.3420, n=3). That claim rests entirely on R2. Zhou, Xue, Zhang & Fu
(arXiv:2601.02213, Jan 2026, rev. Mar 2026, "Quantized SO(3)-Equivariant Graph Neural Networks for
Efficient Molecular Property Prediction") show that naive low-bit quantization of an SO(3)-equivariant
transformer DESTROYS equivariant structure, and they track it with a Local Equivariance Error; keeping it
intact took three separate innovations (magnitude-direction decoupled quantization, branch-separated QAT,
attention normalisation). They do 8-bit weights on QM9/rMD17 energies and forces -- no ternary, no
protein-ligand data -- so they do not scoop our claim. What they do is expose a hole in how we validated
it: a model can keep its R2 while its symmetry quietly degrades, and we never looked.

THE PREDICTION, stated before measuring, because a prediction made afterwards is not a test. Our EGNN
should be IMMUNE by construction, for two architectural reasons visible in models/egnn.py:
  * geometry enters the scalar features only through d_sq = sum(rel_x**2) (line 42) fed to a Gaussian
    smearing -- a function of interatomic DISTANCES, which no rotation or translation changes;
  * the coordinate update (line 63) is delta_x = scatter_sum((xi - xj)/(|d|+1) * x_mlp(mij)), i.e. the
    relative position vectors scaled by a learned SCALAR. Quantizing an nn.Linear changes those scalars
    and nothing else, so the update stays a linear combination of the same relative vectors.
Zhou et al.'s setting is the opposite: there the equivariant VECTOR features are themselves quantized,
which is precisely why decoupling magnitude from direction was necessary. If this holds, the useful
statement is not "ternary happens to work for us" but "whether weight quantization can threaten
equivariance is a property of the architecture", and distance-based message passing is safe for free.

THE POSITIVE CONTROL is the part that makes the result mean anything. A test that reports ~0 deviation
proves nothing unless it can also detect a real violation, so every run includes a deliberately broken
transform: rotating the LIGAND while leaving the protein fixed. That changes the actual interatomic
distances, so any model that reads geometry at all must respond to it. If the broken control also comes
out near zero, the harness is measuring nothing and its clean result must be discarded.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.tnn_qat import convert_to_ternary_qat
from models.property_pred.prop_model import PropPredNet

FP32_CKPT_GLOB = './logs_lp_split_stage0/crossdocked_affinity_egnn_*/checkpoints/best.pt'
QAT_CKPT_GLOB = './logs_a1g_qat_ternary/crossdocked_affinity_egnn_*_a1g_s{seed}/checkpoints/best.pt'


def random_rotation(rng):
    """A uniformly distributed proper rotation (det = +1), via QR of a Gaussian matrix.

    The sign correction matters: QR alone yields O(3), which includes reflections. A reflection is not a
    physical symmetry of a chiral molecule, so including one would make this test ask a different and
    wrong question."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))             # fix QR's sign ambiguity
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return torch.tensor(q, dtype=torch.float32)


def load_model(ckpt_path, ternary, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(device)
    if ternary:
        convert_to_ternary_qat(model)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def predict(model, ppos, pfeat, lpos, lfeat, device):
    bp = torch.zeros(ppos.size(0), dtype=torch.long, device=device)
    bl = torch.zeros(lpos.size(0), dtype=torch.long, device=device)
    out = model(protein_pos=ppos.to(device), protein_atom_feature=pfeat.float().to(device),
                ligand_pos=lpos.to(device), ligand_atom_feature=lfeat.float().to(device),
                batch_protein=bp, batch_ligand=bl, output_kind=None)
    return float(out.view(-1).item())


def check_one_model(name, model, dataset, idxs, n_rot, rng, device):
    """Returns per-complex deviation statistics for the honest transform and for the broken control."""
    dev_rigid, dev_broken, base_preds, per_complex = [], [], [], []
    for i in idxs:
        d = dataset[i]
        ppos, lpos = d.protein_pos, d.ligand_pos
        pfeat, lfeat = d.protein_atom_feature, d.ligand_atom_feature_full
        with torch.no_grad():
            p0 = predict(model, ppos, pfeat, lpos, lfeat, device)
            base_preds.append(p0)
            for _ in range(n_rot):
                R = random_rotation(rng)
                t = torch.tensor(rng.normal(scale=10.0, size=3), dtype=torch.float32)
                # Honest transform: the WHOLE complex moves, so every interatomic distance is unchanged
                # and an E(3)-invariant scalar prediction must not move either.
                p_rot = predict(model, ppos @ R.T + t, pfeat, lpos @ R.T + t, lfeat, device)
                dev_rigid.append(abs(p_rot - p0))
                # Positive control: only the ligand moves, about the ligand's own centroid so it stays in
                # the pocket rather than flying off into empty space (which would be a trivially easy
                # thing to detect and would not test much).
                c = lpos.mean(0, keepdim=True)
                p_bad = predict(model, ppos, pfeat, (lpos - c) @ R.T + c, lfeat, device)
                dev_broken.append(abs(p_bad - p0))
        per_complex.append(max(dev_rigid[-n_rot:]))
    return (np.array(dev_rigid), np.array(dev_broken), np.array(base_preds),
            np.array(per_complex))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_complexes', type=int, default=24)
    ap.add_argument('--n_rotations', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=20261011)
    ap.add_argument('--out', default='./guidance/uncertainty_a1/equivariance_check_results.json')
    args = ap.parse_args()

    device = args.device
    print(f'device: {device}')
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, Compose([pf, lf]))

    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(len(test_set), size=min(args.n_complexes, len(test_set)), replace=False).tolist())
    print(f'test complexes: {len(idxs)} of {len(test_set)}, {args.n_rotations} rotations each\n')

    models = []
    fp32 = sorted(glob.glob(FP32_CKPT_GLOB))
    if fp32:
        models.append(('fp32_stage0', fp32[0], False))
    for s in args.seeds:
        hits = sorted(glob.glob(QAT_CKPT_GLOB.format(seed=s)))
        if hits:
            models.append((f'ternary_qat_s{s}', hits[0], True))
    if not models:
        raise SystemExit('ABORT: no checkpoints found; nothing to compare.')

    results = {}
    for name, path, ternary in models:
        model, ckpt = load_model(path, ternary, device)
        rigid, broken, preds, per_cx = check_one_model(name, model, test_set, idxs, args.n_rotations,
                                                      np.random.default_rng(args.seed), device)
        results[name] = dict(
            checkpoint=path, ternary=bool(ternary), n_complexes=len(idxs), n_rotations=args.n_rotations,
            prediction_sd=float(preds.std(ddof=1)), prediction_range=float(preds.max() - preds.min()),
            rigid_dev_mean=float(rigid.mean()), rigid_dev_max=float(rigid.max()),
            broken_dev_mean=float(broken.mean()), broken_dev_max=float(broken.max()),
            # Per-complex maxima. Needed because an aggregate max cannot distinguish one pathological
            # complex from a uniform degradation, and those two findings mean entirely different things.
            rigid_dev_per_complex=[float(x) for x in per_cx],
            rigid_dev_median_per_complex=float(np.median(per_cx)),
            complexes_above_1e_4=int((per_cx > 1e-4).sum()),
        )
        r = results[name]
        print(f'{name}')
        print(f'  prediction sd across complexes      {r["prediction_sd"]:.4f} pK   '
              f'(range {r["prediction_range"]:.4f})')
        print(f'  rigid-motion deviation   mean {r["rigid_dev_mean"]:.3e}   max {r["rigid_dev_max"]:.3e} pK')
        print(f'    per-complex median {r["rigid_dev_median_per_complex"]:.3e}, '
              f'{r["complexes_above_1e_4"]}/{len(idxs)} complexes above 1e-4 pK')
        print(f'  BROKEN control (ligand only)  mean {r["broken_dev_mean"]:.3e}   '
              f'max {r["broken_dev_max"]:.3e} pK')
        ratio = r['broken_dev_mean'] / max(r['rigid_dev_mean'], 1e-12)
        print(f'  control / invariance ratio          {ratio:.3g}x\n')

    # ---- verdict, stated against explicit thresholds rather than by eyeballing
    print('=' * 100)
    worst_rigid = max(v['rigid_dev_max'] for v in results.values())
    weakest_control = min(v['broken_dev_mean'] for v in results.values())
    sd = np.mean([v['prediction_sd'] for v in results.values()])
    print(f'Largest rigid-motion deviation across all models : {worst_rigid:.3e} pK')
    print(f'Weakest positive control across all models       : {weakest_control:.3e} pK')
    print(f'Mean prediction sd across complexes              : {sd:.4f} pK')
    print()
    if weakest_control < 1e-3:
        print('HARNESS INVALID: the deliberately broken transform barely moved the prediction, so this')
        print('test cannot detect a violation and its clean result below means nothing. Investigate the')
        print('harness before drawing any conclusion.')
    elif worst_rigid > 0.01 * sd:
        print('INVARIANCE IS MEASURABLY BROKEN: rigid motion moved a prediction by more than 1% of the')
        print('across-complex sd. Report which models and by how much; the R2-only validation of the QAT')
        print('claim was insufficient, exactly as Zhou et al. (arXiv:2601.02213) would predict.')
    else:
        print('INVARIANCE HOLDS at float precision, for FP32 and ternary alike, while the positive control')
        print('moves predictions by orders of magnitude more. Combined with the architectural argument')
        print('(geometry enters only through distances; coordinate updates are scalar-weighted relative')
        print('vectors), the conclusion is that weight quantization CANNOT threaten equivariance in a')
        print('distance-based message-passing model -- it is safe by construction, not by luck. That is a')
        print('narrower and more useful claim than "ternary worked", and it is what distinguishes this')
        print('case from the SO(3) attention transformers where vector features are quantized directly.')
    print('=' * 100)

    with open(args.out, 'w') as f:
        json.dump(dict(results=results, n_complexes=len(idxs), n_rotations=args.n_rotations,
                       seed=args.seed, device=device,
                       prior_art='arXiv:2601.02213 Zhou, Xue, Zhang, Fu -- 8-bit SO(3)-equivariant, '
                                 'QM9/rMD17; no ternary, no protein-ligand'), f, indent=2)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
