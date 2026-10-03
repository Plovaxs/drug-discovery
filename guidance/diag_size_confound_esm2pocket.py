"""DIAG1-size-direction for the ESM2-pocket-only checkpoint
(guidance/lp_split/train_egnn_stage0_esm2pocket.py). Same method as
diag_size_confound.py and diag_size_confound_esm2.py -- only the
guidance-model construction differs (pocket-keyed vector lookup instead
of accession-keyed).
"""
import argparse
import json

import torch

from guidance.diag_size_confound import load_model_and_dataset, run_pocket, bootstrap_ci
from guidance.affinity_guidance_esm2 import AffinityGuidanceESM2
from guidance.lp_split.train_egnn_stage0_esm2pocket import build_pocket_vec_for_target

ESM2POCKET_AFFINITY_CHECKPOINT = ('./logs_lp_split_stage0_esm2pocket/'
                                   'crossdocked_affinity_egnn_2026_10_03__14_28_13_v1/'
                                   'checkpoints/best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pockets_file', type=str, default='./guidance/lpsplit_confirmation_pockets.json')
    parser.add_argument('--n_pockets', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--out', type=str, default='./guidance/diag_size_confound_esm2pocket_results.json')
    parser.add_argument('--affinity_ckpt', type=str, default=ESM2POCKET_AFFINITY_CHECKPOINT)
    args = parser.parse_args()

    with open(args.pockets_file) as f:
        pockets = json.load(f)[:args.n_pockets]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, test_set = load_model_and_dataset(device)
    vec_for_target, pocket_cache, n_hit = build_pocket_vec_for_target()
    guidance_model = AffinityGuidanceESM2(args.affinity_ckpt, device=device, vec_for_target=vec_for_target)

    all_size_corrs, all_dist_corrs = [], []
    per_pocket = {}
    for pocket in pockets:
        in_cache = pocket['target'] in pocket_cache
        print(f'=== pocket {pocket["target"]} (in pocket-embedding cache: {in_cache}) ===')
        guidance_model.set_pocket(pocket['target'])
        size_corrs, dist_corrs = run_pocket(model, test_set, guidance_model, pocket, args.n_samples, device)
        print(f'  n_size_corrs={len(size_corrs)}, n_dist_corrs={len(dist_corrs)}')
        per_pocket[pocket['target']] = {'size_corrs': size_corrs, 'dist_corrs': dist_corrs, 'in_cache': in_cache}
        all_size_corrs += size_corrs
        all_dist_corrs += dist_corrs

    size_mean, size_lo, size_hi = bootstrap_ci(all_size_corrs)
    dist_mean, dist_lo, dist_hi = bootstrap_ci(all_dist_corrs)
    print(f'\nSize/heaviness correlation: mean={size_mean:.3f} 95% CI=[{size_lo:.3f}, {size_hi:.3f}] (n={len(all_size_corrs)})')
    print(f'Distance-to-pocket correlation: mean={dist_mean:.3f} 95% CI=[{dist_lo:.3f}, {dist_hi:.3f}] (n={len(all_dist_corrs)})')

    result = {
        'affinity_ckpt': args.affinity_ckpt,
        'per_pocket': per_pocket,
        'size_correlation': {'mean': size_mean, 'ci_lo': size_lo, 'ci_hi': size_hi, 'n': len(all_size_corrs)},
        'distance_correlation': {'mean': dist_mean, 'ci_lo': dist_lo, 'ci_hi': dist_hi, 'n': len(all_dist_corrs)},
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
