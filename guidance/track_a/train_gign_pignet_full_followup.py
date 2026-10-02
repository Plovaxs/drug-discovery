"""Applies the two follow-up fixes that gave the best diagnostic profile on
the EGNN backbone (guidance/lp_split/train_egnn_stage0_vinatarget.py) to the
Track A GIGN+PIGNet2 architecture, to test whether the null-guidance pattern
is specific to EGNN or general across architectures:

  (1) Noise-matched ligand-position curriculum, t ~ Uniform(0, t_max),
      x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*eps, same schedule as the pretrained
      diffusion model (see train_egnn_stage0_noisematched.py's docstring).
      Protein positions are left clean (pocket is fixed conditioning input
      during real guided sampling, never diffused).
  (2) Vina-derived pK anchor target instead of experimental pK (see
      train_egnn_stage0_vinatarget.py's docstring and
      guidance/track_e/core.py's build_anchor_table/pk_vina).

Otherwise identical to guidance/track_a/train_gign_pignet_full.py: same
model (hidden_dim=256), same optimizer, same LP split (full 46,964-entry
train set, no subsampling), same early-stopping protocol (patience on
validation loss). Graph edges (build_intra_inter_edges) are rebuilt from
the NOISED ligand position every step, so the physics terms (distances,
vdW/H-bond/hydrophobic masks by element only, not by noised position) stay
internally consistent with the perturbed geometry.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.utils.tensorboard
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_a.model import GIGNPignetAffinity
from guidance.track_a.features import (
    ligand_interaction_masks, protein_interaction_masks, vdw_radii_for, build_intra_inter_edges,
)
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod
from guidance.lp_split.train_egnn_stage0_vinatarget import build_vina_pk_by_idx


def prepare_sample_noised(data, device, ligand_pos_noised):
    protein_pos = data.protein_pos.to(device)
    ligand_feat = data.ligand_atom_feature_full.float().to(device)
    protein_feat = data.protein_atom_feature.float().to(device)

    edge_index_intra, edge_index_inter = build_intra_inter_edges(
        ligand_pos_noised, protein_pos, data.ligand_bond_index.to(device))

    l_metal, l_donor, l_acceptor, l_hydrophobic = ligand_interaction_masks(
        data.ligand_element.to(device), data.ligand_atom_feature.to(device))
    p_metal, p_donor, p_acceptor, p_hydrophobic = protein_interaction_masks(
        data.protein_element.to(device))

    is_metal = torch.cat([l_metal, p_metal])
    is_h_donor = torch.cat([l_donor, p_donor])
    is_h_acceptor = torch.cat([l_acceptor, p_acceptor])
    is_hydrophobic = torch.cat([l_hydrophobic, p_hydrophobic])
    vdw_radii = torch.cat([
        vdw_radii_for(data.ligand_element).to(device),
        vdw_radii_for(data.protein_element).to(device),
    ])

    n_l, n_p = ligand_pos_noised.size(0), protein_pos.size(0)
    batch_ligand = torch.zeros(n_l, dtype=torch.long, device=device)
    batch_protein = torch.zeros(n_p, dtype=torch.long, device=device)

    return dict(
        protein_pos=protein_pos, protein_feat=protein_feat,
        ligand_pos=ligand_pos_noised, ligand_feat=ligand_feat,
        edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
        batch_ligand=batch_ligand, batch_protein=batch_protein,
        vdw_radii=vdw_radii, is_metal=is_metal, is_h_donor=is_h_donor,
        is_h_acceptor=is_h_acceptor, is_hydrophobic=is_hydrophobic,
    )


def get_loss_noised(model, data, alphas_cumprod, t_max, device, eval_mode=False):
    ligand_pos = data.ligand_pos.to(device)
    if eval_mode:
        ligand_pos_noised = ligand_pos
    else:
        t = int(torch.randint(0, t_max + 1, (1,)).item())
        a_t = alphas_cumprod[t]
        noise = torch.randn_like(ligand_pos)
        ligand_pos_noised = a_t.sqrt() * ligand_pos + (1.0 - a_t).sqrt() * noise
    kwargs = prepare_sample_noised(data, device, ligand_pos_noised)
    pred, energies = model(**kwargs)
    y = data.y.to(device).view(-1)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, pred, energies


def evaluate(model, data_list, device, alphas_cumprod, t_max, logger, prefix):
    model.eval()
    ypred_arr, ytrue_arr = [], []
    sum_loss, sum_n = 0.0, 0
    with torch.no_grad():
        for data in tqdm(data_list, desc=prefix, leave=False):
            loss, pred, _ = get_loss_noised(model, data, alphas_cumprod, t_max, device, eval_mode=True)
            sum_loss += loss.item()
            sum_n += 1
            ypred_arr.append(pred.item())
            ytrue_arr.append(data.y.item())
    avg_loss = sum_loss / max(sum_n, 1)
    ypred_arr = np.array(ypred_arr, dtype=np.float64)
    ytrue_arr = np.array(ytrue_arr, dtype=np.float64)
    logger.info('[%s] n=%d Loss %.6f' % (prefix, sum_n, avg_loss))
    pearson = float('nan')
    if sum_n > 1:
        get_eval_scores(ypred_arr, ytrue_arr, logger, prefix=prefix)
        pearson = float(pearsonr(ytrue_arr, ypred_arr)[0])
    model.train()
    return avg_loss, pearson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_subsample', type=int, default=None,
                        help='None (default) = full 46,964-entry train set.')
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_track_a_followup')
    parser.add_argument('--tag', type=str, default='noisematched_vinatarget')
    args = parser.parse_args()

    misc.seed_all(2021)
    log_dir = misc.get_new_log_dir(args.logdir, prefix='gign_pignet', tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_gign_pignet_followup', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    device = args.device if torch.cuda.is_available() else 'cpu'
    alphas_cumprod = build_alphas_cumprod(device)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    bond_featurizer = utils_trans.FeaturizeLigandBond()
    transform = Compose([protein_featurizer, ligand_featurizer, bond_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits (index/target membership only)...')
    base, splits, _ = build_lp_splits(train_subsample=args.train_subsample)
    logger.info('Building Vina-derived pK anchor labels from the cached Track E anchor table...')
    vina_pk_by_idx = build_vina_pk_by_idx()
    n_missing = sum(1 for part in splits.values() for i in part if i not in vina_pk_by_idx)
    logger.info(f'{n_missing} of {sum(len(v) for v in splits.values())} split indices missing a Vina anchor')
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  Test: {len(splits["test"])}')

    train_set = CrossDockedAffinityDataset(base, splits['train'], vina_pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], vina_pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], vina_pk_by_idx, transform)

    model = GIGNPignetAffinity(
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.95, 0.999))

    best_val_loss = float('inf')
    best_val_pearson = float('-inf')
    patience_count = 0
    best_path = os.path.join(ckpt_dir, 'best.pt')

    t0 = time.time()
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        g = torch.Generator().manual_seed(2021 + epoch)
        perm = torch.randperm(len(train_set), generator=g).tolist()

        epoch_loss_sum, epoch_n = 0.0, 0
        for idx in tqdm(perm, desc=f'Epoch {epoch}', dynamic_ncols=True):
            data = train_set[idx]
            loss, pred, energies = get_loss_noised(model, data, alphas_cumprod, args.t_max, device)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss_sum += loss.item()
            epoch_n += 1

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        elapsed = time.time() - t0
        logger.info(f'Epoch {epoch:03d} | Train loss {train_loss:.6f} | elapsed {elapsed:.0f}s')
        writer.add_scalar('train/epoch_loss', train_loss, epoch)

        val_loss, val_pearson = evaluate(model, val_set, device, alphas_cumprod, args.t_max,
                                         logger, prefix=f'Validate epoch {epoch}')
        writer.add_scalar('val/loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            logger.info(f'Best val LOSS achieved at epoch {epoch}, val loss: {best_val_loss:.6f}')
            test_loss, test_pearson = evaluate(model, test_set, device, alphas_cumprod, args.t_max,
                                               logger, prefix=f'Test (epoch {epoch})')
            torch.save({
                'model': model.state_dict(),
                'protein_atom_feature_dim': protein_featurizer.feature_dim,
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'hidden_dim': args.hidden_dim,
                'epoch': epoch, 'val_loss': best_val_loss, 'val_pearson': val_pearson,
                'test_loss': test_loss, 'test_pearson': test_pearson,
                't_max': args.t_max, 'target': 'vina_derived_pk',
            }, best_path)
            logger.info(f'Model saved to {best_path}')
        else:
            patience_count += 1
            logger.info(f'Val loss did not improve (patience {patience_count}/{args.patience}), '
                       f'best so far: {best_val_loss:.6f}')
            if patience_count >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}.')
                break

    logger.info(f'Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
