"""Phase 3's ESM2-pocket-only variant: identical recipe to
train_egnn_stage0_esm2.py (vina-target + noise-matching + ESM2 global
feature), with ONE change -- the ESM2 embedding is mean-pooled over only
the pocket residues (guidance/lp_split/esm2_pocket_embed.py), not the
whole protein chain. Tests whether a sharper, binding-site-specific
protein-language-model signal does better than the coarse whole-protein
version (test R²=0.460, Pearson=0.731, the best predictive quality in
this investigation) -- and, more importantly given this investigation's
central finding, whether it produces any different real-docking outcome
than any of the other 17 checkpoints tested so far.

847/971 LP-split targets (87%) have a pocket-only embedding; the
remaining 124 fall back to a zero vector (same disclosed convention as
train_egnn_stage0_esm2.py), tracked and reported, not silently merged
into the "real embedding" population.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
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
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod
from guidance.lp_split.train_egnn_stage0_vinatarget import build_vina_pk_by_idx
from guidance.lp_split.train_egnn_stage0_esm2 import PropPredNetESM2

ESM2_DIM = 480
ESM2_PROJ_DIM = 32
POCKET_EMB_PATH = './guidance/lp_split/esm2_pocket_embeddings.pt'


def build_pocket_vec_for_target():
    pocket_cache = torch.load(POCKET_EMB_PATH)
    n_hit = [0]

    def vec_for_target(target_name):
        vec = pocket_cache.get(target_name)
        if vec is None:
            return torch.zeros(ESM2_DIM)
        n_hit[0] += 1
        return vec

    return vec_for_target, pocket_cache, n_hit


def get_loss(model, data, alphas_cumprod, t_max, pos_noise_std_protein, vec_for_target, device, eval_mode=False):
    protein_pos = data.protein_pos.to(device)
    protein_noise = torch.randn_like(protein_pos) * pos_noise_std_protein
    protein_pos_noised = protein_pos + protein_noise

    target_name = data.ligand_filename.split('/')[0]
    esm2_vec = vec_for_target(target_name).to(device).unsqueeze(0)

    ligand_pos = data.ligand_pos.to(device)
    if eval_mode:
        ligand_pos_in = ligand_pos
    else:
        t = int(torch.randint(0, t_max + 1, (1,)).item())
        a_t = alphas_cumprod[t]
        ligand_pos_in = a_t.sqrt() * ligand_pos + (1.0 - a_t).sqrt() * torch.randn_like(ligand_pos)

    n_l, n_p = ligand_pos_in.size(0), protein_pos.size(0)
    pred = model(
        protein_pos=protein_pos_noised,
        protein_atom_feature=data.protein_atom_feature.float().to(device),
        ligand_pos=ligand_pos_in,
        ligand_atom_feature=data.ligand_atom_feature_full.float().to(device),
        batch_protein=torch.zeros(n_p, dtype=torch.long, device=device),
        batch_ligand=torch.zeros(n_l, dtype=torch.long, device=device),
        esm2_vec=esm2_vec,
    )
    y = data.y.to(device).view(-1)
    loss = F.mse_loss(pred.view(-1), y)
    return loss, pred


def evaluate(model, data_list, device, alphas_cumprod, t_max, vec_for_target, logger, prefix):
    model.eval()
    ypred_arr, ytrue_arr = [], []
    sum_loss, sum_n = 0.0, 0
    with torch.no_grad():
        for data in tqdm(data_list, desc=prefix, leave=False):
            loss, pred = get_loss(model, data, alphas_cumprod, t_max, 0.0, vec_for_target, device, eval_mode=True)
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
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_esm2pocket')
    parser.add_argument('--tag', type=str, default='esm2pocket')
    args = parser.parse_args()

    misc.seed_all(2021)
    log_dir = misc.get_new_log_dir(args.logdir, prefix='crossdocked_affinity_egnn', tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_esm2pocket', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    device = args.device if torch.cuda.is_available() else 'cpu'
    config = misc.load_config(args.config)
    alphas_cumprod = build_alphas_cumprod(device)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits...')
    base, splits, _ = build_lp_splits(train_subsample=args.train_subsample)
    logger.info('Building Vina-derived pK anchor labels...')
    pk_by_idx = build_vina_pk_by_idx()
    logger.info('Loading pocket-only ESM2 embeddings...')
    vec_for_target, pocket_cache, n_hit = build_pocket_vec_for_target()
    logger.info(f'{len(pocket_cache)} targets with a cached pocket-only ESM2 embedding')

    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)
    logger.info(f'Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}')

    model = PropPredNetESM2(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.95, 0.999))

    best_val_loss = float('inf')
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
            loss, _ = get_loss(model, data, alphas_cumprod, args.t_max,
                               config.train.pos_noise_std, vec_for_target, device)
            loss.backward()
            clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss_sum += loss.item()
            epoch_n += 1

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        elapsed = time.time() - t0
        logger.info(f'Epoch {epoch:03d} | Train loss {train_loss:.6f} | elapsed {elapsed:.0f}s')
        writer.add_scalar('train/epoch_loss', train_loss, epoch)

        val_loss, val_pearson = evaluate(model, val_set, device, alphas_cumprod, args.t_max,
                                         vec_for_target, logger, prefix=f'Validate epoch {epoch}')
        writer.add_scalar('val/loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            logger.info(f'Best val LOSS achieved at epoch {epoch}, val loss: {best_val_loss:.6f}')
            test_loss, test_pearson = evaluate(model, test_set, device, alphas_cumprod, args.t_max,
                                               vec_for_target, logger, prefix=f'Test (epoch {epoch})')
            torch.save({
                'model': model.state_dict(), 'config': config,
                'protein_atom_feature_dim': protein_featurizer.feature_dim,
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'epoch': epoch, 'val_loss': best_val_loss, 'val_pearson': val_pearson,
                'test_loss': test_loss, 'test_pearson': test_pearson,
                't_max': args.t_max, 'target': 'vina_derived_pk',
                'esm2_dim': ESM2_DIM, 'esm2_proj_dim': ESM2_PROJ_DIM, 'esm2_pocket_only': True,
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
