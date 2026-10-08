"""A1f: EGNN Stage 0 architecture with a Deep Evidential Regression head (Amini et al. 2020, NeurIPS,
"Deep Evidential Regression", arXiv:1910.02600). Same copy-the-training-loop approach as A1e
(heteroscedastic): only output_dim and the loss function change.

Network outputs 4 raw values per complex, transformed into the 4 parameters of a Normal-Inverse-Gamma
(NIG) distribution over (mu, sigma^2):
  gamma = raw[0]                      (predicted mean, unconstrained)
  nu    = softplus(raw[1])            (>0, "virtual observations" / confidence in gamma)
  alpha = softplus(raw[2]) + 1         (>1)
  beta  = softplus(raw[3])            (>0)
Predictive variance (the aleatoric+epistemic combined uncertainty used downstream, matching the official
reference implementation's convention): Var = beta / (nu * (alpha - 1)).

Loss (NIG negative log-likelihood + evidential regularizer, both per the original paper):
  Omega = 2 * beta * (1 + nu)
  nll_i = 0.5*log(pi/nu) - alpha*log(Omega) + (alpha+0.5)*log((y-gamma)^2 * nu + Omega)
          + lgamma(alpha) - lgamma(alpha + 0.5)
  reg_i = |y - gamma| * (2*nu + alpha)
  loss  = mean(nll_i) + LAMBDA * mean(reg_i)
LAMBDA = 0.01, the value used in the original paper's regression experiments -- NOT tuned for this
dataset; reported as a known, disclosed limitation, not presented as optimized.

Same cold-start risk as A1e applies, worse with 4 interacting parameters -- gamma is warm-started from the
Stage 0 MSE checkpoint exactly as A1e's mu was; nu/alpha/beta's raw pre-activation values are zero-
initialized, giving nu=softplus(0)=0.693, alpha=1.693, beta=0.693, so initial Var = 0.693/(0.693*0.693) =
1.443 (sigma~1.2, close to this dataset's label std, consistent with A1e's sigma~1.28 init).

Usage:
  python guidance/uncertainty_a1/train_egnn_evidential.py configs/prop/crossdocked_affinity_egnn.yml --skip_test_logging
"""
import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.train import get_optimizer, get_scheduler
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits

LAMBDA = 0.01
WARM_START_CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'


def nig_params(raw):
    gamma = raw[:, 0]
    nu = F.softplus(raw[:, 1])
    alpha = F.softplus(raw[:, 2]) + 1.0
    beta = F.softplus(raw[:, 3])
    return gamma, nu, alpha, beta


def get_loss(model, batch, pos_noise_std):
    protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
    ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std
    out = model(
        protein_pos=batch.protein_pos + protein_noise,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=batch.ligand_pos + ligand_noise,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        output_kind=None,
    )
    gamma, nu, alpha, beta = nig_params(out)
    y = batch.y
    omega = 2 * beta * (1 + nu)
    nll = (0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    reg = (y - gamma).abs() * (2 * nu + alpha)
    loss = nll.mean() + LAMBDA * reg.mean()
    var = beta / (nu * (alpha - 1))
    return loss, gamma, var


def warm_start(model, ckpt_path):
    old = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    new = model.state_dict()
    for k, v in old.items():
        if k in ('out_block.2.weight', 'out_block.2.bias'):
            continue
        assert new[k].shape == v.shape, (k, new[k].shape, v.shape)
        new[k] = v
    new['out_block.2.weight'][0] = old['out_block.2.weight'][0]
    new['out_block.2.bias'][0] = old['out_block.2.bias'][0]
    new['out_block.2.weight'][1:4] = 0.0
    new['out_block.2.bias'][1:4] = 0.0
    model.load_state_dict(new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_a1f_evidential')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--skip_test_logging', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_warm_start', action='store_true')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    if args.seed is not None:
        config.train.seed = args.seed
    misc.seed_all(config.train.seed)

    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_evidential', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    if not args.resume:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits...')
    base, splits, pk_by_idx = build_lp_splits(train_subsample=args.train_subsample)
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  Test: {len(splits["test"])}')

    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    follow_batch = ['protein_element', 'ligand_element']
    exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, follow_batch=follow_batch,
                              exclude_keys=exclude_keys)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=exclude_keys)

    model = PropPredNet(
        config.model, protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim, output_dim=4,
    ).to(args.device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')
    if not args.resume and not args.no_warm_start:
        warm_start(model, WARM_START_CKPT)
        logger.info(f'Warm-started gamma from {WARM_START_CKPT}; nu/alpha/beta raw init to 0 '
                    f'(nu=0.693, alpha=1.693, beta=0.693, initial Var~1.44).')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    start_epoch = 1
    best_val_loss = float('inf')
    best_val_epoch = 0
    patience_count = 0
    global_it = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_val_epoch = ckpt['best_val_epoch']
        patience_count = ckpt['patience_count']
        global_it = ckpt['global_it']
        logger.info(f'Resumed from {args.resume}: starting at epoch {start_epoch}, '
                    f'best_val_loss={best_val_loss:.4f} at epoch {best_val_epoch}, patience_count={patience_count}')

    def train(epoch):
        nonlocal global_it
        model.train()
        optimizer.zero_grad()
        for it, batch in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}'), start=1):
            batch = batch.to(args.device)
            loss, gamma, var = get_loss(model, batch, pos_noise_std=config.train.pos_noise_std)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | mean_sigma %.4f | Lr %.6f' % (
                    epoch, it, loss.item(), var.detach().sqrt().mean().item(),
                    optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/loss', loss, global_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)

    def validate(epoch, data_loader, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        gamma_arr, var_arr, y_arr = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                loss, gamma, var = get_loss(model, batch, pos_noise_std=0.)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                gamma_arr.append(gamma); var_arr.append(var); y_arr.append(batch.y)
        if sum_n == 0:
            return float('inf'), None
        avg_loss = sum_loss / sum_n
        gamma_arr = torch.cat(gamma_arr).cpu().numpy().astype(np.float64)
        var_arr = torch.cat(var_arr).cpu().numpy().astype(np.float64)
        y_arr = torch.cat(y_arr).cpu().numpy().astype(np.float64)
        sigma_arr = np.sqrt(var_arr)
        rmse = float(np.sqrt(((y_arr - gamma_arr) ** 2).mean()))
        r2 = float(1 - ((y_arr - gamma_arr) ** 2).sum() / ((y_arr - y_arr.mean()) ** 2).sum())
        logger.info('[%s] Epoch %03d | Loss %.6f | RMSE(gamma) %.4f | R2(gamma) %.4f | '
                    'mean_sigma %.4f | std_sigma %.4f' % (
                        prefix, epoch, avg_loss, rmse, r2, sigma_arr.mean(), sigma_arr.std()))
        return avg_loss, (gamma_arr, sigma_arr, y_arr)

    last_path = os.path.join(ckpt_dir, 'last.pt')
    early_stop_patience = config.train.get('early_stop_patience', None)
    try:
        for epoch in range(start_epoch, config.train.max_epochs + 1):
            train(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs:
                val_loss, _ = validate(epoch, val_loader, prefix='Validate')
                if config.train.scheduler.type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                writer.add_scalar('val/loss', val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    patience_count = 0
                    logger.info(f'Best val achieved at epoch {epoch}, val loss: {best_val_loss:.3f}')
                    if not args.skip_test_logging:
                        validate(epoch, test_loader, prefix='Test')
                    ckpt_path = os.path.join(ckpt_dir, 'best.pt')
                    torch.save({
                        'config': config, 'model': model.state_dict(),
                        'protein_atom_feature_dim': protein_featurizer.feature_dim,
                        'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                        'epoch': epoch, 'val_loss': best_val_loss, 'seed': config.train.seed,
                    }, ckpt_path)
                    logger.info(f'Model saved to {ckpt_path}')
                else:
                    patience_count += 1
                    logger.info(f'Val loss did not improve (patience {patience_count}'
                                f'{"/" + str(early_stop_patience) if early_stop_patience else ""}), '
                                f'best so far: {best_val_loss:.3f} at epoch {best_val_epoch}')

                torch.save({
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'global_it': global_it,
                    'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch,
                    'patience_count': patience_count, 'seed': config.train.seed,
                }, last_path)

                if early_stop_patience is not None and patience_count >= early_stop_patience:
                    logger.info(f'Early stopping: no val improvement for {patience_count} epochs.')
                    break
    except KeyboardInterrupt:
        logger.info('Terminating... (last.pt is up to date through the last completed epoch; '
                    'rerun with --resume to continue)')

    logger.info(f'Best val loss: {best_val_loss:.3f} at epoch {best_val_epoch}')


if __name__ == '__main__':
    main()
