"""A1g: ternary quantization-AWARE training (QAT) of the Stage 0 EGNN architecture, via the STE mechanism
in guidance/tnn_qat.py. Answers the question the post-hoc probe (guidance/tnn_feasibility_probe.py) left
open: that probe ternarized an ALREADY-CONVERGED FP32 model with no further training and got R2=-1.50
(from 0.41 FP32) -- a lower bound on what ternary can do here, not an upper bound. This script instead
warm-starts from that same FP32 checkpoint, converts every nn.Linear to a TernaryLinear (STE forward/
identity backward), and continues training under the ternary constraint, same config/data/loss (plain
MSE) as guidance/lp_split/train_egnn_stage0.py -- the only difference from that script is the warm-start +
ternary conversion step.

Deliberately plain MSE on the ORIGINAL (not evidential, not heteroscedastic) architecture: this isolates
the ternary-quantization question from the uncertainty-method questions already answered in A1-A1f. If
QAT recovers most of the FP32 R2 here, ternarizing one of the surviving UQ methods becomes a reasonable
follow-up; if QAT also fails to recover, there is no point compounding it with evidential's own training
fragility.

Usage:
  python guidance/uncertainty_a1/train_egnn_qat_ternary.py configs/prop/crossdocked_affinity_egnn.yml --skip_test_logging
"""
import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from utils.train import get_optimizer, get_scheduler
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.tnn_qat import convert_to_ternary_qat, sparsity_report

WARM_START_CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'


def get_loss(model, batch, pos_noise_std):
    protein_noise = torch.randn_like(batch.protein_pos) * pos_noise_std
    ligand_noise = torch.randn_like(batch.ligand_pos) * pos_noise_std
    pred = model(
        protein_pos=batch.protein_pos + protein_noise,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=batch.ligand_pos + ligand_noise,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        output_kind=None,
    )
    loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
    return loss, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_a1g_qat_ternary')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--skip_test_logging', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
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
    logger = misc.get_logger('train_egnn_qat_ternary', log_dir)
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
        ligand_atom_feature_dim=ligand_featurizer.feature_dim, output_dim=1,
    ).to(args.device)
    if not args.resume:
        ckpt = torch.load(WARM_START_CKPT, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        logger.info(f'Warm-started full model from {WARM_START_CKPT} (FP32 test R^2 0.342).')
    convert_to_ternary_qat(model)
    logger.info(f'Converted to ternary QAT. Sparsity at init: {sparsity_report(model)}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

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
            loss, _ = get_loss(model, batch, pos_noise_std=config.train.pos_noise_std)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | Lr %.6f' % (
                    epoch, it, loss.item(), optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/loss', loss, global_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)

    def validate(epoch, data_loader, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        ypred_arr, ytrue_arr = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                loss, pred = get_loss(model, batch, pos_noise_std=0.)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                ypred_arr.append(pred.view(-1))
                ytrue_arr.append(batch.y)
        if sum_n == 0:
            return float('inf')
        avg_loss = sum_loss / sum_n
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        logger.info('[%s] Epoch %03d | Loss %.6f' % (prefix, epoch, avg_loss))
        get_eval_scores(ypred_arr, ytrue_arr, logger, prefix=prefix)
        return avg_loss

    last_path = os.path.join(ckpt_dir, 'last.pt')
    early_stop_patience = config.train.get('early_stop_patience', None)
    try:
        for epoch in range(start_epoch, config.train.max_epochs + 1):
            train(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs:
                val_loss = validate(epoch, val_loader, prefix='Validate')
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
