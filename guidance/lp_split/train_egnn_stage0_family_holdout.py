"""Phase 2's second closely-connected branch: identical recipe to
train_egnn_stage0_vinatarget.py (noise-matched curriculum + Vina-derived
pK target, the best-performing "simple" recipe in this investigation) --
the ONLY change is the split. Instead of guidance/lp_split/
lp_split_loader.py's build_lp_splits (protein-identity-threshold LP
split), this uses guidance/lp_split/build_family_holdout_split.py's
build_family_holdout_splits: the entire PF00069 (protein kinase domain)
family -- 53 targets, 8,880 entries -- is held out of training entirely
and used as the sole test set, so test-time evaluation is against a
protein family the model never saw any member of, in any split, during
training. See that module's docstring for why this is a strictly
stronger generalization test than the LP-split's identity threshold.

This directly tests whether the null guidance pattern already
established across 16 checkpoints on the original LP split also holds
under a stricter, family-level train/test separation -- not whether a
differently-trained model predicts better (that is not the question this
script is built to answer, and is not expected to differ from the
existing vina-target checkpoint's own quality on a same-distribution
test set).
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
from guidance.lp_split.build_family_holdout_split import build_family_holdout_splits
from guidance.lp_split.train_egnn_stage0_vinatarget import build_vina_pk_by_idx
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod, get_loss_noised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_family_holdout')
    parser.add_argument('--tag', type=str, default='pf00069_kinase_holdout')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_family_holdout', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    alphas_cumprod = build_alphas_cumprod(args.device)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    logger.info('Building family-holdout split (PF00069 protein kinase domain held out entirely)...')
    base, splits, _, info = build_family_holdout_splits(train_subsample=args.train_subsample)
    logger.info(f'Held-out families: {info["held_out_families"]}  '
                f'Held-out targets: {info["n_held_out_targets"]}  '
                f'Remaining targets: {info["n_remaining_targets"]}')
    logger.info('Building Vina-derived pK anchor labels (idx -> pK_Vina)...')
    vina_pk_by_idx = build_vina_pk_by_idx()
    n_missing = sum(1 for part in splits.values() for i in part if i not in vina_pk_by_idx)
    logger.info(f'{n_missing} of {sum(len(v) for v in splits.values())} split indices missing a Vina anchor')
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  '
                f'Test (held-out kinase family): {len(splits["test"])}')

    train_set = CrossDockedAffinityDataset(base, splits['train'], vina_pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], vina_pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], vina_pk_by_idx, transform)

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
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    global_it = 0

    def train(epoch):
        nonlocal global_it
        model.train()
        optimizer.zero_grad()
        for it, batch in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}'), start=1):
            batch = batch.to(args.device)
            loss, _, t_sampled = get_loss_noised(model, batch, alphas_cumprod, args.t_max,
                                                  config.train.pos_noise_std, args.device)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | t %d | Lr %.6f' % (
                    epoch, it, loss.item(), t_sampled, optimizer.param_groups[0]['lr']))
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
                loss, pred, _ = get_loss_noised(model, batch, alphas_cumprod, args.t_max,
                                                 0., args.device, eval_mode=True)
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

    best_val_loss = float('inf')
    best_val_epoch = 0
    patience_count = 0
    early_stop_patience = config.train.get('early_stop_patience', None)
    try:
        for epoch in range(1, config.train.max_epochs + 1):
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
                    validate(epoch, test_loader, prefix='Test')
                    ckpt_path = os.path.join(ckpt_dir, 'best.pt')
                    torch.save({
                        'config': config, 'model': model.state_dict(),
                        'protein_atom_feature_dim': protein_featurizer.feature_dim,
                        'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                        'epoch': epoch, 'val_loss': best_val_loss,
                        't_max': args.t_max, 'target': 'vina_derived_pk',
                        'held_out_families': info['held_out_families'],
                    }, ckpt_path)
                    logger.info(f'Model saved to {ckpt_path}')
                else:
                    patience_count += 1
                    logger.info(f'Val loss did not improve (patience {patience_count}'
                                f'{"/" + str(early_stop_patience) if early_stop_patience else ""}), '
                                f'best so far: {best_val_loss:.3f} at epoch {best_val_epoch}')
                    if early_stop_patience is not None and patience_count >= early_stop_patience:
                        logger.info(f'Early stopping: no val improvement for {patience_count} epochs.')
                        break
    except KeyboardInterrupt:
        logger.info('Terminating...')

    logger.info(f'Best val loss: {best_val_loss:.3f} at epoch {best_val_epoch}')


if __name__ == '__main__':
    main()
