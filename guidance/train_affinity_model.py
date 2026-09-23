"""Trains the Task C affinity guidance model: an EGNN binding-affinity
predictor (models/property_pred/prop_model.py's PropPredNet, reused
unmodified) on CrossDocked2020 poses cross-referenced against
data/affinity_info.pkl's real experimental pKd/pKi/pIC50 labels (see
datasets/crossdocked_affinity.py for why this substitutes for a literal
PDBBind download).

Trained standalone here, then frozen and wrapped by
guidance/affinity_guidance.py for use as an in-loop differentiable
guidance signal during sampling (Task E) -- this script has nothing to do
with the diffusion core, which stays completely untouched.

Deviates from scripts/property_prediction/train_prop.py only where the data
source requires it: output_dim=1 (a single pk regression target, not the
Ki/Kd/IC50 3-head split PDBBind's index.pkl provides) and a from-scratch
train/val/test split (see build_labeled_splits) rather than PDBBind's own.
Model architecture, optimizer/scheduler construction, and the train/eval
loop structure are otherwise the same as train_prop.py.
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
from datasets.crossdocked_affinity import build_labeled_splits, CrossDockedAffinityDataset


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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_affinity')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--max_iters', type=int, default=None,
                         help='If set, stop after this many training iterations total '
                              '(for smoke-testing epoch time before a full run).')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_affinity', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    logger.info('Building affinity-labeled CrossDocked2020 splits...')
    base, splits, pk_by_idx = build_labeled_splits(
        root=config.dataset.root, split_path=config.dataset.split,
        affinity_path=config.dataset.affinity_info, val_fraction=config.dataset.val_fraction,
        seed=config.train.seed, train_subsample=config.dataset.get('train_subsample', None))
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  '
                f'Test: {len(splits["test"])} (test is small by construction -- '
                f'only 27 of the official 100 test pockets carry a real affinity label)')

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

            if args.max_iters is not None and global_it >= args.max_iters:
                return True
        return False

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

    best_val_loss = float('inf')
    best_val_epoch = 0
    patience_count = 0
    early_stop_patience = config.train.get('early_stop_patience', None)
    try:
        for epoch in range(1, config.train.max_epochs + 1):
            hit_iter_cap = train(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs or hit_iter_cap:
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
                        'config': config,
                        'model': model.state_dict(),
                        'protein_atom_feature_dim': protein_featurizer.feature_dim,
                        'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                        'epoch': epoch,
                        'val_loss': best_val_loss,
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
            if hit_iter_cap:
                logger.info(f'Stopping at global_it={global_it} (--max_iters={args.max_iters})')
                break
    except KeyboardInterrupt:
        logger.info('Terminating...')

    logger.info(f'Best val loss: {best_val_loss:.3f} at epoch {best_val_epoch}')


if __name__ == '__main__':
    main()
