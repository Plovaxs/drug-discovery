"""Track D, Step D.2a: retrains the Task D synthesizability guidance model
(SynthPredNet, unchanged architecture) on the leakage-safe RA-score split
(build_synth_lp_splits.py) with a genuinely convergence-seeking epoch
budget -- the original training run (logs_synth/synth_ra_egnn_2026_09_04)
hit a fixed max_epochs=40 ceiling with its patience counter only at 1/6,
i.e. it was cut off, not converged (see guidance/TRACK_D_STATUS_D2.md).

Near-identical structure to guidance/train_synth_model.py -- only the
split source changes (leakage-safe, with a real held-out test set) and a
test-set evaluation is added at the end (the original script never
evaluated on a genuine test set at all), matching this project's
established train_egnn_stage0.py-forks-train_affinity_model.py
convention for split-source changes.
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
from utils.train import get_optimizer, get_scheduler
from guidance.synth_model import SynthPredNet
from datasets.synth_dataset import SynthDataset
from guidance.lp_split.build_synth_lp_splits import build_synth_lp_splits
from guidance.train_synth_model import get_loss, eval_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/prop/synth_ra_egnn_lp.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_synth_lp')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_synth_lp', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([ligand_featurizer])

    logger.info('Building leakage-safe RA-score splits (reusing Stage 0\'s target-level assignment)...')
    base, splits, ra_by_idx, discarded = build_synth_lp_splits(
        labels_path=config.dataset.labels, root=config.dataset.root)
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  '
               f'Test: {len(splits["test"])}  Discarded (target not leakage-safe-assigned): {discarded}')

    train_set = SynthDataset(base, splits['train'], ra_by_idx, transform)
    val_set = SynthDataset(base, splits['val'], ra_by_idx, transform)
    test_set = SynthDataset(base, splits['test'], ra_by_idx, transform)

    follow_batch = ['ligand_element']
    exclude_keys = ['ligand_nbh_list', 'protein_pos', 'protein_element', 'protein_atom_to_aa_type',
                    'protein_is_backbone']
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, follow_batch=follow_batch,
                              exclude_keys=exclude_keys)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=exclude_keys)

    model = SynthPredNet(config.model, ligand_atom_feature_dim=ligand_featurizer.feature_dim).to(args.device)
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
            loss, _ = get_loss(model, batch)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | Lr %.6f' % (
                    epoch, it, loss.item(), optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/loss', loss, global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)

    def validate(epoch, data_loader, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        ypred_arr, ytrue_arr = [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=prefix, leave=False):
                batch = batch.to(args.device)
                loss, pred = get_loss(model, batch)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                ypred_arr.append(pred.view(-1))
                ytrue_arr.append(batch.y)
        avg_loss = sum_loss / sum_n
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        logger.info('[%s] Epoch %03d | Loss %.6f' % (prefix, epoch, avg_loss))
        eval_scores(ypred_arr, ytrue_arr, logger, prefix=prefix)
        model.train()
        return avg_loss

    best_val_loss = float('inf')
    best_val_epoch = 0
    patience_count = 0
    early_stop_patience = config.train.get('early_stop_patience', None)
    for epoch in range(1, config.train.max_epochs + 1):
        train(epoch)
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
            logger.info(f'Best val achieved at epoch {epoch}, val loss: {best_val_loss:.6f}')
            test_loss = validate(epoch, test_loader, prefix='Test')
            ckpt_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'epoch': epoch, 'val_loss': best_val_loss, 'test_loss': test_loss,
            }, ckpt_path)
            logger.info(f'Model saved to {ckpt_path}')
        else:
            patience_count += 1
            logger.info(f'Val loss did not improve (patience {patience_count}/{early_stop_patience}), '
                       f'best so far: {best_val_loss:.6f} at epoch {best_val_epoch}')
            if early_stop_patience is not None and patience_count >= early_stop_patience:
                logger.info(f'Early stopping at epoch {epoch}: no val improvement for {patience_count} epochs.')
                break

    logger.info(f'Done. Best val loss: {best_val_loss:.6f} at epoch {best_val_epoch}.')


if __name__ == '__main__':
    main()
