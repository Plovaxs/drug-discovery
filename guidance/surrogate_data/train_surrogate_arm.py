"""Surrogate experiment arms: trains the Stage 0 EGNN architecture on a configurable mix of CrossDocked
(CD) and BindingNet (BN) rows, always validating and testing on the SAME pure-CrossDocked LP splits so
every arm is scored on identical, untouched data.

Pre-registered design (fixed before any arm was run -- see TASKS.md "Eksperimen surrogate"):
  Arm 0  --cd_rows 6000                      anchor; reproduces the Stage 0 data budget at the new batch
                                             size, so the batch-size change itself can be ruled out as a
                                             cause of any later difference.
  Arm B  --cd_rows 6000  --bn_rows 40964     same TOTAL as Arm A, but most of the budget spent on BN.
  Arm A  --cd_rows 46964                     quantity control: all available CD, no BN.
  Arm C  --cd_rows 46964 --bn_rows <all>     production run, composition/filters guided by A and B.

Why this trio: comparing a BN-trained surrogate against the existing Stage 0 number alone would conflate
DATA SOURCE with DATA QUANTITY (Stage 0 saw only 6,000 of the 46,964 available CD rows -- a time-budget
choice, not a data limit). Arm A supplies the quantity control; Arm 0 re-anchors at the new batch size;
Arm B is the test. Interpreting B requires A, so all three must finish before any claim.

Known limitation, stated up front rather than discovered later: BN poses are TEMPLATE-MODELLED (a ChEMBL
compound aligned into a crystal template) while the LP test set is CrossDocked docked/minimised poses, so
every BN-containing arm crosses a pose-generation domain boundary that the CD-only arms do not. A weak BN
result therefore cannot be attributed to data quality alone. Keeping 6,000 CD rows in Arm B (rather than a
pure BN swap) deliberately preserves a CD-domain foothold so the asymmetry is reduced.

Censored BN labels (2,772 rows whose affinity is recorded as "<value", i.e. truly MORE potent than the
number) are EXCLUDED by default: training on them as exact targets injects label noise in a known
direction. --bn_keep_censored overrides.

Usage (one arm per sitting; resumable via --resume after a cooldown/shutdown):
  python guidance/surrogate_data/train_surrogate_arm.py configs/prop/crossdocked_affinity_egnn.yml \\
      --arm_tag arm0 --cd_rows 6000 --seed 2021 --batch_size 8
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from utils.train import get_optimizer, get_scheduler
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from datasets.pl_pair_dataset import PocketLigandPairDataset
from guidance.lp_split.lp_split_loader import build_lp_splits

BN_ROOT = './data/bindingnet_pocket10'
BN_LABELS = os.path.join(BN_ROOT, 'labels.csv')


def build_bn_train_set(n_rows, transform, seed, keep_censored=False,
                       max_core_rmsd=None, min_similarity=None):
    """BindingNet training subset. Val/test are NEVER drawn from here -- they stay pure CrossDocked."""
    labels = pd.read_csv(BN_LABELS)
    n_all = len(labels)
    if not keep_censored:
        labels = labels[~labels['censored'].astype(bool)]
    if max_core_rmsd is not None:
        labels = labels[labels['core_rmsd'] <= max_core_rmsd]
    if min_similarity is not None:
        labels = labels[labels['similarity'] >= min_similarity]
    print(f'BN pool: {n_all} -> {len(labels)} after filters '
          f'(keep_censored={keep_censored}, max_core_rmsd={max_core_rmsd}, min_similarity={min_similarity})',
          flush=True)

    if n_rows is not None and n_rows < len(labels):
        labels = labels.sample(n=n_rows, random_state=seed)
    base = PocketLigandPairDataset(BN_ROOT)
    indices = labels['idx'].astype(int).tolist()
    pk_by_idx = dict(zip(labels['idx'].astype(int), labels['pk'].astype(float)))
    return CrossDockedAffinityDataset(base, indices, pk_by_idx, transform), len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--arm_tag', type=str, required=True, help='e.g. arm0 / armA / armB / armC')
    parser.add_argument('--cd_rows', type=int, default=6000, help='CrossDocked training rows (0 = none)')
    parser.add_argument('--bn_rows', type=int, default=0, help='BindingNet training rows (0 = none)')
    parser.add_argument('--bn_keep_censored', action='store_true')
    parser.add_argument('--bn_max_core_rmsd', type=float, default=None)
    parser.add_argument('--bn_min_similarity', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None, help='override config.train.batch_size')
    parser.add_argument('--amp', action='store_true',
                        help='bf16 autocast. Measured 1.38x faster at bs=4 (16.0 vs 11.6 rows/sec) with '
                             'slightly lower VRAM; see guidance/surrogate_data/bench_batch_size.py. Changes '
                             'numerics, so Arm 0 doubles as the check that it did not cost accuracy.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_surrogate_arms')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip_test_logging', action='store_true', default=True,
                        help='default ON: test set stays unseen until the final eval script')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    if args.seed is not None:
        config.train.seed = args.seed
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    misc.seed_all(config.train.seed)

    tag = f'{args.arm_tag}_cd{args.cd_rows}_bn{args.bn_rows}_bs{config.train.batch_size}_s{config.train.seed}'
    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_surrogate_arm', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    if not args.resume:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    transform = Compose([pf, lf])

    # Val/test: always the pure-CrossDocked LP splits, never subsampled, never BN -- identical across arms
    # so model selection and final scoring are comparable.
    cd_base, cd_splits, cd_pk = build_lp_splits(train_subsample=None)
    val_set = CrossDockedAffinityDataset(cd_base, cd_splits['val'], cd_pk, transform)
    test_set = CrossDockedAffinityDataset(cd_base, cd_splits['test'], cd_pk, transform)

    train_parts, part_desc = [], []
    if args.cd_rows > 0:
        cd_train_idx = cd_splits['train']
        if args.cd_rows < len(cd_train_idx):
            g = torch.Generator().manual_seed(config.train.seed)
            perm = torch.randperm(len(cd_train_idx), generator=g).tolist()[:args.cd_rows]
            cd_train_idx = [cd_train_idx[i] for i in perm]
        train_parts.append(CrossDockedAffinityDataset(cd_base, cd_train_idx, cd_pk, transform))
        part_desc.append(f'CD={len(cd_train_idx)}')
    if args.bn_rows > 0:
        bn_set, n_bn = build_bn_train_set(
            args.bn_rows, transform, config.train.seed, args.bn_keep_censored,
            args.bn_max_core_rmsd, args.bn_min_similarity)
        train_parts.append(bn_set)
        part_desc.append(f'BN={n_bn}')
    assert train_parts, 'need cd_rows > 0 or bn_rows > 0'
    train_set = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    logger.info(f'TRAIN mix: {" + ".join(part_desc)} = {len(train_set)} rows | '
                f'VAL {len(val_set)} (CD) | TEST {len(test_set)} (CD, unseen)')

    follow_batch = ['protein_element', 'ligand_element']
    exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers, follow_batch=follow_batch,
                              exclude_keys=exclude_keys)
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=exclude_keys)

    model = PropPredNet(config.model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(args.device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    start_epoch, best_val_loss, best_val_epoch, patience_count, global_it = 1, float('inf'), 0, 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss, best_val_epoch = ckpt['best_val_loss'], ckpt['best_val_epoch']
        patience_count, global_it = ckpt['patience_count'], ckpt['global_it']
        logger.info(f'Resumed from {args.resume}: epoch {start_epoch}, best {best_val_loss:.4f} '
                    f'@{best_val_epoch}, patience {patience_count}')

    def get_loss(batch, pos_noise_std):
        pn = torch.randn_like(batch.protein_pos) * pos_noise_std
        ln = torch.randn_like(batch.ligand_pos) * pos_noise_std
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
            pred = model(protein_pos=batch.protein_pos + pn,
                         protein_atom_feature=batch.protein_atom_feature.float(),
                         ligand_pos=batch.ligand_pos + ln,
                         ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                         batch_protein=batch.protein_element_batch,
                         batch_ligand=batch.ligand_element_batch, output_kind=None)
            loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
        return loss, pred

    def train_epoch(epoch):
        nonlocal global_it
        model.train()
        optimizer.zero_grad()
        for it, batch in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}'), start=1):
            batch = batch.to(args.device)
            loss, _ = get_loss(batch, config.train.pos_noise_std)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %05d | Loss %.6f | Lr %.6f' % (
                    epoch, it, loss.item(), optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/loss', loss, global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)

    def validate(epoch, loader, prefix='Validate'):
        model.eval()
        sum_loss, sum_n, yp, yt = 0.0, 0, [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=prefix, dynamic_ncols=True):
                batch = batch.to(args.device)
                loss, pred = get_loss(batch, 0.)
                sum_loss += loss.item() * len(batch.y)
                sum_n += len(batch.y)
                yp.append(pred.view(-1)); yt.append(batch.y)
        avg = sum_loss / max(sum_n, 1)
        # .float() is required, not cosmetic: under bf16 autocast pred comes back as BFloat16 and
        # torch->numpy conversion raises "unsupported ScalarType BFloat16".
        yp = torch.cat(yp).float().cpu().numpy().astype(np.float64)
        yt = torch.cat(yt).float().cpu().numpy().astype(np.float64)
        logger.info('[%s] Epoch %03d | Loss %.6f' % (prefix, epoch, avg))
        get_eval_scores(yp, yt, logger, prefix=prefix)
        return avg

    last_path = os.path.join(ckpt_dir, 'last.pt')
    patience = config.train.get('early_stop_patience', None)
    try:
        for epoch in range(start_epoch, config.train.max_epochs + 1):
            train_epoch(epoch)
            if epoch % config.train.val_freq == 0 or epoch == config.train.max_epochs:
                val_loss = validate(epoch, val_loader)
                scheduler.step(val_loss) if config.train.scheduler.type == 'plateau' else scheduler.step()
                writer.add_scalar('val/loss', val_loss, epoch)
                if val_loss < best_val_loss:
                    best_val_loss, best_val_epoch, patience_count = val_loss, epoch, 0
                    logger.info(f'Best val at epoch {epoch}: {best_val_loss:.4f}')
                    if not args.skip_test_logging:
                        validate(epoch, test_loader, prefix='Test')
                    torch.save({'config': config, 'model': model.state_dict(),
                                'protein_atom_feature_dim': pf.feature_dim,
                                'ligand_atom_feature_dim': lf.feature_dim,
                                'epoch': epoch, 'val_loss': best_val_loss, 'seed': config.train.seed,
                                'arm_tag': args.arm_tag, 'cd_rows': args.cd_rows, 'bn_rows': args.bn_rows,
                                'train_rows': len(train_set)},
                               os.path.join(ckpt_dir, 'best.pt'))
                else:
                    patience_count += 1
                    logger.info(f'No improve (patience {patience_count}/{patience}), best '
                                f'{best_val_loss:.4f} @{best_val_epoch}')
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'global_it': global_it,
                            'best_val_loss': best_val_loss, 'best_val_epoch': best_val_epoch,
                            'patience_count': patience_count, 'seed': config.train.seed}, last_path)
                if patience is not None and patience_count >= patience:
                    logger.info(f'Early stopping at epoch {epoch}.')
                    break
    except KeyboardInterrupt:
        logger.info('Terminating... last.pt is current through the last completed epoch; use --resume.')

    logger.info(f'Done. Best val loss {best_val_loss:.4f} at epoch {best_val_epoch}.')


if __name__ == '__main__':
    main()
