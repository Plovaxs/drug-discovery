"""Track A FULL-TIER training (A2-full): trains GIGN+PIGNet2 to real
convergence (early-stopped on validation loss, not a fixed iteration
cutoff) on the FULL leakage-safe LP-PDBBind-style training split
(46,964 entries, not the exploratory tier's 3,000-entry subsample) --
fixing that tier's actual "underpowered" root cause (see
guidance/STAGE2_PLUS_EXPERIMENT_LOG.md's A2-recon entry): the exploratory
checkpoint tested a model only ~1.67 epochs into training on 6% of the
available data.

Scope, per the user's explicit decision at this cost/rigor tradeoff
(logged as A2-full): NO PDA/NDA data augmentation -- matches Stage 0's
own EGNN training convention (which also used no augmentation and still
doubled its R^2 over the pre-Stage-0 baseline), avoiding the multi-day-
to-multi-week augmentation-pipeline implementation+runtime cost the
original full spec would have required. batch_size=1 is unchanged from
the exploratory tier (no new batched collate_fn written) -- "no
augmentation" alone already brings training time into the ~1.5-3 day
range estimated and accepted at this decision point; a batching pipeline
was explicitly out of scope for this decision.

Resumable by design (per the addendum's Sec 1.5 requirement, since this
run is expected to take 1.5-3+ days unattended): saves a `last.pt`
checkpoint (model + optimizer + epoch + best_val_loss + patience_count)
after EVERY epoch, not just on validation improvement, and --resume
reloads all of this to continue an interrupted run exactly. Per-epoch
shuffling is reseeded as `2021 + epoch` (not a serialized RNG stream) so
a resumed epoch's shuffle order is reproducible without needing to
checkpoint RNG state.
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
from guidance.track_a.train_gign_pignet_stage2 import get_loss


def evaluate(model, data_list, device, logger, prefix):
    model.eval()
    ypred_arr, ytrue_arr = [], []
    sum_loss, sum_n = 0.0, 0
    with torch.no_grad():
        for data in tqdm(data_list, desc=prefix, leave=False):
            loss, pred, _ = get_loss(model, data, device)
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
                        help='None (default) = full 46,964-entry train set. '
                             'Only set for a quick resume-logic smoke test.')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12,
                        help='epochs with no val-loss improvement before early stopping')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_track_a_full')
    parser.add_argument('--tag', type=str, default='full')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to a last.pt checkpoint to resume from')
    args = parser.parse_args()

    misc.seed_all(2021)

    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix='gign_pignet', tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_gign_pignet_full', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    bond_featurizer = utils_trans.FeaturizeLigandBond()
    transform = Compose([protein_featurizer, ligand_featurizer, bond_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits (FULL train set for A2-full)...')
    base, splits, pk_by_idx = build_lp_splits(train_subsample=args.train_subsample)
    logger.info(f'Train: {len(splits["train"])}  Val: {len(splits["val"])}  Test: {len(splits["test"])}')

    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = GIGNPignetAffinity(
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.95, 0.999))

    start_epoch = 1
    best_val_loss = float('inf')
    best_val_pearson = float('-inf')
    patience_count = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_val_pearson = ckpt.get('best_val_pearson', float('-inf'))
        patience_count = ckpt['patience_count']
        logger.info(f'Resumed from {args.resume}: starting at epoch {start_epoch}, '
                   f'best_val_loss={best_val_loss:.4f}, best_val_pearson={best_val_pearson:.4f}, '
                   f'patience_count={patience_count}')

    best_path = os.path.join(ckpt_dir, 'best.pt')
    best_pearson_path = os.path.join(ckpt_dir, 'best_by_pearson.pt')
    last_path = os.path.join(ckpt_dir, 'last.pt')

    t0 = time.time()
    model.train()
    for epoch in range(start_epoch, args.max_epochs + 1):
        g = torch.Generator().manual_seed(2021 + epoch)
        perm = torch.randperm(len(train_set), generator=g).tolist()

        epoch_loss_sum, epoch_n = 0.0, 0
        for idx in tqdm(perm, desc=f'Epoch {epoch}', dynamic_ncols=True):
            data = train_set[idx]
            loss, pred, energies = get_loss(model, data, device)
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

        val_loss, val_pearson = evaluate(model, val_set, device, logger, prefix=f'Validate epoch {epoch}')
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/pearson', val_pearson, epoch)

        # Two independent "best" criteria are tracked and checkpointed
        # separately, not just one: this model's raw physics-summed
        # output has no learned bias/scale head (see model.py's
        # docstring), so val MSE loss can be a noisy proxy for ranking
        # quality even while correlation keeps improving (observed in
        # this run's own early epochs: val loss worsened epoch 2->3 while
        # Pearson kept climbing 0.590->0.593->0.619). Saving only the
        # loss-based "best" would risk losing access to a genuinely
        # better-correlated checkpoint once training moves past it --
        # "last.pt" only ever holds the MOST RECENT epoch, not every
        # epoch in between, so this must be decided per-epoch, not
        # reconstructed after the fact.
        loss_improved = val_loss < best_val_loss
        pearson_improved = val_pearson > best_val_pearson if not np.isnan(val_pearson) else False

        if loss_improved:
            best_val_loss = val_loss
            patience_count = 0
            logger.info(f'Best val LOSS achieved at epoch {epoch}, val loss: {best_val_loss:.6f}')
            test_loss, test_pearson = evaluate(model, test_set, device, logger, prefix=f'Test (best-loss so far, epoch {epoch})')
            torch.save({
                'model': model.state_dict(),
                'protein_atom_feature_dim': protein_featurizer.feature_dim,
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'hidden_dim': args.hidden_dim,
                'epoch': epoch, 'val_loss': best_val_loss, 'val_pearson': val_pearson,
                'test_loss': test_loss, 'test_pearson': test_pearson,
            }, best_path)
            logger.info(f'Model saved to {best_path}')
        else:
            patience_count += 1
            logger.info(f'Val loss did not improve (patience {patience_count}/{args.patience}), '
                       f'best so far: {best_val_loss:.6f}')

        if pearson_improved:
            best_val_pearson = val_pearson
            logger.info(f'Best val PEARSON achieved at epoch {epoch}, val pearson: {best_val_pearson:.6f}')
            test_loss_p, test_pearson_p = evaluate(model, test_set, device, logger, prefix=f'Test (best-pearson so far, epoch {epoch})')
            torch.save({
                'model': model.state_dict(),
                'protein_atom_feature_dim': protein_featurizer.feature_dim,
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'hidden_dim': args.hidden_dim,
                'epoch': epoch, 'val_loss': val_loss, 'val_pearson': best_val_pearson,
                'test_loss': test_loss_p, 'test_pearson': test_pearson_p,
            }, best_pearson_path)
            logger.info(f'Model saved to {best_pearson_path}')

        # Saved every epoch regardless of improvement, so a kill mid-run
        # resumes exactly from the last completed epoch.
        torch.save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'best_val_loss': best_val_loss, 'best_val_pearson': best_val_pearson,
            'patience_count': patience_count,
            'hidden_dim': args.hidden_dim,
            'protein_atom_feature_dim': protein_featurizer.feature_dim,
            'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
        }, last_path)

        if patience_count >= args.patience:
            logger.info(f'Early stopping at epoch {epoch}: no val LOSS improvement for {patience_count} epochs '
                       f'(early stopping still keyed on loss, per this project''s established convention -- '
                       f'best_by_pearson.pt is saved alongside for later comparison, not used as the stopping signal).')
            break

    logger.info(f'Done. Best val loss: {best_val_loss:.6f}. Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
