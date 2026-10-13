"""TODO item 4: within-target y-randomisation for the EGNN itself, not just the descriptor pipeline.

Why this exists. noise_ceiling.py already measures a "class-level prior" via within-target permutation,
but only for the ligand-only ridge-on-14-descriptors model: permute training pK WITHIN each target
(so the SET of labels a target owns is unchanged, only which ligand carries which value is scrambled),
refit on the permuted labels, and see how well the result predicts REAL test labels. That gave R2=0.2468
-- the claim this project leans on hardest is that 70% of a ligand-only model's performance needs no real
SAR, just "which target is this". That number was established for a model with NO protein input at all.
Without this run, the "beyond class-level prior" table always carries a caveat: maybe the EGNN, which DOES
see the actual 3D protein, reaches a prior-matching R2 through a completely different and untested
mechanism, and the 0.2468 benchmark does not actually apply to it.

What this measures for the EGNN SPECIFICALLY, and why the mechanism differs from the ridge case. The
ridge model achieves its 0.2468 because ligand CHEMOTYPE happens to correlate with which target a
compound was tested against (different chemical series cluster toward different targets) -- it has no
access to the target's identity except through that correlation. The EGNN has no such limitation: it
reads the real 3D protein directly. Trained on within-target-permuted labels, the ligand it is shown no
longer predicts the label it must match (that mapping was scrambled), so the only information left that
correlates with the target's label range AT ALL is the protein's own identity -- recognisable from its
actual structure. The EGNN's best available strategy under this regime is to learn roughly the target's
OWN mean label and ignore the ligand, which is the same "how much is just knowing the target" question,
reached by reading the protein directly rather than through a ligand-chemotype correlation. If the EGNN
lands near 0.2468 too, the prior is not an artefact of the descriptor pipeline's blind spot -- it is a
property of the task that a structure-reading model hits by its own, different route.

Implementation: a copy of train_egnn_stage0.py (unchanged architecture/hyperparameters/split) with
EXACTLY one addition -- pk_by_idx for TRAIN indices only is permuted within each target, immediately
after build_lp_splits returns and before any Dataset/DataLoader is built. Validation and test labels are
never touched, matching y_randomise()'s own convention in noise_ceiling.py (model selection and the
final reported R2 must be against REAL labels, or this measures nothing). Target per train index comes
from guidance/track_e/core.py's already-cached anchor_table.npz, not re-derived, to avoid a second,
possibly inconsistent definition of "target" from appearing in the project.

Usage:
  PYTHONPATH=. python guidance/lp_split/train_egnn_stage0_withintarget_yrand.py \\
      configs/prop/crossdocked_affinity_egnn.yml --seed 2021
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
from guidance.track_e.core import build_anchor_table, split_arrays

PERM_SEED = 20260925     # same project-standard seed used everywhere else (BOOT_SEED in noise_ceiling.py
                         # and every bootstrap in this project) -- reused here for the SAME reason: one
                         # documented, grep-able constant rather than a fresh unexplained number.


def permute_pk_within_target(pk_by_idx, train_idx, seed=PERM_SEED):
    """Returns a NEW dict (pk_by_idx is not mutated) where every TRAIN index's pK is replaced by another
    TRAIN index's pK from the SAME target, chosen by permuting within each target's own index list. Val
    and test indices are copied through unchanged -- this function is only ever given train_idx, but the
    explicit copy-then-overwrite structure makes that guarantee visible rather than assumed."""
    anchor = build_anchor_table()
    A = split_arrays(anchor, 'train')
    target_of = dict(zip(A['idx'].tolist(), A['target'].tolist()))
    missing = [i for i in train_idx if i not in target_of]
    if missing:
        raise SystemExit(f'ABORT: {len(missing)} train indices have no target in anchor_table.npz '
                         f'(first few: {missing[:5]}) -- rebuild it with force=True before trusting this.')

    groups = {}
    for i in train_idx:
        groups.setdefault(target_of[i], []).append(i)

    rng = np.random.default_rng(seed)
    out = dict(pk_by_idx)
    n_permuted = 0
    for target, idxs in groups.items():
        idxs = np.array(idxs)
        if len(idxs) < 2:
            continue           # a singleton-ligand target has nothing to permute; left as-is, not an error
        perm = rng.permutation(len(idxs))
        original_pk = np.array([pk_by_idx[i] for i in idxs])
        permuted_pk = original_pk[perm]
        for i, v in zip(idxs, permuted_pk):
            out[int(i)] = float(v)
        n_permuted += len(idxs)
    print(f'within-target y-permutation: {n_permuted} / {len(train_idx)} train pK values reassigned '
         f'across {len(groups)} targets (seed={seed})')
    # Sanity check visible in the log, not just trusted: the SET of labels per target must be the exact
    # same multiset before and after, only the assignment to ligands changed.
    for target, idxs in list(groups.items())[:3]:
        before = sorted(round(pk_by_idx[i], 6) for i in idxs)
        after = sorted(round(out[i], 6) for i in idxs)
        assert before == after, f'target {target}: label SET changed by permutation -- this is a bug'
    return out


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
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_withintarget_yrand')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--skip_test_logging', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--perm_seed', type=int, default=PERM_SEED)
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
    logger = misc.get_logger('train_egnn_stage0_withintarget_yrand', log_dir)
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

    # THE one addition relative to train_egnn_stage0.py: train labels only, permuted within target.
    pk_by_idx = permute_pk_within_target(pk_by_idx, splits['train'], seed=args.perm_seed)
    logger.info(f'Applied within-target y-permutation to the training split (perm_seed={args.perm_seed}). '
               f'Validation and test labels are untouched -- model selection and the reported R2 are '
               f'both against REAL labels, matching noise_ceiling.py\'s y_randomise() convention.')

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
                        'perm_seed': args.perm_seed,
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
