"""A1e: EGNN Stage 0 architecture with a heteroscedastic Gaussian head, trained via NLL instead of MSE.
Direct copy of guidance/lp_split/train_egnn_stage0.py's training loop (same data, split, optimizer,
scheduler, resume mechanics) -- the ONLY changes are output_dim=2 (mu, log_var) and the loss function.
No architecture change beyond that: PropPredNet's forward() already returns the full unmasked (N,
output_dim) vector when output_kind=None, so output_dim=2 directly gives (mu, log_var) per complex.

Gaussian NLL (log_var clamped to [-6, 6], i.e. sigma in [~0.05, ~20] pK units -- wide enough to cover this
dataset's y range (mean/std roughly 6.6/1.2-2.1 per earlier Stage 0 runs) without the exp() over/underflowing):
  loss_i = 0.5 * log_var_i + 0.5 * (y_i - mu_i)^2 * exp(-log_var_i)   (dropping the constant 0.5*log(2*pi))

Evaluated with the SAME 3 tests as A1/A1b/A1c (Spearman(sigma,|err|), calibration ratio, partial Spearman
given n_lig), cluster bootstrap B=2,000 seed 20260925, BH correction -- see guidance/uncertainty_a1/
analyze_heteroscedastic.py (written separately, after this trains, same pre-registration discipline: the
test spec is already fixed by A1's docstring, reused verbatim, not re-derived per method).

Known failure mode to watch for (not just assumed away): heteroscedastic NLL can minimize loss cheaply by
inflating sigma uniformly (variance collapse) rather than learning a sigma that actually varies with
difficulty -- val sigma statistics are logged every epoch so this is visible during training, not just
after the fact.

A1e-beta (added 2026-10-10, --beta_nll): a prior-art audit found that A1e's loss is one the literature
already characterises as pathological. Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic
Uncertainty Estimation with Probabilistic Neural Networks", shows Gaussian NLL under gradient-based
optimisers converges to "very poor but stable" parameter estimates, because the gradient of the mean is
scaled by the predictive variance -- so an example the model has wrongly assigned a large sigma stops
contributing to the mean fit, its error never shrinks, and its sigma stays large. That is exactly A1e's
observed cold-start collapse. Their published remedy is beta-NLL (see beta_nll() below).

This matters for what A1e is allowed to conclude. "Heteroscedastic sigma does not rank error" is a much
weaker statement if the only loss tested is the one already known to fail. So A1e-beta re-runs the same
model, data, split, optimizer and selection criterion with the single factor beta changed, and:
  * if beta-NLL also fails the three pre-registered tests, the negative result becomes considerably
    STRONGER -- it survives the field's own correction;
  * if beta-NLL passes, A1e's conclusion was an artifact of the loss and must be withdrawn.
Either outcome is publishable; neither is available from A1e alone. The analysis plan is NOT re-derived
for this run -- it is A1's three tests verbatim (Spearman(sigma,|err|), calibration ratio, partial
Spearman given n_lig; cluster bootstrap B=2,000 seed 20260925, BH), via analyze_heteroscedastic.py.

Usage:
  python guidance/uncertainty_a1/train_egnn_heteroscedastic.py configs/prop/crossdocked_affinity_egnn.yml --skip_test_logging
  python guidance/uncertainty_a1/train_egnn_heteroscedastic.py configs/prop/crossdocked_affinity_egnn.yml \
      --skip_test_logging --beta_nll 0.5 --logdir ./logs_a1e_beta_heteroscedastic --seed 2021
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
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits

LOG_VAR_CLAMP = (-6.0, 6.0)
WARM_START_CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'


def warm_start(model, ckpt_path, init_log_var=0.5):
    """Loads the existing output_dim=1 MSE-trained Stage 0 checkpoint into this output_dim=2 model: every
    layer matches except the final Linear's output row (1 -> 2). Without this, a cold random mu has large
    initial error, which (as verified directly: see chat) makes the NLL loss blow up and immediately drives
    log_var to the clamp ceiling (variance collapse) within a handful of steps, before mu ever has a chance
    to learn -- a well-documented pathology of heteroscedastic NLL training from scratch. Starting mu near
    its already-good Stage 0 optimum (test R^2 0.342) avoids that regime. init_log_var=0.5 (sigma~1.28,
    close to this dataset's label std) is used only as the new log_var row's starting point, not learned."""
    old = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
    new = model.state_dict()
    for k, v in old.items():
        if k in ('out_block.2.weight', 'out_block.2.bias'):
            continue
        assert new[k].shape == v.shape, (k, new[k].shape, v.shape)
        new[k] = v
    new['out_block.2.weight'][0] = old['out_block.2.weight'][0]
    new['out_block.2.bias'][0] = old['out_block.2.bias'][0]
    new['out_block.2.weight'][1] = 0.0
    new['out_block.2.bias'][1] = init_log_var
    model.load_state_dict(new)


def gaussian_nll(y, mu, log_var):
    """Per-example Gaussian NLL, constant 0.5*log(2*pi) dropped."""
    return 0.5 * log_var + 0.5 * (y - mu) ** 2 * torch.exp(-log_var)


def beta_nll(y, mu, log_var, beta=0.0):
    """beta-NLL (Seitzer et al. 2022): each example's NLL is weighted by its own variance raised to beta,
    with the weight's gradient STOPPED.

        L_i = stopgrad(sigma_i^2)^beta * NLL_i

    Returns (weighted_per_example_loss, plain_per_example_nll) so a run can optimise one and be SELECTED
    on the other -- see below for why that matters.

    The problem beta solves. In plain NLL the gradient of the loss w.r.t. mu carries a factor 1/sigma^2:

        dNLL_i/dmu_i = -(y_i - mu_i) / sigma_i^2

    so every example's pull on the mean is scaled by its own inverse predicted variance. Early in
    training sigma is a bad estimate, and any example the model has (wrongly) assigned a large sigma is
    effectively silenced -- it stops contributing to the mean fit, so its error never shrinks, so its
    sigma stays large. That self-reinforcing loop is the "very poor but stable" fixed point Seitzer et al.
    characterise, and it is precisely the variance collapse A1e hit at cold start (log_var pinned at the
    clamp ceiling within ~3 steps).

    Multiplying by stopgrad(sigma^2)^beta makes the mean gradient scale as sigma^(2*beta - 2):
      beta = 0   -> sigma^-2, i.e. plain NLL, the pathological case
      beta = 0.5 -> sigma^-1, partially restored  (the paper's recommended default)
      beta = 1   -> sigma^0,  mean gradient fully independent of sigma, exactly as in MSE
    The weight MUST be detached: left attached it would also rescale the variance objective, turning a
    reweighting of the mean fit into a different loss whose optimum is no longer the true predictive
    distribution. tests/test_beta_nll.py verifies the scaling law and the detachment numerically rather
    than taking this paragraph's word for it.

    beta = 0.0 is the default so this function reproduces A1e's original loss BIT-FOR-BIT, which is what
    makes A1e vs A1e-beta a controlled comparison of one factor rather than a rewrite.
    """
    nll = gaussian_nll(y, mu, log_var)
    if beta == 0.0:
        return nll, nll
    weight = torch.exp(beta * log_var.detach())      # (sigma^2)^beta, gradient stopped
    return weight * nll, nll


def get_loss(model, batch, pos_noise_std, beta=0.0):
    """Returns (training_loss, plain_nll, mu, log_var).

    Two losses, deliberately. `training_loss` is what gets backpropagated; `plain_nll` is the unweighted
    Gaussian NLL and is what validation reports and early stopping selects on. Keeping selection on the
    plain NLL means A1e and A1e-beta are chosen by the SAME criterion and differ only in the training
    gradient -- if beta changed the selection objective too, a difference in the result could not be
    attributed to beta. It also keeps every val number directly comparable to the A1e run already on
    record, so the two can go in one table without a footnote explaining that the columns mean different
    things."""
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
    mu, log_var = out[:, 0], out[:, 1].clamp(*LOG_VAR_CLAMP)
    weighted, nll = beta_nll(batch.y, mu, log_var, beta)
    return weighted.mean(), nll.mean(), mu, log_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_a1e_heteroscedastic')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--skip_test_logging', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--beta_nll', type=float, default=0.0,
                        help='beta for beta-NLL (Seitzer et al. 2022). 0.0 (default) is plain Gaussian '
                             'NLL, reproducing A1e bit-for-bit; 0.5 is the published recommendation; '
                             '1.0 makes the mean gradient variance-independent, as in MSE. Validation '
                             'always reports plain NLL so runs stay comparable across beta.')
    parser.add_argument('--no_warm_start', action='store_true',
                        help='train mu+log_var from scratch instead of warm-starting mu from the Stage 0 '
                             'MSE checkpoint (NOT recommended -- verified this collapses log_var to the '
                             'clamp ceiling within ~3 steps; see WARM_START_CKPT / warm_start())')
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
    logger = misc.get_logger('train_egnn_heteroscedastic', log_dir)
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
        ligand_atom_feature_dim=ligand_featurizer.feature_dim, output_dim=2,
    ).to(args.device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')
    if not args.resume and not args.no_warm_start:
        warm_start(model, WARM_START_CKPT)
        logger.info(f'Warm-started mu from {WARM_START_CKPT}; log_var row initialized to 0.5 (sigma~1.28).')

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
            loss, nll, mu, log_var = get_loss(model, batch, pos_noise_std=config.train.pos_noise_std,
                                              beta=args.beta_nll)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            if it % config.train.report_iter == 0:
                # Both are logged whenever beta != 0: the optimised objective and the comparable one.
                # Reporting only the weighted loss would make the training curve incomparable to A1e's.
                extra = '' if args.beta_nll == 0.0 else ' | betaNLL %.6f' % loss.item()
                logger.info('[Train] Epoch %03d Iter %04d | NLL %.6f%s | mean_sigma %.4f | Lr %.6f' % (
                    epoch, it, nll.item(), extra, log_var.detach().exp().sqrt().mean().item(),
                    optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/nll', nll, global_it)
            if args.beta_nll != 0.0:
                writer.add_scalar('train/beta_nll', loss, global_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)

    def validate(epoch, data_loader, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        mu_arr, logvar_arr, y_arr = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                # beta=0 here on purpose: validation reports and selects on the PLAIN NLL regardless of
                # how training was weighted, so the number stays comparable to the A1e run on record.
                _, nll, mu, log_var = get_loss(model, batch, pos_noise_std=0., beta=0.0)
                sum_loss += nll.item() * len(batch.y)
                sum_n += len(batch.y)
                mu_arr.append(mu); logvar_arr.append(log_var); y_arr.append(batch.y)
        if sum_n == 0:
            return float('inf'), None
        avg_loss = sum_loss / sum_n
        mu_arr = torch.cat(mu_arr).cpu().numpy().astype(np.float64)
        logvar_arr = torch.cat(logvar_arr).cpu().numpy().astype(np.float64)
        y_arr = torch.cat(y_arr).cpu().numpy().astype(np.float64)
        sigma_arr = np.exp(0.5 * logvar_arr)
        rmse = float(np.sqrt(((y_arr - mu_arr) ** 2).mean()))
        r2 = float(1 - ((y_arr - mu_arr) ** 2).sum() / ((y_arr - y_arr.mean()) ** 2).sum())
        logger.info('[%s] Epoch %03d | NLL %.6f | RMSE(mu) %.4f | R2(mu) %.4f | '
                    'mean_sigma %.4f | std_sigma %.4f' % (
                        prefix, epoch, avg_loss, rmse, r2, sigma_arr.mean(), sigma_arr.std()))
        return avg_loss, (mu_arr, sigma_arr, y_arr)

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
                writer.add_scalar('val/nll', val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    patience_count = 0
                    logger.info(f'Best val achieved at epoch {epoch}, val NLL: {best_val_loss:.3f}')
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
                    logger.info(f'Val NLL did not improve (patience {patience_count}'
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

    logger.info(f'Best val NLL: {best_val_loss:.3f} at epoch {best_val_epoch}')


if __name__ == '__main__':
    main()
