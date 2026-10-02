"""New follow-up branch, not previously tried: an auxiliary training loss
that directly targets DIAG1's diagnosed mechanism (guidance/
DIAG1_SIZE_CONFOUND_FINDING.md: the Stage 0 EGNN's gradient magnitude is
positively correlated with distance to the pocket, +0.240 -- the strongest
pull is on atoms least likely to form binding contacts).

Rather than hoping a different target or a noise-matched input distribution
incidentally reshapes the gradient (tried in train_egnn_stage0_noisematched.py
and train_egnn_stage0_vinatarget.py, both null), this directly adds a loss
term computed from the model's OWN gradient during training:

    L = MSE(pred, y) + alpha * corr(||grad_pos||, dist_to_pocket)

where corr is the per-example Pearson correlation between each ligand
atom's gradient-of-prediction magnitude and its distance to the nearest
protein atom, over the atoms of that one training example. Minimizing this
term pushes the correlation toward zero or negative, i.e. explicitly
discourages the model's gradient from being stronger on atoms farther from
the pocket -- a direct training-time intervention on the mechanism DIAG1
diagnosed, rather than an indirect one.

This requires a double backward (create_graph=True on the first,
inner autograd.grad call, so that call's own computational graph -- which
depends on the model's parameters -- can be differentiated again by the
final loss.backward()). This roughly doubles the memory/compute cost per
training step relative to the single-backward baseline, but is far cheaper
than backpropagating through the diffusion denoiser (which was measured,
in a separate feasibility check, to exceed this GPU's 4GB at a single
denoiser step): the guidance predictor here has 2.5M parameters, not the
diffusion core's full multi-layer transformer.

Combined with the noise-matching curriculum (train_egnn_stage0_noisematched.py)
since that was the other intervention that did not hurt predictive quality;
this is therefore the third disclosed compounding of untested-hypothesis
fixes into one checkpoint this investigation has tried.

Caveats disclosed up front, not discovered after the fact:
  - The alignment term is a per-EXAMPLE (one ligand, N_l atoms) Pearson
    correlation, which is a noisy estimator for small N_l; examples with
    fewer than 3 heavy atoms skip the term (matches DIAG1's own skip rule).
  - alpha was not tuned; a single value (default 1.0) was chosen to be
    comparable in scale to the MSE term's typical magnitude at the start of
    training and is reported as a hyperparameter that was not searched.
  - The correlation is computed against distance to the nearest PROTEIN
    atom only (not typed by interaction relevance), the same proxy DIAG1
    itself used -- it does not know which contacts matter, only how far
    each ligand atom is from any pocket atom.
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
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod


def gradient_alignment_loss(pred, ligand_pos, protein_pos):
    """Per-example Pearson correlation between ||d(pred)/d(ligand_pos_i)||
    and atom i's distance to the nearest protein atom, differentiable
    w.r.t. the MODEL (via create_graph=True on the inner grad call so this
    function's output still depends on model parameters through grad_pos).
    Returns a scalar tensor; 0.0 (no gradient contribution) if the ligand
    has fewer than 3 atoms or either quantity has ~zero variance.
    """
    n_l = ligand_pos.size(0)
    if n_l < 3:
        return torch.zeros((), device=ligand_pos.device)
    grad_pos, = torch.autograd.grad(pred.sum(), ligand_pos, create_graph=True)
    grad_mag = grad_pos.norm(dim=-1)  # (n_l,), still carries grad to model params
    with torch.no_grad():
        dist = torch.cdist(ligand_pos, protein_pos).min(dim=-1).values  # (n_l,), treated as fixed
    if dist.std() < 1e-6:
        return torch.zeros((), device=ligand_pos.device)
    gm = grad_mag - grad_mag.mean()
    dm = dist - dist.mean()
    denom = (gm.pow(2).sum().sqrt() * dm.pow(2).sum().sqrt()).clamp_min(1e-8)
    corr = (gm * dm).sum() / denom
    return corr


def get_loss(model, batch, alphas_cumprod, t_max, pos_noise_std_protein, alpha, device, eval_mode=False):
    protein_pos = batch.protein_pos
    protein_noise = torch.randn_like(protein_pos) * pos_noise_std_protein
    protein_pos_noised = protein_pos + protein_noise

    ligand_pos = batch.ligand_pos.clone()
    if eval_mode:
        # No backward pass happens on this path (mse is returned directly
        # below), so building an autograd graph here is pure waste -- it
        # was previously left on by accident and fragmented this 4GB GPU's
        # memory over an 11,855-example test set until it OOM'd mid-epoch 4.
        with torch.no_grad():
            ligand_pos_in = ligand_pos
            pred = model(
                protein_pos=protein_pos_noised,
                protein_atom_feature=batch.protein_atom_feature.float(),
                ligand_pos=ligand_pos_in,
                ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                batch_protein=batch.protein_element_batch,
                batch_ligand=batch.ligand_element_batch,
                output_kind=None,
            )
            mse = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
        return mse, pred, torch.zeros(())

    ligand_pos.requires_grad_(True)
    t = int(torch.randint(0, t_max + 1, (1,)).item())
    a_t = alphas_cumprod[t]
    ligand_pos_in = a_t.sqrt() * ligand_pos + (1.0 - a_t).sqrt() * torch.randn_like(ligand_pos)

    pred = model(
        protein_pos=protein_pos_noised,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=ligand_pos_in,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        output_kind=None,
    )
    mse = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
    if alpha == 0.0:
        return mse, pred, torch.zeros(())
    align = gradient_alignment_loss(pred, ligand_pos_in, protein_pos_noised)
    loss = mse + alpha * align
    return loss, pred, align.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=1.0,
                         help='weight of the gradient-alignment loss term; not tuned, see module docstring')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_gradalign')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_gradalign', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    alphas_cumprod = build_alphas_cumprod(args.device)

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
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    global_it = 0

    def train(epoch):
        nonlocal global_it
        model.train()
        optimizer.zero_grad()
        align_vals = []
        for it, batch in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}'), start=1):
            batch = batch.to(args.device)
            loss, _, align = get_loss(model, batch, alphas_cumprod, args.t_max,
                                       config.train.pos_noise_std, args.alpha, args.device)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_it += 1
            align_vals.append(align.item())
            if it % config.train.report_iter == 0:
                logger.info('[Train] Epoch %03d Iter %04d | Loss %.6f | align_corr %.4f | Lr %.6f' % (
                    epoch, it, loss.item(), align.item(), optimizer.param_groups[0]['lr']))
            writer.add_scalar('train/loss', loss, global_it)
            writer.add_scalar('train/align_corr', align, global_it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_it)
            writer.add_scalar('train/grad', grad_norm, global_it)
        logger.info(f'[Train] Epoch {epoch:03d} mean align_corr: {np.mean(align_vals):.4f}')

    def validate(epoch, data_loader, prefix='Validate'):
        sum_loss, sum_n = 0, 0
        ypred_arr, ytrue_arr = [], []
        model.eval()
        for batch in tqdm(data_loader, desc=prefix):
            batch = batch.to(args.device)
            loss, pred, _ = get_loss(model, batch, alphas_cumprod, args.t_max, 0., 0.0, args.device, eval_mode=True)
            sum_loss += loss.item() * len(batch.y)
            sum_n += len(batch.y)
            ypred_arr.append(pred.view(-1).detach())
            ytrue_arr.append(batch.y)
        if sum_n == 0:
            return float('inf')
        avg_loss = sum_loss / sum_n
        ypred_arr = torch.cat(ypred_arr).cpu().numpy().astype(np.float64)
        ytrue_arr = torch.cat(ytrue_arr).cpu().numpy().astype(np.float64)
        logger.info('[%s] Epoch %03d | Loss %.6f' % (prefix, epoch, avg_loss))
        get_eval_scores(ypred_arr, ytrue_arr, logger, prefix=prefix)
        model.train()
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
                        't_max': args.t_max, 'alpha': args.alpha,
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
