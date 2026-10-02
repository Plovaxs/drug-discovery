"""Follow-up to DIAG1 (guidance/DIAG1_SIZE_CONFOUND_FINDING.md): retrains the
Stage 0 EGNN affinity model with a ligand-position noise curriculum matched to
TargetDiff's own forward-diffusion schedule, instead of the fixed 0.1 A jitter
used by guidance/lp_split/train_egnn_stage0.py.

Rationale (see guidance/thesis/chapters/09_discussion.txt, "Hypotheses, not
tested", item (a)): the guidance predictor is trained on clean, real docked
poses but evaluated during guided sampling on x0-hat, the diffusion model's
clean-data estimate, which at high noise levels differs substantially from
anything in the training distribution. DIAG1 found the predictor's gradient
magnitude is *positively* correlated with distance to the pocket (+0.240),
i.e. concentrated on atoms least likely to matter for binding -- one
candidate explanation is exactly this train/inference mismatch.

Noise formula (identical to models/molopt_score_model.py's forward process,
confirmed by direct inspection of that file):
    a_t = alphas_cumprod[t]
    x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps,  eps ~ N(0, I)
using the same beta schedule as the pretrained diffusion model
(configs/training.yml: sigmoid, beta_start=1e-7, beta_end=2e-3, T=1000).

Scope decision (disclosed, not silently made): the model architecture
(PropPredNet / EGNN) is NOT modified to take the timestep t as an input --
doing so would require also modifying guidance/affinity_guidance.py's
guidance interface to pass t through, a larger change than fits this
diagnostic's budget. The timestep is therefore sampled from a *bounded*
range [0, t_max] (default 500, half the trajectory) rather than the full
[0, 999]: at high t the ligand position noise scale grows large (measured:
sqrt(1-a_t) = 0.33 at t=500, 0.71 at t=850, 0.80 at t=999, versus 0.002 at
t=0) and a fixed-architecture regressor with no notion of "how much noise
is present" would receive an increasingly ill-posed training signal as t
grows, so training across the full range would risk injecting label noise
rather than testing the intended hypothesis. This bound covers DIAG1's
first two evaluation timesteps (150, 500) directly; DIAG1's third point
(850, sqrt(1-a)=0.71) lies outside the trained noise range, so any change
observed there would reflect generalisation beyond training, not direct
coverage, and is reported as such rather than assumed.
Protein (pocket) positions are left with the same fixed small jitter as the
original Stage 0 recipe, because the pocket is never part of the diffusion
process during real guided sampling (it is fixed conditioning input).

Otherwise identical to guidance/lp_split/train_egnn_stage0.py: same model,
same optimizer/scheduler/early-stopping config, same LP split and subsample,
same validation protocol (validation and test are evaluated at t=0, i.e.
un-noised, for direct comparability with the original Stage 0 checkpoint's
reported numbers).
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
from models.molopt_score_model import get_beta_schedule
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits

# Identical to configs/training.yml (the pretrained TargetDiff model's own schedule).
BETA_SCHEDULE = dict(beta_schedule='sigmoid', beta_start=1.e-7, beta_end=2.e-3,
                      num_diffusion_timesteps=1000)


def build_alphas_cumprod(device):
    betas = get_beta_schedule(**BETA_SCHEDULE)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)


def get_loss_noised(model, batch, alphas_cumprod, t_max, protein_noise_std, device, eval_mode=False):
    protein_noise = torch.randn_like(batch.protein_pos) * protein_noise_std
    if eval_mode:
        # Validation/test: no ligand noise (t=0), for direct comparability
        # with the original Stage 0 checkpoint's reported val/test numbers.
        ligand_pos_perturbed = batch.ligand_pos
        t_sampled = 0
    else:
        t_sampled = int(torch.randint(0, t_max + 1, (1,)).item())
        a_t = alphas_cumprod[t_sampled]
        ligand_noise = torch.randn_like(batch.ligand_pos)
        ligand_pos_perturbed = a_t.sqrt() * batch.ligand_pos + (1.0 - a_t).sqrt() * ligand_noise
    pred = model(
        protein_pos=batch.protein_pos + protein_noise,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=ligand_pos_perturbed,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        output_kind=None,
    )
    loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
    return loss, pred, t_sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500,
                         help='Upper bound (inclusive) of the sampled diffusion timestep for the '
                              'ligand-position noise curriculum. See module docstring for why this '
                              'is bounded rather than the full [0, 999].')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_noisematched')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_noisematched', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    logger.info(f'Noise schedule: {BETA_SCHEDULE}, t_max={args.t_max}')
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    alphas_cumprod = build_alphas_cumprod(args.device)
    logger.info(f'sqrt(1-a_t) at t=0,{args.t_max//2},{args.t_max}: '
                f'{(1 - alphas_cumprod[0]).sqrt().item():.4f}, '
                f'{(1 - alphas_cumprod[args.t_max // 2]).sqrt().item():.4f}, '
                f'{(1 - alphas_cumprod[args.t_max]).sqrt().item():.4f}')

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
        for it, batch in enumerate(tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch}'), start=1):
            batch = batch.to(args.device)
            loss, _, t_sampled = get_loss_noised(model, batch, alphas_cumprod, args.t_max,
                                                  pos_noise_std_protein(config), args.device)
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
            writer.add_scalar('train/t_sampled', t_sampled, global_it)

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
                        't_max': args.t_max, 'beta_schedule': BETA_SCHEDULE,
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


def pos_noise_std_protein(config):
    return config.train.pos_noise_std


if __name__ == '__main__':
    main()
