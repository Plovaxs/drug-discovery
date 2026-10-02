"""Fourth follow-up branch: a TIME-CONDITIONED guidance predictor.

Rationale (gap disclosed in train_egnn_stage0_noisematched.py's own
docstring, not discovered after the fact): that earlier branch trained the
EGNN on ligand positions noised to match the diffusion schedule, but the
architecture itself was never told HOW MUCH noise was present at any given
example -- the same network had to produce one prediction whether the
input was nearly clean (t near 0) or substantially perturbed (t near
t_max). This is unlike real classifier guidance in image diffusion
(Dhariwal & Nichol 2021), where the classifier is always given the
timestep as an input alongside the noised image, precisely so it can
calibrate its answer to the noise level. This branch adds that missing
ingredient: a sinusoidal timestep embedding, concatenated to the pooled
graph representation before the output head.

This is the properly-motivated version of the clean/noisy mismatch fix --
train_egnn_stage0_noisematched.py tested "expose the model to noise during
training", this tests "expose the model to noise AND tell it how much",
which is the actual recipe used in the literature this project cites
(Section II.2 / IX.2) for this exact problem.

Architecture: PropPredNetTimeConditioned wraps the unmodified EGNN encoder
(models/property_pred/prop_egnn.EnEquiEncoder) and protein/ligand
embeddings exactly as PropPredNet does, only the final output block's
input dimension changes (hidden_dim + time_emb_dim instead of hidden_dim).
At inference (guidance time), the current diffusion timestep i is already
available in guided_sampling.py's loop (the loop variable), so wiring this
checkpoint into guidance (not done in this feasibility/training branch)
would need AffinityGuidance.grad_log_score to accept and forward a
timestep argument -- a small, disclosed follow-on change, not implemented
here; this script and its planned evaluation test the predictor's
properties (quality, gradient diagnostic) at known, fixed timesteps only.
"""
import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from utils.train import get_optimizer, get_scheduler
from models.common import compose_context_prop, ShiftedSoftplus
from models.property_pred.prop_model import get_encoder
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod


def sinusoidal_embedding(t, dim, max_period=1000):
    """Standard DDPM-style sinusoidal timestep embedding. t: (B,) long or float."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class PropPredNetTimeConditioned(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, output_dim=1, time_emb_dim=32):
        super().__init__()
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.time_emb_dim = time_emb_dim
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)
        self.encoder = get_encoder(config.encoder)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), ShiftedSoftplus(),
        )
        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim + time_emb_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature,
                batch_protein, batch_ligand, t, output_kind=None):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)
        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein, h_ligand=h_ligand,
            pos_protein=protein_pos, pos_ligand=ligand_pos,
            batch_protein=batch_protein, batch_ligand=batch_ligand,
        )
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (num_graphs, H)

        num_graphs = pre_out.size(0)
        t_vec = t if torch.is_tensor(t) and t.numel() == num_graphs else t.expand(num_graphs)
        t_emb = self.time_mlp(sinusoidal_embedding(t_vec, self.time_emb_dim))
        pre_out = torch.cat([pre_out, t_emb], dim=-1)

        output = self.out_block(pre_out)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output


def get_loss(model, batch, alphas_cumprod, t_max, pos_noise_std_protein, device, eval_mode=False):
    protein_pos = batch.protein_pos
    protein_noise = torch.randn_like(protein_pos) * pos_noise_std_protein
    if eval_mode:
        ligand_pos_in = batch.ligand_pos
        t = torch.zeros(1, device=device, dtype=torch.long)
    else:
        t_val = int(torch.randint(0, t_max + 1, (1,)).item())
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        a_t = alphas_cumprod[t_val]
        ligand_pos_in = a_t.sqrt() * batch.ligand_pos + (1.0 - a_t).sqrt() * torch.randn_like(batch.ligand_pos)
    pred = model(
        protein_pos=protein_pos + protein_noise,
        protein_atom_feature=batch.protein_atom_feature.float(),
        ligand_pos=ligand_pos_in,
        ligand_atom_feature=batch.ligand_atom_feature_full.float(),
        batch_protein=batch.protein_element_batch,
        batch_ligand=batch.ligand_element_batch,
        t=t,
    )
    loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
    return loss, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--time_emb_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_timecond')
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_timecond', log_dir)
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

    model = PropPredNetTimeConditioned(
        config.model, protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim, output_dim=1,
        time_emb_dim=args.time_emb_dim,
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
            loss, _ = get_loss(model, batch, alphas_cumprod, args.t_max, config.train.pos_noise_std, args.device)
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
            for batch in tqdm(data_loader, desc=prefix):
                batch = batch.to(args.device)
                loss, pred = get_loss(model, batch, alphas_cumprod, args.t_max, 0., args.device, eval_mode=True)
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
                        't_max': args.t_max, 'time_emb_dim': args.time_emb_dim,
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
