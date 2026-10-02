"""EGNN + ESM2 global protein feature (see guidance/lp_split/esm2_embed.py's
module docstring for why this is tried and what it discloses). Otherwise
identical to the noise-matched + vina-target recipe
(train_egnn_stage0_vinatarget.py) that gave the best diagnostic profile
among the pure-geometry variants, so this isolates the effect of ADDING the
ESM2 feature on top of that existing best recipe, rather than re-deriving a
new baseline from scratch.

The ESM2 embedding is looked up per-example from its target name (the
"<GENE>_<SPECIES>_..." prefix of `data.ligand_filename`, identical
convention to guidance/lp_split/uniprot_client.py), mapped through
uniprot_cache.json's gene/species-to-accession table, then
esm2_embeddings.pt's accession-to-480-dim-vector cache. It is a FIXED,
non-differentiable input appended to the pooled graph representation before
the output head, exactly like train_egnn_stage0_timecond.py's time
embedding -- never backpropagated into the EGNN's own atom features.

Targets with no cached UniProt match or no cached ESM2 embedding (a small
minority, see reported count at startup) fall back to a zero vector --
disclosed, not silently imputed as if it were a real embedding.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from scipy.stats import pearsonr
from torch.nn.utils import clip_grad_norm_
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from models.common import compose_context_prop, ShiftedSoftplus
from models.property_pred.prop_model import get_encoder
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.lp_split.train_egnn_stage0_noisematched import build_alphas_cumprod
from guidance.lp_split.train_egnn_stage0_vinatarget import build_vina_pk_by_idx

ESM2_DIM = 480
ESM2_PROJ_DIM = 32
UNIPROT_CACHE_PATH = './guidance/lp_split/uniprot_cache.json'
ESM2_CACHE_PATH = './guidance/lp_split/esm2_embeddings.pt'


def build_target_to_esm2():
    """Returns a vec_for_target(target_name) closure resolving a target's
    ESM2 embedding (or a zero vector if unavailable), memoized per-target
    to avoid recomputing the gene_species->accession lookup for every one
    of the ~47k dataset entries that share a much smaller set of targets."""
    with open(UNIPROT_CACHE_PATH) as f:
        uniprot_cache = json.load(f)
    esm2_cache = torch.load(ESM2_CACHE_PATH)

    target_to_vec = {}
    n_hit, n_miss = 0, 0

    def vec_for_target(target_name):
        if target_name in target_to_vec:
            return target_to_vec[target_name]
        gene_species = '_'.join(target_name.split('_')[:2])
        urec = uniprot_cache.get(gene_species)
        vec = None
        if urec and urec.get('found'):
            vec = esm2_cache.get(urec['accession'])
        if vec is None:
            vec = torch.zeros(ESM2_DIM)
        target_to_vec[target_name] = vec
        return vec

    return vec_for_target, uniprot_cache, esm2_cache


class PropPredNetESM2(nn.Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, output_dim=1,
                 esm2_dim=ESM2_DIM, esm2_proj_dim=ESM2_PROJ_DIM):
        super().__init__()
        self.hidden_dim = config.hidden_channels
        self.output_dim = output_dim
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, self.hidden_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_dim)
        self.encoder = get_encoder(config.encoder)
        self.esm2_proj = nn.Sequential(
            nn.Linear(esm2_dim, esm2_proj_dim), ShiftedSoftplus(),
        )
        self.out_block = nn.Sequential(
            nn.Linear(self.hidden_dim + esm2_proj_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature,
                batch_protein, batch_ligand, esm2_vec, output_kind=None):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)
        h_ctx, pos_ctx, batch_ctx = compose_context_prop(
            h_protein=h_protein, h_ligand=h_ligand,
            pos_protein=protein_pos, pos_ligand=ligand_pos,
            batch_protein=batch_protein, batch_ligand=batch_ligand,
        )
        h_ctx = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx)
        pre_out = scatter(h_ctx, index=batch_ctx, dim=0, reduce='sum')  # (num_graphs, H)

        esm2_emb = self.esm2_proj(esm2_vec)  # (num_graphs, esm2_proj_dim)
        pre_out = torch.cat([pre_out, esm2_emb], dim=-1)

        output = self.out_block(pre_out)
        if output_kind is not None:
            output_mask = F.one_hot(output_kind - 1, self.output_dim)
            output = torch.sum(output * output_mask, dim=-1, keepdim=True)
        return output


def get_loss(model, data, alphas_cumprod, t_max, device, vec_for_target, eval_mode=False):
    protein_pos = data.protein_pos.to(device)
    ligand_pos = data.ligand_pos.to(device)
    if eval_mode:
        ligand_pos_in = ligand_pos
    else:
        t = int(torch.randint(0, t_max + 1, (1,)).item())
        a_t = alphas_cumprod[t]
        ligand_pos_in = a_t.sqrt() * ligand_pos + (1.0 - a_t).sqrt() * torch.randn_like(ligand_pos)

    target_name = data.ligand_filename.split('/')[0]
    esm2_vec = vec_for_target(target_name).to(device).unsqueeze(0)

    n_l, n_p = ligand_pos_in.size(0), protein_pos.size(0)
    pred = model(
        protein_pos=protein_pos,
        protein_atom_feature=data.protein_atom_feature.float().to(device),
        ligand_pos=ligand_pos_in,
        ligand_atom_feature=data.ligand_atom_feature_full.float().to(device),
        batch_protein=torch.zeros(n_p, dtype=torch.long, device=device),
        batch_ligand=torch.zeros(n_l, dtype=torch.long, device=device),
        esm2_vec=esm2_vec,
    )
    y = data.y.to(device).view(-1)
    loss = F.mse_loss(pred.view(-1), y)
    return loss, pred


def evaluate(model, data_list, device, alphas_cumprod, t_max, vec_for_target, logger, prefix):
    model.eval()
    ypred_arr, ytrue_arr = [], []
    sum_loss, sum_n = 0.0, 0
    with torch.no_grad():
        for data in tqdm(data_list, desc=prefix, leave=False):
            loss, pred = get_loss(model, data, alphas_cumprod, t_max, device, vec_for_target, eval_mode=True)
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
    parser.add_argument('config', type=str)
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_lp_split_stage0_esm2')
    parser.add_argument('--tag', type=str, default='esm2')
    parser.add_argument('--target', type=str, default='vina', choices=['vina', 'experimental'],
                         help='vina: Vina-derived pK anchor (matches vinatarget.py). '
                              'experimental: real pKd/pKi/pIC50 (matches noisematched.py).')
    args = parser.parse_args()

    misc.seed_all(2021)
    log_dir = misc.get_new_log_dir(args.logdir, prefix='crossdocked_affinity_egnn', tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_egnn_stage0_esm2', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    device = args.device if torch.cuda.is_available() else 'cpu'
    config = misc.load_config(args.config)
    alphas_cumprod = build_alphas_cumprod(device)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits...')
    base, splits, pk_by_idx_exp = build_lp_splits(train_subsample=args.train_subsample)
    if args.target == 'vina':
        logger.info('Building Vina-derived pK anchor labels...')
        pk_by_idx = build_vina_pk_by_idx()
    else:
        pk_by_idx = pk_by_idx_exp

    logger.info('Building target -> ESM2 embedding lookup...')
    vec_for_target, uniprot_cache, esm2_cache = build_target_to_esm2()
    logger.info(f'{len(esm2_cache)} cached ESM2 embeddings, '
                f'{sum(1 for v in uniprot_cache.values() if v.get("found"))} resolvable UniProt accessions')

    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)
    logger.info(f'Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}')

    model = PropPredNetESM2(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.95, 0.999))

    best_val_loss = float('inf')
    patience_count = 0
    best_path = os.path.join(ckpt_dir, 'best.pt')

    t0 = time.time()
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        g = torch.Generator().manual_seed(2021 + epoch)
        perm = torch.randperm(len(train_set), generator=g).tolist()

        epoch_loss_sum, epoch_n = 0.0, 0
        for idx in tqdm(perm, desc=f'Epoch {epoch}', dynamic_ncols=True):
            data = train_set[idx]
            loss, _ = get_loss(model, data, alphas_cumprod, args.t_max, device, vec_for_target)
            loss.backward()
            clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss_sum += loss.item()
            epoch_n += 1

        train_loss = epoch_loss_sum / max(epoch_n, 1)
        elapsed = time.time() - t0
        logger.info(f'Epoch {epoch:03d} | Train loss {train_loss:.6f} | elapsed {elapsed:.0f}s')
        writer.add_scalar('train/epoch_loss', train_loss, epoch)

        val_loss, val_pearson = evaluate(model, val_set, device, alphas_cumprod, args.t_max,
                                         vec_for_target, logger, prefix=f'Validate epoch {epoch}')
        writer.add_scalar('val/loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            logger.info(f'Best val LOSS achieved at epoch {epoch}, val loss: {best_val_loss:.6f}')
            test_loss, test_pearson = evaluate(model, test_set, device, alphas_cumprod, args.t_max,
                                               vec_for_target, logger, prefix=f'Test (epoch {epoch})')
            torch.save({
                'model': model.state_dict(), 'config': config,
                'protein_atom_feature_dim': protein_featurizer.feature_dim,
                'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                'epoch': epoch, 'val_loss': best_val_loss, 'val_pearson': val_pearson,
                'test_loss': test_loss, 'test_pearson': test_pearson,
                't_max': args.t_max, 'target': args.target, 'esm2_dim': ESM2_DIM,
                'esm2_proj_dim': ESM2_PROJ_DIM,
            }, best_path)
            logger.info(f'Model saved to {best_path}')
        else:
            patience_count += 1
            logger.info(f'Val loss did not improve (patience {patience_count}/{args.patience}), '
                       f'best so far: {best_val_loss:.6f}')
            if patience_count >= args.patience:
                logger.info(f'Early stopping at epoch {epoch}.')
                break

    logger.info(f'Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
