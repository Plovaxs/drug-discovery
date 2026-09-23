"""Track A exploratory-tier checkpoint: partial training of the GIGN+PIGNet2
hybrid (guidance/track_a/model.py) on the same leakage-safe LP-PDBBind-style
split used for Stage 0 (guidance/lp_split/leakage_safe_split.json), so its
predictive-quality numbers are directly comparable to Stage 0's EGNN
checkpoint (guidance/FLAGSHIP_ARCHITECTURE_RESULTS.md).

Deliberately NOT a from-scratch DataLoader/batching pipeline: this
architecture's intra/inter edges and interaction masks are built per-sample
(features.py) before any PyG batching could apply, and since this is the
exploratory tier (partial training, not the final/full tier), batch_size=1
is used throughout rather than writing a custom heterogeneous-graph
collate_fn -- correct and simple, at the cost of no batching speedup. A
faster custom collate function would be worth writing if Track A clears
its exploratory-tier gate and proceeds to full-tier training.

"Partial training" here means: a fixed, small iteration budget (not
early-stopped to convergence like Stage 0's EGNN was) -- enough to get a
real, honest predictive-quality readout and a real, honest gradient set for
the guidance wrapper to use, without spending the multi-hour budget a full
Stage 2 training run would need. See guidance/STAGE2_PLUS_EXPERIMENT_LOG.md
for how this checkpoint's results are logged and gated.
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from utils.misc_prop import get_eval_scores
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_a.model import GIGNPignetAffinity
from guidance.track_a.features import (
    ligand_interaction_masks, protein_interaction_masks, vdw_radii_for, build_intra_inter_edges,
)


def prepare_sample(data, device):
    """Builds all the extra per-sample tensors GIGNPignetAffinity needs,
    beyond what CrossDockedAffinityDataset + the standard transforms
    already attach."""
    ligand_pos = data.ligand_pos.to(device)
    protein_pos = data.protein_pos.to(device)
    ligand_feat = data.ligand_atom_feature_full.float().to(device)
    protein_feat = data.protein_atom_feature.float().to(device)

    edge_index_intra, edge_index_inter = build_intra_inter_edges(
        ligand_pos, protein_pos, data.ligand_bond_index.to(device))

    l_metal, l_donor, l_acceptor, l_hydrophobic = ligand_interaction_masks(
        data.ligand_element.to(device), data.ligand_atom_feature.to(device))
    p_metal, p_donor, p_acceptor, p_hydrophobic = protein_interaction_masks(
        data.protein_element.to(device))

    is_metal = torch.cat([l_metal, p_metal])
    is_h_donor = torch.cat([l_donor, p_donor])
    is_h_acceptor = torch.cat([l_acceptor, p_acceptor])
    is_hydrophobic = torch.cat([l_hydrophobic, p_hydrophobic])
    vdw_radii = torch.cat([
        vdw_radii_for(data.ligand_element).to(device),
        vdw_radii_for(data.protein_element).to(device),
    ])

    n_l, n_p = ligand_pos.size(0), protein_pos.size(0)
    batch_ligand = torch.zeros(n_l, dtype=torch.long, device=device)
    batch_protein = torch.zeros(n_p, dtype=torch.long, device=device)

    return dict(
        protein_pos=protein_pos, protein_feat=protein_feat,
        ligand_pos=ligand_pos, ligand_feat=ligand_feat,
        edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
        batch_ligand=batch_ligand, batch_protein=batch_protein,
        vdw_radii=vdw_radii, is_metal=is_metal, is_h_donor=is_h_donor,
        is_h_acceptor=is_h_acceptor, is_hydrophobic=is_hydrophobic,
    )


def get_loss(model, data, device):
    kwargs = prepare_sample(data, device)
    pred, energies = model(**kwargs)
    y = data.y.to(device).view(-1)
    loss = torch.nn.functional.mse_loss(pred, y)
    return loss, pred, energies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_subsample', type=int, default=400,
                        help='exploratory tier: small subsample, not Stage 0''s 6000, '
                             'since batch_size=1 makes a full pass much slower')
    parser.add_argument('--max_iters', type=int, default=2000,
                        help='fixed iteration budget for partial (non-convergent) training')
    parser.add_argument('--val_freq', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_track_a_stage2')
    parser.add_argument('--tag', type=str, default='exploratory')
    args = parser.parse_args()

    misc.seed_all(2021)
    log_dir = misc.get_new_log_dir(args.logdir, prefix='gign_pignet', tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_gign_pignet_stage2', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    bond_featurizer = utils_trans.FeaturizeLigandBond()
    transform = Compose([protein_featurizer, ligand_featurizer, bond_featurizer])

    logger.info('Building Stage 0 leakage-safe LP-PDBBind-style splits (reused for Track A)...')
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

    def evaluate(data_list, prefix):
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
        logger.info('[%s] Loss %.6f' % (prefix, avg_loss))
        if sum_n > 1:
            get_eval_scores(ypred_arr, ytrue_arr, logger, prefix=prefix)
        return avg_loss

    best_val_loss = float('inf')
    model.train()
    optimizer.zero_grad()
    it = 0
    t0 = time.time()
    pbar = tqdm(total=args.max_iters, desc='Track A exploratory training')
    while it < args.max_iters:
        for data in train_set:
            if it >= args.max_iters:
                break
            loss, pred, energies = get_loss(model, data, device)
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            it += 1
            pbar.update(1)
            writer.add_scalar('train/loss', loss.item(), it)
            writer.add_scalar('train/grad', grad_norm, it)
            if it % 50 == 0:
                logger.info(f'Iter {it:05d} | Loss {loss.item():.6f} | '
                           f'elapsed {time.time() - t0:.0f}s')

            if it % args.val_freq == 0 or it == args.max_iters:
                val_loss = evaluate(val_set, prefix='Validate')
                writer.add_scalar('val/loss', val_loss, it)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f'Best val achieved at iter {it}, val loss: {best_val_loss:.3f}')
                    test_loss = evaluate(test_set, prefix='Test')
                    ckpt_path = os.path.join(ckpt_dir, 'best.pt')
                    torch.save({
                        'model': model.state_dict(),
                        'protein_atom_feature_dim': protein_featurizer.feature_dim,
                        'ligand_atom_feature_dim': ligand_featurizer.feature_dim,
                        'hidden_dim': args.hidden_dim,
                        'iter': it, 'val_loss': best_val_loss, 'test_loss': test_loss,
                    }, ckpt_path)
                    logger.info(f'Model saved to {ckpt_path}')
                model.train()
    pbar.close()
    logger.info(f'Done. Best val loss: {best_val_loss:.3f}. Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
