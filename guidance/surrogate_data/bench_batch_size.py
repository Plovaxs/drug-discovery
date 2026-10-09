"""Measures training throughput and peak VRAM for the Stage 0 EGNN at several batch sizes, to decide the
batch size for the surrogate arms. This matters a lot: the whole arm schedule is ~31 h at a workable batch
size vs ~189 h at batch_size=1 (the config's current value), so this ~15 min measurement decides days.

Runs a fixed number of real training steps (forward+backward+step) per batch size on the real mixed
dataset, so the numbers include the actual graph sizes and collation cost -- not a synthetic benchmark.
Reports rows/sec (the quantity that sets epoch time) rather than it/sec, since larger batches do more work
per iteration. OOM is caught and reported rather than crashing the sweep.

Usage:
  python guidance/surrogate_data/bench_batch_size.py
"""
import argparse
import time

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from datasets.pl_pair_dataset import PocketLigandPairDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from models.property_pred.prop_model import PropPredNet
from utils.train import get_optimizer

BN_ROOT = './data/bindingnet_pocket10'


def bench(batch_size, dataset, config, device, n_steps, pf, lf, amp=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    misc.seed_all(2021)
    model = PropPredNet(config.model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(device)
    opt = get_optimizer(config.train.optimizer, model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                        follow_batch=['protein_element', 'ligand_element'],
                        exclude_keys=['ligand_nbh_list'])
    model.train()
    it = iter(loader)
    # warm-up (CUDA context, cuDNN autotune, worker spin-up) excluded from timing
    for _ in range(3):
        batch = next(it).to(device)
        pred = model(protein_pos=batch.protein_pos, protein_atom_feature=batch.protein_atom_feature.float(),
                     ligand_pos=batch.ligand_pos, ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                     batch_protein=batch.protein_element_batch, batch_ligand=batch.ligand_element_batch,
                     output_kind=None)
        loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
        loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()

    rows, t0 = 0, time.time()
    for _ in range(n_steps):
        batch = next(it).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            pred = model(protein_pos=batch.protein_pos,
                         protein_atom_feature=batch.protein_atom_feature.float(),
                         ligand_pos=batch.ligand_pos,
                         ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                         batch_protein=batch.protein_element_batch,
                         batch_ligand=batch.ligand_element_batch, output_kind=None)
            loss = torch.nn.functional.mse_loss(pred.view(-1), batch.y)
        loss.backward(); opt.step(); opt.zero_grad()
        rows += len(batch.y)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, opt, loader
    return rows / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/prop/crossdocked_affinity_egnn.yml')
    ap.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--n_steps', type=int, default=40)
    args = ap.parse_args()

    config = misc.load_config(args.config)
    device = 'cuda'
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    tf = Compose([pf, lf])

    cd_base, cd_splits, cd_pk = build_lp_splits(train_subsample=None)
    cd = CrossDockedAffinityDataset(cd_base, cd_splits['train'][:3000], cd_pk, tf)
    import pandas as pd
    lab = pd.read_csv(f'{BN_ROOT}/labels.csv').head(3000)
    bn_base = PocketLigandPairDataset(BN_ROOT)
    bn = CrossDockedAffinityDataset(bn_base, lab['idx'].astype(int).tolist(),
                                    dict(zip(lab['idx'].astype(int), lab['pk'].astype(float))), tf)
    dataset = ConcatDataset([cd, bn])  # realistic mix, so graph sizes match the arms
    print(f'bench dataset: {len(dataset)} rows (CD+BN mix), {args.n_steps} timed steps per size\n')

    print(f"{'batch':>6} {'amp':>5} {'rows/sec':>10} {'peak GB':>9} {'min/ep @46,964':>16} {'min/ep @172,202':>17}")
    base_rate = None
    for amp in (False, True):
        for bs in args.sizes:
            try:
                rate, peak = bench(bs, dataset, config, device, args.n_steps, pf, lf, amp=amp)
                if base_rate is None:
                    base_rate = rate
                print(f'{bs:>6} {str(amp):>5} {rate:>10.1f} {peak:>9.2f} {46964 / rate / 60:>16.1f} '
                      f'{172202 / rate / 60:>17.1f}   ({rate / base_rate:.2f}x vs bs1-noamp)')
            except torch.cuda.OutOfMemoryError:
                print(f'{bs:>6} {str(amp):>5} {"OOM":>10}')
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f'{bs:>6} {str(amp):>5} {"ERROR":>10}  {type(e).__name__}: {e}')
                torch.cuda.empty_cache()
                break


if __name__ == '__main__':
    main()
