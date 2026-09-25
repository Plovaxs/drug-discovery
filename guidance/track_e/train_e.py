"""Track E resumable trainer (one run = one (arm, seed)).

Families
  gign : Track A GIGN+PIGNet2 backbone (TrackEModel). Arms differ only by --target {abs,delta}, --head {phys,mlp},
         --aux {none,plip,scrambled}, --lam, --pcgrad.
  egnn : Stage 0 EGNN, byte-matched to guidance/lp_split/train_egnn_stage0.py (6000-entry train subsample, pos noise 0.1,
         plateau scheduler, patience 5, max 20 epochs); only the training seed differs (extra baseline seeds).

Robustness (hard requirement): atomic checkpoints (tmp file + fsync + os.replace) every --ckpt_minutes AND at every epoch
boundary AND on SIGTERM/SIGINT; the checkpoint carries model, optimiser, scheduler, epoch, step-in-epoch, best/patience,
history, and all RNG states, so a resumed run continues mid-epoch. Exit code 75 = interrupted (resume me), 0 = done.
"""
import argparse
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.track_a.train_gign_pignet_stage2 import prepare_sample
from guidance.track_e.core import (PRIMARY_CLASSES, PLIP_CLASSES_ALL, PlipLabelStore, build_anchor_table, delta_target,
                                   pk_vina, scramble_permutation, split_arrays)

EXIT_INTERRUPTED = 75
STOP = {'flag': False}


def _on_signal(signum, frame):
    STOP['flag'] = True


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(), 'python': random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s['cuda'].cpu())
    np.random.set_state(s['numpy'])
    random.setstate(s['python'])


class Log:
    def __init__(self, path):
        self.f = open(path, 'a')

    def __call__(self, msg):
        line = time.strftime('[%Y-%m-%d %H:%M:%S] ') + msg
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()


# ---------------------------------------------------------------------------------------------------------------
def build_everything(args, log):
    """Returns a dict with datasets, model, and per-family closures."""
    device = args.device if torch.cuda.is_available() else 'cpu'
    ctx = {'device': device}
    if args.family == 'egnn':
        pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
        transform = Compose([pf, lf])
        base, splits, pk_by_idx = build_lp_splits(train_subsample=args.train_subsample or 6000)
        mk = lambda p: CrossDockedAffinityDataset(base, splits[p], pk_by_idx, transform)
        ctx.update(train=mk('train'), val=mk('val'), test=mk('test'))
        from models.property_pred.prop_model import PropPredNet
        config = misc.load_config('configs/prop/crossdocked_affinity_egnn.yml')
        ctx['config'] = config
        ctx['model'] = PropPredNet(config.model, protein_atom_feature_dim=pf.feature_dim,
                                   ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(device)
        return ctx

    pf, lf, bf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom(), utils_trans.FeaturizeLigandBond()
    transform = Compose([pf, lf, bf])
    table = build_anchor_table()
    A = {p: split_arrays(table, p) for p in ('train', 'val', 'test')}
    vina_by_idx = {int(i): float(v) for p in A for i, v in zip(A[p]['idx'], A[p]['vina'])}
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    keep = {p: [int(i) for i, v in zip(A[p]['idx'], A[p]['vina']) if (v <= 0 or p == 'test')] for p in A}
    if args.train_subsample:
        keep['train'] = keep['train'][:args.train_subsample]
    if args.limit_val:
        keep['val'] = keep['val'][:args.limit_val]
    if args.limit_test:
        keep['test'] = keep['test'][:args.limit_test]
    log(f"filtered Vina>0 from train/val; sizes train={len(keep['train'])} val={len(keep['val'])} test={len(keep['test'])}")
    mk = lambda p: CrossDockedAffinityDataset(base, keep[p], pk_by_idx, transform)
    ctx.update(train=mk('train'), val=mk('val'), test=mk('test'), vina_by_idx=vina_by_idx, pk_by_idx=pk_by_idx)

    # standardisation statistics from the (filtered) training split
    tr_pk = np.array([pk_by_idx[i] for i in keep['train']]); tr_v = np.array([vina_by_idx[i] for i in keep['train']])
    t = tr_pk if args.target == 'abs' else delta_target(tr_pk, tr_v)
    ctx['mu'], ctx['sd'] = float(t.mean()), float(t.std())
    log(f"target={args.target} standardisation mu={ctx['mu']:.4f} sd={ctx['sd']:.4f}")

    n_aux = 0
    if args.aux != 'none':
        classes = args.classes.split(',')
        n_aux = len(classes)
        store = PlipLabelStore()
        ctx['store'], ctx['classes'] = store, classes
        missing = [i for p in keep for i in keep[p] if i not in store]
        if missing and not args.benchmark_steps:
            raise RuntimeError(f'{len(missing)} complexes lack cached PLIP labels (build_plip_cache.py not finished?)')
        lab_tr = [store.get(i, classes) for i in keep['train'] if i in store]
        if lab_tr:
            pos = np.concatenate(lab_tr).mean(0)
            ctx['pos_weight'] = torch.tensor(np.minimum((1 - pos) / np.maximum(pos, 1e-6), 10.0), dtype=torch.float32, device=device)
            log(f"aux classes {classes} train atom-level positive rate {np.round(pos, 4).tolist()} pos_weight {ctx['pos_weight'].tolist()}")
        else:
            ctx['pos_weight'] = torch.ones(n_aux, device=device)
        # complexes with no detected interaction (over the primary classes) are masked out of the aux loss
        ctx['aux_mask_ok'] = {i: bool(store.get(i, classes).any()) for i in keep['train'] + keep['val'] + keep['test'] if i in store}
        if args.aux == 'scrambled':
            n_lig = np.array([store.get(i, classes).shape[0] if i in store else 0 for i in keep['train']])
            src = scramble_permutation(n_lig, seed=9000 + args.seed)
            ctx['donor'] = {keep['train'][k]: keep['train'][int(src[k])] for k in range(len(keep['train']))}
            log(f"scrambled control: {(src == np.arange(len(src))).sum()} of {len(src)} complexes left unscrambled (singleton size groups)")
    from guidance.track_e.models_e import TrackEModel
    ctx['model'] = TrackEModel(pf.feature_dim, lf.feature_dim, hidden_dim=args.hidden, head=args.head, n_aux=n_aux).to(device)
    return ctx


def make_batch(data, device):
    b = Batch.from_data_list([data], follow_batch=['protein_element', 'ligand_element'], exclude_keys=['ligand_nbh_list'])
    return b.to(device)


# ---------------------------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, args, run_dir, log):
        self.args, self.dir, self.log = args, run_dir, log
        misc.seed_all(args.seed)                    # BEFORE model construction: initial weights must depend on the seed
        self.ctx = build_everything(args, log)
        self.model, self.device = self.ctx['model'], self.ctx['device']
        log(f'# trainable parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.4f} M')
        if args.family == 'egnn':
            from utils.train import get_optimizer, get_scheduler
            cfg = self.ctx['config']
            self.opt = get_optimizer(cfg.train.optimizer, self.model)
            self.sched = get_scheduler(cfg.train.scheduler, self.opt)
            self.patience_max, self.max_epochs = cfg.train.early_stop_patience, cfg.train.max_epochs
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.95, 0.999))
            self.sched = None
            self.patience_max, self.max_epochs = args.patience, args.max_epochs
        self.shared = self.model.shared_parameters() if args.family == 'gign' else None
        self.cos_log = os.path.join(run_dir, 'cos_log.jsonl')

    # ---- per-sample loss / prediction --------------------------------------------------------------
    def _targets(self, data):
        idx = int(data.id)
        pk = float(data.y)
        if self.args.target == 'abs':
            t = pk
        else:
            t = float(delta_target(pk, self.ctx['vina_by_idx'][idx]))
        return (t - self.ctx['mu']) / self.ctx['sd']

    def _aux_labels(self, idx):
        src = self.ctx['donor'][idx] if self.args.aux == 'scrambled' and idx in self.ctx.get('donor', {}) else idx
        store = self.ctx['store']
        if src not in store:                                   # benchmark-only fallback
            return torch.zeros(1, len(self.ctx['classes']), device=self.device)
        return torch.tensor(store.get(src, self.ctx['classes']), dtype=torch.float32, device=self.device)

    def forward_loss(self, data, train=True, want_parts=False):
        if self.args.family == 'egnn':
            batch = make_batch(data, self.device)
            noise = self.ctx['config'].train.pos_noise_std if train else 0.0
            pred = self.model(protein_pos=batch.protein_pos + torch.randn_like(batch.protein_pos) * noise,
                              protein_atom_feature=batch.protein_atom_feature.float(),
                              ligand_pos=batch.ligand_pos + torch.randn_like(batch.ligand_pos) * noise,
                              ligand_atom_feature=batch.ligand_atom_feature_full.float(),
                              batch_protein=batch.protein_element_batch, batch_ligand=batch.ligand_element_batch,
                              output_kind=None)
            loss = F.mse_loss(pred.view(-1), batch.y)
            return loss, None, {'pred': pred.view(-1).item()}
        kw = prepare_sample(data, self.device)
        out = self.model(**kw)
        z = torch.tensor([self._targets(data)], dtype=torch.float32, device=self.device)
        z_hat = out['delta_z'].view(-1)
        loss_main = F.mse_loss(z_hat, z)
        loss_aux, probs = None, None
        if self.args.aux != 'none':
            lab = self._aux_labels(int(data.id))
            logits = out['aux_logits']
            if lab.shape[0] == logits.shape[0]:
                loss_aux = F.binary_cross_entropy_with_logits(logits, lab, pos_weight=self.ctx['pos_weight'])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            if not self.ctx['aux_mask_ok'].get(int(data.id), True):
                loss_aux = None                                # zero-interaction complex: masked out of the aux loss
        return loss_main, loss_aux, {'z_hat': z_hat.item(), 'probs': probs}

    def to_original_units(self, data, z_hat):
        idx = int(data.id)
        v = self.ctx['mu'] + self.ctx['sd'] * z_hat
        if self.args.target == 'delta':
            v = v + float(pk_vina(self.ctx['vina_by_idx'][idx]))
        return v

    # ---- one optimisation step -----------------------------------------------------------------------
    def train_step(self, data, global_step):
        a = self.args
        loss_main, loss_aux, _ = self.forward_loss(data, train=True)
        self.opt.zero_grad()
        if a.family == 'egnn' or loss_aux is None or a.aux == 'none':
            loss_main.backward()
        else:
            do_cos = (global_step % 200 == 0) or a.pcgrad
            if do_cos:
                g1 = torch.autograd.grad(loss_main, self.shared, retain_graph=True, allow_unused=True)
                g2 = torch.autograd.grad(a.lam * loss_aux, self.shared, retain_graph=True, allow_unused=True)
                f1 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g1, self.shared)])
                f2 = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g2, self.shared)])
                dot = torch.dot(f1, f2)
                cos = (dot / (f1.norm() * f2.norm() + 1e-12)).item()
                if global_step % 200 == 0:
                    with open(self.cos_log, 'a') as f:
                        f.write(json.dumps({'step': global_step, 'cos': cos}) + '\n')
            if a.pcgrad:
                shared_ids = {id(p) for p in self.shared}
                other = [p for p in self.model.parameters() if id(p) not in shared_ids]
                total = loss_main + a.lam * loss_aux
                go = torch.autograd.grad(total, other, allow_unused=True)
                if dot < 0:
                    f1p = f1 - dot / (f2.norm() ** 2 + 1e-12) * f2
                    f2p = f2 - dot / (f1.norm() ** 2 + 1e-12) * f1
                else:
                    f1p, f2p = f1, f2
                merged, off = f1p + f2p, 0
                for p in self.shared:
                    n = p.numel()
                    p.grad = merged[off:off + n].view_as(p).clone()
                    off += n
                for p, g in zip(other, go):
                    p.grad = g
            else:
                (loss_main + a.lam * loss_aux).backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        return float(loss_main.item())

    # ---- evaluation ------------------------------------------------------------------------------------
    def evaluate(self, dataset, collect=False, check_stop=True):
        self.model.eval()
        ys, yh, loss_sum, n = [], [], 0.0, 0
        idxs, aux_p, aux_y, aux_len = [], [], [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                if check_stop and STOP['flag']:
                    self.model.train()
                    return None
                data = dataset[i]
                loss_main, _, parts = self.forward_loss(data, train=False)
                loss_sum += float(loss_main.item()); n += 1
                if self.args.family == 'egnn':
                    yhat = parts['pred']
                else:
                    yhat = self.to_original_units(data, parts['z_hat'])
                ys.append(float(data.y)); yh.append(yhat); idxs.append(int(data.id))
                if collect and parts.get('probs') is not None:
                    aux_p.append(parts['probs']); aux_y.append(self.ctx['store'].get(int(data.id), self.ctx['classes'])); aux_len.append(len(parts['probs']))
        self.model.train()
        ys, yh = np.array(ys), np.array(yh)
        res = {'loss': loss_sum / max(n, 1), 'n': n, 'r2': float(r2_score(ys, yh)) if n > 1 else float('nan'),
               'pearson': float(pearsonr(ys, yh)[0]) if n > 2 else float('nan')}
        if collect:
            res.update(idx=np.array(idxs), y_true=ys, y_pred=yh)
            if aux_p:
                res.update(aux_p=np.concatenate(aux_p), aux_y=np.concatenate(aux_y), aux_len=np.array(aux_len))
        return res

    # ---- checkpointing -----------------------------------------------------------------------------------
    def save_ckpt(self, st):
        blob = {'model': self.model.state_dict(), 'optimizer': self.opt.state_dict(),
                'scheduler': self.sched.state_dict() if self.sched is not None else None,
                'state': st, 'rng': rng_state(), 'args': vars(self.args), 'ctx_stats': {'mu': self.ctx.get('mu'), 'sd': self.ctx.get('sd')}}
        atomic_save(blob, os.path.join(self.dir, 'ckpt_last.pt'))
        atomic_json({'status': 'running', 'epoch': st['epoch'], 'step': st['step'], 'phase': st['phase'],
                     'best_val': st['best_val'], 'time': time.time()}, os.path.join(self.dir, 'status.json'))

    def load_ckpt(self):
        path = os.path.join(self.dir, 'ckpt_last.pt')
        if not os.path.exists(path):
            return None
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(blob['model']); self.opt.load_state_dict(blob['optimizer'])
        if self.sched is not None and blob['scheduler'] is not None:
            self.sched.load_state_dict(blob['scheduler'])
        set_rng_state(blob['rng'])
        return blob['state']

    # ---- main ----------------------------------------------------------------------------------------------
    def run(self):
        a, log = self.args, self.log
        n_train = len(self.ctx['train'])
        st = self.load_ckpt()
        if st is None:
            st = dict(epoch=1, step=0, phase='train', best_val=float('inf'), patience=0, hist=[], global_step=0, train_loss_sum=0.0)
            log(f'fresh run: {a.family}/{a.arm} seed {a.seed}')
        else:
            log(f"RESUMED at epoch {st['epoch']} step {st['step']} phase {st['phase']} best_val {st['best_val']:.5f}")
        self.model.train()
        last_ckpt = time.time()
        complete_flag = os.path.join(self.dir, 'train_complete.json')
        while not os.path.exists(complete_flag):
            if st['epoch'] > self.max_epochs:
                break
            if st['phase'] == 'train':
                g = torch.Generator().manual_seed(a.seed * 100003 + st['epoch'])
                perm = torch.randperm(n_train, generator=g).tolist()
                t_ep = time.time()
                for s in range(st['step'], n_train):
                    if STOP['flag']:
                        st['step'] = s
                        self.save_ckpt(st)
                        log(f'signal received: checkpointed at epoch {st["epoch"]} step {s}; exiting for resume')
                        return EXIT_INTERRUPTED
                    l = self.train_step(self.ctx['train'][perm[s]], st['global_step'])
                    st['global_step'] += 1; st['train_loss_sum'] += l; st['step'] = s + 1
                    if (s + 1) % 2000 == 0:
                        log(f"epoch {st['epoch']} step {s + 1}/{n_train} running train loss {st['train_loss_sum'] / (s + 1 - 0):.4f} ({(s + 1) / (time.time() - t_ep + 1e-9):.1f} it/s this session)")
                    if time.time() - last_ckpt > a.ckpt_minutes * 60:
                        self.save_ckpt(st); last_ckpt = time.time()
                st['phase'] = 'val'
                st['epoch_train_loss'] = st['train_loss_sum'] / max(n_train, 1)
                self.save_ckpt(st)
            val = self.evaluate(self.ctx['val'])
            if val is None:
                self.save_ckpt(st)
                log('signal received during validation: checkpointed (val pending); exiting for resume')
                return EXIT_INTERRUPTED
            improved = val['loss'] < st['best_val']
            if improved:
                st['best_val'], st['patience'] = val['loss'], 0
                atomic_save({'model': self.model.state_dict(), 'epoch': st['epoch'], 'val': val, 'args': vars(self.args),
                             'ctx_stats': {'mu': self.ctx.get('mu'), 'sd': self.ctx.get('sd')}}, os.path.join(self.dir, 'best.pt'))
            else:
                st['patience'] += 1
            if self.sched is not None:
                self.sched.step(val['loss'])
            st['hist'].append({'epoch': st['epoch'], 'train_loss': st['epoch_train_loss'], 'val_loss': val['loss'],
                               'val_r2': val['r2'], 'val_pearson': val['pearson'], 'improved': improved})
            log(f"epoch {st['epoch']:03d} train {st['epoch_train_loss']:.4f} | val loss {val['loss']:.4f} R2 {val['r2']:.3f} r {val['pearson']:.3f} "
                f"| best {st['best_val']:.4f} patience {st['patience']}/{self.patience_max} | {'BEST' if improved else ''}")
            st['epoch'] += 1; st['step'] = 0; st['phase'] = 'train'; st['train_loss_sum'] = 0.0
            self.save_ckpt(st); last_ckpt = time.time()
            if st['patience'] >= self.patience_max:
                log('early stopping'); break
        atomic_json({'epochs_run': st['epoch'] - 1, 'best_val': st['best_val'], 'hist': st['hist']}, complete_flag)
        return self.final_test()

    def final_test(self):
        log = self.log
        best = torch.load(os.path.join(self.dir, 'best.pt'), map_location=self.device, weights_only=False)
        self.model.load_state_dict(best['model'])
        log(f"final test with best.pt (epoch {best['epoch']}) on {len(self.ctx['test'])} test complexes")
        res = self.evaluate(self.ctx['test'], collect=True)
        if res is None:
            log('signal during final test: will redo on resume'); return EXIT_INTERRUPTED
        extra = {}
        if 'aux_p' in res:
            aucs = {}
            for ci, c in enumerate(self.ctx['classes']):
                y = res['aux_y'][:, ci]
                if 0 < y.sum() < len(y):
                    aucs[c] = float(roc_auc_score(y, res['aux_p'][:, ci]))
            extra['aux_auroc'] = aucs
            extra['aux_auroc_macro'] = float(np.mean(list(aucs.values())))
        np.savez(os.path.join(self.dir, 'test_preds.npz'), **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
        summary = {'test_r2': res['r2'], 'test_pearson': res['pearson'], 'best_epoch': best['epoch'], **extra}
        atomic_json(summary, os.path.join(self.dir, 'DONE.json'))
        atomic_json({'status': 'done', **summary}, os.path.join(self.dir, 'status.json'))
        log(f'DONE {summary}')
        return 0


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    p.add_argument('--family', choices=['gign', 'egnn'], default='gign')
    p.add_argument('--target', choices=['abs', 'delta'], default='delta')
    p.add_argument('--head', choices=['phys', 'mlp'], default='phys')
    p.add_argument('--aux', choices=['none', 'plip', 'scrambled'], default='none')
    p.add_argument('--classes', default=','.join(PRIMARY_CLASSES))
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--pcgrad', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--max_epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--train_subsample', type=int, default=None)
    p.add_argument('--limit_val', type=int, default=None)
    p.add_argument('--limit_test', type=int, default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--run_root', default='./runs_track_e')
    p.add_argument('--ckpt_minutes', type=float, default=5.0)
    p.add_argument('--benchmark_steps', type=int, default=0)
    return p.parse_args()


def main():
    args = parse()
    run_dir = os.path.join(args.run_root, f'{args.arm}_s{args.seed}' + ('_bench' if args.benchmark_steps else ''))
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'DONE.json')):
        print('already done:', run_dir); return 0
    log = Log(os.path.join(run_dir, 'log.txt'))
    cfg_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        for k in ('family', 'target', 'head', 'aux', 'classes', 'lam', 'pcgrad', 'seed', 'lr', 'hidden', 'train_subsample'):
            if old.get(k) != vars(args).get(k):
                raise SystemExit(f'config mismatch on resume for {k}: {old.get(k)} vs {vars(args).get(k)}')
    else:
        json.dump(vars(args), open(cfg_path, 'w'), indent=1)
    runner = Runner(args, run_dir, log)
    if args.benchmark_steps:
        runner.model.train()
        n_tr = len(runner.ctx['train'])
        for i in range(20):                                   # warm-up
            runner.train_step(runner.ctx['train'][i % n_tr], i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        for i in range(args.benchmark_steps):
            runner.train_step(runner.ctx['train'][(20 + i) % n_tr], 20 + i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        t = time.time()
        runner.evaluate(type('Sub', (), {'__len__': lambda s_: 200, '__getitem__': lambda s_, i: runner.ctx['val'][i]})(), check_stop=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        de = time.time() - t
        print(json.dumps({'arm': args.arm, 'train_steps': args.benchmark_steps, 'train_it_per_s': round(args.benchmark_steps / dt, 1),
                          'eval_it_per_s': round(200 / de, 1), 'peak_gpu_MiB': round(peak)}))
        return 0
    code = runner.run()
    return code if isinstance(code, int) else 0


if __name__ == '__main__':
    sys.exit(main())
