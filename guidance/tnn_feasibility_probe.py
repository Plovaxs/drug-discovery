"""Cheap feasibility probe for ternary weight quantization (TWN, Li et al. 2016, arXiv:1605.04711) on
OUR OWN already-trained Stage 0 EGNN affinity model -- post-hoc, no retraining (this is literally what the
original TNN pitch called "TNT: Target None-retraining Ternary"). Answers one question with real numbers
on our own architecture/data, instead of reasoning from other papers' results: does naive post-hoc ternary
quantization survive on this specific model, or does it collapse the way Rasool et al. 2025 (J.
Cheminformatics, quantized-GNN molecular property prediction) found 2-bit quantization already does?

TWN formula (per-layer, applied to every nn.Linear.weight in the model -- biases and BatchNorm/LayerNorm
params, if any, are left at full precision, matching TWN's own scope):
  threshold   Delta = 0.7 * mean(|W|)
  scale       alpha = mean(|W_i|  for i with |W_i| > Delta)
  W_ternary_i = alpha * sign(W_i)  if |W_i| > Delta  else 0

This is deliberately the simplest, least-tuned version (no STE fine-tuning, no per-channel scales, no
calibration data) -- a lower bound on what a properly-trained ternary model could do, not an upper bound.
A result here that's clearly fine does NOT mean QAT is unnecessary; a result that's clearly broken DOES
mean naive post-hoc ternarization is not viable for this architecture at this size, which is itself a
useful, cheap answer before committing to a full QAT implementation.

Usage:
  python guidance/tnn_feasibility_probe.py
"""
import copy
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch_geometric.transforms import Compose

import utils.transforms_prop as utils_trans
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from models.property_pred.prop_model import PropPredNet

DEVICE = 'cpu'  # deliberate: GPU is busy with A1c training; this is a quick inference-only probe
CKPT = './logs_lp_split_stage0/crossdocked_affinity_egnn_2026_09_08__16_12_40/checkpoints/best.pt'
OUT = './guidance/tnn_feasibility_probe_results.json'
VAL_SUBSAMPLE = 1500  # full val (6069) takes ~1h/model on CPU; this keeps the probe to ~15 min/model
# while still giving a stable enough R2/Pearson estimate to tell "survives" from "collapses"


def twn_ternarize(weight, delta_factor=0.7):
    w = weight.detach()
    delta = delta_factor * w.abs().mean()
    mask = w.abs() > delta
    if mask.sum() == 0:
        return torch.zeros_like(w), 0.0, 1.0
    alpha = w[mask].abs().mean()
    w_t = torch.zeros_like(w)
    w_t[mask] = alpha * w[mask].sign()
    sparsity = 1.0 - mask.float().mean().item()
    return w_t, float(alpha), sparsity


def ternarize_model(model):
    sparsities = {}
    n_linear = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w_t, alpha, sparsity = twn_ternarize(module.weight.data)
            module.weight.data.copy_(w_t)
            sparsities[name] = dict(alpha=alpha, sparsity=sparsity, shape=list(w_t.shape))
            n_linear += 1
    return sparsities, n_linear


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    pf, lf = utils_trans.FeaturizeProteinAtom(), utils_trans.FeaturizeLigandAtom()
    model = PropPredNet(ckpt['config'].model, protein_atom_feature_dim=pf.feature_dim,
                        ligand_atom_feature_dim=lf.feature_dim, output_dim=1).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, pf, lf


def evaluate(model, dataset):
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            d = dataset[i]
            batch_protein = torch.zeros(d.protein_pos.size(0), dtype=torch.long)
            batch_ligand = torch.zeros(d.ligand_pos.size(0), dtype=torch.long)
            pred = model(protein_pos=d.protein_pos, protein_atom_feature=d.protein_atom_feature.float(),
                        ligand_pos=d.ligand_pos, ligand_atom_feature=d.ligand_atom_feature_full.float(),
                        batch_protein=batch_protein, batch_ligand=batch_ligand, output_kind=None)
            y_pred.append(pred.view(-1).item())
            y_true.append(float(d.y))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return dict(n=len(y_true), r2=float(r2_score(y_true, y_pred)), pearson=float(pearsonr(y_true, y_pred)[0]),
                rmse=float(np.sqrt(((y_true - y_pred) ** 2).mean())))


def main():
    model, pf, lf = load_model()
    transform = Compose([pf, lf])
    base, splits, pk_by_idx = build_lp_splits(train_subsample=None)
    g = torch.Generator().manual_seed(2021)
    val_idx = [splits['val'][i] for i in torch.randperm(len(splits['val']), generator=g)[:VAL_SUBSAMPLE].tolist()]
    val_set = CrossDockedAffinityDataset(base, val_idx, pk_by_idx, transform)

    print('Evaluating FP32 baseline on val set...', flush=True)
    fp32 = evaluate(model, val_set)
    print(f'FP32: {fp32}', flush=True)

    ternary_model = copy.deepcopy(model)
    sparsities, n_linear = ternarize_model(ternary_model)
    overall_sparsity = np.mean([s['sparsity'] for s in sparsities.values()])
    print(f'Ternarized {n_linear} nn.Linear layers, mean sparsity {overall_sparsity:.3f}', flush=True)

    print('Evaluating post-hoc TWN-ternarized model on val set...', flush=True)
    ternary = evaluate(ternary_model, val_set)
    print(f'Ternary: {ternary}', flush=True)

    out = dict(checkpoint=CKPT, n_linear_layers=n_linear, mean_sparsity=float(overall_sparsity),
               per_layer=sparsities, fp32=fp32, ternary_posthoc=ternary,
               r2_retained_pct=float(100 * ternary['r2'] / fp32['r2']) if fp32['r2'] != 0 else None)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_layer'}, indent=2), flush=True)
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
