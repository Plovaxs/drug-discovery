"""Track B2 (classifier-guidance reformulation): trains a small binary
classification head (a single Linear layer) on top of Stage 0's already-
trained, FROZEN EGNN backbone's pooled embedding
(guidance/frozen_backbone_features.py), rather than training a whole new
backbone from scratch -- a linear probe on an already-learned
representation, chosen specifically to keep this Track B variant cheap
(per the addendum's own cost ranking: B2 is "moderate cost", not
comparable to Track A's from-scratch architecture).

Motivation for trying a classifier reformulation at all (not just reusing
the existing regression model): classic classifier guidance
(Dhariwal & Nichol 2021) uses grad_x log p(y|x) from a model trained via
cross-entropy on a class label, not grad_x of a raw regression output --
the training objective and the resulting gradient's sharpness near a
decision boundary are genuinely different things, even holding the input
representation fixed. This is testable cheaply by keeping Stage 0's
backbone frozen and only swapping the objective/head.

Label: binarized at the TRAINING SET's own median pk (computed once,
documented, not re-derived per-run) -- pk > median = "good binder"
(label 1), matching a roughly-balanced 50/50 split rather than an
arbitrary domain threshold.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms_prop as utils_trans
from models.property_pred.prop_model import PropPredNet
from datasets.crossdocked_affinity import CrossDockedAffinityDataset
from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.frozen_backbone_features import extract_pooled_embedding

BACKBONE_CHECKPOINT = './guidance_models/affinity_egnn_lpsplit.pt'


def load_frozen_backbone(device):
    ckpt = torch.load(BACKBONE_CHECKPOINT, map_location=device, weights_only=False)
    model = PropPredNet(
        ckpt['config'].model,
        protein_atom_feature_dim=ckpt['protein_atom_feature_dim'],
        ligand_atom_feature_dim=ckpt['ligand_atom_feature_dim'],
        output_dim=1,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt['config'].model.hidden_channels


def get_embedding_and_label(backbone, batch, threshold, device):
    with torch.no_grad():
        emb = extract_pooled_embedding(
            backbone, batch.protein_pos, batch.protein_atom_feature.float(),
            batch.ligand_pos, batch.ligand_atom_feature_full.float(),
            batch.protein_element_batch, batch.ligand_element_batch)
    label = (batch.y > threshold).float()
    return emb, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_subsample', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out', type=str, default='./guidance_models/classifier_head.pt')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    misc.seed_all(2021)

    backbone, hidden_dim = load_frozen_backbone(device)

    protein_featurizer = utils_trans.FeaturizeProteinAtom()
    ligand_featurizer = utils_trans.FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])

    base, splits, pk_by_idx = build_lp_splits(train_subsample=args.train_subsample)
    train_pks = np.array([pk_by_idx[i] for i in splits['train']])
    threshold = float(np.median(train_pks))
    print(f'Binarization threshold (train median pk): {threshold:.4f}')

    train_set = CrossDockedAffinityDataset(base, splits['train'], pk_by_idx, transform)
    val_set = CrossDockedAffinityDataset(base, splits['val'], pk_by_idx, transform)
    test_set = CrossDockedAffinityDataset(base, splits['test'], pk_by_idx, transform)

    follow_batch = ['protein_element', 'ligand_element']
    exclude_keys = ['ligand_nbh_list']
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              follow_batch=follow_batch, exclude_keys=exclude_keys)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                            follow_batch=follow_batch, exclude_keys=exclude_keys)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                             follow_batch=follow_batch, exclude_keys=exclude_keys)

    head = nn.Linear(hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    def evaluate(loader, prefix):
        head.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                emb, label = get_embedding_and_label(backbone, batch, threshold, device)
                logit = head(emb).view(-1)
                logits_all.append(logit.cpu())
                labels_all.append(label.cpu())
        logits_all = torch.cat(logits_all).numpy()
        labels_all = torch.cat(labels_all).numpy()
        preds = (logits_all > 0).astype(float)
        acc = (preds == labels_all).mean()
        try:
            auroc = roc_auc_score(labels_all, logits_all)
        except ValueError:
            auroc = float('nan')
        print(f'[{prefix}] n={len(labels_all)} acc={acc:.4f} auroc={auroc:.4f} pos_rate={labels_all.mean():.3f}')
        return acc, auroc

    best_val_auroc = -1.0
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', dynamic_ncols=True):
            batch = batch.to(device)
            emb, label = get_embedding_and_label(backbone, batch, threshold, device)
            logit = head(emb).view(-1)
            loss = loss_fn(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc, val_auroc = evaluate(val_loader, prefix=f'Validate epoch {epoch}')
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            test_acc, test_auroc = evaluate(test_loader, prefix=f'Test (best so far, epoch {epoch})')
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                'head_state_dict': head.state_dict(),
                'hidden_dim': hidden_dim,
                'threshold': threshold,
                'backbone_checkpoint': BACKBONE_CHECKPOINT,
                'epoch': epoch, 'val_auroc': best_val_auroc, 'test_auroc': test_auroc,
            }, args.out)
            print(f'Saved best head to {args.out}')

    print(f'Done. Best val AUROC: {best_val_auroc:.4f}')


if __name__ == '__main__':
    main()
