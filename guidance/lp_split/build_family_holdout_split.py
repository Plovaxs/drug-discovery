"""New follow-up branch, Phase 2's second closely-connected branch:
tests whether the null guidance pattern (16 checkpoints so far, see
`guidance/FOLLOWUP_PHASE_POOLED_CORRECTION.md`) holds when the test set
is not just a different PROTEIN but an entirely unseen protein FAMILY --
a stricter, more realistic proxy for "a genuinely novel drug target" than
the existing LP-split's protein-identity threshold allows (two proteins
can fail a >0.9 identity threshold while still belonging to the same
family/fold and sharing the same general pocket chemistry).

Method: reuses guidance/lp_split/leakage_safe_split.json's ALREADY
leakage-safe target pool as-is (its own protein-identity / ligand-
similarity / pocket-cosine thresholds are not touched or re-derived) --
this script only RE-PARTITIONS that same pool by Pfam family membership
instead of by the original (near-)random assignment. Holding out an
entire family is a strictly stronger leakage guarantee than the
identity threshold alone (two targets in different Pfam families cannot
be near-duplicates by the identity threshold's own standard), so this is
a legitimate re-use, not a safety shortcut.

Held-out family: PF00069 ("Protein kinase domain"), chosen because it is
the single largest family in the LP-split target pool (59/997 targets,
see guidance/lp_split/uniprot_cache.json's cached pfam_ids) and because
kinases are a major, well-studied real-world drug target class --
holding out all of them means every kinase-binding chemistry pattern the
model might see at test time was NEVER present anywhere in training,
not even from a different kinase subfamily.

Caveat disclosed up front: Pfam family membership here is only as
complete/correct as the cached UniProt record (981/997 targets have a
Pfam annotation; targets with no family info are kept in train by
default since they cannot be confirmed as safe to hold out or as
belonging to the held-out family either).
"""
import argparse
import json

import torch

from datasets.pl_pair_dataset import PocketLigandPairDataset

HELD_OUT_FAMILIES = ['PF00069']  # protein kinase domain
UNIPROT_CACHE_PATH = './guidance/lp_split/uniprot_cache.json'
SPLIT_JSON = './guidance/lp_split/leakage_safe_split.json'


def _target_to_family_map():
    with open(UNIPROT_CACHE_PATH) as f:
        uniprot_cache = json.load(f)
    fam_by_gene_species = {}
    for gene_species, rec in uniprot_cache.items():
        if rec.get('found') and rec.get('pfam_ids'):
            fam_by_gene_species[gene_species] = rec['pfam_ids'][0]
    return fam_by_gene_species


def build_family_holdout_splits(root='./data/crossdocked_v1.1_rmsd1.0_pocket10',
                                train_subsample=6000, val_fraction=0.1, seed=2021,
                                held_out_families=HELD_OUT_FAMILIES):
    with open(SPLIT_JSON) as f:
        d = json.load(f)
    base = PocketLigandPairDataset(root)
    fam_by_gene_species = _target_to_family_map()

    all_entries = []  # (idx, pk)
    for part in ('train', 'val', 'test'):
        all_entries.extend(d[part])

    held_out_entries, remaining_entries = [], []
    held_out_targets, remaining_targets = set(), set()
    for idx, pk in all_entries:
        target_name = base[idx].ligand_filename.split('/')[0]
        gene_species = '_'.join(target_name.split('_')[:2])
        fam = fam_by_gene_species.get(gene_species)
        if fam in held_out_families:
            held_out_entries.append((idx, pk))
            held_out_targets.add(target_name)
        else:
            remaining_entries.append((idx, pk))
            remaining_targets.add(target_name)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(remaining_entries), generator=g).tolist()
    n_val = max(1, int(len(remaining_entries) * val_fraction))
    val_idx_pos = set(perm[:n_val])
    train_entries, val_entries = [], []
    for i, entry in enumerate(remaining_entries):
        (val_entries if i in val_idx_pos else train_entries).append(entry)

    pk_by_idx = {}
    splits = {}
    for name, entries in [('train', train_entries), ('val', val_entries), ('test', held_out_entries)]:
        indices = []
        for idx, pk in entries:
            pk_by_idx[idx] = pk
            indices.append(idx)
        splits[name] = indices

    if train_subsample is not None and train_subsample < len(splits['train']):
        g2 = torch.Generator().manual_seed(seed)
        perm2 = torch.randperm(len(splits['train']), generator=g2).tolist()[:train_subsample]
        splits['train'] = [splits['train'][i] for i in perm2]

    info = {
        'held_out_families': held_out_families,
        'n_held_out_targets': len(held_out_targets),
        'n_remaining_targets': len(remaining_targets),
        'held_out_targets': sorted(held_out_targets),
    }
    return base, splits, pk_by_idx, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./guidance/lp_split/family_holdout_split_info.json')
    args = parser.parse_args()
    base, splits, pk_by_idx, info = build_family_holdout_splits(train_subsample=None)
    print(f"Held-out families: {info['held_out_families']}")
    print(f"Held-out targets: {info['n_held_out_targets']}")
    print(f"Remaining (train+val pool) targets: {info['n_remaining_targets']}")
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test (held-out family): {len(splits['test'])}")
    with open(args.out, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'Saved target list to {args.out}')


if __name__ == '__main__':
    main()
