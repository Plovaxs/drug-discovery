"""Phase 3's second item: a sharper version of guidance/lp_split/
esm2_embed.py's global per-target ESM2 feature. That embedding mean-
pools ESM2's per-residue representation over the ENTIRE protein chain --
a coarse descriptor that treats a pocket residue identically to a
surface residue on the opposite side of the protein. This instead
mean-pools only over the residues within POCKET_RADIUS of the docked
ligand (same convention as guidance/alphafold_pocket_robustness.py's own
"pocket10"-matching definition), reusing that module's pocket-extraction
and UniProt-alignment functions directly rather than re-deriving them.

Now feasible for (almost) every LP-split target, not just the ~46 that
had a local raw PDB before guidance/alphafold_pocket_robustness.py's
scope was extended to the full downloaded CrossDocked2020 v1.1 archive
(./data/CrossDocked2020_v1.1).

Targets where the pocket cannot be confidently resolved (no UniProt
match, no local structure, low sequence-alignment identity, pocket too
small) are skipped and counted, not silently defaulted to a zero or
whole-protein fallback -- this is a disclosed coverage limitation, not a
guarantee every target gets a pocket-specific embedding.
"""
import argparse
import json
import os

import torch
import esm

from guidance.lp_split.lp_split_loader import build_lp_splits
from guidance.lp_split.uniprot_client import fetch_records as fetch_uniprot_records
from guidance.alphafold_pocket_robustness import (
    DATA_ROOT, POCKET_RADIUS,
    _chain_sequence_and_residues, _resolve_ligand_source,
    _ligand_atom_coords, _ligand_atom_coords_from_gz,
    _pocket_residue_indices, _align_to_uniprot,
)

SPLIT_JSON = './guidance/lp_split/leakage_safe_split.json'
OUT_PATH = './guidance/lp_split/esm2_pocket_embeddings.pt'
MAX_LEN = 1022


def resolve_pocket_uniprot_indices(target_name, ligand_filename, uniprot_cache):
    """Returns (accession, sorted set of 0-based UniProt-sequence indices
    covering this target's pocket residues) or (None, status_string) on
    failure -- mirrors alphafold_pocket_robustness.process_target's logic
    but stops after resolving indices (no AlphaFold comparison needed
    here)."""
    gene_species = '_'.join(target_name.split('_')[:2])
    urec = uniprot_cache.get(gene_species)
    if not urec or not urec.get('found'):
        return None, 'no_uniprot_match'
    accession = urec['accession']

    target_dir = os.path.join(DATA_ROOT, target_name)
    if not os.path.isdir(target_dir):
        return None, 'no_local_data_dir'
    sdf_basename = os.path.basename(ligand_filename)
    if '_rec_' not in sdf_basename:
        return None, 'unparseable_ligand_filename'
    rec_basename = sdf_basename.split('_rec_')[0] + '_rec.pdb'
    rec_path = os.path.join(target_dir, rec_basename)
    if not os.path.isfile(rec_path):
        return None, 'missing_pdb_or_sdf'
    ligand_source = _resolve_ligand_source(target_dir, sdf_basename)
    if ligand_source is None:
        return None, 'missing_pdb_or_sdf'

    try:
        chain_seq, residues = _chain_sequence_and_residues(rec_path)
        if ligand_source[0] == 'direct':
            ligand_coords = _ligand_atom_coords(ligand_source[1])
        else:
            ligand_coords = _ligand_atom_coords_from_gz(ligand_source[1], ligand_source[2])
        if ligand_coords is None or len(chain_seq) < 10:
            return None, 'parse_failed'
        pocket_idx = _pocket_residue_indices(residues, ligand_coords, POCKET_RADIUS)
        if len(pocket_idx) < 3:
            return None, 'pocket_too_small'

        chain_to_uni, identity = _align_to_uniprot(chain_seq, urec['sequence'])
        if identity < 0.9:
            return None, 'low_alignment_identity'

        uni_indices = sorted({chain_to_uni[int(c_i)] for c_i in pocket_idx if int(c_i) in chain_to_uni})
        if len(uni_indices) < 3:
            return None, 'too_few_matched_pocket_residues'
        return accession, uni_indices
    except Exception as e:
        return None, f'error:{e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    all_targets = sorted(set(split['train_targets'] + split['val_targets'] + split['test_targets']))
    if args.limit:
        all_targets = all_targets[:args.limit]
    print(f'{len(all_targets)} LP-split targets to resolve')

    print('Building target -> ligand_filename map from dataset indices...')
    base, splits, _ = build_lp_splits(train_subsample=None)
    wanted = set(all_targets)
    ligand_filename_by_target = {}
    for part in ('train', 'val', 'test'):
        for idx in splits[part]:
            lf = base[idx].ligand_filename
            t = lf.split('/')[0]
            if t in wanted and t not in ligand_filename_by_target:
                ligand_filename_by_target[t] = lf

    gene_species_tokens = sorted({'_'.join(t.split('_')[:2]) for t in all_targets})
    print(f'Fetching/caching UniProt records for {len(gene_species_tokens)} gene/species tokens...')
    uniprot_cache = fetch_uniprot_records(gene_species_tokens, verbose=False)

    from collections import Counter
    status_counts = Counter()
    pocket_indices_by_target = {}
    accession_by_target = {}
    for i, target in enumerate(all_targets, 1):
        if target not in ligand_filename_by_target:
            status_counts['no_local_data_dir'] += 1
            continue
        accession, result = resolve_pocket_uniprot_indices(
            target, ligand_filename_by_target[target], uniprot_cache)
        if accession is None:
            status_counts[result] += 1
            continue
        pocket_indices_by_target[target] = result
        accession_by_target[target] = accession
        status_counts['ok'] += 1
        if i % 100 == 0:
            print(f'[{i}/{len(all_targets)}] resolved so far: {status_counts["ok"]}')
    print('Status breakdown:', dict(status_counts))

    print('Loading ESM2 (esm2_t12_35M_UR50D)...')
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    seq_by_accession = {}
    for t, acc in accession_by_target.items():
        gene_species = '_'.join(t.split('_')[:2])
        seq_by_accession[acc] = uniprot_cache[gene_species]['sequence']

    print(f'Computing per-residue ESM2 embeddings for {len(seq_by_accession)} unique accessions...')
    per_residue_cache = {}
    for i, (acc, seq) in enumerate(seq_by_accession.items(), 1):
        seq_trunc = seq[:MAX_LEN]
        _, _, toks = batch_converter([('x', seq_trunc)])
        with torch.no_grad():
            out = model(toks, repr_layers=[n_layers])
        rep = out['representations'][n_layers][0, 1:len(seq_trunc) + 1]  # (L, 480)
        per_residue_cache[acc] = rep
        if i % 100 == 0:
            print(f'  [{i}/{len(seq_by_accession)}]')

    print('Pooling pocket-only residues per target...')
    pocket_embeddings = {}
    n_truncated_dropped = 0
    for target, uni_indices in pocket_indices_by_target.items():
        acc = accession_by_target[target]
        rep = per_residue_cache[acc]
        valid_idx = [i for i in uni_indices if i < rep.size(0)]
        if len(valid_idx) < 3:
            n_truncated_dropped += 1
            continue
        pocket_embeddings[target] = rep[valid_idx].mean(dim=0)
    print(f'{len(pocket_embeddings)} targets with a pocket-only ESM2 embedding '
          f'({n_truncated_dropped} dropped: pocket residues fell past the {MAX_LEN}-residue truncation)')

    torch.save(pocket_embeddings, OUT_PATH)
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
