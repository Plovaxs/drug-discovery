"""New follow-up branch: attaches a global protein-language-model feature
(ESM2) to the Stage 0 EGNN, as a genuinely different signal from anything
tried so far -- every prior intervention (noise-matching, vina-target,
gradient-alignment) changed HOW the model is trained on the same inputs
(per-atom element/position features only); this changes WHAT the model is
given to work with, adding sequence-derived evolutionary/functional context
that pure 3D atom coordinates cannot encode (e.g. how conserved a pocket
residue is across homologs).

Model: esm2_t12_35M_UR50D (33.5M params, 480-dim per-residue output,
"t12" = 12 transformer layers) -- the smallest ESM2 checkpoint, chosen
because it already runs in well under a second per sequence on CPU (see
module smoke test), so there is no need to touch the GPU (currently busy
with guidance/lp_split/train_egnn_stage0_gradalign.py) just to compute these
once-per-target, cacheable embeddings. A larger ESM2 would plausibly encode
richer structure but was not tried -- not a claim that 35M is sufficient,
simply the cheapest starting point.

Embedding = mean-pooled per-residue representation from the final
transformer layer, over the real sequence positions only (BOS/EOS and
padding tokens excluded). Mean-pooling discards per-residue/positional
detail (a residue's specific pocket relevance is not distinguished from the
rest of the protein) -- a coarse global descriptor, not a pocket-local one;
sharper (e.g. pocket-residue-only) pooling was not attempted here.

Sequences are truncated to the first 1022 residues before embedding. This
is NOT because ESM2 has a hard positional limit (it uses rotary position
embeddings, unlike ESM-1b) -- it is a disclosed compute-budget cutoff to
keep this CPU-only job fast (attention cost grows with length^2). This
affects 121 of 983 cached sequences (12.3%, see uniprot_cache.json), and for
those the embedding reflects only the first 1022 residues of the full
protein, which may exclude a pocket that sits later in the sequence --
a real, disclosed limitation, not silently absorbed into the result.
"""
import argparse
import json
import os
import time

import torch
import esm

CACHE_PATH = './guidance/lp_split/uniprot_cache.json'
OUT_PATH = './guidance/lp_split/esm2_embeddings.pt'
MAX_LEN = 1022


def compute_embeddings(accessions_to_seq, log=print):
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    embeddings = {}
    t0 = time.time()
    for i, (accession, seq) in enumerate(accessions_to_seq.items(), 1):
        seq_trunc = seq[:MAX_LEN]
        _, _, toks = batch_converter([('x', seq_trunc)])
        with torch.no_grad():
            out = model(toks, repr_layers=[n_layers])
        rep = out['representations'][n_layers][0, 1:len(seq_trunc) + 1].mean(0)
        embeddings[accession] = rep
        if i % 100 == 0:
            log(f'[{i}/{len(accessions_to_seq)}] {time.time()-t0:.0f}s elapsed')
    log(f'Done: {len(embeddings)} embeddings in {time.time()-t0:.0f}s')
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    with open(CACHE_PATH) as f:
        uniprot_cache = json.load(f)

    accessions_to_seq = {}
    for rec in uniprot_cache.values():
        if rec.get('found') and rec.get('sequence'):
            accessions_to_seq[rec['accession']] = rec['sequence']
    print(f'{len(accessions_to_seq)} unique UniProt accessions with a cached sequence')
    n_long = sum(1 for s in accessions_to_seq.values() if len(s) > MAX_LEN)
    print(f'{n_long} exceed {MAX_LEN} residues and will be truncated')

    if args.limit:
        accessions_to_seq = dict(list(accessions_to_seq.items())[:args.limit])

    embeddings = compute_embeddings(accessions_to_seq)
    torch.save(embeddings, OUT_PATH)
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
