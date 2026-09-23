"""Stage 0: assembles the LP-PDBBind-adapted leakage-safe split for
CrossDocked2020's labeled affinity subset (pk != 0.0 and rmsd < 2.0,
1,041 targets / 76,803 entries -- see FLAGSHIP_ARCHITECTURE_RESEARCH.md).

Operates at TARGET granularity (not individual pose/entry), since
CrossDocked cross-docks a small set of ligands into many receptor
structures per target (median 11, mean 74 entries/target) -- the real
combinatorial-leakage question is target-vs-target, not
pose-vs-pose. All entries belonging to a target move together into
whichever split that target is assigned to, guaranteeing target-disjoint
train/val/test (unlike the current build_labeled_splits, whose val
shares ~74% of its targets with train -- see AFFINITY_MODEL_STANDALONE_QUALITY.md).

Algorithm (LP-PDBBind, adapted):
1. Categorize every target via UniProt Pfam/family (guidance/lp_split/uniprot_client.py).
   Uncategorized targets get a singleton category (their own entry name)
   so they never spuriously cluster with anything else.
2. Seed a small random test set per category (5% of category members,
   min 1 if category has >= 2 members).
3. Iterate remaining candidates: move to test if protein identity > 0.9
   OR ligand similarity > 0.99 OR pocket-histogram cosine similarity > 0.95
   to any current test-set member (protein/pocket checks gated to the
   same category only).
4. Seed a small random val set (5%) from what remains, then iterate
   remaining candidates: move to val if protein identity > 0.5 OR ligand
   similarity > 0.99 OR pocket similarity > 0.8 to any current val-set
   member.
5. Everything else is train. Final pass: discard any train target whose
   ligand similarity to ANY val/test target exceeds 0.99 (per the
   addendum's explicit final step) -- these targets are dropped entirely,
   not moved.

Pocket-histogram cosine similarity calibration: an empirical check on
3,000 random target pairs (guidance/lp_split/pocket_sim_distribution.log)
found this proxy is far too coarse for LP-PDBBind-style two-tier
thresholding -- mean 0.69, p90 0.82, p99.9 0.94 among genuinely unrelated
targets (20-dim AA-type composition histograms are simply not very
discriminative). Using 0.8/0.95 as originally sketched would have swept
nearly all candidates into val once val had a handful of members (P(no
random pair exceeds 0.8) drops below 50% past ~7 val members). Recalibrated
to a SINGLE conservative near-duplicate threshold (0.97, comfortably above
the observed p99.9=0.94 and the observed max=0.95 in the 3,000-pair
sample) used identically at both the test and val expansion stages --
functioning as a rare supplementary safety net for pathological
same-pocket-family cases, not a primary clustering signal. Protein
sequence identity and ligand similarity carry the real clustering work,
matching genuine LP-PDBBind practice.
"""

POCKET_SIM_THRESHOLD = 0.97
import argparse
import json
import pickle
import random

from datasets.pl_pair_dataset import PocketLigandPairDataset
from guidance.lp_split.uniprot_client import fetch_records, get_category
from guidance.lp_split.protein_similarity import protein_similarity, sequence_identity
from guidance.lp_split.pocket_similarity import build_target_histograms, pocket_similarity
from guidance.lp_split.ligand_similarity import build_similarity_cache, ligand_similarity_precomputed


def load_labeled_target_index(affinity_path='./data/affinity_info.pkl'):
    with open(affinity_path, 'rb') as f:
        d = pickle.load(f)
    labeled = {k: v for k, v in d.items() if v['pk'] != 0.0 and v['rmsd'] < 2.0}
    target_to_keys = {}
    for k in labeled:
        target_to_keys.setdefault(k.split('/')[0], []).append(k)
    return labeled, target_to_keys


def build_target_metadata(base, target_to_keys, labeled):
    """For each target: representative ligand SMILES set (deduplicated),
    a few representative dataset indices (for pocket histograms), and the
    mnemonic used for UniProt lookup. Also maps target -> (idx, pk) pairs
    for the final label output, matching build_labeled_splits' interface.

    Scans the FULL base dataset (166,500 entries), not just the official
    train+test split (100,090 entries) -- the official
    crossdocked_pocket10_pose_split.pt leaves ~66,410 entries (40%)
    unassigned to either partition, and Stage 0 is building a replacement
    split anyway, so there is no reason to inherit that restriction (doing
    so silently dropped 229/1041 -- 22% -- of labeled targets that only
    exist in the unassigned portion, caught via smoke-test before this
    fix)."""
    key_to_idx = {}
    for idx in range(len(base)):
        lf = base[idx].ligand_filename
        key = lf[:-4] if lf.endswith('.sdf') else lf
        key_to_idx[key] = idx

    target_meta = {}
    for target, keys in target_to_keys.items():
        idx_pk = []
        smiles_set = set()
        for k in keys:
            idx = key_to_idx.get(k)
            if idx is None:
                continue
            idx_pk.append((idx, labeled[k]['pk']))
            smi = base[idx].ligand_smiles
            if smi:
                smiles_set.add(smi)
        if idx_pk:
            target_meta[target] = {
                'idx_pk': idx_pk,
                'smiles': sorted(smiles_set),
                'mnemonic': '_'.join(target.split('_')[:2]),
            }
    return target_meta


def max_ligand_sim(smiles_a, smiles_b, fps, canon):
    best = 0.0
    for a in smiles_a:
        for b in smiles_b:
            s = ligand_similarity_precomputed(a, b, fps, canon)
            if s > best:
                best = s
            if best >= 0.99:
                return best
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--out', type=str, default='./guidance/lp_split/leakage_safe_split.json')
    args = parser.parse_args()
    rng = random.Random(args.seed)

    print('Loading labeled target index...')
    labeled, target_to_keys = load_labeled_target_index()
    print(f'{len(target_to_keys)} targets, {len(labeled)} labeled entries')

    print('Loading base dataset...')
    base = PocketLigandPairDataset('./data/crossdocked_v1.1_rmsd1.0_pocket10')

    print('Building target metadata (ligand SMILES sets, mnemonics)...')
    target_meta = build_target_metadata(base, target_to_keys, labeled)
    targets = sorted(target_meta.keys())
    print(f'{len(targets)} targets with resolvable indices')

    print('Fetching UniProt records (cached)...')
    mnemonics = sorted(set(m['mnemonic'] for m in target_meta.values()))
    records = fetch_records(mnemonics)
    category_by_target = {}
    sequence_by_target = {}
    for t in targets:
        rec = records.get(target_meta[t]['mnemonic'], {})
        cat = get_category(rec)
        category_by_target[t] = cat if cat is not None else f'singleton:{t}'
        sequence_by_target[t] = rec.get('sequence') or ''

    print('Building pocket AA-composition histograms...')
    target_to_indices = {t: [idx for idx, _ in target_meta[t]['idx_pk']] for t in targets}
    histograms = build_target_histograms(base, target_to_indices)

    print('Precomputing ligand fingerprints...')
    all_smiles = [s for t in targets for s in target_meta[t]['smiles']]
    fps, canon = build_similarity_cache(all_smiles)

    # Group targets by category for efficient protein/pocket comparison.
    by_category = {}
    for t in targets:
        by_category.setdefault(category_by_target[t], []).append(t)

    def protein_sim(t_a, t_b):
        if category_by_target[t_a] != category_by_target[t_b]:
            return 0.0
        return sequence_identity(sequence_by_target[t_a], sequence_by_target[t_b])

    def pocket_sim(t_a, t_b):
        return pocket_similarity(histograms[t_a], histograms[t_b])

    def ligand_sim(t_a, t_b):
        return max_ligand_sim(target_meta[t_a]['smiles'], target_meta[t_b]['smiles'], fps, canon)

    remaining = set(targets)
    test_set, val_set = set(), set()

    print('Seeding test set (5% of category-eligible targets, sampled globally)...')
    # NOTE: an earlier version forced "at least 1 seed per category," which
    # with ~600 categories over 1002 targets (170 with >=2 members alone)
    # inflated the seed to ~17% of all targets before any expansion even
    # started, and downstream family-level expansion (a real, intended
    # LP-PDBBind mechanism -- e.g. AKT1/AKT2/AKT3 sharing >85% sequence
    # identity correctly cluster together) then roughly doubled that,
    # producing an unusable 38%-of-targets test set. Sampling 5% from the
    # global pool of category-eligible targets (weighted implicitly toward
    # larger categories, since they contribute more members to the pool)
    # matches LP-PDBBind's actual proportions far better -- documented here
    # since it's a real calibration finding, not a silent tweak.
    seedable_pool = [t for cat, members in by_category.items() if len(members) >= 2 for t in members]
    n_test_seed = max(1, round(len(seedable_pool) * 0.05))
    test_seed = rng.sample(seedable_pool, min(n_test_seed, len(seedable_pool)))
    for t in test_seed:
        test_set.add(t)
        remaining.discard(t)

    print(f'Test seed: {len(test_set)} targets. Expanding via contamination check...')
    for i, t in enumerate(sorted(remaining)):
        for s in test_set:
            if protein_sim(t, s) > 0.9 or ligand_sim(t, s) > 0.99 or pocket_sim(t, s) > POCKET_SIM_THRESHOLD:
                test_set.add(t)
                break
        if (i + 1) % 100 == 0:
            print(f'  test expansion: checked {i + 1}/{len(remaining)}, test size now {len(test_set)}')
    remaining -= test_set
    print(f'Test set final: {len(test_set)} targets')

    print('Seeding val set (5% of category-eligible remaining targets, sampled globally)...')
    remaining_by_cat = {}
    for t in remaining:
        remaining_by_cat.setdefault(category_by_target[t], []).append(t)
    val_seedable_pool = [t for cat, members in remaining_by_cat.items() if len(members) >= 2 for t in members]
    n_val_seed = max(1, round(len(val_seedable_pool) * 0.05))
    val_seed = rng.sample(val_seedable_pool, min(n_val_seed, len(val_seedable_pool)))
    for t in val_seed:
        val_set.add(t)
        remaining.discard(t)

    # NOTE (calibration finding): an earlier version only checked candidates
    # against the growing val set here, not against the already-finalized
    # test set. A ligand-identity leak between val and test (CSK21_HUMAN /
    # PIM1_HUMAN, both kinases but below the 0.5 protein-identity bar,
    # sharing an identical ligand) slipped through as a result --
    # ligand-identity is exactly as much a leak between val and test as
    # within either set, so the check must include test_set here too.
    print(f'Val seed: {len(val_set)} targets. Expanding via contamination check (incl. vs. finalized test set)...')
    for i, t in enumerate(sorted(remaining)):
        for s in val_set:
            if protein_sim(t, s) > 0.5 or ligand_sim(t, s) > 0.99 or pocket_sim(t, s) > POCKET_SIM_THRESHOLD:
                val_set.add(t)
                break
        else:
            for s in test_set:
                if ligand_sim(t, s) > 0.99:
                    val_set.add(t)
                    break
        if (i + 1) % 100 == 0:
            print(f'  val expansion: checked {i + 1}/{len(remaining)}, val size now {len(val_set)}')
    remaining -= val_set
    print(f'Val set final: {len(val_set)} targets')

    print('Safety-net check: any remaining val-vs-test ligand contamination?')
    val_test_contaminated = sorted(set(v for v in val_set for s in test_set if ligand_sim(v, s) > 0.99))
    if val_test_contaminated:
        print(f'  discarding {len(val_test_contaminated)} val targets still contaminated vs test')
        val_set -= set(val_test_contaminated)
        # discarded, not returned to `remaining` -- these targets are dropped
        # entirely, matching the train-side final-pass convention below.
    else:
        print('  none found')

    print('Final pass: discarding train targets with ligand similarity > 0.99 to val/test...')
    train_set = set()
    discarded = []
    for t in remaining:
        contaminated = False
        for s in test_set | val_set:
            if ligand_sim(t, s) > 0.99:
                contaminated = True
                break
        if contaminated:
            discarded.append(t)
        else:
            train_set.add(t)
    all_discarded = sorted(set(discarded) | set(val_test_contaminated))
    print(f'Train set final: {len(train_set)} targets ({len(discarded)} discarded for train-side ligand '
         f'contamination; {len(val_test_contaminated)} discarded for val-vs-test contamination; '
         f'{len(all_discarded)} total discarded)')

    def flatten(target_set):
        out = []
        for t in target_set:
            out.extend(target_meta[t]['idx_pk'])
        return out

    result = {
        'train': flatten(train_set), 'val': flatten(val_set), 'test': flatten(test_set),
        'train_targets': sorted(train_set), 'val_targets': sorted(val_set), 'test_targets': sorted(test_set),
        'discarded_targets': all_discarded,
        'seed': args.seed,
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved to {args.out}')
    print(f'Train: {len(result["train"])} entries / {len(train_set)} targets')
    print(f'Val:   {len(result["val"])} entries / {len(val_set)} targets')
    print(f'Test:  {len(result["test"])} entries / {len(test_set)} targets')


if __name__ == '__main__':
    main()
