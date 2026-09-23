"""LP-PDBBind protein similarity: Needleman-Wunsch global alignment
percent identity, computed only within the same protein functional
category (cross-category pairs are similarity 0 by definition -- a
kinase and a protease should never be treated as similar regardless of
raw alignment score noise).

Uses Bio.Align.PairwiseAligner (the modern, non-deprecated biopython
API) in global-alignment mode with BLOSUM62 substitution scores, matching
standard Needleman-Wunsch practice.
"""
from Bio.Align import PairwiseAligner, substitution_matrices

_aligner = PairwiseAligner()
_aligner.mode = 'global'
_aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
_aligner.open_gap_score = -10
_aligner.extend_gap_score = -0.5


def sequence_identity(seq_a, seq_b):
    """Percent identity over the global alignment length (matches /
    alignment length), in [0, 1]."""
    if not seq_a or not seq_b:
        return 0.0
    alignments = _aligner.align(seq_a, seq_b)
    best = alignments[0]
    aligned_a, aligned_b = best[0], best[1]
    matches = sum(1 for a, b in zip(aligned_a, aligned_b) if a == b and a != '-')
    aln_len = len(aligned_a)
    return matches / aln_len if aln_len > 0 else 0.0


def protein_similarity(seq_a, category_a, seq_b, category_b):
    """Category-gated similarity: 0 if categories differ (or either is
    None, i.e. unclassifiable -- treated as its own singleton category so
    it never matches anything else), else Needleman-Wunsch identity."""
    if category_a is None or category_b is None or category_a != category_b:
        return 0.0
    return sequence_identity(seq_a, seq_b)


def build_category_grouped_pairs(targets, category_by_target):
    """Returns {category: [target, ...]} for targets with a real
    (non-None) category -- pairwise comparison only needs to happen
    within these groups, which is what makes this tractable at ~1000
    targets (Needleman-Wunsch is O(n*m) per pair)."""
    groups = {}
    for t in targets:
        cat = category_by_target.get(t)
        if cat is not None:
            groups.setdefault(cat, []).append(t)
    return groups
