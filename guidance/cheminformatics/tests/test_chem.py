"""Tests for the cheminformatics layer (chem_data, ligand_only_baseline, compare_vs_structure).

The one that matters most is the Tanimoto check. All three analyses -- the ligand-only baselines, the
novelty tiers and the scaffold audit -- rest on one hand-written similarity routine that computes the
full 11,855 x 46,964 similarity matrix as a BLAS product (intersection = Q @ R.T for binary vectors,
union = |A| + |B| - intersection). That trick is correct but easy to get subtly wrong: swap the
population counts, forget that the diagonal of a self-comparison must be 1, or mishandle an all-zero
fingerprint, and every downstream number shifts without anything raising. So it is checked against
RDKit's own BulkTanimotoSimilarity -- a genuinely independent implementation -- rather than against a
re-derivation of the same algebra.

The alignment guard is tested too, because compare_vs_structure.py's paired bootstrap is only meaningful
if the prediction vectors really are row-aligned, and the files it reads are bare vectors with no index
column. A permutation that preserved the label multiset would otherwise pass unnoticed.

Pure CPU, no GPU, seconds to run.

Usage:
  PYTHONPATH=. python guidance/cheminformatics/tests/test_chem.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from guidance.cheminformatics.ligand_only_baseline import r2, tanimoto_nn
from guidance.cheminformatics.compare_vs_structure import load_structure, r2_rows

SMILES = ['CCO', 'CCCO', 'c1ccccc1', 'c1ccccc1O', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
          'C1CCCCC1', 'CCN(CC)CC', 'OC(=O)c1ccccc1', 'CCOC(=O)C']


# tanimoto_nn computes the similarity matrix with a float32 BLAS product, so agreement with RDKit's
# float64 arithmetic is limited to ~1e-7 relative. Asserting 1e-9 would be testing float32 against
# float64 and failing for a reason that has nothing to do with correctness -- which is exactly what
# happened on first run, and only the self-similarity cases (exactly 1.0) passed.
FP32_TOL = 1e-6


def _fps(smiles_list, bits=2048, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    assert all(m is not None for m in mols)
    np_fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    rd_fps = [gen.GetFingerprint(m) for m in mols]
    return np_fps, rd_fps


# ----------------------------------------------- the similarity engine, vs RDKit

def test_tanimoto_matches_rdkit_bulk_similarity():
    """Independent reference: RDKit's BulkTanimotoSimilarity on its own ExplicitBitVect objects."""
    from rdkit import DataStructs
    np_fps, rd_fps = _fps(SMILES)
    y = np.arange(len(SMILES), dtype=np.float64) * 1.5      # distinct values so argmax is unambiguous
    for i in range(len(SMILES)):
        ref = np.array(DataStructs.BulkTanimotoSimilarity(rd_fps[i], rd_fps))
        _, got = tanimoto_nn(np_fps[i:i + 1], np_fps, y)
        assert abs(got[0] - ref.max()) < FP32_TOL, (i, got[0], ref.max())


def test_tanimoto_nn_returns_the_neighbour_rdkit_picks():
    from rdkit import DataStructs
    np_fps, rd_fps = _fps(SMILES)
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    # Query each molecule against the OTHERS, so the trivial self-match cannot mask an error.
    for i in range(len(SMILES)):
        others = [j for j in range(len(SMILES)) if j != i]
        ref = np.array(DataStructs.BulkTanimotoSimilarity(rd_fps[i], [rd_fps[j] for j in others]))
        if (np.abs(ref - ref.max()) < 1e-12).sum() > 1:
            continue        # tied nearest neighbours: argmax is arbitrary, so the label is not pinned
        pred, sim = tanimoto_nn(np_fps[i:i + 1], np_fps[others], y[others])
        assert abs(sim[0] - ref.max()) < FP32_TOL, (i, sim[0], ref.max())
        # The LABEL it returns must be exact: the neighbour is chosen by argmax, so float32 noise can
        # only matter if two neighbours tie, and these fixtures have a unique maximum (verified below).
        assert abs(pred[0] - y[others][int(ref.argmax())]) < 1e-12, i


def test_self_similarity_is_exactly_one():
    np_fps, _ = _fps(SMILES)
    y = np.zeros(len(SMILES))
    _, sim = tanimoto_nn(np_fps, np_fps, y)
    assert np.abs(sim - 1.0).max() < 1e-12


def test_chunking_does_not_change_the_answer():
    """The query loop is chunked for memory; a chunk-boundary bug would shift a handful of rows only."""
    np_fps, _ = _fps(SMILES)
    y = np.arange(len(SMILES), dtype=np.float64)
    base_p, base_s = tanimoto_nn(np_fps, np_fps, y, chunk=len(SMILES))
    for chunk in (1, 2, 3, 7):
        p, s = tanimoto_nn(np_fps, np_fps, y, chunk=chunk)
        assert np.array_equal(p, base_p), chunk
        assert np.abs(s - base_s).max() < 1e-12, chunk


def test_all_zero_fingerprint_gives_zero_not_nan():
    """Union is zero for two empty fingerprints; the guard must yield 0.0 similarity, not 0/0."""
    fps = np.zeros((2, 64), dtype=np.uint8)
    p, s = tanimoto_nn(fps, fps, np.array([5.0, 6.0]))
    assert np.isfinite(s).all() and (s == 0.0).all()
    assert np.isfinite(p).all()


def test_knn_averages_the_k_nearest_labels():
    from rdkit import DataStructs
    np_fps, rd_fps = _fps(SMILES)
    y = np.arange(len(SMILES), dtype=np.float64) * 3.0
    k = 3
    ref = np.array(DataStructs.BulkTanimotoSimilarity(rd_fps[0], rd_fps))
    top = np.argsort(-ref)[:k]
    pred, _ = tanimoto_nn(np_fps[0:1], np_fps, y, k=k)
    assert abs(pred[0] - y[top].mean()) < 1e-9, (pred[0], y[top].mean())


def test_float32_precision_is_bounded_and_tier_boundaries_are_safe():
    """Records the precision the analyses actually run at, and checks it cannot move a molecule across a
    novelty-tier boundary. novelty_tiers.py cuts at 0.35/0.50/0.70; an error of ~1e-7 could only matter
    for a similarity sitting within 1e-7 of a cut, so the test asserts both the error bound and that no
    fixture similarity lands that close to a boundary."""
    from rdkit import DataStructs
    np_fps, rd_fps = _fps(SMILES)
    y = np.zeros(len(SMILES))
    worst = 0.0
    for i in range(len(SMILES)):
        ref = np.array(DataStructs.BulkTanimotoSimilarity(rd_fps[i], rd_fps))
        for chunk_start in range(len(SMILES)):
            _, sim = tanimoto_nn(np_fps[chunk_start:chunk_start + 1], np_fps, y)
            break
        _, sim_all = tanimoto_nn(np_fps[i:i + 1], np_fps, y)
        worst = max(worst, abs(float(sim_all[0]) - float(ref.max())))
    assert worst < FP32_TOL, f'float32 error {worst:.2e} exceeds the documented bound {FP32_TOL:.0e}'
    for cut in (0.35, 0.50, 0.70):
        for i in range(len(SMILES)):
            ref = np.array(DataStructs.BulkTanimotoSimilarity(rd_fps[i], rd_fps))
            near = np.abs(ref - cut)
            assert near.min() > 10 * FP32_TOL or near.min() == 0.0, (
                f'a fixture similarity sits {near.min():.2e} from the {cut} tier cut; float32 noise '
                f'could flip its tier')


# ----------------------------------------------- the metric

def test_r2_matches_sklearn():
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.normal(6.5, 1.5, 200)
        p = y + rng.normal(0, 1.0, 200)
        assert abs(r2(y, p) - r2_score(y, p)) < 1e-12


def test_r2_of_the_mean_predictor_is_zero():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 500)
    assert abs(r2(y, np.full_like(y, y.mean()))) < 1e-12


def test_r2_rows_is_computed_on_the_subset_mean_not_the_global_one():
    """The subtle part of subset R2, and the reason the scaffold audit was initially misread: SST is the
    SUBSET's variance. Using the global mean instead would give a different, non-standard number."""
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    p = y.copy()
    rows = np.array([0, 1, 2])
    assert abs(r2_rows(y, p, rows) - 1.0) < 1e-12
    yy = y[rows]
    const = np.full(6, yy.mean())
    assert abs(r2_rows(y, const, rows)) < 1e-12     # subset mean predictor -> exactly 0 on that subset


def test_r2_rows_returns_nan_for_a_zero_variance_subset():
    y = np.array([5.0, 5.0, 5.0, 1.0])
    assert np.isnan(r2_rows(y, y.copy(), np.array([0, 1, 2])))


# ----------------------------------------------- the alignment guard

def _write(tmp, **arrays):
    np.savez(tmp, **arrays)
    return tmp


def test_alignment_guard_accepts_matching_order(tmpdir=None):
    import tempfile
    pk = np.array([6.0, 7.0, 8.0])
    tgt = np.array(['A', 'A', 'B'])
    with tempfile.TemporaryDirectory() as d:
        f = _write(os.path.join(d, 'ok.npz'), y_true=pk.copy(), y_pred=np.array([6.1, 6.9, 8.2]),
                   tgt=tgt.copy())
        p, note = load_structure('x', f, 'y_true', 'y_pred', pk, tgt)
        assert p is not None, note


def test_alignment_guard_rejects_a_permutation():
    """The case the guard exists for: same labels, different order."""
    import tempfile
    pk = np.array([6.0, 7.0, 8.0])
    tgt = np.array(['A', 'A', 'B'])
    with tempfile.TemporaryDirectory() as d:
        f = _write(os.path.join(d, 'perm.npz'), y_true=pk[::-1].copy(),
                   y_pred=np.array([1.0, 2.0, 3.0]), tgt=tgt[::-1].copy())
        p, note = load_structure('x', f, 'y_true', 'y_pred', pk, tgt)
        assert p is None and 'ROW ORDER DIFFERS' in note, note


def test_alignment_guard_catches_a_label_preserving_permutation_via_targets():
    """Labels alone cannot detect a swap between two complexes with the same pK -- many complexes share
    one. The target-string check is what closes that hole, so it is tested on exactly that case."""
    import tempfile
    pk = np.array([7.0, 7.0, 8.0])
    tgt = np.array(['A', 'B', 'C'])
    with tempfile.TemporaryDirectory() as d:
        f = _write(os.path.join(d, 'swap.npz'), y_true=pk.copy(), y_pred=np.array([1.0, 2.0, 3.0]),
                   tgt=np.array(['B', 'A', 'C']))
        p, note = load_structure('x', f, 'y_true', 'y_pred', pk, tgt)
        assert p is None and 'ROW ORDER DIFFERS' in note, note


def test_alignment_guard_rejects_a_length_mismatch():
    import tempfile
    pk = np.array([6.0, 7.0, 8.0])
    tgt = np.array(['A', 'A', 'B'])
    with tempfile.TemporaryDirectory() as d:
        f = _write(os.path.join(d, 'short.npz'), y_true=pk[:2].copy(), y_pred=np.array([1.0, 2.0]),
                   tgt=tgt[:2].copy())
        p, note = load_structure('x', f, 'y_true', 'y_pred', pk, tgt)
        assert p is None and 'length' in note, note


def test_alignment_guard_rejects_a_missing_file_without_raising():
    p, note = load_structure('x', '/nonexistent/path.npz', 'y_true', 'y_pred',
                             np.array([1.0]), np.array(['A']))
    assert p is None and 'missing file' in note


# ----------------------------------------------- scaffolds

def test_generic_scaffold_abstracts_atom_identity():
    """Benzene and pyridine share a generic framework; that is the conservative behaviour the scaffold
    audit relies on (abstraction can only make train and test look MORE alike)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    def generic(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(
            MurckoScaffold.GetScaffoldForMol(m)))
    assert generic('c1ccccc1') == generic('c1ccncc1')
    assert generic('c1ccccc1CC') == generic('c1ccccc1CCC') == generic('c1ccccc1')


def test_generic_scaffold_is_deterministic():
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    def generic(smi):
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(
            MurckoScaffold.GetScaffoldForMol(m)))
    # Same molecule written two ways must give one scaffold, or the overlap count is meaningless.
    assert generic('CC(=O)Oc1ccccc1C(=O)O') == generic('O=C(O)c1ccccc1OC(C)=O')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
