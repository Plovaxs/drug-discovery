"""Tests for guidance/gradient_normalization.py -- Track B1's gradient-norm transform.

Why this file exists. This module exists to answer a specific confound: every guidance model in the
project has a different raw gradient scale (Track A's physics model fragments molecules by lambda=3.0,
Stage 0's EGNN is stable to lambda 10-30), so "no guidance effect" is ambiguous between an uninformative
gradient DIRECTION and a badly-scaled magnitude that never got a fair chance. The module's whole value
is that it removes the second explanation, and that rests on two properties its docstring asserts:

  1. It rescales MAGNITUDE ONLY, never direction. If direction moved, the transform would not be a
     reparameterisation of lambda -- it would be a different guidance signal, and every lambda sweep run
     through it would be measuring something else.
  2. Normalisation is PER SAMPLE, not per batch. The docstring argues this at length: a batch-wide norm
     would make guidance strength depend on ligand size, reintroducing the size confound the project
     controls for elsewhere. That argument is only sound if the code does what it says.

Neither had a test. Both are checked here as properties over random inputs rather than on one example,
because a single example can satisfy both by accident.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from guidance.gradient_normalization import normalize_unit_per_atom

TOL = 1e-4          # the module's own eps is 1e-6; its effect on the mean is of that order
EPS = 1e-6


def _batch(sizes, scale_pos=1.0, scale_v=1.0, seed=0, n_classes=5):
    """One graph per entry in `sizes`, with that many atoms."""
    g = torch.Generator().manual_seed(seed)
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    total = int(sum(sizes))
    gp = torch.randn(total, 3, generator=g) * scale_pos
    gv = torch.randn(total, n_classes, generator=g) * scale_v
    return gp, gv, batch


def _mean_per_atom_norm(grad, batch, i):
    return float(grad[batch == i].norm(dim=-1).mean())


# ----------------------------------------------------------- property 1: magnitude only, not direction

def test_direction_is_preserved_exactly_for_every_atom():
    """The transform divides by a positive per-sample scalar, so each atom's gradient must stay parallel
    to what it was. Checked via cosine similarity, which is 1 iff the direction is unchanged."""
    gp, gv, batch = _batch([7, 13, 4], seed=1)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    cos_p = torch.nn.functional.cosine_similarity(gp, np_, dim=-1)
    cos_v = torch.nn.functional.cosine_similarity(gv, nv, dim=-1)
    assert torch.allclose(cos_p, torch.ones_like(cos_p), atol=1e-5), float(cos_p.min())
    assert torch.allclose(cos_v, torch.ones_like(cos_v), atol=1e-5), float(cos_v.min())


def test_the_transform_is_invariant_to_the_input_gradient_scale():
    """THE point of the module: a model whose raw gradients are 1000x larger must come out identical, so
    lambda means the same thing for every guidance model. Without this, a lambda sweep is not portable."""
    # Confined to the valid regime on purpose: at k=1e-3 the raw mean per-atom norm is ~1.6e-3 and
    # the eps floor already costs 0.06%, larger than this test's tolerance. The boundary itself is
    # the subject of the eps-floor test below.
    for k in (1e-1, 1.0, 1e3):
        gp, gv, batch = _batch([9, 6], seed=2)
        a_p, a_v = normalize_unit_per_atom(gp, gv, batch)
        b_p, b_v = normalize_unit_per_atom(gp * k, gv * k, batch)
        assert torch.allclose(a_p, b_p, atol=TOL), f'pos changed at scale {k}: {float((a_p-b_p).abs().max()):.2e}'
        assert torch.allclose(a_v, b_v, atol=TOL), f'v changed at scale {k}'


# ----------------------------------------------------------- property 2: per sample, not per batch

def test_every_sample_reaches_mean_per_atom_norm_one():
    gp, gv, batch = _batch([5, 17, 2, 9], seed=3)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    for i in range(4):
        assert abs(_mean_per_atom_norm(np_, batch, i) - 1.0) < TOL, (i, _mean_per_atom_norm(np_, batch, i))
        assert abs(_mean_per_atom_norm(nv, batch, i) - 1.0) < TOL, (i, _mean_per_atom_norm(nv, batch, i))


def test_a_sample_with_a_wildly_different_magnitude_is_still_brought_to_one():
    """This is the test that distinguishes per-sample from per-batch normalisation. Under a single batch
    norm the loud sample would land far above 1 and the quiet one far below; both must land at 1."""
    g = torch.Generator().manual_seed(4)
    # Both scales sit INSIDE the transform's valid regime (raw mean |g| per atom >> 1e-4). The
    # first version of this test used 1e-4 for the quiet sample and failed at 0.99503 -- which was
    # not a fault in per-sample normalisation but the eps floor, now measured and pinned in its own
    # test below. Keeping the two apart is the point: this test is about independence between
    # samples, that one is about accuracy of the target norm.
    quiet = torch.randn(8, 3, generator=g) * 1e-2
    loud = torch.randn(8, 3, generator=g) * 1e4
    gp = torch.cat([quiet, loud])
    gv = torch.cat([quiet, loud])
    batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])
    np_, _ = normalize_unit_per_atom(gp, gv, batch)
    for i in (0, 1):
        assert abs(_mean_per_atom_norm(np_, batch, i) - 1.0) < TOL, (i, _mean_per_atom_norm(np_, batch, i))


def test_guidance_strength_does_not_depend_on_ligand_size():
    """The size confound, stated directly: a 4-atom and a 120-atom ligand in the same batch must both end
    at mean per-atom norm 1. A shared norm would spread the larger molecule's gradient thinner per atom."""
    gp, gv, batch = _batch([4, 120], seed=5)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    small, big = _mean_per_atom_norm(np_, batch, 0), _mean_per_atom_norm(np_, batch, 1)
    assert abs(small - 1.0) < TOL and abs(big - 1.0) < TOL, (small, big)
    assert abs(small - big) < TOL, f'per-atom strength differs by size: {small:.6f} vs {big:.6f}'


def test_one_samples_gradient_cannot_affect_another_sample():
    """Independence, checked by perturbation: rescaling sample 1 must leave sample 0's output bit-stable.
    This is the strongest form of 'per sample' and it is what a scatter bug would break."""
    gp, gv, batch = _batch([6, 6], seed=6)
    before, _ = normalize_unit_per_atom(gp, gv, batch)
    gp2 = gp.clone()
    gp2[batch == 1] *= 500.0
    after, _ = normalize_unit_per_atom(gp2, gv, batch)
    assert torch.allclose(before[batch == 0], after[batch == 0], atol=1e-6), 'sample 0 moved'


# ----------------------------------------------------------- independence of the two components

def test_position_and_type_components_are_normalised_independently():
    """The docstring says position and categorical components are normalised SEPARATELY because they have
    different scales and meanings. So scaling one must not move the other at all."""
    gp, gv, batch = _batch([10, 5], seed=7)
    a_p, a_v = normalize_unit_per_atom(gp, gv, batch)
    b_p, b_v = normalize_unit_per_atom(gp, gv * 1e5, batch)
    assert torch.allclose(a_p, b_p, atol=1e-6), 'scaling grad_v moved grad_pos'
    c_p, c_v = normalize_unit_per_atom(gp * 1e5, gv, batch)
    assert torch.allclose(a_v, c_v, atol=1e-6), 'scaling grad_pos moved grad_v'


# ----------------------------------------------------------- degenerate inputs

def test_an_all_zero_gradient_yields_zeros_and_never_nan():
    """A dead guidance model must produce a zero push, not NaN. eps in the denominator is what makes this
    work, and a NaN here would silently poison a whole sampling run."""
    batch = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    gp = torch.zeros(10, 3)
    gv = torch.zeros(10, 4)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    assert torch.isfinite(np_).all() and torch.isfinite(nv).all(), 'produced non-finite values'
    assert torch.equal(np_, torch.zeros_like(np_)) and torch.equal(nv, torch.zeros_like(nv))


def test_a_single_atom_sample_is_normalised_to_norm_one():
    """n=1 is where a mean-based normaliser is most likely to be written wrongly."""
    gp = torch.tensor([[3.0, 4.0, 0.0]])
    gv = torch.tensor([[1.0, 1.0]])
    batch = torch.zeros(1, dtype=torch.long)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    assert abs(float(np_.norm(dim=-1)[0]) - 1.0) < TOL, float(np_.norm(dim=-1)[0])
    assert abs(float(nv.norm(dim=-1)[0]) - 1.0) < TOL


def test_an_empty_batch_returns_empty_tensors_without_raising():
    """n_samples is computed as max()+1 and guarded for an empty batch; this pins that the guard works."""
    gp, gv = torch.zeros(0, 3), torch.zeros(0, 4)
    batch = torch.zeros(0, dtype=torch.long)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    assert np_.shape == (0, 3) and nv.shape == (0, 4), (np_.shape, nv.shape)


def test_a_gap_in_the_sample_indices_is_harmless():
    """batch_ligand is contiguous in practice, but n_samples = max()+1 means a gap creates a sample with
    no atoms and a mean norm of 0. That row is never indexed, since no atom refers to it, so the present
    samples must be unaffected. Pinned so the reasoning is on record rather than assumed."""
    gp = torch.randn(6, 3, generator=torch.Generator().manual_seed(8))
    gv = torch.randn(6, 4, generator=torch.Generator().manual_seed(9))
    batch = torch.tensor([0, 0, 0, 2, 2, 2])          # sample 1 has no atoms
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    assert torch.isfinite(np_).all() and torch.isfinite(nv).all()
    for i in (0, 2):
        assert abs(_mean_per_atom_norm(np_, batch, i) - 1.0) < TOL, (i, _mean_per_atom_norm(np_, batch, i))


def test_the_eps_floor_bounds_the_output_for_a_vanishing_gradient():
    """With a tiny but nonzero gradient, eps dominates the denominator and the output must stay small
    rather than exploding to 1/eps. Bounds the worst case instead of asserting it never happens."""
    batch = torch.zeros(4, dtype=torch.long)
    gp = torch.full((4, 3), 1e-12)
    gv = torch.full((4, 2), 1e-12)
    np_, nv = normalize_unit_per_atom(gp, gv, batch)
    assert torch.isfinite(np_).all() and torch.isfinite(nv).all()
    assert float(np_.abs().max()) <= 1e-12 / EPS + TOL, float(np_.abs().max())


def test_the_eps_floor_attenuates_small_gradients_by_a_known_amount():
    """The transform's validity boundary, measured rather than assumed.

    The achieved per-atom norm is mean/(mean+eps), not 1, so the shortfall is eps/(mean+eps). That is
    invisible for ordinary gradients and severe for tiny ones, and it matters because this module exists
    to remove exactly one ambiguity -- "is a null guidance result an uninformative direction, or a badly
    scaled magnitude?" -- while below about 1e-5 it reintroduces that same ambiguity itself.

    Pinned as a table so the boundary is a tested fact and not a remark in a docstring. The closed form
    is the real claim; the tabulated numbers are there to keep it concrete."""
    expected = {1e-2: 0.0001, 1e-3: 0.0006, 1e-4: 0.0061, 1e-5: 0.0577, 1e-6: 0.3798}
    for k, want in expected.items():
        g = torch.Generator().manual_seed(4)
        gp = torch.randn(64, 3, generator=g) * k
        batch = torch.zeros(64, dtype=torch.long)
        raw = float(gp.norm(dim=-1).mean())
        got = float(normalize_unit_per_atom(gp, gp, batch)[0].norm(dim=-1).mean())
        shortfall = 1.0 - got
        predicted = EPS / (raw + EPS)
        assert abs(shortfall - predicted) < 1e-3, (k, shortfall, predicted)
        assert abs(shortfall - want) < 5e-4, f'at scale {k:.0e}: {shortfall:.4f} vs tabulated {want}'


def test_our_measured_gradient_magnitudes_are_inside_the_valid_regime():
    """Ties the boundary to this project's actual models, so that adding a guidance model with weaker
    gradients trips a test rather than quietly losing the scale-invariance this module promises.

    Magnitudes come from guidance/diag_gradient_magnitude_results.json. All three sit between 3.4e-4 and
    1.0e-3, giving shortfalls of 0.10% to 0.30% -- negligible against a between-seed sd of 0.0377 R2,
    which is why no reported result is affected by the boundary above."""
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(os.path.dirname(here)),
                        'guidance/diag_gradient_magnitude_results.json')
    if not os.path.exists(path):
        return                      # the diagnostic has not been run on this machine
    with open(path) as f:
        res = json.load(f)
    assert res, 'results file is empty'
    for name, rec in res.items():
        mag = rec.get('grad_mag_mean')
        assert mag is not None, f'{name} has no grad_mag_mean'
        shortfall = EPS / (mag + EPS)
        assert shortfall < 0.01, (
            f'{name}: mean per-atom gradient {mag:.3e} gives a {100 * shortfall:.2f}% shortfall under '
            f'normalize_unit_per_atom, above the 1% boundary. Beyond it the transform no longer delivers '
            f'the scale-invariance that makes lambda comparable across guidance models.')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
