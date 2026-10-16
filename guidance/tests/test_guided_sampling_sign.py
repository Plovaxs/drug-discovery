"""Fast, CPU-only regression guard on the guidance injection sign convention in
guidance/guided_sampling.py.

This is the single most consequential piece of arithmetic in the thesis. Every reading of the
gradient-direction result (DIAG2: "mean cosine toward the nearest contact is -0.019, CI includes
zero"; Appendix E.5's trace of how far guidance actually moves an atom; the F1/F2/F3 failure-
mechanism framework of Chapter III) is stated relative to one fixed convention: guidance adds
`lambda_affinity * grad_pos` to the clean-data estimate, so a positive cosine between the gradient
and the direction to the nearest pocket contact means the atom is pushed *toward* that contact. If
that `+` were ever flipped to a `-` by an edit, by a merge, or by refactoring, every one of those
readings would silently invert without any number in this thesis changing in a way a reader could
notice: the diagnostic measurements (DIAG2, the per-atom trace) read the ACTUAL gradient and ACTUAL
injected delta off a live run, so they would simply report the flipped convention as if it were
correct, and the only thing that would still say "pushed toward" is prose nobody would think to
re-derive by hand a second time.

The only existing verification of this (guidance/verify_baseline_equivalence.py) checks the
lambda=0 no-op case against the real diffusion model on GPU; it is correctly excluded from the
lightweight suite because it needs a loaded checkpoint. It does not check the lambda != 0 case at
all, so the sign convention itself -- the part DIAG2 depends on -- was previously unverified by
anything that runs automatically. This test covers exactly that gap with a fully deterministic,
single-step, CPU-only mock: no real diffusion model, no GPU, no checkpoint. It constructs the
minimal fake `model` the function actually touches when `pos_only=True`, `model_mean_type='C0'`,
`num_steps=1` and `t=0` (which zeroes the posterior noise term, making the whole step exact), and
drives the real, unmodified `sample_diffusion_ligand_guided` through it.

Run:  PYTHONPATH=. python guidance/tests/test_guided_sampling_sign.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from guidance.guided_sampling import sample_diffusion_ligand_guided


class _FakeModel:
    """The minimal stand-in for ScorePosNet3D that one deterministic, pos_only, C0, t=0,
    num_steps=1 guided step actually touches. `base_pos0` is the clean-data estimate the
    (fake) diffusion core "predicts" at its only step, fixed and known so the test can compute
    the expected guided output by hand rather than trusting the function under test to grade
    itself."""

    model_mean_type = 'C0'
    num_classes = 1  # unused when pos_only=True, but accessed defensively by some call paths

    def __init__(self, base_pos0):
        self.base_pos0 = base_pos0
        self.num_timesteps = 1
        # t == 0 everywhere, so nonzero_mask is exactly 0 and no posterior noise is added:
        # ligand_pos_next == pos_model_mean == q_pos_posterior's return value, exactly.
        self.posterior_logvar = torch.zeros(1, 1)

    def __call__(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v,
                batch_ligand, time_step):
        return {'pred_ligand_pos': self.base_pos0, 'pred_ligand_v': init_ligand_v}

    def q_pos_posterior(self, x0, xt, t, batch):
        # Identity: whatever x0 (pos0_from_e, after guidance has been added) comes in is
        # exactly what comes out, so the test can assert on it directly.
        return x0


class _FakeGuidance:
    """A guidance model whose gradient is fixed and known, so the expected injected delta
    (lambda * grad) can be computed independently of the function under test."""

    def __init__(self, grad_pos):
        self.grad_pos = grad_pos

    def grad_log_score(self, pos0, v0, batch_ligand, protein_pos, protein_v, batch_protein):
        return self.grad_pos.clone(), torch.zeros_like(v0)


def _run(lambda_affinity, base_pos0, grad_pos):
    n = base_pos0.shape[0]
    # A single protein atom placed at the ligand's own centroid: center_pos_mode='protein' (the
    # mode every real caller in this project actually uses) subtracts the protein centroid before
    # the step and adds it back after, so placing it at the base centroid makes that round trip an
    # identity and keeps the expected-output arithmetic in the test readable.
    protein_pos = base_pos0.mean(dim=0, keepdim=True)
    protein_v = torch.zeros(1, 1)
    batch_protein = torch.zeros(1, dtype=torch.long)
    batch_ligand = torch.zeros(n, dtype=torch.long)
    init_ligand_v = torch.zeros(n, 1)

    model = _FakeModel(base_pos0 - protein_pos)  # the model "sees" the centered frame
    affinity = _FakeGuidance(grad_pos)
    out = sample_diffusion_ligand_guided(
        model, protein_pos.clone(), protein_v, batch_protein,
        init_ligand_pos=base_pos0.clone(), init_ligand_v=init_ligand_v, batch_ligand=batch_ligand,
        num_steps=1, center_pos_mode='protein', pos_only=True,
        affinity_model=affinity, lambda_affinity=lambda_affinity,
    )
    return out['pos']


def test_lambda_zero_is_an_exact_noop_on_the_clean_data_estimate():
    """With lambda_affinity == 0, guidance must not touch pos0_from_e at all."""
    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    grad = torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])  # large, to make a leak obvious
    got = _run(0.0, base, grad)
    assert torch.allclose(got, base, atol=1e-6), (
        f'lambda_affinity=0 changed the output: got {got}, expected the untouched base {base}')


def test_guidance_is_added_not_subtracted():
    """The documented, pre-registered convention (guided_sampling.py's own docstring, and the
    interpretation DIAG2 was pre-registered against) is `pos0_from_e + lambda * grad_pos`. This
    is the one assertion that would catch an accidental sign flip of that `+`."""
    base = torch.tensor([[0.0, 0.0, 0.0]])
    grad = torch.tensor([[1.0, 2.0, 3.0]])
    got = _run(1.0, base, grad)
    expected_plus = base + grad
    expected_minus = base - grad
    assert torch.allclose(got, expected_plus, atol=1e-6), (
        f'guidance was not added as documented: got {got}, expected base + grad = {expected_plus} '
        f'(NOT base - grad = {expected_minus})')


def test_lambda_scales_linearly_including_sign():
    """lambda_affinity=-1 must flip the injected delta relative to lambda_affinity=+1, and
    lambda_affinity=2 must double it -- catches an accidental abs() or a hardcoded sign that
    would make the result insensitive to the caller's chosen lambda."""
    base = torch.tensor([[0.0, 0.0, 0.0]])
    grad = torch.tensor([[1.0, 0.0, 0.0]])
    pos = _run(1.0, base, grad)
    neg = _run(-1.0, base, grad)
    double = _run(2.0, base, grad)
    assert torch.allclose(pos - base, -(neg - base), atol=1e-6), (
        f'lambda=+1 delta {pos - base} is not the negation of lambda=-1 delta {neg - base}')
    assert torch.allclose(double - base, 2 * (pos - base), atol=1e-6), (
        f'lambda=2 delta {double - base} is not double the lambda=1 delta {pos - base}')


def test_zero_gradient_is_a_noop_even_with_nonzero_lambda():
    """A sanity check in the other direction: a real (nonzero) lambda with a zero gradient must
    not perturb the output, ruling out a bug where guidance_active alone injects a spurious
    constant."""
    base = torch.tensor([[1.0, -2.0, 3.5]])
    grad = torch.zeros(1, 3)
    got = _run(5.0, base, grad)
    assert torch.allclose(got, base, atol=1e-6), (
        f'zero gradient with lambda=5 still changed the output: got {got}, expected {base}')


def test_center_pos_none_mode_returns_an_indexable_offset_not_a_bare_float():
    """Regression guard for a found-and-fixed bug, not a hypothetical one.

    `models/molopt_score_model.py`'s `center_pos(mode='none')` used to return a bare Python
    `0.` as the offset. Every caller that consumes the offset, including
    `guided_sampling.py`'s own `ori_ligand_pos = ligand_pos + offset[batch_ligand]`, indexes it
    unconditionally, so this crashed with `TypeError: 'float' object is not subscriptable` the
    moment anything actually selected mode='none'. No config or call site in this project ever
    did (verified by a project-wide grep, 2026-10-16), so the crash never affected a committed
    result, but the function was not correct, only unreachable, and a future caller trying
    mode='none' as its name implies it should work would hit this immediately.
    """
    from models.molopt_score_model import center_pos
    protein_pos = torch.randn(7, 3)
    ligand_pos = torch.randn(4, 3)
    batch_protein = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch_ligand = torch.tensor([0, 0, 1, 1])
    out_protein, out_ligand, offset = center_pos(
        protein_pos, ligand_pos, batch_protein, batch_ligand, mode='none')
    # 'none' must not move anything...
    assert torch.equal(out_protein, protein_pos)
    assert torch.equal(out_ligand, ligand_pos)
    # ...and the offset must be indexable by both batch vectors without crashing, and be zero.
    assert torch.all(offset[batch_protein] == 0)
    assert torch.all(offset[batch_ligand] == 0)


def test_center_pos_protein_mode_is_unchanged_by_the_none_mode_fix():
    """The 'none' and 'protein' branches are disjoint in the source (an if/elif), so fixing the
    former cannot touch the latter -- this pins the 'protein' path's own well-known formula
    (subtract the protein centroid) so that claim is checked, not just argued."""
    from models.molopt_score_model import center_pos
    protein_pos = torch.randn(7, 3)
    ligand_pos = torch.randn(4, 3)
    batch_protein = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch_ligand = torch.tensor([0, 0, 1, 1])
    out_protein, out_ligand, offset = center_pos(
        protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein')
    expected_offset_0 = protein_pos[:3].mean(dim=0)
    expected_offset_1 = protein_pos[3:].mean(dim=0)
    assert torch.allclose(offset[0], expected_offset_0, atol=1e-6)
    assert torch.allclose(offset[1], expected_offset_1, atol=1e-6)
    assert torch.allclose(out_protein, protein_pos - offset[batch_protein], atol=1e-6)
    assert torch.allclose(out_ligand, ligand_pos - offset[batch_ligand], atol=1e-6)


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
