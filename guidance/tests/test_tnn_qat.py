"""Tests for guidance/tnn_qat.py -- the ternary QAT mechanism behind a thesis claim.

Why this file exists. The thesis reports that ternary QAT is indistinguishable from FP32 on this
project's EGNN (0.3437 +/- 0.0377 vs 0.3420, n=3). That claim rests entirely on this module doing what
its docstring says, and the module had no tests. Three of its statements are load-bearing and each is
checked here against the thing itself rather than against a re-implementation:

  1. The TWN threshold rule (Li et al. 2016): delta = 0.7 * mean|w|, alpha = mean|w| over the survivors.
     Pinned against a worked example computed on paper, so a change to the rule fails here instead of
     silently shifting what "ternary" means.
  2. The straight-through estimator passes gradient UNCHANGED, including to weights the forward pass
     zeroed. This is the whole difference between QAT and the post-hoc probe that collapsed R2 from
     0.41 to -1.50: if gradient did not reach the pruned weights they could never come back.
  3. convert_to_ternary_qat SHARES nn.Parameter objects rather than copying them, which is what makes a
     checkpoint load correctly either before or after conversion. A copy would load without error and
     train the wrong tensor -- the kind of fault that produces plausible numbers.

One known limitation is pinned rather than fixed, because fixing it silently would be worse than
documenting it: a model that is ITSELF an nn.Linear is not converted, since the walk only inspects
named_children(). Every model in this project is a container, so this never bites, but a future caller
deserves to find it stated.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from guidance.tnn_qat import (TernaryLinear, convert_to_ternary_qat, sparsity_report, ste_ternarize,
                              twn_ternary)

FP32_TOL = 1e-6


def _net(bias=True):
    """A container with nesting, so the recursive walk is actually exercised."""
    return nn.Sequential(
        nn.Linear(4, 8, bias=bias),
        nn.ReLU(),
        nn.Sequential(nn.Linear(8, 8, bias=bias), nn.ReLU(), nn.Linear(8, 2, bias=bias)),
    )


# ----------------------------------------------------------------- the TWN rule itself

def test_twn_matches_a_hand_computed_example():
    """w = [0.1, -0.2, 0.9, -1.0, 0.0]
       mean|w| = 2.2/5 = 0.44      -> delta = 0.7 * 0.44 = 0.308
       survivors are |w| > 0.308   -> {0.9, 1.0}
       alpha   = (0.9 + 1.0)/2 = 0.95
       result  = [0, 0, +0.95, -0.95, 0]"""
    w = torch.tensor([0.1, -0.2, 0.9, -1.0, 0.0])
    got = twn_ternary(w)
    expect = torch.tensor([0.0, 0.0, 0.95, -0.95, 0.0])
    assert torch.allclose(got, expect, atol=FP32_TOL), f'{got.tolist()} vs {expect.tolist()}'


def test_twn_output_takes_at_most_three_values():
    """'Ternary' is the claim; this is the claim stated as a property."""
    g = torch.Generator().manual_seed(11)
    for _ in range(50):
        w = torch.randn(6, 7, generator=g)
        vals = torch.unique(twn_ternary(w))
        assert len(vals) <= 3, f'{len(vals)} distinct values: {vals.tolist()}'
        nz = vals[vals != 0].abs().unique()
        assert len(nz) <= 1, f'more than one magnitude: {nz.tolist()}'


def test_alpha_is_the_mean_absolute_value_of_the_survivors():
    """Not merely 'some positive scale'. TWN specifies this particular alpha, and it is what makes the
    ternary tensor a least-squares fit to the full-precision one."""
    g = torch.Generator().manual_seed(3)
    for _ in range(30):
        w = torch.randn(40, generator=g)
        t = twn_ternary(w)
        delta = 0.7 * w.abs().mean()
        survivors = w.abs() > delta
        expected_alpha = w[survivors].abs().mean()
        got_alpha = t[t != 0].abs().unique()
        assert len(got_alpha) == 1
        assert abs(float(got_alpha[0]) - float(expected_alpha)) < FP32_TOL


def test_twn_preserves_the_sign_of_every_surviving_weight():
    g = torch.Generator().manual_seed(5)
    w = torch.randn(200, generator=g)
    t = twn_ternary(w)
    live = t != 0
    assert torch.equal(torch.sign(t[live]), torch.sign(w[live])), 'a surviving weight changed sign'


def test_all_zero_weights_give_zeros_and_not_nan():
    """mask.sum() == 0 makes alpha undefined; the guard must return zeros rather than 0/0."""
    t = twn_ternary(torch.zeros(5, 5))
    assert torch.equal(t, torch.zeros(5, 5))
    assert torch.isfinite(t).all()


def test_sparsity_report_counts_the_zeros_of_the_ternary_weights():
    """Hand-checkable: the example above zeroes 3 of 5 entries."""
    lin = TernaryLinear(5, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[0.1, -0.2, 0.9, -1.0, 0.0]]))
    rep = sparsity_report(nn.Sequential(lin))
    assert set(rep) == {'0'}, rep
    assert abs(rep['0'] - 0.6) < FP32_TOL, rep


# ----------------------------------------------------------------- the straight-through estimator

def test_ste_forward_value_matches_the_ternary_tensor_to_within_one_ulp():
    """`w + (ternary(w) - w).detach()` equals ternary(w) in real arithmetic but NOT bit-for-bit in
    float32: the addition rounds. This test originally asserted exact equality and FAILED, which is
    how the discrepancy was found. Measured: 0.1% of weights differ, by at most 1.192e-07 -- exactly
    one unit in the last place of float32. The module docstring was corrected accordingly.

    One ULP is asserted rather than a loose tolerance, because the bound is what makes the deviation
    harmless: it is the smallest representable nonzero difference, against a between-seed sd of
    0.0377 R2 in the result this module supports. A regression that widened it would mean something
    real had changed, and a loose tolerance would hide exactly that."""
    one_ulp = torch.finfo(torch.float32).eps
    g = torch.Generator().manual_seed(7)
    for _ in range(50):
        w = torch.randn(9, 9, generator=g, requires_grad=True)
        dev = (ste_ternarize(w).detach() - twn_ternary(w)).abs().max()
        assert float(dev) <= one_ulp, f'forward value drifted {float(dev):.3e}, above one ULP'


def test_ste_zeros_are_exactly_zero_so_the_sparsity_pattern_is_exact():
    """The companion to the ULP result, and the half that must be exact. Where ternary(w) is 0 the
    expression is w + (0 - w) = w - w, which float32 evaluates to exactly 0. So the SPARSITY a
    ternary network claims is not approximate: a zeroed weight contributes nothing at all, rather
    than something of order 1e-7. Were this to fail, the compression claim is what to re-examine."""
    g = torch.Generator().manual_seed(23)
    for _ in range(50):
        w = torch.randn(9, 9, generator=g, requires_grad=True)
        st, t = ste_ternarize(w).detach(), twn_ternary(w)
        zeros = t == 0
        assert bool(zeros.any()), 'no weights were zeroed; example does not exercise the property'
        assert torch.equal(st[zeros], torch.zeros_like(st[zeros])), 'a zeroed weight is not exact'


def test_ste_gradient_is_identity():
    """d(w_t)/dw == 1 everywhere. Checked through autograd, not by reading the formula."""
    g = torch.Generator().manual_seed(13)
    w = torch.randn(30, generator=g, requires_grad=True)
    ste_ternarize(w).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w), atol=FP32_TOL), w.grad.tolist()


def test_ste_gradient_reaches_the_weights_the_forward_pass_zeroed():
    """The property that makes QAT trainable. A weight below the threshold contributes nothing to the
    forward value, yet must still receive gradient so it can grow back above the threshold. If this
    failed, QAT would degenerate into the post-hoc pruning that collapsed R2 to -1.50."""
    w = torch.tensor([0.01, 0.02, 1.0, -1.0], requires_grad=True)
    t = ste_ternarize(w)
    assert float(t[0]) == 0.0 and float(t[1]) == 0.0, 'expected the small weights to be zeroed'
    (t * torch.tensor([3.0, 5.0, 1.0, 1.0])).sum().backward()
    assert abs(float(w.grad[0]) - 3.0) < FP32_TOL, f'pruned weight got gradient {float(w.grad[0])}'
    assert abs(float(w.grad[1]) - 5.0) < FP32_TOL, f'pruned weight got gradient {float(w.grad[1])}'


def test_ternary_linear_forward_equals_an_explicit_ternary_matmul():
    g = torch.Generator().manual_seed(17)
    lin = TernaryLinear(6, 3)
    x = torch.randn(5, 6, generator=g)
    got = lin(x)
    expect = F.linear(x, twn_ternary(lin.weight), lin.bias)
    assert torch.allclose(got, expect, atol=FP32_TOL)


def test_bias_is_not_ternarized():
    """Standard TWN/TTQ practice, and stated in the module docstring. Checked by giving the bias values
    that ternarization would visibly destroy."""
    lin = TernaryLinear(3, 3, bias=True)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor([0.001, 0.5, -7.0]))
    out = lin(torch.zeros(1, 3))
    assert torch.allclose(out[0], torch.tensor([0.001, 0.5, -7.0]), atol=FP32_TOL), out[0].tolist()


# ----------------------------------------------------------------- conversion

def test_conversion_replaces_every_linear_including_nested_ones():
    m = convert_to_ternary_qat(_net())
    linears = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == 3, len(linears)
    assert all(isinstance(mod, TernaryLinear) for mod in linears), 'a nested Linear was missed'


def test_conversion_shares_the_parameter_objects_rather_than_copying_them():
    """The checkpoint-compatibility claim. A copy would still load and still train -- but it would train
    a tensor the forward pass never reads, and the failure would look like 'QAT just works worse'."""
    m = _net()
    before = {n: p for n, p in m.named_parameters()}
    convert_to_ternary_qat(m)
    after = {n: p for n, p in m.named_parameters()}
    assert set(before) == set(after), (sorted(before), sorted(after))
    for n in before:
        assert before[n] is after[n], f'{n} was copied, not shared'


def test_a_checkpoint_loads_identically_before_or_after_conversion():
    """Stated explicitly in the docstring: 'loads either before or after conversion with no
    special-casing'. Tested in both orders, with the forward output as the witness."""
    torch.manual_seed(2021)
    src = _net()
    sd = {k: v.clone() for k, v in src.state_dict().items()}
    x = torch.randn(4, 4)

    load_then_convert = _net()
    load_then_convert.load_state_dict(sd)
    convert_to_ternary_qat(load_then_convert)

    convert_then_load = convert_to_ternary_qat(_net())
    convert_then_load.load_state_dict(sd)

    a, b = load_then_convert(x), convert_then_load(x)
    assert torch.allclose(a, b, atol=FP32_TOL), 'conversion order changed the model'


def test_conversion_preserves_the_absence_of_a_bias():
    m = convert_to_ternary_qat(_net(bias=False))
    for mod in m.modules():
        if isinstance(mod, TernaryLinear):
            assert mod.bias is None, 'conversion invented a bias'


def test_converting_twice_is_a_no_op():
    """Idempotence matters because conversion can plausibly be called in both a builder and a loader."""
    m = convert_to_ternary_qat(_net())
    ids = [id(mod) for mod in m.modules() if isinstance(mod, TernaryLinear)]
    convert_to_ternary_qat(m)
    ids2 = [id(mod) for mod in m.modules() if isinstance(mod, TernaryLinear)]
    assert ids == ids2, 'a second conversion re-wrapped already-ternary layers'


def test_a_bare_linear_passed_at_the_top_level_is_NOT_converted():
    """A documented limitation, pinned so it cannot be discovered the hard way. The walk inspects
    named_children(), and an nn.Linear has none, so it is returned unchanged. Every model in this
    project is a container, so this does not affect any reported result."""
    bare = nn.Linear(3, 3)
    out = convert_to_ternary_qat(bare)
    assert not isinstance(out, TernaryLinear), (
        'behaviour changed: a top-level Linear is now converted. That is an improvement, but this test '
        'and the module docstring both describe the old behaviour -- update them together.')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
