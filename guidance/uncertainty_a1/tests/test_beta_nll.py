"""Verifies the beta-NLL loss (Seitzer et al. 2022) added to train_egnn_heteroscedastic.py.

Why these tests and not just a careful read of the loss: the entire justification for running A1e-beta is
a claim about GRADIENTS -- that plain Gaussian NLL scales each example's pull on the mean by 1/sigma^2,
silencing exactly the examples it has mispredicted, and that multiplying by stopgrad(sigma^2)^beta
restores them. If the weight were attached to the graph instead of detached, or if the exponent were
applied to sigma rather than sigma^2, the loss would still train and still produce plausible numbers --
it just would not be beta-NLL, and the comparison against A1e would be measuring something nobody
intended. None of that is visible in a training curve. It is visible in autograd, so that is what is
checked here.

The scaling law asserted (the paper's central mechanism, derived independently below):
    dL_i/dmu_i = -(y_i - mu_i) * sigma_i^(2*beta - 2)
  beta = 0   -> sigma^-2  (plain NLL; the pathological regime)
  beta = 0.5 -> sigma^-1
  beta = 1   -> sigma^0   (variance-independent, as in MSE)

Pure CPU, no model, no data, no GPU.

Usage:
  PYTHONPATH=. python guidance/uncertainty_a1/tests/test_beta_nll.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from guidance.uncertainty_a1.train_egnn_heteroscedastic import (LOG_VAR_CLAMP, beta_nll,
                                                                gaussian_nll)

TOL = 1e-10


def fixture(n=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    mu = torch.randn(n, generator=g, dtype=torch.float64)
    log_var = torch.randn(n, generator=g, dtype=torch.float64)       # spans ~[-2, 2]
    return y, mu, log_var


# ------------------------------------------------- the base NLL, against an outside reference

def test_gaussian_nll_matches_scipy_logpdf_up_to_the_dropped_constant():
    """Checked against scipy's normal log-density rather than against a re-typed version of the same
    formula, so an algebra slip cannot agree with itself. The implementation drops 0.5*log(2*pi), so the
    comparison restores it."""
    from scipy.stats import norm
    y, mu, log_var = fixture()
    ours = gaussian_nll(y, mu, log_var).numpy()
    ref = -norm.logpdf(y.numpy(), loc=mu.numpy(), scale=np.exp(0.5 * log_var.numpy()))
    assert np.abs(ours - (ref - 0.5 * math.log(2 * math.pi))).max() < 1e-12


def test_nll_is_minimised_in_log_var_at_the_true_squared_error():
    """Sanity on the variance term: for fixed residual r, NLL(log_var) is minimised at log_var = log(r^2).
    If the 0.5 factors were mismatched this optimum would move."""
    r = 1.7
    lv = torch.tensor([2 * math.log(r)], dtype=torch.float64, requires_grad=True)
    loss = gaussian_nll(torch.tensor([r], dtype=torch.float64),
                        torch.zeros(1, dtype=torch.float64), lv).sum()
    loss.backward()
    assert abs(lv.grad.item()) < 1e-12


# ------------------------------------------------- beta = 0 must be A1e, bit-for-bit

def test_beta_zero_reproduces_plain_nll_exactly():
    """A1e vs A1e-beta is only a controlled comparison if beta=0 is the original loss unchanged. Asserted
    as exact equality, not approximate: beta=0 short-circuits before any arithmetic is applied."""
    y, mu, log_var = fixture()
    weighted, plain = beta_nll(y, mu, log_var, beta=0.0)
    expected = gaussian_nll(y, mu, log_var)
    assert torch.equal(weighted, expected)
    assert torch.equal(plain, expected)


def test_beta_zero_gradients_are_identical_to_plain_nll():
    y, mu0, log_var0 = fixture()
    grads = []
    for beta in (0.0, None):
        mu = mu0.clone().requires_grad_(True)
        lv = log_var0.clone().requires_grad_(True)
        loss = (beta_nll(y, mu, lv, beta=0.0)[0] if beta == 0.0 else gaussian_nll(y, mu, lv)).sum()
        loss.backward()
        grads.append((mu.grad.clone(), lv.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])


# ------------------------------------------------- the weight itself

def test_weight_is_variance_to_the_beta_not_sigma_to_the_beta():
    """The weight is (sigma^2)^beta = exp(beta*log_var). Applying beta to sigma instead would halve the
    exponent and silently make beta=0.5 behave like beta=0.25 -- a difference no training curve shows."""
    y, mu, log_var = fixture()
    for beta in (0.25, 0.5, 1.0, 2.0):
        weighted, plain = beta_nll(y, mu, log_var, beta=beta)
        implied = (weighted / plain).numpy()
        expected = np.exp(beta * log_var.numpy())
        assert np.abs(implied - expected).max() < TOL, beta
        # and NOT the sigma^beta variant
        wrong = np.exp(0.5 * beta * log_var.numpy())
        assert np.abs(implied - wrong).max() > 1e-6, f'beta={beta} looks like sigma^beta, not var^beta'


def test_plain_nll_returned_is_never_weighted():
    """The second return value feeds model selection. If beta ever leaked into it, early stopping would
    optimise a different objective than the one reported, and A1e/A1e-beta val columns would stop being
    comparable."""
    y, mu, log_var = fixture()
    reference = gaussian_nll(y, mu, log_var)
    for beta in (0.0, 0.5, 1.0):
        _, plain = beta_nll(y, mu, log_var, beta=beta)
        assert torch.allclose(plain, reference, atol=0, rtol=0)


# ------------------------------------------------- the central claim: the gradient scaling law

def _dmu(beta, y, mu0, log_var0):
    mu = mu0.clone().requires_grad_(True)
    beta_nll(y, mu, log_var0.clone(), beta=beta)[0].sum().backward()
    return mu.grad.clone()


def test_mean_gradient_follows_sigma_to_the_2beta_minus_2():
    """The mechanism, verified by autograd against the closed form derived in the module docstring."""
    y, mu0, log_var0 = fixture()
    resid = (y - mu0).numpy()
    for beta in (0.0, 0.25, 0.5, 1.0):
        got = _dmu(beta, y, mu0, log_var0).numpy()
        expected = -resid * np.exp(log_var0.numpy() * (beta - 1.0))   # sigma^(2beta-2)
        assert np.abs(got - expected).max() < 1e-9, beta


def test_at_beta_one_the_mean_gradient_is_variance_independent():
    """beta=1 is the limiting case that makes the claim concrete: the pull on the mean becomes exactly the
    MSE gradient, identical for two examples with wildly different predicted sigma."""
    y = torch.tensor([1.0, 1.0], dtype=torch.float64)
    mu0 = torch.zeros(2, dtype=torch.float64)
    log_var0 = torch.tensor([-4.0, 4.0], dtype=torch.float64)        # sigma 0.135 vs 7.39
    g = _dmu(1.0, y, mu0, log_var0).numpy()
    assert abs(g[0] - g[1]) < 1e-12, 'beta=1 must equalise the two'
    assert np.abs(g - np.array([-1.0, -1.0])).max() < 1e-12, 'and must equal the MSE gradient'


def test_plain_nll_silences_high_sigma_examples_which_is_the_pathology():
    """Demonstrates the failure mode A1e hit, so the fix is tested against a reproduction of the problem
    rather than against nothing. Two identical residuals, sigma differing by e^4: under plain NLL the
    high-sigma example's influence on the mean is ~3000x smaller."""
    y = torch.tensor([1.0, 1.0], dtype=torch.float64)
    mu0 = torch.zeros(2, dtype=torch.float64)
    log_var0 = torch.tensor([-4.0, 4.0], dtype=torch.float64)
    g0 = _dmu(0.0, y, mu0, log_var0).numpy()
    ratio = abs(g0[0]) / abs(g0[1])
    assert ratio > 1000, f'expected plain NLL to suppress the high-sigma example, ratio was {ratio:.1f}'
    g_half = _dmu(0.5, y, mu0, log_var0).numpy()
    assert abs(g_half[0]) / abs(g_half[1]) < ratio, 'beta=0.5 must reduce the imbalance'


# ------------------------------------------------- detachment

def test_weight_gradient_is_stopped():
    """If the weight were left attached, d/dlog_var would pick up an extra beta*w*NLL term. Compared here
    against the closed form for the DETACHED case, and separately shown to differ from the attached one,
    so the test fails whichever way the mistake is made."""
    y, mu0, log_var0 = fixture()
    beta = 0.5
    lv = log_var0.clone().requires_grad_(True)
    beta_nll(y, mu0.clone(), lv, beta=beta)[0].sum().backward()
    got = lv.grad.numpy()

    yv, muv, lvv = y.numpy(), mu0.numpy(), log_var0.numpy()
    w = np.exp(beta * lvv)
    nll = 0.5 * lvv + 0.5 * (yv - muv) ** 2 * np.exp(-lvv)
    d_nll = 0.5 - 0.5 * (yv - muv) ** 2 * np.exp(-lvv)
    detached = w * d_nll
    attached = w * d_nll + beta * w * nll

    assert np.abs(got - detached).max() < 1e-9, 'weight is not detached'
    assert np.abs(got - attached).max() > 1e-6, 'gradient matches the attached form -- weight leaked'


def test_detachment_holds_for_every_beta_tested():
    y, mu0, log_var0 = fixture(seed=3)
    for beta in (0.25, 0.5, 1.0, 2.0):
        lv = log_var0.clone().requires_grad_(True)
        beta_nll(y, mu0.clone(), lv, beta=beta)[0].sum().backward()
        yv, muv, lvv = y.numpy(), mu0.numpy(), log_var0.numpy()
        w = np.exp(beta * lvv)
        d_nll = 0.5 - 0.5 * (yv - muv) ** 2 * np.exp(-lvv)
        assert np.abs(lv.grad.numpy() - w * d_nll).max() < 1e-9, beta


# ------------------------------------------------- numerics at the clamp boundaries

def test_no_overflow_or_nan_at_the_log_var_clamp_limits():
    """log_var is clamped to [-6, 6] upstream. At beta=2 the weight reaches e^12 ~ 1.6e5, and the NLL term
    at the low end carries exp(6); both must stay finite in float32, which is what training uses."""
    y = torch.tensor([0.0, 10.0], dtype=torch.float32)
    mu = torch.zeros(2, dtype=torch.float32)
    for lv_val in LOG_VAR_CLAMP:
        lv = torch.full((2,), float(lv_val), dtype=torch.float32, requires_grad=True)
        for beta in (0.0, 0.5, 1.0, 2.0):
            w, p = beta_nll(y, mu, lv, beta=beta)
            assert torch.isfinite(w).all(), (lv_val, beta, 'weighted loss not finite')
            assert torch.isfinite(p).all(), (lv_val, beta, 'plain nll not finite')
            lv.grad = None
            w.sum().backward()
            assert torch.isfinite(lv.grad).all(), (lv_val, beta, 'gradient not finite')


def test_shape_and_dtype_are_preserved():
    for dtype in (torch.float32, torch.float64):
        y = torch.zeros(5, dtype=dtype)
        mu = torch.zeros(5, dtype=dtype)
        lv = torch.zeros(5, dtype=dtype)
        for beta in (0.0, 0.5):
            w, p = beta_nll(y, mu, lv, beta=beta)
            assert w.shape == (5,) and p.shape == (5,)
            assert w.dtype == dtype and p.dtype == dtype


def test_loss_is_per_example_not_reduced():
    """get_loss does the .mean(); beta_nll must not reduce, or the weighting would be applied to an
    already-averaged scalar and become a no-op constant."""
    y, mu, log_var = fixture(n=11)
    w, p = beta_nll(y, mu, log_var, beta=0.5)
    assert w.numel() == 11 and p.numel() == 11


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
