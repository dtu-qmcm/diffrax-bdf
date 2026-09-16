"""Unit tests for the coefficient tables and the backward-difference array.

scipy's BDF is pure Python, so it is used here as a white-box oracle: we compare
against `scipy.integrate._ivp.bdf` directly rather than against recorded values.
"""

import numpy as np
import pytest
import scipy.integrate._ivp.bdf as scipy_bdf
from scipy.integrate import BDF as ScipyBDF

from diffrax_bdf._coeffs import (
    MAX_ORDER,
    change_D,
    compute_R,
    make_tables,
    weighted_row_sum,
)


ORDERS = list(range(1, MAX_ORDER + 1))
FACTORS = [0.5, 1.0, 2.3, 0.37, 10.0]


def _scipy_instance():
    """A scipy BDF object, purely to read its runtime-computed coefficients."""
    return ScipyBDF(lambda t, y: -y, 0.0, np.array([1.0]), t_bound=1.0)


def test_tables_match_scipy():
    solver = _scipy_instance()
    gamma, alpha, error_const = make_tables(use_ndf=True)
    np.testing.assert_allclose(gamma, solver.gamma, rtol=0, atol=0)
    np.testing.assert_allclose(alpha, solver.alpha, rtol=0, atol=0)
    np.testing.assert_allclose(error_const, solver.error_const, rtol=0, atol=0)


def test_error_const_is_not_monotonic():
    """error_const[4] > error_const[3] is genuine, not a transcription slip."""
    _, _, error_const = make_tables(use_ndf=True)
    assert error_const[4] > error_const[3]
    assert error_const[4] == pytest.approx(0.1135416666, rel=1e-9)
    assert error_const[3] == pytest.approx(0.0991166666, rel=1e-9)


def test_kappa_zero_recovers_classical_bdf():
    """With the NDF correction off we get the coefficients CVODE uses."""
    gamma, alpha, error_const = make_tables(use_ndf=False)
    np.testing.assert_allclose(alpha, gamma, rtol=0, atol=0)
    np.testing.assert_allclose(
        alpha[1:], np.cumsum(1 / np.arange(1, MAX_ORDER + 1)), atol=1e-15
    )
    np.testing.assert_allclose(
        error_const, 1 / np.arange(1, MAX_ORDER + 2), atol=1e-15
    )


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("factor", FACTORS)
def test_compute_R_active_block_matches_scipy(order, factor):
    got = np.asarray(compute_R(factor, order))
    expected = scipy_bdf.compute_R(order, factor)
    np.testing.assert_allclose(got[: order + 1, : order + 1], expected, atol=1e-14)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("factor", FACTORS)
def test_compute_R_is_identity_outside_active_block(order, factor):
    got = np.asarray(compute_R(factor, order))
    eye = np.eye(MAX_ORDER + 1)
    np.testing.assert_array_equal(got[order + 1 :, order + 1 :], eye[order + 1 :, order + 1 :])
    np.testing.assert_array_equal(got[: order + 1, order + 1 :], 0.0)
    np.testing.assert_array_equal(got[order + 1 :, : order + 1], 0.0)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("factor", FACTORS)
def test_change_D_matches_scipy(order, factor):
    rng = np.random.default_rng(0)
    d = rng.standard_normal((MAX_ORDER + 3, 4))
    expected = d.copy()
    scipy_bdf.change_D(expected, order, factor)
    got = np.asarray(change_D(d, order, factor))
    np.testing.assert_allclose(got, expected, atol=1e-13)


@pytest.mark.parametrize("order", ORDERS)
def test_change_D_identity_at_unit_factor(order):
    """Catches a transposed or mis-ordered `R @ U`."""
    rng = np.random.default_rng(1)
    d = rng.standard_normal((MAX_ORDER + 3, 3))
    np.testing.assert_allclose(np.asarray(change_D(d, order, 1.0)), d, atol=1e-14)


@pytest.mark.parametrize("order", ORDERS)
def test_change_D_leaves_inactive_rows_untouched(order):
    rng = np.random.default_rng(2)
    d = rng.standard_normal((MAX_ORDER + 3, 3))
    got = np.asarray(change_D(d, order, 0.5))
    np.testing.assert_array_equal(got[order + 1 :], d[order + 1 :])


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("factor", [0.5, 2.3])
def test_change_D_resamples_a_polynomial(order, factor):
    """The property that actually defines `change_D`.

    Build `D` from a degree-`order` polynomial sampled at spacing `h`, rescale by
    `factor`, and require the result to equal `D` built directly at spacing
    `factor * h`. Exact to roundoff, and independent of scipy.
    """
    rng = np.random.default_rng(3)
    coeffs = rng.standard_normal(order + 1)
    poly = np.polynomial.Polynomial(coeffs)

    def build(step):
        # D[j] is the j-th backward difference of y at t=0, spacing `step`.
        samples = np.array([poly(-i * step) for i in range(order + 1)])
        d = np.zeros((MAX_ORDER + 3, 1))
        col = samples.copy()
        for j in range(order + 1):
            d[j, 0] = col[0]
            col = col[:-1] - col[1:]
        return d

    got = np.asarray(change_D(build(1.0), order, factor))
    np.testing.assert_allclose(got[: order + 1], build(factor)[: order + 1], atol=1e-11)


@pytest.mark.parametrize("order", ORDERS)
def test_change_D_composes(order):
    """change_D(change_D(D, a), b) == change_D(D, a * b)."""
    rng = np.random.default_rng(4)
    d = rng.standard_normal((MAX_ORDER + 3, 3))
    twice = change_D(change_D(d, order, 0.4), order, 2.5)
    once = change_D(d, order, 0.4 * 2.5)
    np.testing.assert_allclose(np.asarray(twice), np.asarray(once), atol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_weighted_row_sum_predictor_matches_scipy(order):
    """`sum(D[:order + 1])` is scipy's predictor."""
    rng = np.random.default_rng(5)
    d = rng.standard_normal((MAX_ORDER + 3, 6))
    mask = (np.arange(MAX_ORDER + 1) <= order).astype(float)
    got = np.asarray(weighted_row_sum(mask, d))
    np.testing.assert_allclose(got, np.sum(d[: order + 1], axis=0), atol=1e-14)
