"""Numerical-correctness tests for the BDF solver itself."""

from math import comb

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
import scipy.integrate._ivp.bdf as scipy_bdf

from diffrax_bdf import BDF
from diffrax_bdf._bdf import _bdf_residual
from diffrax_bdf._coeffs import D_ROWS, MAX_ORDER, make_tables

from helpers import observed_order, seed_state


TIGHT = diffrax.VeryChord(rtol=1e-13, atol=1e-13)

# The finest usable step: beyond this the error hits the float64 roundoff floor and
# the measured slope collapses. High-order formulas reach the floor sooner.
FINEST_REFINEMENT = {1: 11, 2: 11, 3: 10, 4: 9, 5: 8}


def _decay_problem(lam=-1.0):
    exact = lambda t: jnp.array([jnp.exp(lam * t)])
    term = diffrax.ODETerm(lambda t, y, args: lam * y)
    return term, exact


def _rotating_problem():
    """scipy's `fun_linear`: a genuinely 2-D Jacobian, which catches index bugs."""
    term = diffrax.ODETerm(
        lambda t, y, args: jnp.stack([-y[0] - 5 * y[1], y[0] + y[1]])
    )
    exact = lambda t: jnp.stack(
        [-5 * jnp.sin(2 * t), 2 * jnp.cos(2 * t) + jnp.sin(2 * t)]
    )
    return term, exact


def _fixed_step_error(term, exact, order, h, t1, use_ndf=True):
    solver = BDF(bdf_order=order, use_ndf=use_ndf, root_finder=TIGHT)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=h,
        y0=exact(0.0),
        stepsize_controller=diffrax.ConstantStepSize(),
        solver_state=seed_state(solver, term, exact, 0.0, h, order),
        max_steps=1_000_000,
    )
    return float(jnp.linalg.norm(sol.ys[0] - exact(t1)))


@pytest.mark.parametrize("use_ndf", [True, False])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_convergence_order_scalar(order, use_ndf):
    """A BDF of order k must converge at order k when the history is exact.

    Seeding the history from the analytic solution is essential: left to bootstrap
    itself the solver ramps the order up from 1, contributing `O(h ** 2)` error that
    swamps the slope of every higher-order formula.
    """
    term, exact = _decay_problem()
    finest = FINEST_REFINEMENT[order]
    hs = [1.0 / 2**m for m in range(finest - 4, finest)]
    errors = [_fixed_step_error(term, exact, order, h, 1.0, use_ndf) for h in hs]
    assert observed_order(hs, errors) == pytest.approx(order, abs=0.15)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_convergence_order_two_dimensional(order):
    term, exact = _rotating_problem()
    hs = [1.0 / 2**m for m in range(6, 10)]
    errors = [_fixed_step_error(term, exact, order, h, 1.0) for h in hs]
    assert observed_order(hs, errors) == pytest.approx(order, abs=0.15)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_backward_integration(order):
    """Integrating t0 > t1 gives a negative step from `terms.contr`.

    Every formula is consistent under that sign provided the step is only ever
    obtained from `terms.contr` and never recomputed as `t1 - t0`. The assertion
    compares against the *forward* error at the same step size rather than against a
    fixed tolerance, so it tests the sign handling rather than the order's accuracy.
    """
    lam = -1.0
    term = diffrax.ODETerm(lambda t, y, args: lam * y)
    exact = lambda t: jnp.array([jnp.exp(lam * t)])
    h = 1e-3

    def solve(t0, t1, dt0):
        return diffrax.diffeqsolve(
            term,
            BDF(bdf_order=order, root_finder=TIGHT),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=exact(t0),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=100_000,
        ).ys[0]

    forward = solve(0.0, 1.0, h)
    backward = solve(1.0, 0.0, -h)
    forward_error = float(abs(forward[0] - exact(1.0)[0]) / exact(1.0)[0])
    backward_error = float(abs(backward[0] - exact(0.0)[0]) / exact(0.0)[0])

    # A sign error would diverge or run the wrong way, not merely lose accuracy.
    assert np.isfinite(backward_error)
    assert backward_error < 5 * forward_error
    assert backward_error < 10.0 ** (-order)


def _max_root_modulus(hlam, order, use_ndf=False):
    """Largest root of the BDF characteristic polynomial at `h * lambda`.

    Built from this package's own `gamma` table -- `gamma[j] - gamma[j - 1] == 1/j`
    are exactly the coefficients of the backward differences in the formula -- so
    this validates the table rather than a hard-coded copy of it.
    """
    gamma, _, _ = make_tables(use_ndf=use_ndf)
    weights = np.diff(gamma)[:order]
    coeffs = np.zeros(order + 1)
    for j in range(1, order + 1):
        for i in range(j + 1):
            coeffs[i] += weights[j - 1] * comb(j, i) * (-1) ** i
    poly = coeffs.astype(complex)
    poly[0] -= hlam
    return np.max(np.abs(np.roots(poly)))


# Published A(alpha) angles of the classical BDF methods, in degrees.
BDF_ANGLES = {1: 90.0, 2: 90.0, 3: 86.03, 4: 73.35, 5: 51.84}


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 75.0, 85.0, 89.0])
def test_stability_angle_matches_published(order, theta):
    """The tables must reproduce the textbook A(alpha) angles.

    `theta` is `arg(-lambda)`: zero is the negative real axis, 90 degrees is the
    imaginary axis. A test on the negative real axis alone would be vacuous, since
    every BDF from 1 to 6 is stable there.
    """
    radians = np.deg2rad(theta)
    worst = max(
        _max_root_modulus(m * np.exp(1j * (np.pi + radians)), order)
        for m in np.logspace(-2, 4, 400)
    )
    if theta > BDF_ANGLES[order]:
        assert worst > 1 + 1e-6
    else:
        assert worst <= 1 + 1e-6


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_solver_stability_matches_angle(order):
    """The solver itself, on a complex pair just outside BDF3's stability wedge.

    At `arg(-lambda) = 85` degrees orders 1-3 are stable and orders 4-5 are not, so
    this discriminates exactly where the published angles say it should.
    """
    theta, modulus, h, steps = np.deg2rad(85.0), 2.0, 1.0, 120
    lam = (modulus / h) * np.exp(1j * (np.pi + theta))
    a, b = float(np.real(lam)), float(np.imag(lam))
    matrix = jnp.array([[a, b], [-b, a]])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: matrix @ y),
        BDF(bdf_order=order, use_ndf=False, root_finder=TIGHT),
        t0=0.0,
        t1=steps * h,
        dt0=h,
        y0=jnp.array([1.0, 0.0]),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(steps=True),
        max_steps=steps + 1,
    )
    ys = np.asarray(sol.ys)
    ys = ys[np.isfinite(ys).all(axis=1)]
    norms = np.linalg.norm(ys, axis=1)
    growth = norms[-1] / norms[len(norms) // 2]
    if order <= 3:
        assert growth < 1.0
    else:
        assert growth > 1.0


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_single_step_solves_the_same_equation_as_scipy(order):
    """Pin the inputs and check our step against scipy's, component by component.

    Step-for-step agreement over a whole solve is not achievable -- the Newton
    iteration count never reaches diffrax's controller, so the safety factor
    differs, and we rescale the difference array on entry where scipy does it on
    exit. A single step with identical inputs is the strongest thing that *is*
    well defined.
    """
    rng = np.random.default_rng(7)
    n = 4
    matrix = rng.standard_normal((n, n)) - n * np.eye(n)
    vf = lambda t, y, args: jnp.asarray(matrix) @ y
    gamma, alpha, error_const = make_tables(use_ndf=True)

    d_array = np.zeros((D_ROWS, n))
    d_array[: order + 1] = rng.standard_normal((order + 1, n)) * 0.1
    d_array[0] += 1.0
    h = 0.05
    t0, t1 = 0.0, h

    # --- scipy's formulation, done by hand with its own helpers ---
    y_predict = np.sum(d_array[: order + 1], axis=0)
    psi = np.dot(d_array[1 : order + 1].T, gamma[1 : order + 1]) / alpha[order]
    c = h / alpha[order]
    lu = scipy.linalg.lu_factor(np.eye(n) - c * matrix)
    scale = 1e-12 + 1e-12 * np.abs(y_predict)
    converged, _, y_new, correction = scipy_bdf.solve_bdf_system(
        lambda t, y: matrix @ y,
        t1,
        y_predict,
        c,
        psi,
        lu,
        lambda lu_, b: scipy.linalg.lu_solve(lu_, b),
        scale,
        1e-12,
    )
    assert converged

    # --- ours ---
    solver = BDF(bdf_order=order, root_finder=TIGHT)
    term = diffrax.ODETerm(vf)
    state = eqx.tree_at(
        lambda s: (s.d_array, s.order, s.n_equal_steps, s.h_prev),
        solver.init(term, t0, t1, jnp.asarray(d_array[0]), None),
        (
            jnp.asarray(d_array),
            jnp.asarray(order, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(h),
        ),
    )
    y1, y_error, _, new_state, result = solver.step(
        term, t0, t1, jnp.asarray(d_array[0]), None, state, False
    )
    assert result == diffrax.RESULTS.successful
    np.testing.assert_allclose(np.asarray(y1), y_new, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(y_error), error_const[order] * correction, rtol=1e-8, atol=1e-12
    )

    # The residual we hand the root finder must be the equation scipy solves.
    residual = _bdf_residual(
        jnp.asarray(correction),
        (
            diffrax.ODETerm(vf).vf_prod,
            t1,
            jnp.asarray(y_predict),
            None,
            h,
            1 / alpha[order],
            jnp.asarray(psi),
        ),
    )
    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-11)

    # And the difference-array update must match scipy's recurrence.
    expected = d_array.copy()
    expected[order + 2] = correction - expected[order + 1]
    expected[order + 1] = correction
    for i in reversed(range(order + 1)):
        expected[i] += expected[i + 1]
    np.testing.assert_allclose(
        np.asarray(new_state.d_array), expected, rtol=1e-9, atol=1e-11
    )


def test_difference_array_tracks_the_solution():
    """Row 0 of the difference array is the solution itself."""
    term, exact = _decay_problem()
    sol = diffrax.diffeqsolve(
        term,
        BDF(bdf_order=3, root_finder=TIGHT),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=exact(0.0),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(t1=True, solver_state=True),
        max_steps=1000,
    )
    np.testing.assert_allclose(
        np.asarray(sol.solver_state.d_array)[0], np.asarray(sol.ys[0]), atol=1e-14
    )
