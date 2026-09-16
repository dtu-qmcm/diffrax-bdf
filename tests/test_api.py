"""The acceptance test: does `BDF` work in the configuration enzax actually uses?

This is the test that closes the "can a multistep solver live inside diffrax's
integration loop without patching it" question. A fixed-step solve would not: it
never rejects a step (so the `solver_state` rollback is untested), and it skips the
`filter_eval_shape(solver.step, ...)` tracing that `diffrax` only performs when
`saveat.dense` is set or an event is present.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from diffrax_bdf import BDF

from helpers import (
    FAST_EQUILIBRIUM_ARGS,
    FAST_EQUILIBRIUM_Y0,
    ROBERTSON_ARGS,
    ROBERTSON_Y0,
    fast_equilibrium_vf,
    robertson_vf,
    steady_state_solve,
    stiff_linear_steady_state,
    stiff_linear_vf,
)


ORDERS = [1, 2, 3, 4, 5]
B = jnp.array([1.0, 2.0, 3.0])
Y0 = jnp.zeros(3)


@pytest.mark.parametrize("order", ORDERS)
def test_steady_state_matches_exact(order):
    """The whole enzax configuration: t1=inf, steady-state event, implicit adjoint."""
    sol = steady_state_solve(BDF(bdf_order=order), stiff_linear_vf, Y0, B)
    assert sol.result == diffrax.RESULTS.event_occurred
    np.testing.assert_allclose(
        sol.ys[0], stiff_linear_steady_state(B), rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize("order", ORDERS)
def test_steady_state_gradient_matches_exact(order):
    """`ImplicitAdjoint` differentiates via `solver.func`, so this should be exact.

    For the linear problem `dy*/db = -inv(A)`, known in closed form.
    """

    def steady(b):
        return steady_state_solve(BDF(bdf_order=order), stiff_linear_vf, Y0, b).ys[0]

    got = jax.jacrev(steady)(B)
    expected = jax.jacrev(stiff_linear_steady_state)(B)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_steady_state_agrees_with_kvaerno5(order):
    """Cross-check against the solver enzax uses today, on a nonlinear problem.

    Note this problem is chosen to have an *isolated* steady state. Robertson is
    unsuitable here despite being the canonical stiff test: it conserves total mass,
    so its Jacobian is singular at the steady state and `|dy/dt|` drops below any
    tolerance while `y` is still far from its limit. That is precisely why enzax
    eliminates conserved moieties structurally before solving.
    """
    bdf = steady_state_solve(
        BDF(bdf_order=order), fast_equilibrium_vf, FAST_EQUILIBRIUM_Y0, FAST_EQUILIBRIUM_ARGS
    )
    kvaerno = steady_state_solve(
        diffrax.Kvaerno5(), fast_equilibrium_vf, FAST_EQUILIBRIUM_Y0, FAST_EQUILIBRIUM_ARGS
    )
    assert bdf.result == diffrax.RESULTS.event_occurred
    np.testing.assert_allclose(bdf.ys[0], kvaerno.ys[0], rtol=1e-6, atol=1e-8)
    # The residual really is at the steady state, not merely slowly varying.
    residual = fast_equilibrium_vf(0.0, bdf.ys[0], FAST_EQUILIBRIUM_ARGS)
    assert float(jnp.linalg.norm(residual)) < 1e-8


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_robertson_transient_matches_scipy(order):
    """Robertson as an initial value problem, against scipy's BDF.

    This is Robertson's proper role here: a stiff transient spanning many decades,
    where the reference is another BDF implementation rather than a steady state.
    """
    t1 = 1e4
    reference = solve_ivp(
        lambda t, y: np.asarray(robertson_vf(t, jnp.asarray(y), ROBERTSON_ARGS)),
        (0.0, t1),
        np.asarray(ROBERTSON_Y0),
        method="BDF",
        rtol=1e-10,
        atol=1e-12,
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(robertson_vf),
        BDF(bdf_order=order),
        t0=0.0,
        t1=t1,
        dt0=1e-8,
        y0=ROBERTSON_Y0,
        args=ROBERTSON_ARGS,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=200_000,
    )
    np.testing.assert_allclose(sol.ys[0], reference.y[:, -1], rtol=1e-5, atol=1e-10)
    # Total mass is conserved exactly by the vector field.
    np.testing.assert_allclose(float(jnp.sum(sol.ys[0])), 1.0, rtol=0, atol=1e-9)


def test_solver_is_jittable():
    """The solver state must survive as a `lax.while_loop` carry under `jit`."""

    @jax.jit
    def run(b):
        return steady_state_solve(BDF(bdf_order=2), stiff_linear_vf, Y0, b).ys[0]

    np.testing.assert_allclose(run(B), stiff_linear_steady_state(B), rtol=1e-6, atol=1e-9)


def test_solver_is_vmappable():
    bs = jnp.stack([B, 2 * B, 0.5 * B])
    run = jax.vmap(
        lambda b: steady_state_solve(BDF(bdf_order=2), stiff_linear_vf, Y0, b).ys[0]
    )
    expected = jax.vmap(stiff_linear_steady_state)(bs)
    np.testing.assert_allclose(run(bs), expected, rtol=1e-6, atol=1e-9)


def test_rejects_non_scalar_control():
    """A `ControlTerm` has no well-defined step-size ratio; fail loudly, not silently."""
    term = diffrax.ControlTerm(
        lambda t, y, args: jnp.eye(3),
        diffrax.VirtualBrownianTree(0.0, 1.0, tol=1e-3, shape=(3,), key=jax.random.key(0)),
    )
    with pytest.raises(Exception):
        diffrax.diffeqsolve(
            term, BDF(bdf_order=2), t0=0.0, t1=1.0, dt0=0.1, y0=Y0,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
