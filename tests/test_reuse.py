"""Tests for the Jacobian / factorisation reuse policy and the step-size deadband.

These are the point of the solver: without them a BDF does the same amount of
linear algebra per step as `Kvaerno5` and has no advantage over it.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffrax_bdf import BDF, BDFController

from helpers import (
    FAST_EQUILIBRIUM_ARGS,
    FAST_EQUILIBRIUM_Y0,
    fast_equilibrium_vf,
    stiff_linear_vf,
)


TOLS = diffrax.VeryChord(rtol=1e-8, atol=1e-10)
B = jnp.array([1.0, 2.0, 3.0])


def _stepped(solver, term, t0, t1, y0, args, state):
    return solver.step(term, t0, t1, y0, args, state, False)


def _initial(solver, term, t0, h, y0, args):
    return solver.init(term, t0, t0 + h, y0, args)


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def _state_with(solver, term, y0, args, **overrides):
    state = _initial(solver, term, 0.0, 0.01, y0, args)
    for name, value in overrides.items():
        state = eqx.tree_at(lambda s, n=name: getattr(s, n), state, value)
    return state


@pytest.mark.parametrize(
    "gamma_change, steps_since_lu, expect_lu",
    [
        (0.0, 0, False),  # nothing has moved: reuse
        (0.29, 0, False),  # inside CVODE's DGMAX deadband
        (0.31, 0, True),  # outside it
        (0.0, 19, False),  # MSBP not yet reached
        (0.0, 20, True),  # MSBP reached
    ],
)
def test_lu_gate(gamma_change, steps_since_lu, expect_lu):
    solver = BDF(bdf_order=2, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    c = 0.5
    state = _state_with(
        solver,
        term,
        jnp.zeros(3),
        B,
        c_at_last_lu=jnp.asarray(c / (1 + gamma_change)),
        steps_since_lu=jnp.asarray(steps_since_lu, jnp.int32),
        steps_since_jac=jnp.asarray(0, jnp.int32),
    )
    need_jac, need_lu, _ = solver._should_update(state, jnp.asarray(c))
    assert bool(need_jac) is False
    assert bool(need_lu) is expect_lu


@pytest.mark.parametrize("steps_since_jac, expect", [(50, False), (51, True)])
def test_jacobian_gate(steps_since_jac, expect):
    solver = BDF(bdf_order=2, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    state = _state_with(
        solver,
        term,
        jnp.zeros(3),
        B,
        c_at_last_lu=jnp.asarray(0.5),
        steps_since_lu=jnp.asarray(0, jnp.int32),
        steps_since_jac=jnp.asarray(steps_since_jac, jnp.int32),
    )
    need_jac, need_lu, _ = solver._should_update(state, jnp.asarray(0.5))
    assert bool(need_jac) is expect
    # Refreshing the Jacobian always forces a refactorisation.
    assert bool(need_lu) is expect


def test_jacobian_outlives_its_factorisations():
    """The asymmetry that makes this worthwhile.

    A change in `c` rebuilds `I - c * J` from the *stored* `J` without touching the
    vector field. The Jacobian -- the expensive part for an AD-traced vector field --
    is left alone.
    """
    solver = BDF(bdf_order=2, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y0 = jnp.zeros(3)
    state = _initial(solver, term, 0.0, 1e-4, y0, B)
    _, _, _, state, _ = _stepped(solver, term, 0.0, 1e-4, y0, B, state)
    assert int(state.steps_since_jac) == 0
    jacobian = np.asarray(state.jac)

    # A step twice as long moves `c` by 100%, far outside the deadband.
    _, _, _, moved, _ = _stepped(solver, term, 1e-4, 3e-4, y0, B, state)
    assert int(moved.steps_since_lu) == 0, "should have refactorised"
    assert int(moved.steps_since_jac) == 1, "should NOT have rebuilt the Jacobian"
    np.testing.assert_array_equal(np.asarray(moved.jac), jacobian)
    assert not np.array_equal(np.asarray(moved.lu), np.asarray(state.lu))


def test_factorisation_is_reused_when_the_step_is_frozen():
    """With `h` and the order both settled, `c` is constant and nothing is rebuilt.

    The opening steps are excluded deliberately: while the order ramps from 1 to
    `bdf_order` it changes `alpha[order]`, and hence `c`, by as much as 29% -- so a
    refactorisation there is correct, not a failure of reuse.
    """
    solver = BDF(bdf_order=2, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y = jnp.zeros(3)
    h = 1e-4
    state = _initial(solver, term, 0.0, h, y, B)
    warmup = 3
    for i in range(warmup):
        y, _, _, state, _ = _stepped(solver, term, i * h, (i + 1) * h, y, B, state)
    assert int(state.order) == 2, "order should have settled"
    baseline = int(state.steps_since_lu)
    factorisation = np.asarray(state.lu)

    for i in range(warmup, warmup + 4):
        y, _, _, state, _ = _stepped(solver, term, i * h, (i + 1) * h, y, B, state)
        baseline += 1
        assert int(state.steps_since_lu) == baseline, "step unchanged, so no refactorise"
        np.testing.assert_array_equal(np.asarray(state.lu), factorisation)


def test_stale_jacobian_is_refreshed_within_the_step():
    """A useless cached Jacobian must not cause a rejected step.

    Rejecting would roll the solver state back, so the retry would arrive with the
    same useless Jacobian. scipy and CVODE both refresh and retry inside the step;
    so do we.
    """
    solver = BDF(bdf_order=2, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y0 = jnp.ones(3)
    h = 1e-3
    size = 3
    state = _initial(solver, term, 0.0, h, y0, B)
    # Pretend we hold a recent but completely wrong Jacobian.
    state = eqx.tree_at(
        lambda s: (s.jac, s.lu, s.piv, s.c_at_last_lu, s.steps_since_lu, s.steps_since_jac),
        state,
        (
            jnp.zeros((size, size)),
            jnp.eye(size),
            jnp.arange(size, dtype=jnp.int32),
            jnp.asarray(h / 1.185),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        ),
    )
    _, y_error, _, new_state, _ = _stepped(solver, term, 0.0, h, y0, B, state)
    assert np.all(np.isfinite(np.asarray(y_error))), "retry should have rescued the step"
    assert int(new_state.steps_since_jac) == 0, "retry should have rebuilt the Jacobian"
    assert not np.allclose(np.asarray(new_state.jac), 0.0)


# --------------------------------------------------------------------------
# The controller deadband
# --------------------------------------------------------------------------


def _adapt(controller, error_scale, h=0.1, error_order=3):
    """Run one `adapt_step_size` with an error estimate of a chosen magnitude."""
    y0 = jnp.ones(3)
    y1 = jnp.ones(3)
    y_error = error_scale * jnp.ones(3)
    _, state = controller.init(
        diffrax.ODETerm(stiff_linear_vf), 0.0, h, y0, h, B, lambda *a: y0, error_order
    )
    keep, next_t0, next_t1, _, _, _ = controller.adapt_step_size(
        0.0, h, y0, y1, B, y_error, error_order, state
    )
    return bool(keep), float(next_t1 - next_t0)


@pytest.mark.parametrize("error_scale", [1e-7, 5e-7, 9e-7])
def test_controller_freezes_modest_growth(error_scale):
    """Whatever the controller wants, growth below `eta_max_fx` keeps `h` put."""
    h = 0.1
    pid = diffrax.PIDController(rtol=1e-6, atol=1e-9)
    bdf = BDFController(rtol=1e-6, atol=1e-9)
    _, pid_dt = _adapt(pid, error_scale, h)
    keep, bdf_dt = _adapt(bdf, error_scale, h)
    assert keep
    if pid_dt / h < bdf.eta_max_fx:
        assert bdf_dt == pytest.approx(h), "inside the deadband, so should be frozen"
    else:
        assert bdf_dt == pytest.approx(pid_dt), "outside it, so should match the PID"


def test_controller_never_shrinks_an_accepted_step():
    """CVODE's deadband has no lower edge: an accepted step never shrinks."""
    h = 0.1
    # An error just inside tolerance: the PID wants a slightly smaller step.
    keep, dt = _adapt(BDFController(rtol=1e-6, atol=1e-9), 9e-7, h)
    assert keep
    assert dt >= h


def test_controller_still_shrinks_a_rejected_step():
    keep, dt = _adapt(BDFController(rtol=1e-6, atol=1e-9), 1e-2, 0.1)
    assert not keep
    assert dt < 0.1


def test_controller_matches_pid_when_deadband_disabled():
    for scale in (1e-4, 1e-7, 1e-2):
        _, pid_dt = _adapt(diffrax.PIDController(rtol=1e-6, atol=1e-9), scale)
        _, bdf_dt = _adapt(BDFController(rtol=1e-6, atol=1e-9, eta_max_fx=1.0), scale)
        assert bdf_dt == pytest.approx(pid_dt)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def _steady_state(solver, controller):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(fast_equilibrium_vf),
        solver,
        t0=0.0,
        t1=jnp.inf,
        dt0=1e-6,
        y0=FAST_EQUILIBRIUM_Y0,
        args=FAST_EQUILIBRIUM_ARGS,
        stepsize_controller=controller,
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
        adjoint=diffrax.ImplicitAdjoint(),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=100_000,
    )


def test_deadband_does_not_change_the_answer():
    """The deadband trades step-size optimality for reuse; it must not cost accuracy."""
    pid = _steady_state(
        BDF(bdf_order=3), diffrax.PIDController(rtol=1e-8, atol=1e-10, dtmax=1e6)
    )
    deadband = _steady_state(
        BDF(bdf_order=3), BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6)
    )
    assert deadband.result == diffrax.RESULTS.event_occurred
    np.testing.assert_allclose(deadband.ys[0], pid.ys[0], rtol=1e-6, atol=1e-9)


def test_gradients_still_exact_with_reuse():
    """Reuse must not perturb the steady-state gradient.

    It cannot, in principle: `ImplicitAdjoint` differentiates the converged steady
    state through `solver.func` alone and stop-gradients the solver state. This pins
    that reasoning down.
    """

    def steady(b):
        return diffrax.diffeqsolve(
            diffrax.ODETerm(stiff_linear_vf),
            BDF(bdf_order=3),
            t0=0.0,
            t1=jnp.inf,
            dt0=1e-6,
            y0=jnp.zeros(3),
            args=b,
            stepsize_controller=BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6),
            event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
            adjoint=diffrax.ImplicitAdjoint(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=100_000,
        ).ys[0]

    from helpers import stiff_linear_steady_state

    np.testing.assert_allclose(
        jax.jacrev(steady)(B),
        jax.jacrev(stiff_linear_steady_state)(B),
        rtol=1e-5,
        atol=1e-9,
    )
