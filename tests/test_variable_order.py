"""Variable-order BDF: order selection, and the channel that carries the order.

diffrax asks a solver for its error order exactly once, outside the integration
loop, so a solver whose order changes per step has to send the current order to the
controller some other way. It travels inside `y_error`, which the loop treats
opaquely. These tests pin both the selection heuristic and that channel.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from diffrax_bdf import BDF, BDFController, ErrorWithOrder

from helpers import (
    FAST_EQUILIBRIUM_ARGS,
    FAST_EQUILIBRIUM_Y0,
    ROBERTSON_ARGS,
    ROBERTSON_Y0,
    fast_equilibrium_vf,
    robertson_vf,
    stiff_linear_steady_state,
    stiff_linear_vf,
)


B = jnp.array([1.0, 2.0, 3.0])
# Calling `solver.step` directly bypasses diffrax, which would otherwise inject the
# controller's tolerances into the root finder, so supply them here.
TOLS = diffrax.VeryChord(rtol=1e-8, atol=1e-10)


def _steady_state(solver, controller=None, vf=fast_equilibrium_vf, y0=None, args=None):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver,
        t0=0.0,
        t1=jnp.inf,
        dt0=1e-6,
        y0=FAST_EQUILIBRIUM_Y0 if y0 is None else y0,
        args=FAST_EQUILIBRIUM_ARGS if args is None else args,
        stepsize_controller=controller
        or BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6),
        event=diffrax.Event(diffrax.steady_state_event(rtol=1e-10, atol=1e-10)),
        adjoint=diffrax.ImplicitAdjoint(),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=200_000,
    )


def _order_trajectory(solver, vf, y0, args, max_steps=400):
    """Step the solver by hand, recording the order of every accepted step."""
    controller = BDFController(rtol=1e-8, atol=1e-10, dtmax=1e6)
    term = diffrax.ODETerm(vf)
    error_order = solver.error_order(term)
    h = 1e-6
    state = solver.init(term, 0.0, h, y0, args)
    _, controller_state = controller.init(
        term, 0.0, h, y0, h, args, solver.func, error_order
    )

    @eqx.filter_jit
    def advance(t0, t1, y, state, controller_state):
        y_new, y_error, _, new_state, _ = solver.step(term, t0, t1, y, args, state, False)
        keep, next_t0, next_t1, _, controller_state, _ = controller.adapt_step_size(
            t0, t1, y, y_new, args, y_error, error_order, controller_state
        )
        return keep, y_new, new_state, next_t0, next_t1, controller_state

    t0, t1, y = 0.0, h, y0
    orders = []
    for _ in range(max_steps):
        keep, y_new, new_state, t0_next, t1_next, controller_state = advance(
            t0, t1, y, state, controller_state
        )
        if bool(keep):
            y, state = y_new, new_state
            orders.append(int(state.order))
        t0, t1 = t0_next, t1_next
        if float(jnp.linalg.norm(vf(0.0, y, args))) < 1e-10:
            break
    return np.array(orders), y


# --------------------------------------------------------------------------
# Order selection
# --------------------------------------------------------------------------


def test_order_ramps_up_then_falls_back_towards_one():
    """The characteristic shape of an adaptive-order run on a stiff steady state.

    The order climbs through the transient, then drops back as the residual
    collapses and the step grows. That tail matters: BDF1 with an enormous step is
    exactly Newton's method on `f(y) = 0`, so a correct implementation converges to
    a steady-state solver all by itself.
    """
    orders, _ = _order_trajectory(
        BDF(bdf_order=None, root_finder=TOLS),
        fast_equilibrium_vf,
        FAST_EQUILIBRIUM_Y0,
        FAST_EQUILIBRIUM_ARGS,
    )
    assert orders[0] == 1, "must start at order 1; there is no history yet"
    assert orders.max() >= 4, "should reach high order during the transient"
    peak = int(orders.argmax())
    assert orders[-1] < orders[peak], "should come back down as the residual collapses"
    assert orders[-1] <= 2, "should approach the Newton limit at order 1"


def test_order_stays_within_bounds():
    orders, _ = _order_trajectory(
        BDF(bdf_order=None, root_finder=TOLS),
        fast_equilibrium_vf,
        FAST_EQUILIBRIUM_Y0,
        FAST_EQUILIBRIUM_ARGS,
    )
    assert orders.min() >= 1
    assert orders.max() <= 5


@pytest.mark.parametrize("max_order", [1, 2, 3, 4])
def test_max_order_is_respected(max_order):
    orders, _ = _order_trajectory(
        BDF(bdf_order=None, max_order=max_order, root_finder=TOLS),
        fast_equilibrium_vf,
        FAST_EQUILIBRIUM_Y0,
        FAST_EQUILIBRIUM_ARGS,
    )
    assert orders.max() <= max_order


def test_order_changes_only_after_enough_equal_steps():
    """scipy gates order changes on `order + 1` steps at one step size.

    Both the error constants and the difference array assume an equally spaced
    history, so an order change before then is comparing invalid estimates.
    """
    solver = BDF(bdf_order=None, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y0 = jnp.zeros(3)
    state = solver.init(term, 0.0, 1e-4, y0, B)
    # Order 3, but only two equal steps behind us: the gate must hold it still.
    state = eqx.tree_at(
        lambda s: (s.order, s.n_equal_steps),
        state,
        (jnp.asarray(3, jnp.int32), jnp.asarray(2, jnp.int32)),
    )
    _, _, _, held, _ = solver.step(term, 0.0, 1e-4, y0, B, state, False)
    assert int(held.order) == 3

    opened = eqx.tree_at(lambda s: s.n_equal_steps, state, jnp.asarray(9, jnp.int32))
    _, _, _, moved, _ = solver.step(term, 0.0, 1e-4, y0, B, opened, False)
    assert int(moved.order) != 3, "gate is open, so the heuristic should act"


# --------------------------------------------------------------------------
# The order channel
# --------------------------------------------------------------------------


def test_variable_order_reports_the_order_it_used():
    solver = BDF(bdf_order=None, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y0 = jnp.zeros(3)
    state = solver.init(term, 0.0, 1e-4, y0, B)
    state = eqx.tree_at(lambda s: s.order, state, jnp.asarray(2, jnp.int32))
    _, y_error, _, new_state, _ = solver.step(term, 0.0, 1e-4, y0, B, state, False)
    assert isinstance(y_error, ErrorWithOrder)
    # The order that produced the estimate, not the one chosen for the next step.
    assert int(y_error.order) == 2
    assert jax.tree.structure(y_error.error) == jax.tree.structure(y0)


def test_fixed_order_reports_a_plain_error():
    """Fixed order must stay usable with any adaptive controller."""
    solver = BDF(bdf_order=3, root_finder=TOLS)
    term = diffrax.ODETerm(stiff_linear_vf)
    y0 = jnp.zeros(3)
    state = solver.init(term, 0.0, 1e-4, y0, B)
    _, y_error, _, _, _ = solver.step(term, 0.0, 1e-4, y0, B, state, False)
    assert not isinstance(y_error, ErrorWithOrder)
    assert jax.tree.structure(y_error) == jax.tree.structure(y0)


def test_controller_uses_the_reported_order():
    """The step-size exponent must follow the reported order."""
    controller = BDFController(rtol=1e-6, atol=1e-9)
    y0 = y1 = jnp.ones(3)
    error = 1e-7 * jnp.ones(3)
    term = diffrax.ODETerm(stiff_linear_vf)
    _, state = controller.init(term, 0.0, 0.1, y0, 0.1, B, lambda *a: y0, 6)

    def dt_for(order):
        wrapped = ErrorWithOrder(error=error, order=jnp.asarray(float(order)))
        _, t0, t1, _, _, _ = controller.adapt_step_size(
            0.0, 0.1, y0, y1, B, wrapped, 99.0, state
        )
        return float(t1 - t0)

    # A low-order method has a large exponent 1/(p+1), so it grows more aggressively.
    assert dt_for(1) > dt_for(4)
    # And the `error_order` argument diffrax passes in is ignored in favour of ours.
    plain = controller.adapt_step_size(0.0, 0.1, y0, y1, B, error, 2.0, state)
    assert float(plain[2] - plain[1]) == pytest.approx(dt_for(1))


def test_variable_order_rejects_a_plain_pid_controller():
    """A clear failure is better than a silently wrong exponent."""
    with pytest.raises(TypeError, match="BDFController"):
        _steady_state(
            BDF(bdf_order=None),
            diffrax.PIDController(rtol=1e-8, atol=1e-10, dtmax=1e6),
        )


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_variable_order_agrees_with_fixed_order():
    variable = _steady_state(BDF(bdf_order=None))
    fixed = _steady_state(BDF(bdf_order=5))
    assert variable.result == diffrax.RESULTS.event_occurred
    np.testing.assert_allclose(variable.ys[0], fixed.ys[0], rtol=1e-6, atol=1e-9)


def test_variable_order_is_not_slower_in_steps():
    variable = _steady_state(BDF(bdf_order=None))
    for order in (2, 3, 4, 5):
        fixed = _steady_state(BDF(bdf_order=order))
        assert int(variable.stats["num_steps"]) <= int(fixed.stats["num_steps"])


def test_variable_order_robertson_matches_scipy():
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
        BDF(bdf_order=None),
        t0=0.0,
        t1=t1,
        dt0=1e-8,
        y0=ROBERTSON_Y0,
        args=ROBERTSON_ARGS,
        stepsize_controller=BDFController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=200_000,
    )
    np.testing.assert_allclose(sol.ys[0], reference.y[:, -1], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(float(jnp.sum(sol.ys[0])), 1.0, rtol=0, atol=1e-9)


def test_variable_order_gradients_are_exact():
    def steady(b):
        return _steady_state(
            BDF(bdf_order=None), vf=stiff_linear_vf, y0=jnp.zeros(3), args=b
        ).ys[0]

    np.testing.assert_allclose(
        jax.jacrev(steady)(B),
        jax.jacrev(stiff_linear_steady_state)(B),
        rtol=1e-5,
        atol=1e-9,
    )


def test_variable_order_is_jittable_and_vmappable():
    run = jax.jit(
        lambda b: _steady_state(
            BDF(bdf_order=None), vf=stiff_linear_vf, y0=jnp.zeros(3), args=b
        ).ys[0]
    )
    np.testing.assert_allclose(run(B), stiff_linear_steady_state(B), rtol=1e-6, atol=1e-9)
    bs = jnp.stack([B, 2 * B, 0.5 * B])
    np.testing.assert_allclose(
        jax.vmap(run)(bs), jax.vmap(stiff_linear_steady_state)(bs), rtol=1e-6, atol=1e-9
    )
